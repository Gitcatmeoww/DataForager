"""Run HySE over NQ-Tables as a transfer test.

The paper's operational HySE, unchanged: N=2 relational hypothetical schemas per
query from the same prompt and the same model the KaggleDS runs used, fused with
the query embedding at equal weight, searched by cosine similarity.

    e = 0.5 * e(query) + 0.5 * mean_i e(schema_i)

The prompt is deliberately *not* adapted to NQ. NQ questions are factoid rather
than analytical, so asking for "a schema to implement the task of who has the
most water in the world" is a strange request. That strangeness is the
experiment: the claim under test is that schema-shaped expansion transfers
without per-corpus tuning, and rewriting the prompt would concede the point.

Text representations match the KaggleDS harness exactly, which was verified
against a stored corpus vector (cosine 0.995):

    corpus table     f"{table_name} {header and two rows as markdown}"
    HySE component   f"{hypo table_name} {hypo two rows as markdown}"

For NQ the Wikipedia page title plays the table name. Everything runs in memory
rather than through pgvector, so no NQ data is written to Postgres.

Stages are cached separately and are resumable, since generation is the only
expensive step:

    python -m evaluation.nq.hyse_nq --data-dir <openti release> --split dev
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.dtr.evaluate_dtr import DEFAULT_RECALL_AT, group_by_question
from evaluation.nq.adapters import NQTablesAdapter, NQTablesFullAdapter
from evaluation.significance import bootstrap_recall, compare_recall, hits_at_k

# Queries are scored against the full corpus in blocks, because the score matrix
# for 169,898 tables by a thousand queries does not want to exist all at once.
SEARCH_BLOCK = 128

# The paper's operational configuration.
DEFAULT_NUM_SCHEMAS = 2
DEFAULT_QUERY_WEIGHT = 0.5
DEFAULT_SCHEMA_APPROACH = "relational"

DEFAULT_WORKERS = 8

# text-embedding-3-small rejects anything longer. Two of the 8,205 NQ tables
# exceed it, both from very long cell contents rather than column count, so the
# tail is trimmed rather than the table narrowed. At 0.02% of the corpus this
# cannot move a recall number, but it must be deterministic to stay
# reproducible.
EMBEDDING_TOKEN_LIMIT = 8192
EMBEDDING_TOKEN_MARGIN = 192

# The embeddings endpoint accepts a list per request, which is what keeps the
# corpus run under the 5,000 requests per minute limit: 169,898 single-text
# calls exceed it even at modest concurrency, while batching needs a few
# hundred requests. Both caps are well inside the endpoint's own limits.
EMBED_BATCH_ITEMS = 256
EMBED_BATCH_TOKENS = 120_000

# Tokens per minute the embeddings endpoint allows, less a margin. Batching
# alone is not enough: the corpus is roughly 33M tokens, and eight concurrent
# 120k-token requests offer them at around 29M per minute. Requests are paced
# against this budget instead of relying on retries, which only absorb bursts.
EMBED_TOKENS_PER_MINUTE = 4_500_000

# Embedding the corpus takes minutes and had been losing all of its work on any
# failure, so progress is checkpointed this often (in requests).
EMBED_CHECKPOINT_EVERY = 25

# USD per million tokens. gpt-4o-mini is the harness default and therefore what
# the KaggleDS HySE schemas were generated with; keeping it makes the transfer
# numbers comparable to the paper's.
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


@dataclass
class Meter:
    """Wall time, call counts, and tokens for one stage, for the cost table."""

    name: str
    seconds: float = 0.0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    truncated: int = 0
    latencies: list = field(default_factory=list)

    def record(self, seconds, prompt_tokens=0, completion_tokens=0):
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.latencies.append(seconds)

    @property
    def cost(self) -> float:
        rates = PRICING.get(self.model)
        if not rates:
            return 0.0
        return (self.prompt_tokens * rates["input"]
                + self.completion_tokens * rates["output"]) / 1_000_000

    def summary(self) -> dict:
        latencies = np.array(self.latencies) if self.latencies else np.zeros(1)
        return {
            "stage": self.name,
            "model": self.model,
            "wall_seconds": round(self.seconds, 2),
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "usd": round(self.cost, 4),
            "truncated": self.truncated,
            "p50_call_seconds": round(float(np.percentile(latencies, 50)), 4),
            "p95_call_seconds": round(float(np.percentile(latencies, 95)), 4),
        }


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def truncate_to_tokens(text: str, limit=EMBEDDING_TOKEN_LIMIT - EMBEDDING_TOKEN_MARGIN):
    """Trim text to a token budget the embedding endpoint will accept.

    Args:
        text: The text to embed.
        limit: Maximum tokens to keep.

    Returns:
        The text and whether it had to be trimmed.
    """
    encoding = _encoding()
    tokens = encoding.encode(text)
    if len(tokens) <= limit:
        return text, False
    return encoding.decode(tokens[:limit]), True


def to_markdown(frame: pd.DataFrame) -> str:
    """Render a table the way the KaggleDS corpus stores example_2rows_md.

    Rows are read positionally rather than through itertuples, which builds a
    namedtuple from the column names and so applies Python identifier rules to
    them. One NQ table is headed with U+212A KELVIN SIGN alongside an ASCII K;
    those are distinct strings, but identifiers are NFKC-normalized, so
    namedtuple collapses them and rejects the table as having duplicate fields.
    """
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in frame.to_numpy(dtype=object)
    ]
    return "\n".join([header, rule] + rows)


def corpus_text(record) -> str:
    """The text embedded for one corpus table."""
    return f"{record.title} {to_markdown(record.table)}"


def top_k_by_cosine(queries: np.ndarray, corpus: np.ndarray, k: int) -> np.ndarray:
    """Rank the corpus for each query, best first.

    argpartition finds the top k without sorting the rest, which matters at
    169,898 tables; only the k survivors are then ordered.

    Args:
        queries: (Q, dim) query vectors.
        corpus: (N, dim) corpus vectors.
        k: How many to keep per query.

    Returns:
        (Q, k) corpus positions, best first.
    """
    corpus_normed = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    k = min(k, corpus.shape[0])
    out = np.empty((queries.shape[0], k), dtype=np.int64)

    for start in range(0, queries.shape[0], SEARCH_BLOCK):
        block = queries[start : start + SEARCH_BLOCK]
        block = block / np.linalg.norm(block, axis=1, keepdims=True)
        scores = block @ corpus_normed.T
        candidates = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        ordered = np.take_along_axis(
            candidates,
            np.argsort(-np.take_along_axis(scores, candidates, axis=1), axis=1),
            axis=1,
        )
        out[start : start + len(block)] = ordered
    return out


def _client():
    from dataforager.table_representation.openai_client import OpenAIClient

    return OpenAIClient()


@dataclass(frozen=True)
class Batch:
    """One embedding request: the texts it carries and their token cost."""

    items: list
    tokens: int

    def __len__(self):
        return len(self.items)


class TokenPacer:
    """Paces requests against a tokens-per-minute budget.

    The endpoint enforces TPM as well as RPM, and retries only absorb bursts:
    a corpus run offers tokens faster than the budget for minutes on end, so
    the 429s never stop coming. This hands out permits at the allowed rate.
    """

    def __init__(self, tokens_per_minute=EMBED_TOKENS_PER_MINUTE):
        self.capacity = float(tokens_per_minute)
        self.available = float(tokens_per_minute)
        self.rate = tokens_per_minute / 60.0
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, tokens: int) -> None:
        """Block until `tokens` may be sent."""
        while True:
            with self.lock:
                now = time.monotonic()
                self.available = min(
                    self.capacity, self.available + (now - self.updated) * self.rate
                )
                self.updated = now
                # A request larger than the whole budget can never fit, so let
                # it through and let the endpoint's own retry handle it.
                if self.available >= tokens or tokens > self.capacity:
                    self.available -= tokens
                    return
                wait = (tokens - self.available) / self.rate
            time.sleep(min(wait, 5.0))


def plan_batches(texts, max_items=EMBED_BATCH_ITEMS, max_tokens=EMBED_BATCH_TOKENS):
    """Group texts into embedding requests, truncating as needed.

    One request per table would need 169,898 of them, well past the 5,000
    requests per minute the embeddings endpoint allows. The endpoint accepts a
    list, so batching by both item count and token budget turns the corpus into
    a few hundred requests and removes the rate limit as a constraint.

    Args:
        texts: Texts to embed, in order.
        max_items: Most texts per request.
        max_tokens: Most tokens per request.

    Returns:
        Batches of (position, text) pairs, and how many texts were truncated.
    """
    encoding = _encoding()
    batches, batch, tokens, truncated = [], [], 0, 0

    for position, text in enumerate(texts):
        text, trimmed = truncate_to_tokens(text)
        truncated += int(trimmed)
        size = len(encoding.encode(text))
        if batch and (len(batch) >= max_items or tokens + size > max_tokens):
            batches.append(Batch(batch, tokens))
            batch, tokens = [], 0
        batch.append((position, text))
        tokens += size

    if batch:
        batches.append(Batch(batch, tokens))
    return batches, truncated


def embed_many(texts, meter: Meter, workers=DEFAULT_WORKERS, model="text-embedding-3-small",
               checkpoint_path=None):
    """Embed texts in paced, batched, concurrent requests, recording usage.

    The shared helper embeds one text per call and discards token usage, so the
    endpoint is called directly here: the cost accounting needs the usage, and
    the corpus run needs batching, pacing, and checkpointing.

    Args:
        texts: Texts to embed, in order.
        meter: Usage accumulator.
        workers: Concurrent requests.
        model: Embedding model.
        checkpoint_path: When given, partial results are saved here and reused
            on a later run, so a rate limit or a crash does not discard minutes
            of completed work.

    Returns:
        A (len(texts), dim) float32 array.
    """
    client = _client()
    meter.model = model

    batches, truncated = plan_batches(texts)
    meter.truncated += truncated
    if truncated:
        print(f"  {truncated} text(s) trimmed to the embedding token limit")

    vectors = None
    done = np.zeros(len(texts), dtype=bool)
    if checkpoint_path and Path(checkpoint_path).exists():
        blob = np.load(checkpoint_path)
        vectors, done = blob["vectors"], blob["done"]
        print(f"  resuming: {int(done.sum())}/{len(texts)} already embedded")

    pending = [b for b in batches if not all(done[p] for p, _ in b.items)]
    if len(batches) > 1:
        print(f"  {len(texts)} texts in {len(batches)} batched requests"
              f"{f', {len(pending)} outstanding' if len(pending) != len(batches) else ''}")
    if not pending:
        return vectors

    pacer = TokenPacer()
    lock = threading.Lock()

    def one(batch):
        pacer.acquire(batch.tokens)
        started = time.time()
        response = client.client.embeddings.create(
            model=model, input=[text.replace("\n", " ") for _, text in batch.items]
        )
        usage = getattr(response, "usage", None)
        return (
            [position for position, _ in batch.items],
            [np.array(item.embedding, dtype=np.float32) for item in response.data],
            time.time() - started,
            getattr(usage, "prompt_tokens", 0) or 0,
        )

    def save():
        """Checkpoint progress, never failing the run if that is not possible."""
        if checkpoint_path is None or vectors is None:
            return
        try:
            # savez appends .npz to a path that lacks it, so write through a
            # handle rather than letting it rename the temp file out from under
            # the replace below.
            tmp = Path(f"{checkpoint_path}.tmp")
            with open(tmp, "wb") as handle:
                np.savez(handle, vectors=vectors, done=done)
            tmp.replace(checkpoint_path)
        except OSError as error:
            print(f"  checkpoint failed, continuing: {error}")

    started = time.time()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for completed, (positions, embeddings, seconds, tokens) in enumerate(
                pool.map(one, pending), start=1
            ):
                with lock:
                    if vectors is None:
                        vectors = np.zeros((len(texts), embeddings[0].shape[0]), dtype=np.float32)
                    for position, vector in zip(positions, embeddings):
                        vectors[position] = vector
                        done[position] = True
                    meter.record(seconds, prompt_tokens=tokens)
                if completed % EMBED_CHECKPOINT_EVERY == 0:
                    save()
                    print(f"  embedded {int(done.sum())}/{len(texts)}")
    finally:
        meter.seconds += time.time() - started
        save()

    missing = int((~done).sum())
    if missing:
        raise RuntimeError(f"{missing} text(s) never embedded")
    return vectors


def generate_schemas(queries, cache_path: Path, meter: Meter,
                     num_schemas=DEFAULT_NUM_SCHEMAS,
                     schema_approach=DEFAULT_SCHEMA_APPROACH,
                     workers=DEFAULT_WORKERS):
    """Generate hypothetical schemas, resuming from a jsonl cache.

    Args:
        queries: Distinct query strings.
        cache_path: Append-only jsonl of {query, slot, table_name, markdown}.
        meter: Usage accumulator.
        num_schemas: Schemas per query.
        schema_approach: relational or non_relational.
        workers: Concurrent generations.

    Returns:
        Mapping of query to a list of (table_name, markdown) pairs.
    """
    from dataforager.hyse.hypo_schema_search import (
        infer_single_hypothetical_schema_with_examples,
    )
    from dataforager.table_representation.openai_client import OpenAIClient

    meter.model = OpenAIClient().text_generation_model_default

    cached: dict[str, dict[int, tuple]] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                cached.setdefault(row["query"], {})[row["slot"]] = (
                    row["table_name"], row["markdown"]
                )

    todo = [
        (query, slot)
        for query in queries
        for slot in range(num_schemas)
        if slot not in cached.get(query, {})
    ]
    print(f"schemas: {sum(len(v) for v in cached.values())} cached, {len(todo)} to generate")

    def one(job):
        query, slot = job
        started = time.time()
        result, usage = infer_single_hypothetical_schema_with_examples(
            initial_query=query, schema_approach=schema_approach, return_usage=True
        )
        return query, slot, result, time.time() - started, usage

    started = time.time()
    if todo:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a", encoding="utf-8") as handle, \
                ThreadPoolExecutor(max_workers=workers) as pool:
            for done, (query, slot, result, seconds, usage) in enumerate(
                pool.map(one, todo), start=1
            ):
                meter.record(seconds, usage["prompt_tokens"], usage["completion_tokens"])
                if result is None:
                    continue
                entry = (result.table_name, result.get_example_2rows_markdown())
                cached.setdefault(query, {})[slot] = entry
                handle.write(json.dumps({
                    "query": query, "slot": slot,
                    "table_name": entry[0], "markdown": entry[1],
                }) + "\n")
                if done % 200 == 0:
                    handle.flush()
                    print(f"  generated {done}/{len(todo)}")
    meter.seconds += time.time() - started

    return {query: [slots[s] for s in sorted(slots)] for query, slots in cached.items()}


def load_cached_embeddings(path: Path, expected_rows: int):
    """Read a cached embedding matrix, rejecting one of the wrong length.

    Caches go stale in a way that is easy to miss: a generation failure leaves
    one fewer schema component, the next run regenerates it, and the previously
    saved matrix is then one row short of what the caller will index into. That
    surfaces far from its cause, so the shape is checked on load instead.

    Args:
        path: Cache file.
        expected_rows: Rows the caller expects.

    Returns:
        The array, or None when absent or stale.
    """
    if not path.exists():
        return None
    cached = np.load(path)
    if cached.shape[0] != expected_rows:
        print(f"  {path.name} holds {cached.shape[0]} rows, expected "
              f"{expected_rows}; recomputing")
        return None
    return cached


def load_corpus_texts(data_dir, corpus):
    """Corpus recall keys and the text to embed for each, in index order.

    The full corpus is streamed so that only one table's DataFrame is alive at
    a time; materializing 169,898 of them costs gigabytes.
    """
    if corpus == "full":
        adapter = NQTablesFullAdapter(data_dir)
        keys, texts = [], []
        for record in adapter.iter_records():
            keys.append(record.recall_key)
            texts.append(corpus_text(record))
        return np.array(keys, dtype=object), texts, adapter

    adapter = NQTablesAdapter(data_dir)
    tables = adapter._load_corpus()
    return (
        np.array([t.recall_key for t in tables], dtype=object),
        [corpus_text(t) for t in tables],
        adapter,
    )


def run(data_dir, split="dev", cache_dir=None, num_schemas=DEFAULT_NUM_SCHEMAS,
        query_weight=DEFAULT_QUERY_WEIGHT, schema_approach=DEFAULT_SCHEMA_APPROACH,
        ks=DEFAULT_RECALL_AT, workers=DEFAULT_WORKERS, limit=None, corpus="gold"):
    """Evaluate HySE and the query-only baseline over one NQ split."""
    cache_dir = Path(cache_dir or Path(data_dir) / "hyse_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    recall_keys, corpus_texts, adapter = load_corpus_texts(data_dir, corpus)
    if corpus == "full":
        examples = adapter.examples(split)
    else:
        _, examples = adapter.load(split)

    queries, gold_ids = group_by_question(examples)
    if limit:
        queries, gold_ids = queries[:limit], gold_ids[:limit]
    # For both corpora the table id and the recall key coincide or map through
    # the corpus, so gold ids are already recall keys for the full corpus.
    if corpus == "full":
        gold_keys = [set(ids) for ids in gold_ids]
    else:
        tables = adapter._load_corpus()
        by_id = {t.table_id: t.recall_key for t in tables}
        gold_keys = [{by_id[i] for i in ids} for ids in gold_ids]

    known = set(recall_keys.tolist())
    unknown = {g for gold in gold_keys for g in gold} - known
    if unknown:
        raise ValueError(
            f"{len(unknown)} gold table(s) absent from the {corpus} corpus, "
            f"for example {sorted(unknown)[:3]}"
        )

    meters = {
        "corpus_embed": Meter("corpus_embed"),
        "query_embed": Meter("query_embed"),
        "schema_generate": Meter("schema_generate"),
        "schema_embed": Meter("schema_embed"),
        "search": Meter("search"),
    }

    # Corpus embeddings are split-independent, so one cache serves every split.
    corpus_path = cache_dir / f"corpus_embeddings_{corpus}.npy"
    corpus_vectors = load_cached_embeddings(corpus_path, len(corpus_texts))
    if corpus_vectors is not None:
        print(f"corpus: loaded {corpus_vectors.shape} from cache")
    else:
        print(f"corpus: embedding {len(corpus_texts)} tables")
        corpus_vectors = embed_many(
            corpus_texts, meters["corpus_embed"], workers,
            checkpoint_path=cache_dir / f"corpus_embeddings_{corpus}.partial.npz",
        )
        np.save(corpus_path, corpus_vectors)

    query_path = cache_dir / f"query_embeddings_{split}.npy"
    query_vectors = None if limit else load_cached_embeddings(query_path, len(queries))
    if query_vectors is None:
        query_vectors = embed_many(queries, meters["query_embed"], workers)
        if not limit:
            np.save(query_path, query_vectors)

    schemas = generate_schemas(
        sorted(set(queries)), cache_dir / f"schemas_{split}_{schema_approach}.jsonl",
        meters["schema_generate"], num_schemas, schema_approach, workers,
    )

    # Embed each distinct schema once, then gather per query.
    texts, index = [], {}
    for query in queries:
        for slot, (name, markdown) in enumerate(schemas.get(query, [])):
            index[(query, slot)] = len(texts)
            texts.append(f"{name} {markdown}")
    schema_path = cache_dir / f"schema_embeddings_{split}_{schema_approach}.npy"
    schema_vectors = None if limit else load_cached_embeddings(schema_path, len(texts))
    if schema_vectors is None:
        print(f"schemas: embedding {len(texts)} components")
        schema_vectors = embed_many(texts, meters["schema_embed"], workers)
        if not limit:
            np.save(schema_path, schema_vectors)

    fused = np.zeros_like(query_vectors)
    missing = 0
    for position, query in enumerate(queries):
        slots = [index[(query, s)] for s in range(len(schemas.get(query, [])))]
        if not slots:
            missing += 1
            fused[position] = query_vectors[position]
            continue
        hypo = schema_vectors[slots].mean(axis=0)
        fused[position] = query_weight * query_vectors[position] + (1 - query_weight) * hypo

    def search(vectors):
        started = time.time()
        top = top_k_by_cosine(vectors, corpus_vectors, max(ks))
        elapsed = time.time() - started
        meters["search"].record(elapsed)
        meters["search"].seconds += elapsed
        return recall_keys[top]

    retrieved = {"semantic": search(query_vectors), "hyse": search(fused)}

    results = {}
    for method, keys in retrieved.items():
        results[method] = {
            k: bootstrap_recall(hits_at_k(keys, gold_keys, k)) for k in ks
        }
    comparisons = {
        k: compare_recall(
            hits_at_k(retrieved["semantic"], gold_keys, k),
            hits_at_k(retrieved["hyse"], gold_keys, k),
        )
        for k in ks
    }

    return {
        "split": split,
        "corpus": corpus,
        "corpus_tables": len(corpus_texts),
        "queries": len(queries),
        "num_schemas": num_schemas,
        "query_weight": query_weight,
        "schema_approach": schema_approach,
        "uncached_queries": missing,
        "recall": results,
        "comparisons": comparisons,
        "cost": [m.summary() for m in meters.values() if m.calls],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--num-schemas", type=int, default=DEFAULT_NUM_SCHEMAS)
    parser.add_argument("--query-weight", type=float, default=DEFAULT_QUERY_WEIGHT)
    parser.add_argument("--schema-approach", default=DEFAULT_SCHEMA_APPROACH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--corpus", default="gold", choices=("gold", "full"),
                        help="gold = 8,205 gold-only tables; full = all 169,898")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    out = run(
        data_dir=args.data_dir, split=args.split, cache_dir=args.cache_dir,
        num_schemas=args.num_schemas, query_weight=args.query_weight,
        schema_approach=args.schema_approach, workers=args.workers, limit=args.limit,
        corpus=args.corpus,
    )

    print(f"\nNQ-Tables  split={out['split']}  corpus={out['corpus']}  "
          f"queries={out['queries']}  "
          f"tables={out['corpus_tables']}  N={out['num_schemas']}  "
          f"lambda={1 - out['query_weight']}  uncached={out['uncached_queries']}")
    print(f"\n{'k':>4}  {'semantic':>26}  {'HySE':>26}  {'difference':>28}")
    for k in DEFAULT_RECALL_AT:
        comparison = out["comparisons"][k]
        print(f"{k:>4}  {str(out['recall']['semantic'][k]):>26}  "
              f"{str(out['recall']['hyse'][k]):>26}  "
              f"{str(comparison.difference):>28}  p={comparison.p_value:.4g}"
              f"{'  *' if comparison.significant else ''}")

    print("\ncost and latency")
    total = 0.0
    for row in out["cost"]:
        total += row["usd"]
        print(f"  {row['stage']:16} {row['model']:24} calls={row['calls']:<6} "
              f"wall={row['wall_seconds']:>7.1f}s p50={row['p50_call_seconds']:.3f}s "
              f"p95={row['p95_call_seconds']:.3f}s  ${row['usd']:.4f}")
    print(f"  {'TOTAL':16} {'':24} {'':13}"
          f"{'':>8}{'':>14}  ${total:.4f}")

    if args.output:
        payload = {
            **{key: out[key] for key in
               ("split", "corpus", "corpus_tables", "queries", "num_schemas", "query_weight",
                "schema_approach", "uncached_queries", "cost")},
            "recall": {
                method: {str(k): {"estimate": v.estimate, "low": v.low, "high": v.high}
                         for k, v in per_k.items()}
                for method, per_k in out["recall"].items()
            },
            "comparisons": {
                str(k): {
                    "baseline": c.baseline, "treatment": c.treatment,
                    "difference": c.difference.estimate,
                    "ci_low": c.difference.low, "ci_high": c.difference.high,
                    "p_value": c.p_value,
                    "wins": c.discordant_wins, "losses": c.discordant_losses,
                }
                for k, c in out["comparisons"].items()
            },
        }
        Path(args.output).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
