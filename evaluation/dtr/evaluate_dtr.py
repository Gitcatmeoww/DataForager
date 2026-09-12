"""Measure Recall@k for a DTR checkpoint over a corpus split.

Recall is scored on recall_key, the bare table name, so that the numbers line up
with the baselines in evaluation/eval_methods.py.

    python -m evaluation.dtr.evaluate_dtr \
        --checkpoint evaluation/dtr/checkpoints/tapas_dual_encoder_proj_256_medium \
        --split test --csv-dir evaluation/huggingface
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.dtr.adapters import KaggleDSAdapter
from evaluation.dtr.index import DTRIndex, encode_query_batch
from evaluation.dtr.modeling import load_dual_encoder, resolve_device

DEFAULT_RECALL_AT = (1, 10, 20, 30, 40, 50)


def recall_at_k(index: DTRIndex, ranking: np.ndarray, gold_keys, ks=DEFAULT_RECALL_AT):
    """Fraction of queries whose gold table appears in the top k.

    Args:
        index: The searched index, used to map positions back to recall keys.
        ranking: Corpus positions per query, best first, shape (Q, >=max(ks)).
        gold_keys: Per query, either the one gold recall key or a collection of
            them. NQ-Tables has questions with two gold tables; KaggleDS has one
            per query and passes bare strings.
        ks: Cutoffs to report.

    Returns:
        Mapping of k to recall.
    """
    keys = np.array(index.recall_keys, dtype=object)
    retrieved = keys[ranking]  # (Q, top_k)

    # A query counts at k if any of its gold keys appears in the first k. A bare
    # string is the one-gold case and gives exactly the same answer as before.
    hits = np.array(
        [
            [cell == gold if isinstance(gold, str) else cell in gold for cell in row]
            for row, gold in zip(retrieved, gold_keys)
        ],
        dtype=bool,
    ).reshape(len(retrieved), -1)

    return {k: float(hits[:, :k].any(axis=1).mean()) for k in ks}


def group_by_question(examples):
    """Collapse examples that belong to one question.

    A question with several gold tables arrives as several examples sharing a
    query_id, and must be scored once against the whole gold set rather than
    once per gold. Examples with no query_id stay separate, which is the
    single-gold case KaggleDS produces.

    Args:
        examples: RetrievalExamples, in order.

    Returns:
        The query text per question and the gold table ids per question, both
        in first-seen order.
    """
    queries, golds, seen = [], [], {}
    for example in examples:
        if example.query_id is None:
            queries.append(example.query)
            golds.append([example.table_id])
            continue
        position = seen.get(example.query_id)
        if position is None:
            seen[example.query_id] = len(queries)
            queries.append(example.query)
            golds.append([example.table_id])
        else:
            golds[position].append(example.table_id)
    return queries, golds


def build_adapter(corpus: str, csv_dir=None, data_dir=None):
    """Construct a corpus adapter by name.

    Args:
        corpus: kaggleds or nq.
        csv_dir: KaggleDS split CSVs; omit to read from the Hub.
        data_dir: The OpenTI release directory, required for nq.

    Returns:
        A CorpusAdapter.
    """
    if corpus == "kaggleds":
        return KaggleDSAdapter(csv_dir=csv_dir)
    if corpus == "nq":
        if not data_dir:
            raise ValueError("--data-dir is required for the nq corpus")
        # Imported here because evaluation.nq.adapters imports this package,
        # and a module-level import would close the cycle.
        from evaluation.nq.adapters import NQTablesAdapter

        return NQTablesAdapter(data_dir)
    raise ValueError(f"unknown corpus {corpus!r}; expected kaggleds or nq")


def evaluate(checkpoint, split="test", csv_dir=None, device=None, batch_size=32,
             ks=DEFAULT_RECALL_AT, limit=None, index_path=None, corpus="kaggleds",
             data_dir=None):
    """Build (or reuse) an index for a split and report Recall@k."""
    device = resolve_device(device)
    model, tokenizer = load_dual_encoder(checkpoint, device=device)

    tables, examples = build_adapter(corpus, csv_dir=csv_dir, data_dir=data_dir).load(split)
    if limit:
        examples = examples[:limit]

    if index_path and Path(index_path).exists():
        index = DTRIndex.load(index_path)
        print(f"Loaded index from {index_path}")
    else:
        index = DTRIndex.build(model, tokenizer, tables, device, batch_size=batch_size)
        if index_path:
            index.save(index_path)
            print(f"Saved index to {index_path}")

    by_id = {t.table_id: t.recall_key for t in tables}
    queries, gold_ids = group_by_question(examples)
    gold_keys = [{by_id[table_id] for table_id in ids} for ids in gold_ids]

    query_embeddings = encode_query_batch(
        model, tokenizer, queries, device,
        batch_size=batch_size, show_progress=True,
    )
    ranking = index.search(query_embeddings, top_k=max(ks))
    return recall_at_k(index, ranking, gold_keys, ks=ks), len(queries), len(tables)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, help="Converted checkpoint or run directory")
    parser.add_argument("--split", default="test")
    parser.add_argument("--corpus", default="kaggleds", choices=("kaggleds", "nq"))
    parser.add_argument("--csv-dir", default=None, help="Local split CSVs; omit to use the Hub")
    parser.add_argument("--data-dir", default=None, help="OpenTI release directory, for --corpus nq")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N queries")
    parser.add_argument("--index-path", default=None, help="Cache the table embeddings here")
    parser.add_argument("--output", default=None, help="Write the metrics to this JSON file")
    args = parser.parse_args()

    recalls, n_queries, n_tables = evaluate(
        checkpoint=args.checkpoint, split=args.split, csv_dir=args.csv_dir,
        device=args.device, batch_size=args.batch_size, limit=args.limit,
        index_path=args.index_path, corpus=args.corpus, data_dir=args.data_dir,
    )

    print(f"\n{args.checkpoint}  corpus={args.corpus}  split={args.split}  "
          f"queries={n_queries}  tables={n_tables}")
    for k, value in recalls.items():
        print(f"  R@{k:<3} {value:.4f}")

    if args.output:
        payload = {"checkpoint": args.checkpoint, "corpus": args.corpus,
                   "split": args.split, "queries": n_queries, "tables": n_tables,
                   "recall": {str(k): v for k, v in recalls.items()}}
        Path(args.output).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()