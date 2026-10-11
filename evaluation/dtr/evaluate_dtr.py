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


def query_hits(index: DTRIndex, ranking: np.ndarray, gold_keys) -> np.ndarray:
    """Mark, per query and rank, whether the retrieved table is a gold one.

    Args:
        index: The searched index, used to map positions back to recall keys.
        ranking: Corpus positions per query, best first, shape (Q, top_k).
        gold_keys: Per query, either the one gold recall key or a collection of
            them. NQ-Tables has questions with two gold tables; KaggleDS has one
            per query and passes bare strings.

    Returns:
        Boolean array of shape (Q, top_k).
    """
    keys = np.array(index.recall_keys, dtype=object)
    retrieved = keys[ranking]  # (Q, top_k)

    # A bare string is the one-gold case and gives exactly the same answer as a
    # one-element set.
    return np.array(
        [
            [cell == gold if isinstance(gold, str) else cell in gold for cell in row]
            for row, gold in zip(retrieved, gold_keys)
        ],
        dtype=bool,
    ).reshape(len(retrieved), -1)


def recall_from_hits(hits: np.ndarray, ks=DEFAULT_RECALL_AT):
    """Fraction of queries with a hit anywhere in the first k, per cutoff."""
    return {k: float(hits[:, :k].any(axis=1).mean()) for k in ks}


def recall_at_k(index: DTRIndex, ranking: np.ndarray, gold_keys, ks=DEFAULT_RECALL_AT):
    """Fraction of queries whose gold table appears in the top k.

    Args:
        index: The searched index, used to map positions back to recall keys.
        ranking: Corpus positions per query, best first, shape (Q, >=max(ks)).
        gold_keys: Per query, the gold recall key or a collection of them.
        ks: Cutoffs to report.

    Returns:
        Mapping of k to recall.
    """
    return recall_from_hits(query_hits(index, ranking, gold_keys), ks=ks)


def recall_by_partition(hits: np.ndarray, labels, ks=DEFAULT_RECALL_AT):
    """Recall@k computed separately for each partition of the queries.

    Args:
        hits: Output of query_hits, shape (Q, top_k).
        labels: One partition label per query, in the same order as hits.
        ks: Cutoffs to report.

    Returns:
        Mapping of label to {"queries": count, "recall": {k: recall}}, with
        labels in sorted order.
    """
    labels = np.asarray(labels, dtype=object)
    if len(labels) != len(hits):
        raise ValueError(f"{len(labels)} labels for {len(hits)} queries")
    return {
        label: {
            "queries": int((labels == label).sum()),
            "recall": recall_from_hits(hits[labels == label], ks=ks),
        }
        for label in sorted(set(labels))
    }


PARTITION_SCHEMES = ("dataset", "name")


def table_partitions(tables, labeled_rows, scheme="dataset"):
    """Assign each corpus table a partition label.

    Args:
        tables: The split's TableRecords.
        labeled_rows: (database_name, table_name, label) triples, one per table.
        scheme: "dataset" matches a table on its dataset and table name, which
            is exact. "name" matches on the bare table name alone with the last
            row winning, which is what table_count_eval/table_count_evaluator.py
            does and therefore reproduces the partition counts in the paper.
            Under "name", every table sharing a file name (train.csv, data.csv,
            ...) takes whichever label was read last.

    Returns:
        Mapping of table_id to label.
    """
    if scheme == "dataset":
        lookup = {(db.strip(), name.strip()): label for db, name, label in labeled_rows}
        key_of = lambda table: (table.group_id, table.recall_key)
    elif scheme == "name":
        lookup = {}
        for _, name, label in labeled_rows:
            lookup[name.strip()] = label
        key_of = lambda table: table.recall_key
    else:
        raise ValueError(f"unknown partition scheme {scheme!r}; expected {PARTITION_SCHEMES}")

    missing = [t.table_id for t in tables if key_of(t) not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} tables have no partition label, e.g. {missing[0]!r}")
    return {t.table_id: lookup[key_of(t)] for t in tables}


def query_partitions(gold_ids, partition_of):
    """Label each query with the partition of its gold table(s).

    Raises if a question's gold tables fall in different partitions, since the
    question then has no single label.
    """
    labels = []
    for ids in gold_ids:
        found = {partition_of[table_id] for table_id in ids}
        if len(found) != 1:
            raise ValueError(f"gold tables {ids} span partitions {sorted(found)}")
        labels.append(found.pop())
    return labels


def load_table_count_rows(split):
    """Read (database_name, table_name, table_count_type) for a KaggleDS split.

    The labels come from the eval_data_<split> Postgres table populated by
    table_count_eval/add_table_count_column.py, the same source the HySE
    single- vs multi-table breakdown used. Rows are read in storage order, with
    no ORDER BY, to match how that evaluator resolved duplicate table names.
    """
    # Imported here so the evaluator needs no database unless partitions are asked for.
    from dataforager.db.connect_db import DatabaseConnection

    with DatabaseConnection() as db:
        db.cursor.execute(
            f"SELECT database_name, table_name, table_count_type FROM eval_data_{split} "
            "WHERE table_count_type IS NOT NULL"
        )
        rows = db.cursor.fetchall()
    return [(r["database_name"] or "", r["table_name"] or "", r["table_count_type"]) for r in rows]


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
    if corpus in ("nq", "nq_full"):
        if not data_dir:
            raise ValueError(f"--data-dir is required for the {corpus} corpus")
        # Imported here because evaluation.nq.adapters imports this package,
        # and a module-level import would close the cycle.
        from evaluation.nq.adapters import NQTablesAdapter, NQTablesFullAdapter

        if corpus == "nq":
            return NQTablesAdapter(data_dir)
        return NQTablesFullAdapter(data_dir)
    raise ValueError(
        f"unknown corpus {corpus!r}; expected kaggleds, nq or nq_full"
    )


def evaluate(checkpoint, split="test", csv_dir=None, device=None, batch_size=32,
             ks=DEFAULT_RECALL_AT, limit=None, index_path=None, corpus="kaggleds",
             data_dir=None, partition_by=None):
    """Build (or reuse) an index for a split and report Recall@k.

    Args:
        partition_by: None, or a scheme from PARTITION_SCHEMES to additionally
            report recall for single- vs multi-table KaggleDS queries.

    Returns:
        Recall per cutoff, the number of queries, the number of tables, and the
        per-partition recall (None unless partition_by is given).
    """
    if partition_by and corpus != "kaggleds":
        raise ValueError("--partition-by is only defined for the kaggleds corpus")
    device = resolve_device(device)
    model, tokenizer = load_dual_encoder(checkpoint, device=device)

    adapter = build_adapter(corpus, csv_dir=csv_dir, data_dir=data_dir)

    # Corpora large enough to matter expose iter_records, so the encoder can be
    # fed a batch at a time rather than from a fully materialized list.
    streaming = hasattr(adapter, "iter_records") and hasattr(adapter, "examples")
    if streaming:
        examples = adapter.examples(split)
        tables = None
    else:
        tables, examples = adapter.load(split)
    if limit:
        examples = examples[:limit]

    if index_path and Path(index_path).exists():
        index = DTRIndex.load(index_path)
        print(f"Loaded index from {index_path}")
    elif streaming:
        index = DTRIndex.build_streaming(
            model, tokenizer, adapter.iter_records(), device, batch_size=batch_size
        )
        if index_path:
            index.save(index_path)
            print(f"Saved index to {index_path}")
    else:
        index = DTRIndex.build(model, tokenizer, tables, device, batch_size=batch_size)
        if index_path:
            index.save(index_path)
            print(f"Saved index to {index_path}")

    # For a streamed corpus the table id is the recall key, so the mapping comes
    # from the index rather than from records we no longer hold.
    by_id = (dict(zip(index.table_ids, index.recall_keys)) if streaming
             else {t.table_id: t.recall_key for t in tables})
    queries, gold_ids = group_by_question(examples)
    gold_keys = [{by_id[table_id] for table_id in ids} for ids in gold_ids]

    query_embeddings = encode_query_batch(
        model, tokenizer, queries, device,
        batch_size=batch_size, show_progress=True,
    )
    ranking = index.search(query_embeddings, top_k=max(ks))
    n_tables = len(index.table_ids)
    hits = query_hits(index, ranking, gold_keys)

    by_partition = None
    if partition_by:
        partition_of = table_partitions(tables, load_table_count_rows(split), scheme=partition_by)
        by_partition = recall_by_partition(hits, query_partitions(gold_ids, partition_of), ks=ks)
    return recall_from_hits(hits, ks=ks), len(queries), n_tables, by_partition


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, help="Converted checkpoint or run directory")
    parser.add_argument("--split", default="test")
    parser.add_argument("--corpus", default="kaggleds", choices=("kaggleds", "nq", "nq_full"))
    parser.add_argument("--csv-dir", default=None, help="Local split CSVs; omit to use the Hub")
    parser.add_argument("--data-dir", default=None, help="OpenTI release directory, for --corpus nq")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N queries")
    parser.add_argument("--index-path", default=None, help="Cache the table embeddings here")
    parser.add_argument("--output", default=None, help="Write the metrics to this JSON file")
    parser.add_argument(
        "--partition-by", default=None, choices=PARTITION_SCHEMES,
        help="Also report single- vs multi-table recall (kaggleds only; reads labels from Postgres)",
    )
    args = parser.parse_args()

    recalls, n_queries, n_tables, by_partition = evaluate(
        checkpoint=args.checkpoint, split=args.split, csv_dir=args.csv_dir,
        device=args.device, batch_size=args.batch_size, limit=args.limit,
        index_path=args.index_path, corpus=args.corpus, data_dir=args.data_dir,
        partition_by=args.partition_by,
    )

    print(f"\n{args.checkpoint}  corpus={args.corpus}  split={args.split}  "
          f"queries={n_queries}  tables={n_tables}")
    for k, value in recalls.items():
        print(f"  R@{k:<3} {value:.4f}")
    for label, part in (by_partition or {}).items():
        print(f"\n  {label} ({args.partition_by} labels, queries={part['queries']})")
        for k, value in part["recall"].items():
            print(f"    R@{k:<3} {value:.4f}")

    if args.output:
        payload = {"checkpoint": args.checkpoint, "corpus": args.corpus,
                   "split": args.split, "queries": n_queries, "tables": n_tables,
                   "recall": {str(k): v for k, v in recalls.items()}}
        if by_partition:
            payload["partition_by"] = args.partition_by
            payload["by_partition"] = {
                label: {"queries": part["queries"],
                        "recall": {str(k): v for k, v in part["recall"].items()}}
                for label, part in by_partition.items()
            }
        Path(args.output).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()