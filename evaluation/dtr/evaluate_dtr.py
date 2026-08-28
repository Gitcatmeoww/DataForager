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
        gold_keys: The gold recall key for each query.
        ks: Cutoffs to report.

    Returns:
        Mapping of k to recall.
    """
    keys = np.array(index.recall_keys, dtype=object)
    retrieved = keys[ranking]  # (Q, top_k)

    hits = retrieved == np.array(gold_keys, dtype=object)[:, None]
    # A query counts at k if the gold key appears anywhere in the first k.
    return {k: float(hits[:, :k].any(axis=1).mean()) for k in ks}


def evaluate(checkpoint, split="test", csv_dir=None, device=None, batch_size=32,
             ks=DEFAULT_RECALL_AT, limit=None, index_path=None):
    """Build (or reuse) an index for a split and report Recall@k."""
    device = resolve_device(device)
    model, tokenizer = load_dual_encoder(checkpoint, device=device)

    tables, examples = KaggleDSAdapter(csv_dir=csv_dir).load(split)
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
    gold_keys = [by_id[e.table_id] for e in examples]

    query_embeddings = encode_query_batch(
        model, tokenizer, [e.query for e in examples], device,
        batch_size=batch_size, show_progress=True,
    )
    ranking = index.search(query_embeddings, top_k=max(ks))
    return recall_at_k(index, ranking, gold_keys, ks=ks), len(examples), len(tables)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, help="Converted checkpoint or run directory")
    parser.add_argument("--split", default="test")
    parser.add_argument("--csv-dir", default=None, help="Local split CSVs; omit to use the Hub")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N queries")
    parser.add_argument("--index-path", default=None, help="Cache the table embeddings here")
    parser.add_argument("--output", default=None, help="Write the metrics to this JSON file")
    args = parser.parse_args()

    recalls, n_queries, n_tables = evaluate(
        checkpoint=args.checkpoint, split=args.split, csv_dir=args.csv_dir,
        device=args.device, batch_size=args.batch_size, limit=args.limit,
        index_path=args.index_path,
    )

    print(f"\n{args.checkpoint}  split={args.split}  queries={n_queries}  tables={n_tables}")
    for k, value in recalls.items():
        print(f"  R@{k:<3} {value:.4f}")

    if args.output:
        payload = {"checkpoint": args.checkpoint, "split": args.split,
                   "queries": n_queries, "tables": n_tables,
                   "recall": {str(k): v for k, v in recalls.items()}}
        Path(args.output).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()