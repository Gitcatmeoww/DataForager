"""Mine one hard negative per training question with a trained DTR model.

Follows Herzig et al. (2021): retrieve the most similar tables for each training
question with the current retriever, discard the false negatives, and keep the
highest-scoring survivor as that question's negative. Mining with DTR itself
beat mining with BM25 in the paper (+hn 81.13 vs +hnbm25 80.51 R@10).

Their false-negative filter drops any candidate that contains the reference
answer. The analogue here is to drop any candidate sharing the gold table's
group_id, since tables from one Kaggle dataset are often near-identical and a
sibling would otherwise be selected almost every time.

    python -m evaluation.dtr.mine_hard_negatives --model evaluation/dtr/runs/dtr \
        --csv-dir evaluation/huggingface --output evaluation/dtr/runs/hard_negatives.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.dtr.adapters import KaggleDSAdapter
from evaluation.dtr.index import DTRIndex, encode_query_batch
from evaluation.dtr.modeling import load_dual_encoder, resolve_device

# Enough depth that a question whose neighbourhood is all siblings still has a
# candidate left after filtering.
DEFAULT_CANDIDATES = 50


def mine(model_dir, output, csv_dir=None, split="train", device=None, batch_size=32,
         num_candidates=DEFAULT_CANDIDATES):
    """Mine negatives and write them as JSON records.

    Returns:
        The list of {query, table_id, negative_id} records written.
    """
    device = resolve_device(device)
    model, tokenizer = load_dual_encoder(model_dir, device=device)

    tables, examples = KaggleDSAdapter(csv_dir=csv_dir).load(split)
    index = DTRIndex.build(model, tokenizer, tables, device, batch_size=batch_size)

    embeddings = encode_query_batch(
        model, tokenizer, [e.query for e in examples], device,
        batch_size=batch_size, show_progress=True,
    )
    ranking = index.search(embeddings, top_k=num_candidates)

    groups = np.array(index.group_ids, dtype=object)
    table_ids = np.array(index.table_ids, dtype=object)
    group_by_id = {t.table_id: t.group_id for t in tables}

    records, skipped = [], 0
    for example, candidates in zip(examples, ranking):
        gold_group = group_by_id[example.table_id]
        # Drop siblings, which may legitimately answer the same query.
        survivors = candidates[groups[candidates] != gold_group]
        if survivors.size == 0:
            skipped += 1
            continue
        records.append({
            "query": example.query,
            "table_id": example.table_id,
            "negative_id": str(table_ids[survivors[0]]),
        })

    Path(output).write_text(json.dumps(records, indent=2))
    print(f"Mined {len(records)} negatives from {len(examples)} examples "
          f"({skipped} skipped, no non-sibling candidate) to {output}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="Trained run directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-dir", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-candidates", type=int, default=DEFAULT_CANDIDATES)
    args = parser.parse_args()

    mine(
        model_dir=args.model, output=args.output, csv_dir=args.csv_dir, split=args.split,
        device=args.device, batch_size=args.batch_size, num_candidates=args.num_candidates,
    )


if __name__ == "__main__":
    main()
