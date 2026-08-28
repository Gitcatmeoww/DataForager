"""Retrieve with HySE hypothetical schemas encoded by the DTR table tower.

HySE normally embeds an LLM-generated hypothetical schema with the same frozen
encoder used for the corpus, then fuses it with the query embedding. This runs
the identical idea on top of DTR: the hypothetical schema goes through DTR's
table tower, the query through its query tower, and the two are fused before
searching the DTR index.

    h = (1 - lambda) * h_q + lambda * mean_i h_T(schema_i)

Fusing across towers is sound because both are trained into one inner-product
space. If this beats DTR on its own, HySE is complementary to a trained
retriever rather than a substitute for one.

Schemas come from the eval_hyse_components cache that the existing harness
already populated, so no LLM calls are made here.

    python -m evaluation.dtr.hyse_over_dtr --model evaluation/dtr/runs/dtr \
        --csv-dir evaluation/huggingface --split test
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.dtr.adapters import KaggleDSAdapter
from evaluation.dtr.evaluate_dtr import DEFAULT_RECALL_AT, recall_at_k
from evaluation.dtr.index import DTRIndex, encode_query_batch
from evaluation.dtr.modeling import load_dual_encoder, resolve_device
from evaluation.dtr.serialization import DEFAULT_MAX_TABLE_LENGTH, encode_tables, parse_markdown_table

# The paper's operational HySE configuration: two relational schemas, fused with
# the task query at equal weight.
DEFAULT_NUM_SCHEMAS = 2
DEFAULT_LAMBDA = 0.5
DEFAULT_SCHEMA_APPROACH = "relational"


def fetch_cached_schemas(queries, num_schemas=DEFAULT_NUM_SCHEMAS,
                         schema_approach=DEFAULT_SCHEMA_APPROACH):
    """Read hypothetical schemas for each query from eval_hyse_components.

    Args:
        queries: Distinct query strings to look up.
        num_schemas: Schemas to keep per query, lowest hypo_schema_id first.
        schema_approach: Which generation template to read.

    Returns:
        Mapping of query to a list of (table_name, markdown) pairs. Queries with
        no cache entry are absent.
    """
    from dataforager.db.connect_db import DatabaseConnection

    schemas = {}
    with DatabaseConnection() as db:
        db.cursor.execute(
            """
            SELECT query, table_name_comp, example_2rows_comp
            FROM (
                SELECT query, table_name_comp, example_2rows_comp,
                       ROW_NUMBER() OVER (PARTITION BY query ORDER BY hypo_schema_id ASC) AS rank
                FROM eval_hyse_components
                WHERE schema_approach = %s AND query = ANY(%s)
            ) ranked
            WHERE rank <= %s;
            """,
            (schema_approach, list(queries), num_schemas),
        )
        for row in db.cursor.fetchall():
            schemas.setdefault(row["query"], []).append(
                (row["table_name_comp"], row["example_2rows_comp"])
            )
    return schemas


def encode_schemas(model, tokenizer, schemas_by_query, queries, device, batch_size=32,
                   max_length=DEFAULT_MAX_TABLE_LENGTH, max_rows=2):
    """Embed each query's hypothetical schemas with the table tower and average.

    Returns:
        Array of shape (len(queries), proj_dim). Rows for queries with no cached
        schema are left as zeros, which makes the fusion fall back to the query
        embedding alone.
    """
    flat_titles, flat_tables, owners = [], [], []
    for position, query in enumerate(queries):
        for table_name, markdown in schemas_by_query.get(query, []):
            flat_titles.append(table_name or "")
            flat_tables.append(parse_markdown_table(markdown, max_rows=max_rows))
            owners.append(position)

    embeddings = np.zeros((len(queries), model.projection_dim), dtype=np.float32)
    if not flat_titles:
        return embeddings

    import torch

    encoded = []
    with torch.no_grad():
        for start in range(0, len(flat_titles), batch_size):
            batch = encode_tables(
                tokenizer,
                flat_titles[start : start + batch_size],
                flat_tables[start : start + batch_size],
                max_length=max_length,
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            encoded.append(model.encode_tables(**batch).float().cpu().numpy())
    encoded = np.concatenate(encoded, axis=0)

    counts = np.zeros(len(queries), dtype=np.float32)
    for row, position in enumerate(owners):
        embeddings[position] += encoded[row]
        counts[position] += 1
    np.divide(embeddings, np.maximum(counts, 1)[:, None], out=embeddings)
    return embeddings


def evaluate(model_dir, split="test", csv_dir=None, device=None, batch_size=32,
             num_schemas=DEFAULT_NUM_SCHEMAS, fusion_lambda=DEFAULT_LAMBDA,
             schema_approach=DEFAULT_SCHEMA_APPROACH, ks=DEFAULT_RECALL_AT,
             limit=None, index_path=None):
    """Report Recall@k for HySE fused with DTR over a split."""
    device = resolve_device(device)
    model, tokenizer = load_dual_encoder(model_dir, device=device)

    tables, examples = KaggleDSAdapter(csv_dir=csv_dir).load(split)
    if limit:
        examples = examples[:limit]

    if index_path and Path(index_path).exists():
        index = DTRIndex.load(index_path)
    else:
        index = DTRIndex.build(model, tokenizer, tables, device, batch_size=batch_size)
        if index_path:
            index.save(index_path)

    queries = [e.query for e in examples]
    distinct = sorted(set(queries))
    schemas_by_query = fetch_cached_schemas(distinct, num_schemas, schema_approach)
    missing = len(distinct) - len(schemas_by_query)
    print(f"Cached schemas for {len(schemas_by_query)}/{len(distinct)} distinct queries")

    query_embeddings = encode_query_batch(
        model, tokenizer, queries, device, batch_size=batch_size, show_progress=True
    )
    schema_embeddings = encode_schemas(
        model, tokenizer, schemas_by_query, queries, device, batch_size=batch_size
    )

    # Where no schema was cached the row is zero, so fall back to the query
    # embedding alone rather than shrinking it toward the origin.
    has_schema = (np.abs(schema_embeddings).sum(axis=1) > 0)[:, None]
    fused = np.where(
        has_schema,
        (1 - fusion_lambda) * query_embeddings + fusion_lambda * schema_embeddings,
        query_embeddings,
    )

    by_id = {t.table_id: t.recall_key for t in tables}
    gold_keys = [by_id[e.table_id] for e in examples]
    ranking = index.search(fused.astype(np.float32), top_k=max(ks))
    return recall_at_k(index, ranking, gold_keys, ks=ks), len(examples), len(tables), missing


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--csv-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-schemas", type=int, default=DEFAULT_NUM_SCHEMAS)
    parser.add_argument("--lambda", dest="fusion_lambda", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--schema-approach", default=DEFAULT_SCHEMA_APPROACH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    recalls, n_queries, n_tables, missing = evaluate(
        model_dir=args.model, split=args.split, csv_dir=args.csv_dir, device=args.device,
        batch_size=args.batch_size, num_schemas=args.num_schemas,
        fusion_lambda=args.fusion_lambda, schema_approach=args.schema_approach,
        limit=args.limit, index_path=args.index_path,
    )

    print(f"\nHySE-over-DTR  {args.model}  split={args.split}  queries={n_queries}  tables={n_tables}")
    print(f"  N={args.num_schemas}  lambda={args.fusion_lambda}  uncached_queries={missing}")
    for k, value in recalls.items():
        print(f"  R@{k:<3} {value:.4f}")

    if args.output:
        Path(args.output).write_text(json.dumps({
            "model": args.model, "split": args.split, "method": "hyse_over_dtr",
            "num_schemas": args.num_schemas, "lambda": args.fusion_lambda,
            "queries": n_queries, "tables": n_tables,
            "recall": {str(k): v for k, v in recalls.items()},
        }, indent=2))


if __name__ == "__main__":
    main()
