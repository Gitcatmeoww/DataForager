# DataForager evaluation

This directory contains the evaluation harness used in the DataForager paper:
the code that measures how well **HySE** retrieves task-relevant tables compared
to semantic and keyword baselines, plus the subgroup experiments and the scripts
that build and publish the **KaggleDS** corpus.

## What gets measured

The harness compares several retrieval methods over the same corpus and reports
**Recall@k** (and retrieval time) per method. The methods live in
[`eval_methods.py`](eval_methods.py):

| Method | Description |
| ------ | ----------- |
| `semantic_search` | Embed the task/keyword query directly and do vector search |
| `syntactic_search` | Keyword (BM25) search via Elasticsearch — the lexical baseline |
| `single_hyse_search` | HySE with a single hypothetical schema |
| `multi_component_hyse_search` / `multi_hyse_search` | HySE with multiple hypothetical schemas |
| `metadata_search` | NL metadata-filter retrieval (granularity, size, …) |
| `dtr_search` | Fine-tuned dense table retriever (DTR) — the learned baseline, see [`dtr/`](dtr/) |
| `hyse_over_dtr_search` | HySE hypothetical schemas encoded by DTR's table tower |

[`evaluator.py`](evaluator.py) orchestrates these over a data split and also
supports weighted fusion, multi-stage retrieval, and metadata-refinement
evaluation.

## Prerequisites

1. Install the eval extras: `pip install -e ".[eval]"` (from the repo root).
2. A **PostgreSQL** (pgvector) database populated with the corpus — the harness
   queries DB tables with precomputed embedding columns (e.g.
   `example_2rows_table_name_embed`), not the raw CSVs.
3. *(Optional)* **Elasticsearch** for the `syntactic_search` baseline. Select the
   backend with `ES_MODE` (`local` default, or `azure`). The client connects
   lazily, so the rest of the harness imports fine without ES.
4. A configured `.env` (see [`../.env.example`](../.env.example)).

## Getting the data

The processed **KaggleDS** splits are published on the HuggingFace Hub, not
vendored here. For inspection / model training:

```python
from evaluation.data.load_kaggleds import load_kaggleds
test = load_kaggleds("test")
```

To run the harness you must load a split into Postgres first (the data-import
helpers under [`schema_type_eval/`](schema_type_eval/), e.g.
`import_test_data.py`, do this; `create_hnsw_index.py` adds the vector index).

## Running the base evaluation

```bash
python -m evaluation.run_evaluator_on_test_set
```

Key knobs (set in that script, or when constructing `Evaluator` directly):

- `data_split` — DB table to evaluate (e.g. `eval_data_test`)
- `embed_col` — embedding column to search against
- `k` — recall cutoff
- `num_embed` — number of hypothetical schemas for multi-HySE
- `filter_should_include` — restrict to rows flagged for inclusion

Results are written to `evaluation/results/` (gitignored): `per_row_results.csv`,
`failed_rows.csv`, `failed_queries.csv`, and average recalls/times are printed.

## The DTR baseline

[`dtr/`](dtr/) holds a re-implementation of the dense table retriever from
Herzig et al. (2021), fine-tuned on the KaggleDS train split. It is the one
baseline that learns from the corpus rather than using frozen embeddings, and it
has its own README covering the pipeline, results, and deviations. It needs the
optional `dtr` extra (`pip install -e ".[dtr]"`) and no database.

## Subgroup experiments

| Directory | Question | Entry point |
| --------- | -------- | ----------- |
| [`schema_type_eval/`](schema_type_eval/) | Does normalized vs. denormalized schema affect HySE? | `python -m evaluation.schema_type_eval.schema_type_evaluator` |
| [`table_count_eval/`](table_count_eval/) | Single-table vs. multi-table databases? | `python -m evaluation.table_count_eval.table_count_evaluator` |

Each has a one-time data-prep step before the evaluator:

- **schema_type_eval** — see [`schema_type_eval/README_SCHEMA_EVALUATION.md`](schema_type_eval/README_SCHEMA_EVALUATION.md)
  for the full pipeline (migrate → populate schema types → import test data →
  index to ES → build HNSW index). `setup_schema_evaluation.py` wires it together.
- **table_count_eval** — run `python -m evaluation.table_count_eval.add_table_count_column`
  to label each row `single_table` / `multi_table`, then run the evaluator.

## Building & publishing the corpus

- [`dataset_construction/kaggle_eval_datasets.ipynb`](dataset_construction/kaggle_eval_datasets.ipynb)
  builds KaggleDS end to end (filtering → metadata enrichment → query synthesis →
  database-level train/val/test split).
- [`huggingface/push_to_hub.py`](huggingface/push_to_hub.py) uploads the splits;
  [`huggingface/README.md`](huggingface/README.md) is the dataset card.

## Notes

- The `test_*.py` scripts under `schema_type_eval/` and `table_count_eval/` are
  **infra-dependent smoke checks** (they need Postgres/Elasticsearch) and are run
  manually — they are intentionally excluded from `pytest` (see `pyproject.toml`).
