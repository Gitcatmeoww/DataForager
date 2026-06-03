# DataForager

**DataForager** is a system for **task-driven dataset search** — finding relevant
tables from a natural-language description of an analytical *goal* (e.g.
_"Analyze trends in the California real estate market over the past decade"_)
rather than from keywords or exact schema matches.

Its core method is **HySE (Hypothetical Schema Embeddings)**: given a task query,
DataForager asks an LLM to imagine the database schema(s) that *would* support the
task, embeds those hypothetical schemas, and retrieves real tables whose metadata
embeddings are most similar. Results can then be interactively refined with
natural-language filters over dataset metadata (temporal/geographic granularity,
size, popularity, tags, …).

This repository accompanies the paper *"DataForager: Enabling Flexible
Need-Aligned Dataset Search"* (under review). The evaluation corpus, **KaggleDS**,
is published on the HuggingFace Hub:
[`trl-lab/kaggleds-corpus-task-based-search-bench`](https://huggingface.co/datasets/trl-lab/kaggleds-corpus-task-based-search-bench).

---

## Repository layout

```
src/dataforager/          Installable Python library
  api/app.py              Flask API service (search + refinement endpoints)
  hyse/                   HySE: hypothetical-schema generation + vector search
  table_representation/   Metadata inference & embedding (OpenAI/Azure client)
  actions/                Query intent + NL metadata-filter handling
  chat/                   In-memory chat/session history
  db/                     Postgres (pgvector) connection & corpus construction
  utils/                  Shared helpers
evaluation/               Evaluation harness + experiments
  evaluator.py            Base evaluator (HySE vs. semantic/keyword baselines)
  schema_type_eval/       Normalized vs. denormalized analysis
  table_count_eval/       Single- vs. multi-table analysis
  elastic_search/         Keyword-search baseline (Elasticsearch)
  dataset_construction/   Notebook that builds the KaggleDS corpus
  huggingface/            Script + dataset card for publishing KaggleDS
  data/load_kaggleds.py   Loader that pulls KaggleDS from the Hub
scripts/build_demo_corpus/  Pipeline to build the small data.gov demo corpus
mock_data/                Demo-corpus JSON artifacts
frontend/my-app/          React UI (optional)
tests/                    Unit / smoke tests
```

## Requirements

- Python **3.10+**
- **PostgreSQL** with the [pgvector](https://github.com/pgvector/pgvector) extension
- An **Azure OpenAI** (or OpenAI) API key — used for schema generation & embeddings
- *(Optional)* **Elasticsearch** — only for the keyword-search evaluation baseline
- *(Optional)* **Node.js + npm** — only for the React frontend

## Installation

```bash
git clone https://github.com/Gitcatmeoww/DataForager.git
cd DataForager

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[eval,dev]"                          # library + eval + dev extras
```

> Editable install (`pip install -e .`) puts the `dataforager` package on your
> path, so the old `PYTHONPATH`/`setenv.sh` workaround is no longer needed.

## Configuration

Copy the template and fill in your credentials:

```bash
cp .env.example .env
```

See [`.env.example`](.env.example) for every variable (OpenAI/Azure keys, Postgres
connection, optional Elasticsearch, Flask). At minimum you need the Azure OpenAI
keys and the `DB_*` / `EVAL_DB_NAME` settings.

## Quickstart (demo corpus)

The fastest way to try DataForager without the full KaggleDS corpus is the small
**data.gov demo corpus**:

```bash
# 1. Build the demo corpus (see scripts/build_demo_corpus/README.md for details)
python scripts/build_demo_corpus/1_generate_mock_data.py
python scripts/build_demo_corpus/2_infer_metadata.py
python scripts/build_demo_corpus/3_embed_metadata.py

# 2. Create the database schema and load the corpus
python -m dataforager.db.construct_db

# 3. Run the API
dataforager-api          # or: python -m dataforager.api.app
```

The API listens on `http://localhost:5000`. To use the web UI:

```bash
cd frontend/my-app
npm install
npm start                # serves http://localhost:3000
```

## API endpoints

| Method | Route | Purpose |
| ------ | ----- | ------- |
| `POST` | `/api/start_chat` | Start a session, returns a `thread_id` |
| `POST` | `/api/hyse_search` | Initial HySE search for a task query |
| `POST` | `/api/refine_search_space` | Refine results with NL metadata filters |
| `POST` | `/api/reset_search_space` | Reset the working result set |
| `POST` | `/api/update_chat_history` | Append a user query to the session |
| `GET`  | `/api/get_chat_history` | Fetch a session's chat history |

## Evaluation

The processed KaggleDS splits are **not** vendored in this repo (they are ~300 MB
and live on the Hub). Pull them on demand:

```python
from evaluation.data.load_kaggleds import load_kaggleds

test = load_kaggleds("test")     # a single split
corpus = load_kaggleds()         # all splits (DatasetDict)
```

Run the evaluators (require a populated Postgres corpus; see each module):

```bash
python -m evaluation.run_evaluator_on_test_set                 # base HySE evaluation
python -m evaluation.schema_type_eval.schema_type_evaluator    # normalized vs. denormalized
python -m evaluation.table_count_eval.table_count_evaluator    # single- vs. multi-table
```

To rebuild the corpus from scratch, see
[`evaluation/dataset_construction/kaggle_eval_datasets.ipynb`](evaluation/dataset_construction/kaggle_eval_datasets.ipynb).

## Tests

```bash
pytest        # unit/smoke tests under tests/
```

(The `test_*.py` scripts under `evaluation/` are infra-dependent smoke checks that
require Postgres/Elasticsearch and are run manually, not by `pytest`.)

## Citation

If you use DataForager or the KaggleDS corpus, please cite the paper (currently
under review):

```bibtex
@misc{dataforager,
  title  = {DataForager: Enabling Flexible Need-Aligned Dataset Search},
  note   = {Under review},
  year   = {2026}
}
```

## License

Code is released under the [MIT License](LICENSE). The KaggleDS dataset is
released separately under CC BY-NC 4.0 (see its
[dataset card](evaluation/huggingface/README.md)).
