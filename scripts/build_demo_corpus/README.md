# Demo corpus build pipeline

These scripts build the small **data.gov demo corpus** used to try DataForager
without the full KaggleDS evaluation corpus. They produce the JSON artifacts in
[`mock_data/`](../../mock_data/), which `construct_db.py` then loads into Postgres.

Run them in order **from the repo root** (so the `mock_data/` relative paths
resolve), with the package installed (`pip install -e .`) and your `.env`
configured:

| Step | Script | Reads | Writes |
| ---- | ------ | ----- | ------ |
| 1 | `1_generate_mock_data.py` | data.gov API | `mock_data/data_gov_mock_data.json` |
| 2 | `2_infer_metadata.py` | `data_gov_mock_data.json` | `mock_data/updated_data_gov_mock_data.json` |
| 3 | `3_embed_metadata.py` | `updated_data_gov_mock_data.json` | `mock_data/mock_data_with_embedding.json` |
| 4 | `dataforager.db.construct_db` (`MockData`) | `mock_data_with_embedding.json` | Postgres |

```bash
python scripts/build_demo_corpus/1_generate_mock_data.py
python scripts/build_demo_corpus/2_infer_metadata.py
python scripts/build_demo_corpus/3_embed_metadata.py
# then load into the database (see construct_db.py)
```

`inspect_granularities.py` is an optional sanity check that prints the temporal
and geographic granularities inferred in step 2.

> These are standalone scripts, intentionally kept **out** of the `dataforager`
> library package so that importing the library has no side effects.
