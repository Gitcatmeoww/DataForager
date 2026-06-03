"""Inspect inferred temporal/geographic granularities in the demo corpus.

Reads the metadata-enriched demo corpus produced by ``2_infer_metadata.py`` and
prints the temporal and geographic granularities extracted for each dataset.
Useful as a quick sanity check of the granularity-inference step.

Run from the repo root:
    python scripts/build_demo_corpus/inspect_granularities.py
"""

from dataforager.utils import extract_granularities, load_json_file

INPUT_PATH = "mock_data/updated_data_gov_mock_data.json"


def main():
    json_data = load_json_file(INPUT_PATH)
    time_granu, geo_granu = extract_granularities(json_data)

    print("Temporal Granularities:", time_granu)
    print("Geographic Granularities:", geo_granu)


if __name__ == "__main__":
    main()
