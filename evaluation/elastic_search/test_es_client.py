import pytest
import logging
from evaluation.elastic_search.es_client import es_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@pytest.fixture(scope="module")
def test_index():
    index = "pytest_evals_index"
    yield index
    # Cleanup after tests
    try:
        es_client.client.indices.delete(index=index)
    except Exception as e:
        print(f"Error deleting index during teardown: {e}")

def test_create_index(test_index):
    es_client.create_index(test_index)
    exists = es_client.client.indices.exists(index=test_index)
    assert exists, f"Index '{test_index}' was not created successfully."

def test_index_data(test_index):
    sample_records = [
        {
            "table_name": "test table 1",
            "example_rows_md": "| col1 | col2 | col3 |\n| --- | --- | --- |\n| val1 | val2 | val3 |"
        },
        {
            "table_name": "test table 2",
            "example_rows_md": "| col1 | col2 | col3 |\n| --- | --- | --- |\n| val1 | val2 | val3 |"
        }
    ]
    es_client.index_data(test_index, sample_records)
    count_response = es_client.client.count(index=test_index)
    assert count_response['count'] == len(sample_records), f"Expected {len(sample_records)} documents, found {count_response['count']}."

def test_search(test_index):
    es_query = {
        "size": 1,
        "query": {
            "match": {
                "table_name": "test table 1"
            }
        }
    }
    response = es_client.client.search(index=test_index, body=es_query)
    assert len(response['hits']['hits']) == 1, "Search did not return the expected number of results."
    assert response['hits']['hits'][0]['_source']['table_name'] == "test table 1"