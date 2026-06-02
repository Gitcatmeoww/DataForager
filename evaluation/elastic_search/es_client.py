import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
import logging

load_dotenv()

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

# Implement ElasticSearchClient as a Singleton
class ElasticSearchClient(metaclass=SingletonMeta):
    def __init__(self):
        # Decide which mode to use based on an environment variable ES_MODE
        es_mode = os.getenv("ES_MODE", "local").lower()
        
        if es_mode == "azure":
            azure_es_endpoint = os.environ["AZURE_ES_ENDPOINT"]
            azure_es_api_key  = os.environ["AZURE_ES_API_KEY"]

            logging.info("Using Azure Elasticsearch deployment.")

            self.client = Elasticsearch(
                azure_es_endpoint,
                api_key=azure_es_api_key
            )
        else:
            # Default: local mode
            logging.info("Using local Elasticsearch deployment.")
            self.client = Elasticsearch("http://localhost:9200", verify_certs=False)

        self.test_connection()

    def test_connection(self):
        try:
            health = self.client.cluster.health()
            logging.info(f"Elasticsearch Cluster Health: {health['status']}")
        except Exception as e:
            logging.error(f"Error connecting to Elasticsearch: {e}")
            raise e

    def get_index_mapping(self):
        # Index fields used for syntactic search: table_name, table_header, example_2rows_md, example_3rows_md
        return {
            "mappings": {
                "properties": {
                    "table_name": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
                    "table_header": {"type": "text"},
                    "example_2rows_md": {"type": "text"},
                    "example_3rows_md": {"type": "text"}
                }
            }
        }

    def create_index(self, index_name):
        try:
            if not self.client.indices.exists(index=index_name):
                self.client.indices.create(index=index_name, body=self.get_index_mapping())
                logging.info(f"Created Elasticsearch index: {index_name}")
            else:
                logging.info(f"Elasticsearch index already exists: {index_name}")
        except Exception as e:
            logging.error(f"Error creating index '{index_name}': {e}")
            raise e

    # Index a list of records into the specified Elasticsearch index using bulk indexing
    def index_data(self, index_name, records):
        try:
            actions = [
                {
                    "_index": index_name,
                    "_source": record
                }
                for record in records
            ]
            success, errors = helpers.bulk(self.client, actions, raise_on_error=False, refresh=True)
            if errors:
                logging.error(f"Encountered errors during bulk indexing: {errors}")
            logging.info(f"Indexed {success} records into '{index_name}' with {len(errors)} errors.")
        except Exception as e:
            logging.error(f"Error indexing data into '{index_name}': {e}")
            raise e

# Initialize the Singleton Elasticsearch client
es_client = ElasticSearchClient()