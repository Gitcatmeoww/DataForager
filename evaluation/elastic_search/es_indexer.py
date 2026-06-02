from dotenv import load_dotenv
import logging
from dataforager.db.connect_db import DatabaseConnection
from evaluation.elastic_search.es_client import es_client

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Extract data from PostgreSQL evaluation tables & indexes it into Elasticsearch
def index_data():
    # tables = ['eval_data_all', 'eval_data_test', 'eval_data_train', 'eval_data_validation']
    tables = ['eval_data_validation']
    
    with DatabaseConnection() as db:
        for table in tables:
            logging.info(f"Processing table: {table}")
            
            # Ensure the Elasticsearch index exists
            es_client.create_index(table)
            
            # Fetch all fields needed for the new mapping
            query = f"SELECT table_name, table_header, example_2rows_md, example_3rows_md FROM {table};"
            db.cursor.execute(query)
            records = db.cursor.fetchall()
            
            if not records:
                logging.warning(f"No records found in PostgreSQL table: {table}")
                continue
            
            # Index the fetched records into Elasticsearch
            es_client.index_data(table, records)

    logging.info("✅ Data indexing into Elasticsearch completed successfully")


if __name__ == "__main__":
    try:
        index_data()
    except Exception as e:
        logging.error(f"Data indexing failed: {e}")
