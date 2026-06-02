"""
Script to index eval_data_test into Elasticsearch for syntactic keyword search.

This script handles test data which only has:
- table_name
- example_2rows_md

Unlike validation data which has table_header and example_3rows_md.
"""

from dotenv import load_dotenv
import logging
from backend.app.evals.elastic_search.es_client import es_client
import os
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_db_connection(db_name):
    """Get database connection to specified database"""
    return psycopg2.connect(
        dbname=db_name,
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        cursor_factory=RealDictCursor
    )


def get_table_columns(cursor, table_name):
    """Get list of columns in a table"""
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position
    """
    cursor.execute(query, (table_name,))
    columns = [row['column_name'] for row in cursor.fetchall()]
    return columns


def index_test_data():
    """Index eval_data_test into Elasticsearch"""

    # Use schema database for test data
    db_name = 'HITS-eval-data-corpus-exp-opt-schema'

    conn = get_db_connection(db_name)
    cursor = conn.cursor()

    try:
        table = 'eval_data_test'
        logging.info(f"Processing table: {table} from database: {db_name}")

        # Get available columns
        available_columns = get_table_columns(cursor, table)
        logging.info(f"Available columns: {available_columns}")

        # Ensure the Elasticsearch index exists
        es_client.create_index(table)

        # Build query based on available columns
        # We want: table_name (required), table_header (optional), example_2rows_md (optional), example_3rows_md (optional)
        select_fields = []
        if 'table_name' in available_columns:
            select_fields.append('table_name')
        if 'table_header' in available_columns:
            select_fields.append('table_header')
        else:
            select_fields.append("NULL as table_header")
        if 'example_2rows_md' in available_columns:
            select_fields.append('example_2rows_md')
        else:
            select_fields.append("NULL as example_2rows_md")
        if 'example_3rows_md' in available_columns:
            select_fields.append('example_3rows_md')
        else:
            select_fields.append("NULL as example_3rows_md")

        query = f"SELECT {', '.join(select_fields)} FROM {table};"
        logging.info(f"Query: {query}")

        cursor.execute(query)
        records = cursor.fetchall()

        if not records:
            logging.warning(f"No records found in PostgreSQL table: {table}")
            return

        logging.info(f"Fetched {len(records)} records from PostgreSQL")

        # Sample first record to see what we're indexing
        if records:
            logging.info(f"Sample record fields: {list(records[0].keys())}")
            logging.info(f"Sample record (first 200 chars): {str(records[0])[:200]}...")

        # Index the fetched records into Elasticsearch
        es_client.index_data(table, records)

        logging.info("✅ Test data indexing into Elasticsearch completed successfully")

    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    try:
        print("=" * 70)
        print("Indexing eval_data_test into Elasticsearch")
        print("=" * 70)
        print()

        index_test_data()

        print()
        print("✅ Indexing completed!")
        print("You can now run syntactic keyword search on eval_data_test")

    except Exception as e:
        logging.error(f"Data indexing failed: {e}")
        import traceback
        traceback.print_exc()
