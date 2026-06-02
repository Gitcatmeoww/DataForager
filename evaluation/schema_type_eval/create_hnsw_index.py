"""
Script to create HNSW index on eval_data_test table for efficient similarity search.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_hnsw_index(target_db='HITS-eval-data-corpus-exp-opt-schema'):
    """Create HNSW index on eval_data_test.example_2rows_table_name_embed"""

    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')

    try:
        conn = psycopg2.connect(
            dbname=target_db,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
            cursor_factory=RealDictCursor
        )

        cursor = conn.cursor()

        logging.info(f"Creating HNSW index on eval_data_test.example_2rows_table_name_embed...")
        logging.info("This may take a few minutes for large datasets...")

        # Check if index already exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1
                FROM pg_indexes
                WHERE tablename = 'eval_data_test'
                AND indexname = 'eval_data_test_example_2rows_table_name_embed_idx'
            )
        """)

        index_exists = cursor.fetchone()['exists']

        if index_exists:
            logging.warning("Index 'eval_data_test_example_2rows_table_name_embed_idx' already exists")

            # Ask if user wants to recreate
            response = input("Do you want to drop and recreate the index? (y/n): ").lower().strip()
            if response == 'y':
                logging.info("Dropping existing index...")
                cursor.execute("DROP INDEX eval_data_test_example_2rows_table_name_embed_idx")
                conn.commit()
                logging.info("Existing index dropped")
            else:
                logging.info("Keeping existing index")
                cursor.close()
                conn.close()
                return True

        # Create HNSW index
        # Parameters:
        #   m = 16: Number of connections per layer (higher = more accurate but slower)
        #   ef_construction = 64: Size of dynamic candidate list (higher = better quality but slower build)
        index_query = """
        CREATE INDEX IF NOT EXISTS eval_data_test_example_2rows_table_name_embed_idx
        ON eval_data_test USING hnsw (example_2rows_table_name_embed vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """

        cursor.execute(index_query)
        conn.commit()

        logging.info("✅ HNSW index created successfully!")

        # Verify index was created
        cursor.execute("""
            SELECT schemaname, tablename, indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'eval_data_test'
            AND indexname = 'eval_data_test_example_2rows_table_name_embed_idx'
        """)

        index_info = cursor.fetchone()
        if index_info:
            logging.info(f"Index verified: {index_info['indexname']}")
            logging.info(f"Index definition: {index_info['indexdef']}")

        cursor.close()
        conn.close()

        return True

    except Exception as e:
        logging.error(f"Error creating HNSW index: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Create HNSW index on eval_data_test')
    parser.add_argument('--database', help='Target database name', default='HITS-eval-data-corpus-exp-opt-schema')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')

    args = parser.parse_args()

    print("HNSW Index Creation Tool")
    print("=" * 50)
    print(f"Target Database: {args.database}")
    print(f"Table: eval_data_test")
    print(f"Column: example_2rows_table_name_embed")
    print()
    print("This will create an HNSW index for efficient similarity search.")
    print("The index uses the following parameters:")
    print("  - m = 16 (connections per layer)")
    print("  - ef_construction = 64 (candidate list size)")
    print()

    if not args.yes:
        choice = input("Proceed with index creation? (y/n): ").lower().strip()
        if choice != 'y':
            print("Index creation cancelled.")
            return

    if create_hnsw_index(args.database):
        print("\n✅ HNSW index created successfully!")
        print("The eval_data_test table is now optimized for similarity search.")
    else:
        print("\n❌ Index creation failed. Check logs for details.")


if __name__ == "__main__":
    main()
