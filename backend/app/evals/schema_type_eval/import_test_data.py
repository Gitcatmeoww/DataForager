"""
Script to import test data into the schema database.

This script:
1. Creates eval_data_test table with columns matching the CSV
2. Imports data from eval_data_test_embed_shcema_type.csv
3. Validates the data integrity
"""

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import os
import logging
from dotenv import load_dotenv
import numpy as np
import ast
import json

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TestDataImporter:
    def __init__(self, csv_path=None, target_db=None):
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        self.db_host = os.getenv('DB_HOST')
        self.db_port = os.getenv('DB_PORT')

        # Default paths and database
        self.csv_path = csv_path or 'eval/eval_data_processed_exp_opt/eval_data_test_embed_schema_type.csv'
        self.target_db = target_db or 'HITS-eval-data-corpus-exp-opt-schema'

        if not all([self.db_user, self.db_password, self.db_host, self.db_port]):
            raise ValueError("Missing required database environment variables")

    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            dbname=self.target_db,
            user=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port,
            cursor_factory=RealDictCursor
        )

    def table_exists(self):
        """Check if eval_data_test table already exists"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'eval_data_test'
                )
            """)
            exists = cursor.fetchone()['exists']

            cursor.close()
            conn.close()
            return exists
        except Exception as e:
            logging.error(f"Error checking table existence: {e}")
            return False

    def create_table(self):
        """Create eval_data_test table with columns matching the CSV"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Drop table if exists (for clean import)
            logging.info("Checking if eval_data_test table exists...")
            if self.table_exists():
                logging.warning("Table eval_data_test already exists. Dropping and recreating...")
                cursor.execute("DROP TABLE eval_data_test")
                conn.commit()

            # Create table with columns matching CSV
            # Note: keywords, task_queries, and tags should be TEXT[] arrays, metadata_queries should be JSONB
            create_table_sql = """
                CREATE TABLE eval_data_test (
                    table_name TEXT,
                    database_name TEXT,
                    example_2rows_md TEXT,
                    schema_type VARCHAR(20),
                    time_granu TEXT,
                    geo_granu TEXT,
                    db_description TEXT,
                    col_num INTEGER,
                    row_num INTEGER,
                    popularity NUMERIC,
                    usability_rating NUMERIC,
                    tags TEXT[],
                    file_size_in_byte BIGINT,
                    keywords TEXT[],
                    task_queries TEXT[],
                    metadata_queries JSONB,
                    example_2rows_table_name_embed vector(1536)
                )
            """

            cursor.execute(create_table_sql)
            conn.commit()

            logging.info("✅ Created eval_data_test table successfully")

            # Create HNSW index on embedding column for efficient similarity search
            logging.info("Creating HNSW index on example_2rows_table_name_embed...")
            index_query = """
            CREATE INDEX IF NOT EXISTS eval_data_test_example_2rows_table_name_embed_idx
            ON eval_data_test USING hnsw (example_2rows_table_name_embed vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
            """
            cursor.execute(index_query)
            conn.commit()

            logging.info("✅ Created HNSW index successfully")

            cursor.close()
            conn.close()
            return True
        except Exception as e:
            logging.error(f"Error creating table: {e}")
            if 'conn' in locals():
                conn.rollback()
            return False

    def load_and_validate_csv(self):
        """Load and validate CSV data"""
        try:
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

            logging.info(f"Loading CSV data from: {self.csv_path}")
            df = pd.read_csv(self.csv_path)

            # Log basic info
            logging.info(f"Loaded {len(df)} records from CSV")
            logging.info(f"Columns: {df.columns.tolist()}")

            # Validate required columns
            required_columns = ['table_name', 'database_name', 'schema_type', 'example_2rows_table_name_embed']
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Validate schema_type values
            valid_schema_types = {'normalized', 'denormalized'}
            schema_type_counts = df['schema_type'].value_counts()
            logging.info(f"Schema type distribution:")
            for schema_type, count in schema_type_counts.items():
                logging.info(f"  {schema_type}: {count}")

            invalid_types = set(df['schema_type'].unique()) - valid_schema_types - {np.nan}
            if invalid_types:
                raise ValueError(f"Invalid schema_type values found: {invalid_types}")

            return df

        except Exception as e:
            logging.error(f"Error loading CSV data: {e}")
            raise

    def parse_vector(self, vector_value):
        """Parse a vector column from string to list of floats"""
        if pd.isna(vector_value) or vector_value is None or vector_value == '':
            return None
        try:
            if isinstance(vector_value, str):
                # Use ast.literal_eval to safely parse the string representation
                vector_list = ast.literal_eval(vector_value)
                # Convert to list of floats
                return [float(x) for x in vector_list]
            elif isinstance(vector_value, list):
                return [float(x) for x in vector_value]
            else:
                return None
        except Exception as e:
            logging.warning(f"Error parsing vector: {e}")
            return None

    def parse_text_array(self, array_value):
        """Parse a text array column from string to list of strings"""
        if pd.isna(array_value) or array_value is None or array_value == '':
            return None
        try:
            if isinstance(array_value, str):
                # PostgreSQL array format: {"item1","item2","item3"} or {item1,item2,item3}
                # Can have mixed quoted/unquoted values
                if array_value.startswith('{') and array_value.endswith('}'):
                    # Remove outer braces
                    inner = array_value[1:-1]

                    # Split by comma, but respect quotes
                    items = []
                    current = []
                    in_quotes = False

                    for char in inner:
                        if char == '"':
                            in_quotes = not in_quotes
                        elif char == ',' and not in_quotes:
                            item = ''.join(current).strip()
                            if item:
                                items.append(item)
                            current = []
                        else:
                            current.append(char)

                    # Add the last item
                    item = ''.join(current).strip()
                    if item:
                        items.append(item)

                    return items if items else None
                else:
                    # Try to parse as JSON directly
                    return json.loads(array_value)
            elif isinstance(array_value, list):
                return array_value
            else:
                return None
        except Exception as e:
            logging.warning(f"Error parsing text array '{str(array_value)[:100]}': {e}")
            return None

    def parse_jsonb(self, jsonb_value):
        """Parse a JSONB column from string to dict/list"""
        if pd.isna(jsonb_value) or jsonb_value is None or jsonb_value == '':
            return None
        try:
            if isinstance(jsonb_value, str):
                return json.loads(jsonb_value)
            elif isinstance(jsonb_value, (dict, list)):
                return jsonb_value
            else:
                return None
        except Exception as e:
            logging.warning(f"Error parsing JSONB: {e}")
            return None

    def import_data(self, df):
        """Import data from DataFrame to database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            logging.info(f"Importing {len(df)} records into eval_data_test...")

            # Prepare data for insertion
            records = []
            skipped_count = 0
            for idx, row in df.iterrows():
                # Parse the embedding vector
                embedding = self.parse_vector(row.get('example_2rows_table_name_embed'))

                # Parse array columns
                keywords = self.parse_text_array(row.get('keywords'))
                task_queries = self.parse_text_array(row.get('task_queries'))
                tags = self.parse_text_array(row.get('tags'))
                metadata_queries = self.parse_jsonb(row.get('metadata_queries'))

                record = (
                    row.get('table_name'),
                    row.get('database_name'),
                    row.get('example_2rows_md'),
                    row.get('schema_type') if pd.notna(row.get('schema_type')) else None,
                    row.get('time_granu') if pd.notna(row.get('time_granu')) else None,
                    row.get('geo_granu') if pd.notna(row.get('geo_granu')) else None,
                    row.get('db_description') if pd.notna(row.get('db_description')) else None,
                    int(row.get('col_num')) if pd.notna(row.get('col_num')) else None,
                    int(row.get('row_num')) if pd.notna(row.get('row_num')) else None,
                    float(row.get('popularity')) if pd.notna(row.get('popularity')) else None,
                    float(row.get('usability_rating')) if pd.notna(row.get('usability_rating')) else None,
                    tags,
                    int(row.get('file_size_in_byte')) if pd.notna(row.get('file_size_in_byte')) else None,
                    keywords,
                    task_queries,
                    Json(metadata_queries) if metadata_queries is not None else None,
                    embedding
                )
                records.append(record)

            # Insert in batches
            batch_size = 100
            inserted_count = 0

            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]

                cursor.executemany("""
                    INSERT INTO eval_data_test (
                        table_name, database_name, example_2rows_md, schema_type,
                        time_granu, geo_granu, db_description, col_num, row_num,
                        popularity, usability_rating, tags, file_size_in_byte,
                        keywords, task_queries, metadata_queries, example_2rows_table_name_embed
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, batch)

                inserted_count += len(batch)
                conn.commit()

                if (i // batch_size + 1) % 10 == 0:
                    logging.info(f"Inserted {inserted_count}/{len(records)} records...")

            logging.info(f"✅ Successfully inserted {inserted_count} records")

            cursor.close()
            conn.close()

            return inserted_count

        except Exception as e:
            logging.error(f"Error importing data: {e}")
            if 'conn' in locals():
                conn.rollback()
            raise

    def verify_import(self):
        """Verify the import was successful"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Check row count
            cursor.execute("SELECT COUNT(*) as count FROM eval_data_test")
            total_count = cursor.fetchone()['count']

            # Check schema type distribution
            cursor.execute("""
                SELECT schema_type, COUNT(*) as count
                FROM eval_data_test
                WHERE schema_type IS NOT NULL
                GROUP BY schema_type
                ORDER BY schema_type
            """)
            schema_distribution = cursor.fetchall()

            # Check for NULL schema_types
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM eval_data_test
                WHERE schema_type IS NULL
            """)
            null_schema_count = cursor.fetchone()['count']

            # Sample a few records
            cursor.execute("""
                SELECT table_name, database_name, schema_type
                FROM eval_data_test
                LIMIT 5
            """)
            sample_records = cursor.fetchall()

            logging.info("Import verification:")
            logging.info(f"  Total records: {total_count}")
            logging.info(f"  NULL schema_type records: {null_schema_count}")
            logging.info("  Schema type distribution:")
            for row in schema_distribution:
                logging.info(f"    {row['schema_type']}: {row['count']}")

            logging.info("\n  Sample records:")
            for record in sample_records:
                logging.info(f"    {record['table_name']} | {record['database_name']} | {record['schema_type']}")

            cursor.close()
            conn.close()

            return {
                'total_count': total_count,
                'schema_distribution': schema_distribution,
                'null_schema_count': null_schema_count
            }

        except Exception as e:
            logging.error(f"Error verifying import: {e}")
            raise

    def import_test_data(self):
        """Run complete import process"""
        logging.info("Starting test data import...")
        logging.info(f"Target database: {self.target_db}")
        logging.info(f"CSV file: {self.csv_path}")

        try:
            # Step 1: Create table
            if not self.create_table():
                logging.error("Failed to create table")
                return False

            # Step 2: Load CSV
            df = self.load_and_validate_csv()

            # Step 3: Import data
            inserted_count = self.import_data(df)

            # Step 4: Verify import
            stats = self.verify_import()

            logging.info("✅ Test data import completed successfully!")
            logging.info(f"Imported {inserted_count} records into eval_data_test")

            return True

        except Exception as e:
            logging.error(f"❌ Import failed: {e}")
            return False


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Import test data into schema database')
    parser.add_argument('--csv-path', help='Path to CSV file')
    parser.add_argument('--database', help='Target database name')
    parser.add_argument('--verify-only', action='store_true', help='Only verify current import status')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')

    args = parser.parse_args()

    importer = TestDataImporter(csv_path=args.csv_path, target_db=args.database)

    print("Test Data Import Tool")
    print("=" * 50)
    print(f"Target Database: {importer.target_db}")
    print(f"CSV File: {importer.csv_path}")
    print()

    if args.verify_only:
        if importer.table_exists():
            print("Verifying test data import...")
            try:
                stats = importer.verify_import()
                print("✅ Verification completed")
            except Exception as e:
                print(f"❌ Verification failed: {e}")
        else:
            print("❌ eval_data_test table does not exist")
        return

    if not args.yes:
        choice = input("Proceed with import? This will drop existing eval_data_test table if it exists. (y/n): ").lower().strip()
        if choice != 'y':
            print("Import cancelled.")
            return

    if importer.import_test_data():
        print("\n✅ Import completed successfully!")
        print("The eval_data_test table is now ready for evaluation.")
    else:
        print("\n❌ Import failed. Check logs for details.")


if __name__ == "__main__":
    main()
