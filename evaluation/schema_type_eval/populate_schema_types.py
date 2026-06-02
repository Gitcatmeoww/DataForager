"""
Data integration script to populate schema_type values from CSV file.

This script:
1. Reads the CSV file with schema_type annotations
2. Updates the eval_data_validation table in the new database
3. Validates the data integrity
4. Provides statistics on the populated data
"""

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SchemaTypePopulator:
    def __init__(self, csv_path=None, target_db=None):
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        self.db_host = os.getenv('DB_HOST')
        self.db_port = os.getenv('DB_PORT')

        # Default paths and database
        self.csv_path = csv_path
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

    def load_csv_data(self):
        """Load and validate CSV data"""
        try:
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

            logging.info(f"Loading CSV data from: {self.csv_path}")
            df = pd.read_csv(self.csv_path)

            # Validate required columns
            required_columns = ['table_name', 'schema_type', 'should_include']
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Clean and validate data
            df = df.dropna(subset=['table_name', 'schema_type', 'should_include'])

            # Normalize whitespace in table_name and database_name
            df['table_name'] = df['table_name'].astype(str).str.strip()
            if 'database_name' in df.columns:
                df['database_name'] = df['database_name'].astype(str).str.strip()
                # Create composite key: table_name + database_name
                df['composite_key'] = df['table_name'] + '|' + df['database_name']
            else:
                # If no database_name column, use table_name as key but log warning
                logging.warning("No 'database_name' column found in CSV. Using only table_name as identifier.")
                df['composite_key'] = df['table_name']

            # Validate schema_type values
            valid_schema_types = {'normalized', 'denormalized'}
            invalid_types = set(df['schema_type'].unique()) - valid_schema_types
            if invalid_types:
                raise ValueError(f"Invalid schema_type values found: {invalid_types}")

            # Validate should_include values
            valid_should_include = {'Y', 'N'}
            df['should_include'] = df['should_include'].astype(str).str.strip().str.upper()
            invalid_should_include = set(df['should_include'].unique()) - valid_should_include
            if invalid_should_include:
                raise ValueError(f"Invalid should_include values found: {invalid_should_include}")

            # Check for duplicates in composite key but keep all records for sequential approach
            duplicates = df[df.duplicated(subset=['composite_key'], keep=False)]
            if not duplicates.empty:
                logging.warning(f"Found {len(duplicates)} duplicate composite keys in CSV:")
                for idx, row in duplicates.head(10).iterrows():
                    logging.warning(f"  {row['composite_key']} -> {row['schema_type']}")

                # Check if duplicates have different schema_types
                conflicting_duplicates = df.groupby('composite_key')['schema_type'].nunique()
                conflicts = conflicting_duplicates[conflicting_duplicates > 1]
                if not conflicts.empty:
                    logging.error(f"Found {len(conflicts)} composite keys with conflicting schema types:")
                    for composite_key in conflicts.index[:5]:
                        subset = df[df['composite_key'] == composite_key][['composite_key', 'schema_type']].drop_duplicates()
                        logging.error(f"  {composite_key}: {subset['schema_type'].tolist()}")
                    raise ValueError("Cannot proceed with conflicting schema types for the same composite key")

                # For sequential approach, keep ALL records including duplicates
                logging.info(f"Keeping all {len(df)} records (including duplicates) for sequential matching")

            logging.info(f"Loaded {len(df)} records from CSV (sequential approach - keeping duplicates)")
            logging.info(f"Schema type distribution:")
            distribution = df['schema_type'].value_counts()
            for schema_type, count in distribution.items():
                logging.info(f"  {schema_type}: {count}")

            logging.info(f"Should include distribution:")
            should_include_dist = df['should_include'].value_counts()
            for value, count in should_include_dist.items():
                logging.info(f"  {value}: {count}")

            return df

        except Exception as e:
            logging.error(f"Error loading CSV data: {e}")
            raise

    def get_database_tables(self):
        """Get list of tables in the database with composite keys"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT table_name, database_name FROM eval_data_validation ORDER BY table_name, database_name")
            rows = cursor.fetchall()

            # Create composite keys and normalize whitespace
            db_composite_keys = set()
            db_table_to_composite = {}

            for row in rows:
                table_name = str(row['table_name']).strip() if row['table_name'] else ''
                database_name = str(row['database_name']).strip() if row['database_name'] else ''

                # Create composite key
                composite_key = f"{table_name}|{database_name}"
                db_composite_keys.add(composite_key)
                db_table_to_composite[composite_key] = {
                    'table_name': table_name,
                    'database_name': database_name
                }

            logging.info(f"Found {len(db_composite_keys)} unique table/database combinations in database")

            cursor.close()
            conn.close()

            return db_composite_keys, db_table_to_composite

        except Exception as e:
            logging.error(f"Error retrieving database tables: {e}")
            raise

    def validate_data_consistency(self, csv_df, db_composite_keys):
        """Validate consistency between CSV and database using composite keys"""
        csv_composite_keys = set(csv_df['composite_key'].unique())

        # Find mismatches
        csv_only = csv_composite_keys - db_composite_keys
        db_only = db_composite_keys - csv_composite_keys
        common_keys = csv_composite_keys & db_composite_keys

        logging.info(f"Data consistency check:")
        logging.info(f"  Composite keys in CSV only: {len(csv_only)}")
        logging.info(f"  Composite keys in DB only: {len(db_only)}")
        logging.info(f"  Common composite keys: {len(common_keys)}")

        if csv_only:
            logging.warning(f"Composite keys in CSV but not in DB (first 10): {list(csv_only)[:10]}")

        if db_only:
            logging.warning(f"Composite keys in DB but not in CSV (first 10): {list(db_only)[:10]}")

        return common_keys, csv_only, db_only

    def populate_schema_types_by_composite_key(self, csv_df):
        """Populate schema_type values using table_name + database_name + row_num as unique identifier"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Get all records from database with their row numbers and additional fields
            cursor.execute("""
                SELECT table_name, database_name, row_num,
                       TRIM(table_name) as trimmed_table,
                       TRIM(database_name) as trimmed_db
                FROM eval_data_validation
                ORDER BY table_name, database_name
            """)
            db_records = cursor.fetchall()

            logging.info(f"Creating composite key mapping for {len(db_records)} database records...")

            # Create database lookup using composite key: table_name|database_name|row_num
            db_lookup = {}
            for record in db_records:
                # Create multiple possible keys to handle whitespace variations
                original_key = f"{record['table_name']}|{record['database_name']}|{record['row_num']}"
                trimmed_key = f"{record['trimmed_table']}|{record['trimmed_db']}|{record['row_num']}"

                db_lookup[original_key] = record
                db_lookup[trimmed_key] = record

            logging.info(f"Created {len(db_lookup)} lookup entries")

            # Process CSV records and match using composite keys
            logging.info(f"Matching {len(csv_df)} CSV records to database...")

            matched_updates = []
            unmatched_count = 0

            for idx, csv_row in csv_df.iterrows():
                csv_table = str(csv_row['table_name']).strip()
                csv_db = str(csv_row.get('database_name', '')).strip()
                csv_row_num = csv_row.get('row_num', None)

                # Create composite key for this CSV record
                csv_key = f"{csv_table}|{csv_db}|{csv_row_num}"

                # Try to find matching database record
                db_record = db_lookup.get(csv_key)
                if not db_record:
                    # Try with original (non-trimmed) values from CSV
                    original_csv_key = f"{csv_row['table_name']}|{csv_row.get('database_name', '')}|{csv_row_num}"
                    db_record = db_lookup.get(original_csv_key)

                if db_record:
                    # Found a match - prepare update
                    matched_updates.append((
                        csv_row['schema_type'],
                        csv_row['should_include'],
                        db_record['table_name'],  # Use exact database values
                        db_record['database_name'],
                        db_record['row_num']
                    ))
                else:
                    unmatched_count += 1
                    if unmatched_count <= 10:  # Log first 10 unmatched records
                        logging.warning(f"No match found for CSV record: '{csv_key}'")

            logging.info(f"Matched {len(matched_updates)} records, {unmatched_count} unmatched")

            if len(matched_updates) == 0:
                raise ValueError("No records could be matched between CSV and database")

            # Execute updates in batches
            updated_count = 0
            batch_size = 100

            for i in range(0, len(matched_updates), batch_size):
                batch = matched_updates[i:i+batch_size]

                cursor.executemany("""
                    UPDATE eval_data_validation
                    SET schema_type = %s, should_include = %s
                    WHERE table_name = %s AND database_name = %s AND row_num = %s
                """, batch)

                batch_updated = cursor.rowcount
                updated_count += batch_updated
                conn.commit()

                logging.info(f"Updated batch {(i//batch_size) + 1}, batch size: {len(batch)}, updated: {batch_updated}, total: {updated_count}")

            logging.info(f"✅ Successfully updated {updated_count} records using composite key matching")

            cursor.close()
            conn.close()

            return updated_count

        except Exception as e:
            logging.error(f"Error in composite key population: {e}")
            if 'conn' in locals():
                conn.rollback()
            raise

    def verify_population(self):
        """Verify the population was successful"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Check overall statistics
            cursor.execute("""
                SELECT
                    COUNT(*) as total_records,
                    COUNT(schema_type) as populated_records,
                    COUNT(*) - COUNT(schema_type) as null_records,
                    COUNT(should_include) as should_include_populated
                FROM eval_data_validation
            """)
            stats = cursor.fetchone()

            # Check schema type distribution
            cursor.execute("""
                SELECT schema_type, COUNT(*) as count
                FROM eval_data_validation
                WHERE schema_type IS NOT NULL
                GROUP BY schema_type
                ORDER BY count DESC
            """)
            distribution = cursor.fetchall()

            # Check should_include distribution
            cursor.execute("""
                SELECT should_include, COUNT(*) as count
                FROM eval_data_validation
                WHERE should_include IS NOT NULL
                GROUP BY should_include
                ORDER BY should_include
            """)
            should_include_dist = cursor.fetchall()

            logging.info("Population verification:")
            logging.info(f"  Total records: {stats['total_records']}")
            logging.info(f"  Populated records: {stats['populated_records']}")
            logging.info(f"  NULL records: {stats['null_records']}")
            logging.info(f"  Should include populated: {stats['should_include_populated']}")
            logging.info("  Schema type distribution:")
            for row in distribution:
                logging.info(f"    {row['schema_type']}: {row['count']}")
            logging.info("  Should include distribution:")
            for row in should_include_dist:
                logging.info(f"    {row['should_include']}: {row['count']}")

            cursor.close()
            conn.close()

            return stats, distribution, should_include_dist

        except Exception as e:
            logging.error(f"Error verifying population: {e}")
            raise

    def populate(self):
        """Run complete population process using composite key approach"""
        logging.info("Starting schema type population (composite key approach)...")
        logging.info(f"Target database: {self.target_db}")
        logging.info(f"CSV file: {self.csv_path}")

        try:
            # Step 1: Load CSV data
            csv_df = self.load_csv_data()

            # Step 2: Use composite key population approach
            updated_count = self.populate_schema_types_by_composite_key(csv_df)

            # Step 3: Verify population
            stats, distribution, should_include_dist = self.verify_population()

            logging.info("✅ Schema type population completed successfully!")
            logging.info(f"Final results: Updated {updated_count} records")
            logging.info(f"Schema type distribution from database:")
            for row in distribution:
                logging.info(f"  {row['schema_type']}: {row['count']}")
            logging.info(f"Should include distribution from database:")
            for row in should_include_dist:
                logging.info(f"  {row['should_include']}: {row['count']}")

            return True

        except Exception as e:
            logging.error(f"❌ Population failed: {e}")
            return False

    def reset_schema_types(self):
        """Reset all schema_type and should_include values to NULL (for testing)"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("UPDATE eval_data_validation SET schema_type = NULL, should_include = NULL")
            affected_rows = cursor.rowcount
            conn.commit()

            logging.info(f"Reset schema_type and should_include for {affected_rows} records")

            cursor.close()
            conn.close()
            return True

        except Exception as e:
            logging.error(f"Error resetting schema types: {e}")
            return False


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Populate schema_type values from CSV')
    parser.add_argument('--csv-path', help='Path to CSV file with schema types')
    parser.add_argument('--database', help='Target database name')
    parser.add_argument('--reset', action='store_true', help='Reset all schema_type values to NULL')
    parser.add_argument('--verify-only', action='store_true', help='Only verify current population status')

    args = parser.parse_args()

    populator = SchemaTypePopulator(csv_path=args.csv_path, target_db=args.database)

    print("Schema Type Population Tool")
    print("===========================")
    print(f"Target Database: {populator.target_db}")
    print(f"CSV File: {populator.csv_path}")
    print()

    if args.reset:
        choice = input("Reset all schema_type values to NULL? (y/n): ").lower().strip()
        if choice == 'y':
            if populator.reset_schema_types():
                print("✅ Schema types reset successfully")
            else:
                print("❌ Failed to reset schema types")
        return

    if args.verify_only:
        print("Verifying current population status...")
        try:
            stats, distribution = populator.verify_population()
            print("✅ Verification completed")
        except Exception as e:
            print(f"❌ Verification failed: {e}")
        return

    choice = input("Proceed with population? (y/n): ").lower().strip()
    if choice != 'y':
        print("Population cancelled.")
        return

    if populator.populate():
        print("\n✅ Population completed successfully!")
        print("The database is now ready for schema type evaluation.")
    else:
        print("\n❌ Population failed. Check logs for details.")


if __name__ == "__main__":
    main()