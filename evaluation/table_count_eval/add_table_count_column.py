"""
Script to add and populate the table_count_type column in eval_data_test.

This script:
1. Adds a `table_count_type` column to `eval_data_test` in the schema database
2. Populates it by counting how many rows share each `database_name`:
   - count == 1  →  'single_table'
   - count  > 1  →  'multi_table'
3. Verifies and prints the resulting distribution

No CSV input needed — classification is derived entirely from the eval data.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TableCountColumnAdder:
    def __init__(self, target_db=None):
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        self.db_host = os.getenv('DB_HOST')
        self.db_port = os.getenv('DB_PORT')
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

    def column_exists(self, table_name, column_name):
        """Check if a column already exists in a table"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = %s AND column_name = %s
                )
            """, (table_name, column_name))
            exists = cursor.fetchone()['exists']
            cursor.close()
            conn.close()
            return exists
        except Exception as e:
            logging.error(f"Error checking column existence: {e}")
            return False

    def add_column(self):
        """Add table_count_type column to eval_data_test if it doesn't exist"""
        if self.column_exists('eval_data_test', 'table_count_type'):
            logging.warning("Column 'table_count_type' already exists in eval_data_test — skipping ADD COLUMN")
            return True

        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                ALTER TABLE eval_data_test
                ADD COLUMN table_count_type VARCHAR(20)
            """)
            conn.commit()
            logging.info("Added 'table_count_type' column to eval_data_test")
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            logging.error(f"Error adding column: {e}")
            return False

    def populate_column(self):
        """
        Populate table_count_type based on how many rows share each database_name.

        single_table: database_name appears exactly once in eval_data_test
        multi_table:  database_name appears more than once
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE eval_data_test t1
                SET table_count_type = CASE
                    WHEN counts.cnt = 1 THEN 'single_table'
                    ELSE 'multi_table'
                END
                FROM (
                    SELECT database_name, COUNT(*) AS cnt
                    FROM eval_data_test
                    GROUP BY database_name
                ) counts
                WHERE t1.database_name = counts.database_name
            """)

            updated = cursor.rowcount
            conn.commit()
            logging.info(f"Populated table_count_type for {updated} rows")
            cursor.close()
            conn.close()
            return updated
        except Exception as e:
            logging.error(f"Error populating column: {e}")
            if 'conn' in locals():
                conn.rollback()
            return 0

    def verify(self):
        """Print distribution of table_count_type values"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) AS total_records,
                    COUNT(table_count_type) AS populated_records,
                    COUNT(*) - COUNT(table_count_type) AS null_records
                FROM eval_data_test
            """)
            stats = cursor.fetchone()

            cursor.execute("""
                SELECT table_count_type, COUNT(*) AS count
                FROM eval_data_test
                WHERE table_count_type IS NOT NULL
                GROUP BY table_count_type
                ORDER BY table_count_type
            """)
            distribution = cursor.fetchall()

            # Also show how many distinct database_names fall into each category
            cursor.execute("""
                SELECT table_count_type, COUNT(DISTINCT database_name) AS distinct_dbs
                FROM eval_data_test
                WHERE table_count_type IS NOT NULL
                GROUP BY table_count_type
                ORDER BY table_count_type
            """)
            db_distribution = cursor.fetchall()

            logging.info("Verification results:")
            logging.info(f"  Total records:     {stats['total_records']}")
            logging.info(f"  Populated records: {stats['populated_records']}")
            logging.info(f"  NULL records:      {stats['null_records']}")
            logging.info("  table_count_type distribution (rows):")
            for row in distribution:
                logging.info(f"    {row['table_count_type']}: {row['count']} rows")
            logging.info("  table_count_type distribution (distinct databases):")
            for row in db_distribution:
                logging.info(f"    {row['table_count_type']}: {row['distinct_dbs']} databases")

            cursor.close()
            conn.close()
            return stats, distribution
        except Exception as e:
            logging.error(f"Error verifying column: {e}")
            raise

    def run(self):
        """Run the full add + populate + verify pipeline"""
        logging.info(f"Starting table_count_type column setup on '{self.target_db}'...")

        if not self.add_column():
            logging.error("Failed to add column")
            return False

        updated = self.populate_column()
        if updated == 0:
            logging.error("No rows were updated — check database connectivity and data")
            return False

        self.verify()
        logging.info("table_count_type column setup complete!")
        return True


def main():
    adder = TableCountColumnAdder()

    print("Table Count Column Setup")
    print("========================")
    print(f"Target database: {adder.target_db}")
    print()

    choice = input("Proceed? (y/n): ").lower().strip()
    if choice != 'y':
        print("Cancelled.")
        return

    if adder.run():
        print("\nSetup complete. eval_data_test now has 'table_count_type' column.")
    else:
        print("\nSetup failed. Check logs for details.")


if __name__ == "__main__":
    main()
