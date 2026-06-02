"""
Database migration script to create a new evaluation database with schema_type support.

This script:
1. Creates a new database: HITS-eval-data-corpus-exp-opt-schema
2. Copies all tables and data from the original database
3. Adds schema_type column to eval_data_validation table
4. Provides rollback functionality
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
import subprocess
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatabaseMigrator:
    def __init__(self):
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        self.db_host = os.getenv('DB_HOST')
        self.db_port = os.getenv('DB_PORT')
        self.original_db = os.getenv('EVAL_DB_NAME')
        self.new_db = 'HITS-eval-data-corpus-exp-opt-schema'

        if not all([self.db_user, self.db_password, self.db_host, self.db_port, self.original_db]):
            raise ValueError("Missing required database environment variables")

    def get_connection(self, database):
        """Get database connection"""
        return psycopg2.connect(
            dbname=database,
            user=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port,
            cursor_factory=RealDictCursor
        )

    def database_exists(self, database_name):
        """Check if database exists"""
        try:
            # Connect to default postgres database to check existence
            conn = psycopg2.connect(
                dbname='postgres',
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port
            )
            conn.autocommit = True
            cursor = conn.cursor()

            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database_name,))
            exists = cursor.fetchone() is not None

            cursor.close()
            conn.close()
            return exists
        except Exception as e:
            logging.error(f"Error checking database existence: {e}")
            return False

    def create_database(self):
        """Create new database"""
        try:
            # Connect to default postgres database to create new one
            conn = psycopg2.connect(
                dbname='postgres',
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port
            )
            conn.autocommit = True
            cursor = conn.cursor()

            # Check if database already exists
            if self.database_exists(self.new_db):
                logging.warning(f"Database {self.new_db} already exists. Skipping creation.")
                cursor.close()
                conn.close()
                return True

            # Create new database
            cursor.execute(f'CREATE DATABASE "{self.new_db}"')
            logging.info(f"Created database: {self.new_db}")

            cursor.close()
            conn.close()
            return True
        except Exception as e:
            logging.error(f"Error creating database: {e}")
            return False

    def copy_database_structure_and_data(self):
        """Copy database structure and data using pg_dump and pg_restore"""
        try:
            # Set environment variables for pg_dump/pg_restore
            env = os.environ.copy()
            env['PGPASSWORD'] = self.db_password

            dump_file = f"{self.original_db}_backup.sql"

            # Dump original database
            logging.info(f"Dumping database {self.original_db}...")
            dump_cmd = [
                'pg_dump',
                '-h', self.db_host,
                '-p', str(self.db_port),
                '-U', self.db_user,
                '-d', self.original_db,
                '-f', dump_file,
                '--verbose'
            ]

            result = subprocess.run(dump_cmd, env=env, capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"pg_dump failed: {result.stderr}")
                return False

            logging.info("Database dump completed successfully")

            # Restore to new database
            logging.info(f"Restoring to database {self.new_db}...")
            restore_cmd = [
                'psql',
                '-h', self.db_host,
                '-p', str(self.db_port),
                '-U', self.db_user,
                '-d', self.new_db,
                '-f', dump_file,
                '--quiet'
            ]

            result = subprocess.run(restore_cmd, env=env, capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"psql restore failed: {result.stderr}")
                return False

            logging.info("Database restore completed successfully")

            # Clean up dump file
            os.remove(dump_file)
            logging.info("Cleaned up temporary dump file")

            return True
        except Exception as e:
            logging.error(f"Error copying database: {e}")
            return False

    def add_schema_type_column(self):
        """Add schema_type and should_include columns to eval_data_validation table"""
        try:
            conn = self.get_connection(self.new_db)
            cursor = conn.cursor()

            # Check if schema_type column already exists
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'eval_data_validation'
                AND column_name = 'schema_type'
            """)

            if not cursor.fetchone():
                # Add schema_type column
                cursor.execute("""
                    ALTER TABLE eval_data_validation
                    ADD COLUMN schema_type VARCHAR(20)
                """)
                logging.info("Added schema_type column to eval_data_validation table")
            else:
                logging.warning("schema_type column already exists")

            # Check if should_include column already exists
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'eval_data_validation'
                AND column_name = 'should_include'
            """)

            if not cursor.fetchone():
                # Add should_include column
                cursor.execute("""
                    ALTER TABLE eval_data_validation
                    ADD COLUMN should_include VARCHAR(1)
                """)
                logging.info("Added should_include column to eval_data_validation table")
            else:
                logging.warning("should_include column already exists")

            conn.commit()

            cursor.close()
            conn.close()
            return True
        except Exception as e:
            logging.error(f"Error adding columns: {e}")
            return False

    def verify_migration(self):
        """Verify the migration was successful"""
        try:
            conn = self.get_connection(self.new_db)
            cursor = conn.cursor()

            # Check tables exist
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = [row['table_name'] for row in cursor.fetchall()]

            expected_tables = [
                'eval_data_validation',
                'eval_hyse_components',
                'eval_keyword_embeds',
                'eval_metadata_sqlclauses',
                'eval_query_embeds'
            ]

            missing_tables = set(expected_tables) - set(tables)
            if missing_tables:
                logging.error(f"Missing tables: {missing_tables}")
                return False

            # Check row counts
            cursor.execute("SELECT COUNT(*) FROM eval_data_validation")
            validation_count = cursor.fetchone()['count']

            # Check schema_type column exists
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'eval_data_validation'
                AND column_name = 'schema_type'
            """)

            if not cursor.fetchone():
                logging.error("schema_type column not found")
                return False

            # Check should_include column exists
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'eval_data_validation'
                AND column_name = 'should_include'
            """)

            if not cursor.fetchone():
                logging.error("should_include column not found")
                return False

            logging.info(f"Migration verification successful:")
            logging.info(f"  - All expected tables present: {tables}")
            logging.info(f"  - eval_data_validation rows: {validation_count}")
            logging.info(f"  - schema_type column added successfully")
            logging.info(f"  - should_include column added successfully")

            cursor.close()
            conn.close()
            return True
        except Exception as e:
            logging.error(f"Error verifying migration: {e}")
            return False

    def rollback(self):
        """Drop the new database (rollback)"""
        try:
            conn = psycopg2.connect(
                dbname='postgres',
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port
            )
            conn.autocommit = True
            cursor = conn.cursor()

            # Terminate existing connections to the database
            cursor.execute(f"""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = '{self.new_db}' AND pid <> pg_backend_pid()
            """)

            # Drop database
            cursor.execute(f'DROP DATABASE IF EXISTS "{self.new_db}"')
            logging.info(f"Dropped database: {self.new_db}")

            cursor.close()
            conn.close()
            return True
        except Exception as e:
            logging.error(f"Error during rollback: {e}")
            return False

    def migrate(self):
        """Run complete migration"""
        logging.info("Starting database migration...")
        logging.info(f"Source: {self.original_db}")
        logging.info(f"Target: {self.new_db}")

        # Step 1: Create new database
        if not self.create_database():
            logging.error("Failed to create database")
            return False

        # Step 2: Copy structure and data
        if not self.copy_database_structure_and_data():
            logging.error("Failed to copy database")
            self.rollback()
            return False

        # Step 3: Add schema_type column
        if not self.add_schema_type_column():
            logging.error("Failed to add schema_type column")
            self.rollback()
            return False

        # Step 4: Verify migration
        if not self.verify_migration():
            logging.error("Migration verification failed")
            self.rollback()
            return False

        logging.info("Migration completed successfully!")
        logging.info(f"New database '{self.new_db}' is ready for schema type evaluation")
        return True


def main():
    """Main function"""
    migrator = DatabaseMigrator()

    print("Database Migration Tool")
    print("=======================")
    print(f"Source Database: {migrator.original_db}")
    print(f"Target Database: {migrator.new_db}")
    print()

    choice = input("Proceed with migration? (y/n): ").lower().strip()
    if choice != 'y':
        print("Migration cancelled.")
        return

    if migrator.migrate():
        print("\n✅ Migration completed successfully!")
        print(f"New database '{migrator.new_db}' is ready for use.")
        print("\nNext steps:")
        print("1. Run the data integration script to populate schema_type values")
        print("2. Update your environment variables to use the new database")
    else:
        print("\n❌ Migration failed. Check logs for details.")


if __name__ == "__main__":
    main()