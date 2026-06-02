"""
Enhanced database connection module with support for schema-type-aware database.

This module provides utilities for connecting to both the original evaluation database
and the new schema-type-enhanced database.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

class SchemaAwareDatabaseConnection:
    """Database connection class that can switch between different evaluation databases"""

    def __init__(self, use_schema_db=False):
        self.use_schema_db = use_schema_db
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        self.db_host = os.getenv('DB_HOST')
        self.db_port = os.getenv('DB_PORT')

        # Database names
        self.original_db = os.getenv('EVAL_DB_NAME', 'HITS-eval-data-corpus-exp-opt')
        self.schema_db = 'HITS-eval-data-corpus-exp-opt-schema'

        # Select database based on flag
        self.current_db = self.schema_db if use_schema_db else self.original_db

        self.conn = None
        self.cursor = None

    def get_db_connection(self, database_name=None):
        """Get database connection for specified database"""
        db_name = database_name or self.current_db

        try:
            connection = psycopg2.connect(
                dbname=db_name,
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port,
                cursor_factory=RealDictCursor
            )
            return connection
        except Exception as error:
            print(f"Error connecting to database {db_name}: {error}")
            raise error

    def __enter__(self):
        self.conn = self.get_db_connection()
        self.cursor = self.conn.cursor()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.conn.rollback()
        else:
            self.conn.commit()
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def switch_database(self, use_schema_db):
        """Switch between original and schema database"""
        self.use_schema_db = use_schema_db
        self.current_db = self.schema_db if use_schema_db else self.original_db

    def test_connections(self):
        """Test connections to both databases"""
        results = {}

        # Test original database
        try:
            with self.get_db_connection(self.original_db) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM eval_data_validation")
                original_count = cursor.fetchone()['count']
                results['original'] = {
                    'status': 'success',
                    'database': self.original_db,
                    'record_count': original_count
                }
                cursor.close()
        except Exception as e:
            results['original'] = {
                'status': 'error',
                'database': self.original_db,
                'error': str(e)
            }

        # Test schema database
        try:
            with self.get_db_connection(self.schema_db) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM eval_data_validation")
                schema_count = cursor.fetchone()['count']

                # Check if schema_type column exists
                cursor.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'eval_data_validation'
                    AND column_name = 'schema_type'
                """)
                has_schema_type = cursor.fetchone() is not None

                # Get schema type distribution
                schema_distribution = {}
                if has_schema_type:
                    cursor.execute("""
                        SELECT schema_type, COUNT(*) as count
                        FROM eval_data_validation
                        WHERE schema_type IS NOT NULL
                        GROUP BY schema_type
                    """)
                    schema_distribution = {row['schema_type']: row['count'] for row in cursor.fetchall()}

                results['schema'] = {
                    'status': 'success',
                    'database': self.schema_db,
                    'record_count': schema_count,
                    'has_schema_type': has_schema_type,
                    'schema_distribution': schema_distribution
                }
                cursor.close()
        except Exception as e:
            results['schema'] = {
                'status': 'error',
                'database': self.schema_db,
                'error': str(e)
            }

        return results


def get_db_connection_for_evaluation(use_schema_db=False):
    """Convenience function to get appropriate database connection for evaluation"""
    connection_manager = SchemaAwareDatabaseConnection(use_schema_db=use_schema_db)
    return connection_manager.get_db_connection()


if __name__ == "__main__":
    # Test both database connections
    print("Testing Database Connections")
    print("=" * 40)

    connection_manager = SchemaAwareDatabaseConnection()
    test_results = connection_manager.test_connections()

    for db_type, result in test_results.items():
        print(f"\n{db_type.upper()} Database:")
        print(f"  Database: {result['database']}")
        print(f"  Status: {result['status']}")

        if result['status'] == 'success':
            print(f"  Records: {result['record_count']}")
            if 'has_schema_type' in result:
                print(f"  Has schema_type column: {result['has_schema_type']}")
                if result['has_schema_type'] and result['schema_distribution']:
                    print(f"  Schema distribution:")
                    for schema_type, count in result['schema_distribution'].items():
                        print(f"    {schema_type}: {count}")
        else:
            print(f"  Error: {result['error']}")

    print(f"\nConnection manager configured for: {connection_manager.current_db}")