"""
Setup script for schema type evaluation pipeline.

This script helps users set up and run the complete schema type evaluation pipeline:
1. Migrate database
2. Populate schema types
3. Test connections
4. Run schema type evaluation
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from backend.app.evals.schema_type_eval.migrate_database import DatabaseMigrator
from backend.app.evals.schema_type_eval.populate_schema_types import SchemaTypePopulator
from backend.app.db.schema_db_connect import SchemaAwareDatabaseConnection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SchemaEvaluationSetup:
    def __init__(self):
        self.migrator = DatabaseMigrator()
        self.populator = SchemaTypePopulator()
        self.connection_manager = SchemaAwareDatabaseConnection()

    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        print("Checking prerequisites...")

        # Check environment variables
        required_vars = ['DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT', 'EVAL_DB_NAME']
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            print(f"❌ Missing environment variables: {missing_vars}")
            return False

        # Check if CSV file exists
        if not os.path.exists(self.populator.csv_path):
            print(f"❌ CSV file not found: {self.populator.csv_path}")
            return False

        # Check if original database is accessible
        try:
            test_results = self.connection_manager.test_connections()
            if test_results['original']['status'] != 'success':
                print(f"❌ Cannot connect to original database: {test_results['original'].get('error', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"❌ Error testing database connection: {e}")
            return False

        print("✅ All prerequisites met")
        return True

    def run_migration(self):
        """Run database migration"""
        print("\n" + "="*50)
        print("STEP 1: Database Migration")
        print("="*50)

        if not self.migrator.migrate():
            print("❌ Database migration failed")
            return False

        print("✅ Database migration completed successfully")
        return True

    def run_population(self):
        """Run schema type population"""
        print("\n" + "="*50)
        print("STEP 2: Schema Type Population")
        print("="*50)

        if not self.populator.populate():
            print("❌ Schema type population failed")
            return False

        print("✅ Schema type population completed successfully")
        return True

    def test_setup(self):
        """Test the complete setup"""
        print("\n" + "="*50)
        print("STEP 3: Testing Setup")
        print("="*50)

        test_results = self.connection_manager.test_connections()

        print("Database Connection Test Results:")
        for db_type, result in test_results.items():
            print(f"\n{db_type.upper()} Database:")
            print(f"  Status: {result['status']}")
            if result['status'] == 'success':
                print(f"  Records: {result['record_count']}")
                if 'has_schema_type' in result:
                    print(f"  Has schema_type: {result['has_schema_type']}")
                    if result['schema_distribution']:
                        print(f"  Schema distribution:")
                        for schema_type, count in result['schema_distribution'].items():
                            print(f"    {schema_type}: {count}")
            else:
                print(f"  Error: {result['error']}")

        # Verify schema database is properly set up
        schema_result = test_results.get('schema', {})
        if schema_result.get('status') != 'success':
            print("❌ Schema database setup failed")
            return False

        if not schema_result.get('has_schema_type', False):
            print("❌ Schema database missing schema_type column")
            return False

        if not schema_result.get('schema_distribution', {}):
            print("❌ Schema database has no populated schema_type data")
            return False

        print("✅ Setup testing completed successfully")
        return True

    def run_sample_evaluation(self):
        """Run a sample evaluation to test the pipeline"""
        print("\n" + "="*50)
        print("STEP 4: Sample Evaluation")
        print("="*50)

        try:
            print("Running sample schema type evaluation...")

            # Import and run schema type evaluator
            from backend.app.evals.schema_type_eval.schema_type_evaluator import SchemaTypeEvaluator

            evaluator = SchemaTypeEvaluator(
                data_split="eval_data_validation",
                embed_col="example_2rows_table_name_embed",
                k=10,
                limit=10,  # Small sample for testing
                num_embed=2,
                use_schema_db=True
            )

            results = evaluator.run_schema_type_analysis()

            if results and results['comparison_results']:
                print("✅ Sample evaluation completed successfully")
                print("\nSample Results:")
                for result in results['comparison_results']:
                    print(f"  {result['method']}:")
                    print(f"    Normalized: {result['normalized_recall']:.4f}")
                    print(f"    Denormalized: {result['denormalized_recall']:.4f}")
                    print(f"    Difference: {result['recall_difference']:.4f}")

                return True
            else:
                print("❌ Sample evaluation failed - no results returned")
                return False

        except Exception as e:
            print(f"❌ Sample evaluation failed: {e}")
            return False

    def setup_complete_pipeline(self):
        """Run the complete setup pipeline"""
        print("Schema Type Evaluation Setup")
        print("=" * 60)
        print("This script will set up the complete schema type evaluation pipeline:")
        print("1. Migrate database (create new database with schema_type column)")
        print("2. Populate schema types from CSV file")
        print("3. Test database connections")
        print("4. Run sample evaluation")
        print()

        # Check prerequisites
        if not self.check_prerequisites():
            print("\n❌ Prerequisites not met. Please fix the issues above and try again.")
            return False

        # Confirm with user
        choice = input("Proceed with complete setup? (y/n): ").lower().strip()
        if choice != 'y':
            print("Setup cancelled.")
            return False

        # Run migration
        if not self.run_migration():
            return False

        # Run population
        if not self.run_population():
            return False

        # Test setup
        if not self.test_setup():
            return False

        # Run sample evaluation
        if not self.run_sample_evaluation():
            return False

        print("\n" + "="*60)
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nThe schema type evaluation pipeline is now ready!")
        print("\nNext steps:")
        print("1. Run full evaluation using schema_type_evaluator.py")
        print("2. Check results in backend/app/evals/eval/results/schema_type_results/")
        print("3. Use SchemaTypeEvaluator class in your own scripts")
        print("\nExample usage:")
        print("  from schema_type_evaluator import SchemaTypeEvaluator")
        print("  evaluator = SchemaTypeEvaluator(use_schema_db=True)")
        print("  results = evaluator.run_schema_type_analysis()")

        return True


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Setup schema type evaluation pipeline')
    parser.add_argument('--migrate-only', action='store_true', help='Only run database migration')
    parser.add_argument('--populate-only', action='store_true', help='Only run schema type population')
    parser.add_argument('--test-only', action='store_true', help='Only test the setup')
    parser.add_argument('--sample-eval-only', action='store_true', help='Only run sample evaluation')

    args = parser.parse_args()

    setup = SchemaEvaluationSetup()

    if args.migrate_only:
        setup.run_migration()
    elif args.populate_only:
        setup.run_population()
    elif args.test_only:
        setup.test_setup()
    elif args.sample_eval_only:
        setup.run_sample_evaluation()
    else:
        # Run complete pipeline
        setup.setup_complete_pipeline()


if __name__ == "__main__":
    main()