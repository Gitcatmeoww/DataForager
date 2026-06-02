"""
Script to update the database with should_include column from the new CSV file.

This script:
1. Adds the should_include column to the database (if not already present)
2. Populates both schema_type and should_include from the CSV file
3. Verifies the population
4. Provides instructions for running evaluations with the filter

Usage:
    python update_should_include.py
"""

import os
import sys
import logging
from pathlib import Path

# Add backend to path
current_dir = Path(__file__).parent
backend_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(backend_dir))

from backend.app.evals.schema_type_eval.migrate_database import DatabaseMigrator
from backend.app.evals.schema_type_eval.populate_schema_types import SchemaTypePopulator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("="*70)
    print("UPDATE DATABASE WITH should_include COLUMN")
    print("="*70)
    print()

    # Path to the new CSV file
    csv_path = "eval/eval_data_processed_exp_opt/eval_data_validation_lean_schema_type_filter_by_table_name.csv"

    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        print("Please make sure the file exists in the correct location.")
        return

    print(f"CSV file: {csv_path}")
    print()

    # Step 1: Add should_include column to database
    print("STEP 1: Adding should_include column to database")
    print("-"*70)

    migrator = DatabaseMigrator()
    print(f"Target database: {migrator.new_db}")

    choice = input("Add should_include column to database? (y/n): ").lower().strip()
    if choice != 'y':
        print("Cancelled.")
        return

    if not migrator.add_schema_type_column():
        print("❌ Failed to add should_include column")
        return

    print("✅ Column added successfully")
    print()

    # Step 2: Populate data from CSV
    print("STEP 2: Populating schema_type and should_include from CSV")
    print("-"*70)

    populator = SchemaTypePopulator(csv_path=csv_path, target_db=migrator.new_db)

    choice = input("Populate data from CSV? (y/n): ").lower().strip()
    if choice != 'y':
        print("Cancelled.")
        return

    if not populator.populate():
        print("❌ Failed to populate data")
        return

    print("✅ Data populated successfully")
    print()

    # Step 3: Show summary and next steps
    print("="*70)
    print("✅ DATABASE UPDATE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print()
    print("Next steps:")
    print("1. Run evaluations with the filter enabled:")
    print()
    print("   from backend.app.evals.evaluator import Evaluator")
    print()
    print("   # Filter to only semantically meaningful tables")
    print("   evaluator = Evaluator(")
    print("       data_split='eval_data_validation',")
    print("       embed_col='example_2rows_table_name_embed',")
    print("       k=10,")
    print("       num_embed=2,")
    print("       filter_should_include='Y'  # Only meaningful tables")
    print("   )")
    print()
    print("   avg_recalls, avg_times = evaluator.evaluate()")
    print()
    print("2. Compare results with and without the filter:")
    print()
    print("   # Without filter (all tables)")
    print("   evaluator_all = Evaluator(..., filter_should_include=None)")
    print()
    print("   # With filter (only meaningful tables)")
    print("   evaluator_filtered = Evaluator(..., filter_should_include='Y')")
    print()
    print("3. You can also filter for non-meaningful tables:")
    print()
    print("   evaluator_non_meaningful = Evaluator(..., filter_should_include='N')")
    print()

if __name__ == "__main__":
    main()
