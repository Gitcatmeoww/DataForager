"""
Test script to verify SchemaTypeEvaluator works with eval_data_test.
"""

import logging

from evaluation.schema_type_eval.schema_type_evaluator import SchemaTypeEvaluator

# Force logging config
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_with_test_set():
    """Test SchemaTypeEvaluator with eval_data_test"""
    print("\n" + "=" * 70)
    print("Testing SchemaTypeEvaluator with eval_data_test (Test Set)")
    print("=" * 70 + "\n")

    try:
        # Create evaluator with test set configuration
        evaluator = SchemaTypeEvaluator(
            data_split="eval_data_test",
            embed_col="example_2rows_table_name_embed",
            k=50,
            limit=10,  # Small sample for testing
            num_embed=2,
            use_schema_db=True,
            filter_should_include=None  # Test set doesn't have `should_include` column
        )

        print(f"\n✅ SchemaTypeEvaluator initialized successfully!")
        print(f"   Data split: {evaluator.data_split}")
        print(f"   Embed column: {evaluator.embed_col}")
        print(f"   K: {evaluator.k}")
        print(f"   Limit: {evaluator.limit}")
        print(f"   Ground truths loaded: {len(evaluator.ground_truths)}")
        print(f"   Schema type data loaded: {len(evaluator.schema_type_data)}")

        # Check schema type distribution
        normalized_count = sum(1 for st in evaluator.schema_type_data.values() if st == 'normalized')
        denormalized_count = sum(1 for st in evaluator.schema_type_data.values() if st == 'denormalized')
        print(f"\n   Schema type distribution:")
        print(f"     Normalized: {normalized_count}")
        print(f"     Denormalized: {denormalized_count}")

        # Run a small evaluation to verify everything works
        print(f"\n   Running schema type analysis (limited to {evaluator.limit} records)...")
        results = evaluator.run_schema_type_analysis()

        if results:
            print("\n✅ Schema type analysis completed successfully!")
            print("\n   Results:")
            for result in results['comparison_results']:
                print(f"\n   Method: {result['method']}")
                print(f"     Normalized Recall@{evaluator.k}: {result['normalized_recall']:.4f}")
                print(f"     Denormalized Recall@{evaluator.k}: {result['denormalized_recall']:.4f}")
                print(f"     Difference: {result['recall_difference']:.4f}")
                if result['p_value'] is not None:
                    print(f"     P-value: {result['p_value']:.4f}")
                    print(f"     Significant: {'Yes' if result['significant'] else 'No'}")

            print(f"\n   Detailed results saved to: {evaluator.schema_results_dir}")
            return True
        else:
            print("\n❌ Schema type analysis failed - no results returned")
            return False

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logging.exception("Full error details:")
        return False


if __name__ == "__main__":
    success = test_with_test_set()

    print("\n" + "=" * 70)
    if success:
        print("✅ TEST PASSED: SchemaTypeEvaluator works with eval_data_test!")
        print("\nYou can now run the full evaluation on the test set by:")
        print("1. Updating the limit parameter (or remove it for full test set)")
        print("2. Running schema_type_evaluator.py with:")
        print("   evaluator = SchemaTypeEvaluator(")
        print("       data_split='eval_data_test',")
        print("       embed_col='example_2rows_table_name_embed',")
        print("       k=50,")
        print("       num_embed=2,")
        print("       use_schema_db=True,")
        print("       filter_should_include=None")
        print("   )")
    else:
        print("❌ TEST FAILED: Check logs above for details")
    print("=" * 70 + "\n")
