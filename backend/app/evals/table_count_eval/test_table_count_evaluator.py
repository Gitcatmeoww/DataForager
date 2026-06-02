"""
Smoke-test script to verify TableCountEvaluator works end-to-end with eval_data_test.

Run this after add_table_count_column.py to confirm the pipeline is healthy
before launching a full evaluation.
"""

import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from backend.app.evals.table_count_eval.table_count_evaluator import TableCountEvaluator

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def test_with_test_set():
    print("\n" + "=" * 70)
    print("Testing TableCountEvaluator with eval_data_test (Test Set)")
    print("=" * 70 + "\n")

    try:
        evaluator = TableCountEvaluator(
            data_split="eval_data_test",
            embed_col="example_2rows_table_name_embed",
            k=50,
            limit=10,
            num_embed=2,
            use_schema_db=True,
            filter_should_include=None
        )

        print(f"\nTableCountEvaluator initialized successfully!")
        print(f"  Data split:              {evaluator.data_split}")
        print(f"  Embed column:            {evaluator.embed_col}")
        print(f"  K:                       {evaluator.k}")
        print(f"  Limit:                   {evaluator.limit}")
        print(f"  Ground truths loaded:    {len(evaluator.ground_truths)}")
        print(f"  Table count data loaded: {len(evaluator.table_count_data)}")

        single_count = sum(1 for v in evaluator.table_count_data.values() if v == 'single_table')
        multi_count = sum(1 for v in evaluator.table_count_data.values() if v == 'multi_table')
        print(f"\n  table_count_type distribution (full test set):")
        print(f"    single_table: {single_count}")
        print(f"    multi_table:  {multi_count}")

        print(f"\n  Running analysis (limited to {evaluator.limit} records)...")
        results = evaluator.run_table_count_analysis()

        if results:
            print("\nAnalysis completed successfully!")
            print("\n  Results:")
            for r in results['comparison_results']:
                print(f"\n  Method: {r['method']}")
                print(f"    Single Table Recall@{evaluator.k}: {r['single_table_recall']:.4f}")
                print(f"    Multi Table Recall@{evaluator.k}:  {r['multi_table_recall']:.4f}")
                print(f"    Difference: {r['recall_difference']:.4f}")
                if r['p_value'] is not None:
                    print(f"    P-value: {r['p_value']:.4f}")
                    print(f"    Significant: {'Yes' if r['significant'] else 'No'}")

            print(f"\n  Detailed results saved to: {evaluator.table_count_results_dir}")
            return True
        else:
            print("\nAnalysis failed — no results returned")
            return False

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        logging.exception("Full error details:")
        return False


if __name__ == "__main__":
    success = test_with_test_set()

    print("\n" + "=" * 70)
    if success:
        print("TEST PASSED: TableCountEvaluator works with eval_data_test!")
        print("\nTo run the full evaluation, use table_count_evaluator.py with limit=None.")
    else:
        print("TEST FAILED: Check logs above for details.")
    print("=" * 70 + "\n")
