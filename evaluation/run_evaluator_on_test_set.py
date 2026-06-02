"""
Run the base Evaluator on eval_data_test
"""

import logging
from evaluation.evaluator import Evaluator

# Configure logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    print("=" * 70)
    print("Running Evaluator on eval_data_test")
    print("=" * 70)

    # Initialize evaluator with test set
    evaluator = Evaluator(
        data_split="eval_data_test",
        embed_col="example_2rows_table_name_embed",
        k=10,
        # limit=50,  # Uncomment to test with a subset
        num_embed=2,
        filter_should_include=None  # Test set doesn't have should_include column
    )

    print(f"\nEvaluator Configuration:")
    print(f"  Data split: {evaluator.data_split}")
    print(f"  Embed column: {evaluator.embed_col}")
    print(f"  K: {evaluator.k}")
    print(f"  Ground truths: {len(evaluator.ground_truths)}")
    print(f"  Num embed: {evaluator.num_embed}")
    print()

    # Run evaluation
    print("Starting evaluation...")
    print("This may take a while depending on the dataset size and API rate limits...")
    print()

    results = evaluator.evaluate()

    # Print results
    if results:
        print("\n" + "=" * 70)
        print("Evaluation Results:")
        print("=" * 70)

        for method_name, avg_recall in results['avg_recalls'].items():
            print(f"\n{method_name}:")
            print(f"  Average Recall@{evaluator.k}: {avg_recall:.4f}")

            if method_name in results.get('avg_retrieval_times', {}):
                avg_time = results['avg_retrieval_times'][method_name]
                if avg_time > 0:
                    print(f"  Average Retrieval Time: {avg_time:.4f} seconds")

        print(f"\nResults saved to: {evaluator.results_dir}")
        print(f"  - Per-row results: {evaluator.results_file}")
        print(f"  - Failed rows: {evaluator.failed_rows_file}")
        print(f"  - Failed queries: {evaluator.failed_queries_file}")
    else:
        print("\n❌ Evaluation failed. Check logs for details.")
