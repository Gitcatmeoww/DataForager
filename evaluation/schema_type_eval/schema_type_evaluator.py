"""
Schema Type Evaluator for analyzing Hyse performance on normalized vs denormalized tables.

This evaluator extends the base Evaluator class to provide schema-type-aware evaluation
capabilities, allowing for detailed subgroup analysis and performance comparison.
"""

import logging
import os
import csv
import numpy as np
from scipy import stats
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv

from evaluation.evaluator import Evaluator
from dataforager.db.connect_db import DatabaseConnection

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SchemaTypeEvaluator(Evaluator):
    def __init__(self, data_split="eval_data_validation", embed_col="example_rows_embed", k=10, limit=None, num_embed=1, use_schema_db=True, filter_should_include=None):
        self.use_schema_db = use_schema_db

        # If using schema database, override the database connection
        if use_schema_db:
            self.original_db_name = os.getenv('EVAL_DB_NAME')
            os.environ['EVAL_DB_NAME'] = 'HITS-eval-data-corpus-exp-opt-schema'

        # Initialize parent class
        super().__init__(data_split, embed_col, k, limit, num_embed, filter_should_include)

        # Schema type specific attributes
        self.schema_type_data = {}
        self.schema_type_results = {
            'normalized': defaultdict(list),
            'denormalized': defaultdict(list)
        }
        self.schema_type_stats = {}

        # Results directories
        self.schema_results_dir = os.path.join(self.results_dir, "schema_type_results")
        os.makedirs(self.schema_results_dir, exist_ok=True)

        # Load schema type information
        self.load_schema_type_data()

        # Initialize schema-specific result files
        self.initialize_schema_results_files()

    def check_column_exists(self, table_name, column_name):
        """Check if a column exists in a table"""
        try:
            with self.db_connection as db:
                db.cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_name = %s AND column_name = %s
                    )
                """, (table_name, column_name))
                result = db.cursor.fetchone()
                return result['exists'] if result else False
        except Exception as e:
            logging.warning(f"Error checking if column exists: {e}")
            return False

    def load_schema_type_data(self):
        """Load schema type information from the database"""
        try:
            # Check if should_include column exists
            has_should_include = self.check_column_exists(self.data_split, 'should_include')

            with self.db_connection as db:
                query = f"""
                SELECT table_name, schema_type
                FROM {self.data_split}
                WHERE schema_type IS NOT NULL
                """

                # Add filter for should_include if specified and column exists
                if self.filter_should_include is not None and has_should_include:
                    query += f" AND should_include = '{self.filter_should_include}'"
                elif self.filter_should_include is not None and not has_should_include:
                    logging.warning(f"Column 'should_include' does not exist in {self.data_split}, skipping filter")

                logging.info(f"Loading schema type data with query: {query}")
                db.cursor.execute(query)
                rows = db.cursor.fetchall()

                for row in rows:
                    self.schema_type_data[row['table_name']] = row['schema_type']

                # Get distribution
                normalized_count = sum(1 for st in self.schema_type_data.values() if st == 'normalized')
                denormalized_count = sum(1 for st in self.schema_type_data.values() if st == 'denormalized')

                logging.info(f"Loaded schema type data for {len(self.schema_type_data)} tables")
                logging.info(f"  Normalized: {normalized_count}")
                logging.info(f"  Denormalized: {denormalized_count}")
                if self.filter_should_include is not None:
                    logging.info(f"  Filtered by should_include = '{self.filter_should_include}'")

        except Exception as e:
            logging.exception(f"Error loading schema type data: {e}")
            self.schema_type_data = {}

    def initialize_schema_results_files(self):
        """Initialize schema-type-specific result files"""
        # Schema-specific per-row results
        self.normalized_results_file = os.path.join(self.schema_results_dir, "normalized_results.csv")
        self.denormalized_results_file = os.path.join(self.schema_results_dir, "denormalized_results.csv")
        self.comparison_results_file = os.path.join(self.schema_results_dir, "schema_comparison.csv")

        # Initialize normalized results file
        if not os.path.exists(self.normalized_results_file):
            with open(self.normalized_results_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = [
                    'Index', 'Table Name', 'Method', 'Query Type', 'Query',
                    'Recall', 'Schema Type', 'Ground Truth Header', 'Hypothetical Schema'
                ]
                writer.writerow(header)

        # Initialize denormalized results file
        if not os.path.exists(self.denormalized_results_file):
            with open(self.denormalized_results_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = [
                    'Index', 'Table Name', 'Method', 'Query Type', 'Query',
                    'Recall', 'Schema Type', 'Ground Truth Header', 'Hypothetical Schema'
                ]
                writer.writerow(header)

        # Initialize comparison results file
        if not os.path.exists(self.comparison_results_file):
            with open(self.comparison_results_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = [
                    'Method', 'Normalized Recall', 'Denormalized Recall', 'Recall Difference',
                    'Normalized Count', 'Denormalized Count', 'P Value', 'Significant'
                ]
                writer.writerow(header)

    def save_schema_row_result(self, idx, table_name, method_name, query_type, query, recall, schema_type, ground_truth_header='', hypothetical_schema=''):
        """Save result to schema-type-specific file"""
        try:
            # Determine which file to use
            result_file = self.normalized_results_file if schema_type == 'normalized' else self.denormalized_results_file

            with open(result_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx, table_name, method_name, query_type, query, recall,
                    schema_type, ground_truth_header, hypothetical_schema
                ])
        except Exception as e:
            logging.exception(f"Error saving schema result for index {idx}: {e}")

    def evaluate_by_schema_type(self):
        """Run evaluation separately for each schema type"""
        logging.info("Starting schema type evaluation...")

        # Best configuration determined on validation set:
        # normalized schema generation (relational), N=2, corpus embed = example_2rows_table_name_embed
        methods = [
            # {'name': 'Multi-Component HySE (Relational)', 'function': self.eval_methods.multi_component_hyse_search, 'query_type': 'task', 'schema_approach': 'relational'},
            {'name': 'Semantic Task Search', 'function': self.eval_methods.semantic_search, 'query_type': 'task'},
        ]

        recalls_by_schema = {
            'normalized': {method['name']: [] for method in methods},
            'denormalized': {method['name']: [] for method in methods}
        }

        retrieval_times_by_schema = {
            'normalized': {method['name']: [] for method in methods},
            'denormalized': {method['name']: [] for method in methods}
        }

        # Iterate over each entry
        for idx, ground_truth_table in tqdm(enumerate(self.ground_truths), total=len(self.ground_truths), desc="Evaluating by Schema Type", unit="entry"):
            try:
                # Skip if table doesn't have schema type information
                if ground_truth_table not in self.schema_type_data:
                    logging.warning(f"No schema type for table {ground_truth_table}, skipping")
                    continue

                schema_type = self.schema_type_data[ground_truth_table]
                task_queries = self.task_queries[idx]

                for method in methods:
                    method_name = method['name']
                    search_function = method['function']
                    query_type = method['query_type']
                    queries = task_queries if query_type == 'task' else self.keywords[idx]

                    for query in queries:
                        try:
                            retrieval_time = 0
                            if method_name.startswith('HySE Search') or method_name.startswith('Multi-Component HySE'):
                                schema_approach = method.get('schema_approach', 'relational')
                                results, retrieval_time = search_function(
                                    query=query,
                                    num_embed=self.num_embed,
                                    schema_approach=schema_approach,
                                    return_timing=True
                                )
                            else:
                                results = search_function(query=query, query_type=query_type)

                            recall = self.compute_recall_at_k(results, ground_truth_table)

                            # Store by schema type
                            recalls_by_schema[schema_type][method_name].append(recall)
                            retrieval_times_by_schema[schema_type][method_name].append(retrieval_time)

                            # Save to both general and schema-specific files
                            self.save_row_result(
                                idx, ground_truth_table, method_name, query_type, query, recall,
                                '', ''
                            )

                            self.save_schema_row_result(
                                idx, ground_truth_table, method_name, query_type, query, recall,
                                schema_type, '', ''
                            )

                        except Exception as e:
                            logging.exception(f"Error in {method_name} with query '{query}' at index {idx}: {e}")
                            recalls_by_schema[schema_type][method_name].append(0)
                            retrieval_times_by_schema[schema_type][method_name].append(0)

            except Exception as e:
                logging.exception(f"Error processing row {idx} (table: {ground_truth_table}): {e}")

        return recalls_by_schema, retrieval_times_by_schema

    def compare_schema_performance(self, recalls_by_schema):
        """Compare performance between normalized and denormalized tables"""
        logging.info("Comparing schema type performance...")

        comparison_results = []

        for method_name in recalls_by_schema['normalized'].keys():
            normalized_recalls = recalls_by_schema['normalized'][method_name]
            denormalized_recalls = recalls_by_schema['denormalized'][method_name]

            if not normalized_recalls or not denormalized_recalls:
                logging.warning(f"Insufficient data for {method_name}")
                continue

            # Calculate statistics
            norm_mean = np.mean(normalized_recalls)
            denorm_mean = np.mean(denormalized_recalls)
            recall_diff = norm_mean - denorm_mean

            # Statistical significance test (Mann-Whitney U test for non-parametric comparison)
            try:
                statistic, p_value = stats.mannwhitneyu(
                    normalized_recalls, denormalized_recalls,
                    alternative='two-sided'
                )
                significant = p_value < 0.05
            except Exception as e:
                logging.warning(f"Could not perform statistical test for {method_name}: {e}")
                p_value = None
                significant = False

            # Store results
            result = {
                'method': method_name,
                'normalized_recall': norm_mean,
                'denormalized_recall': denorm_mean,
                'recall_difference': recall_diff,
                'normalized_count': len(normalized_recalls),
                'denormalized_count': len(denormalized_recalls),
                'p_value': p_value,
                'significant': significant
            }

            comparison_results.append(result)

            # Log results
            logging.info(f"\nResults for {method_name}:")
            logging.info(f"  Normalized Recall@{self.k}: {norm_mean:.4f} (n={len(normalized_recalls)})")
            logging.info(f"  Denormalized Recall@{self.k}: {denorm_mean:.4f} (n={len(denormalized_recalls)})")
            logging.info(f"  Difference: {recall_diff:.4f}")
            if p_value is not None:
                logging.info(f"  P-value: {p_value:.4f} ({'Significant' if significant else 'Not significant'})")

        # Save comparison results
        self.save_comparison_results(comparison_results)

        return comparison_results

    def save_comparison_results(self, comparison_results):
        """Save comparison results to CSV"""
        try:
            with open(self.comparison_results_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header
                header = [
                    'Method', 'Normalized Recall', 'Denormalized Recall', 'Recall Difference',
                    'Normalized Count', 'Denormalized Count', 'P Value', 'Significant'
                ]
                writer.writerow(header)

                # Write data
                for result in comparison_results:
                    writer.writerow([
                        result['method'],
                        f"{result['normalized_recall']:.4f}",
                        f"{result['denormalized_recall']:.4f}",
                        f"{result['recall_difference']:.4f}",
                        result['normalized_count'],
                        result['denormalized_count'],
                        f"{result['p_value']:.4f}" if result['p_value'] is not None else 'N/A',
                        result['significant']
                    ])

            logging.info(f"Comparison results saved to: {self.comparison_results_file}")

        except Exception as e:
            logging.exception(f"Error saving comparison results: {e}")

    def run_schema_type_analysis(self):
        """Run complete schema type analysis"""
        logging.info("Starting comprehensive schema type analysis...")

        # Check if we have schema type data
        if not self.schema_type_data:
            logging.error("No schema type data available. Please ensure the database has been properly migrated and populated.")
            return None

        # Run evaluation by schema type
        recalls_by_schema, retrieval_times_by_schema = self.evaluate_by_schema_type()

        # Compare performance
        comparison_results = self.compare_schema_performance(recalls_by_schema)

        # Generate summary
        self.generate_analysis_summary(recalls_by_schema, retrieval_times_by_schema, comparison_results)

        return {
            'recalls_by_schema': recalls_by_schema,
            'retrieval_times_by_schema': retrieval_times_by_schema,
            'comparison_results': comparison_results
        }

    def generate_analysis_summary(self, recalls_by_schema, retrieval_times_by_schema, comparison_results):
        """Generate a comprehensive analysis summary"""
        summary_file = os.path.join(self.schema_results_dir, "analysis_summary.txt")

        try:
            with open(summary_file, 'w') as f:
                f.write("Schema Type Analysis Summary\n")
                f.write("=" * 50 + "\n\n")

                # Dataset statistics
                normalized_count = sum(1 for st in self.schema_type_data.values() if st == 'normalized')
                denormalized_count = sum(1 for st in self.schema_type_data.values() if st == 'denormalized')

                f.write(f"Dataset Statistics:\n")
                f.write(f"  Total tables: {len(self.schema_type_data)}\n")
                f.write(f"  Normalized tables: {normalized_count}\n")
                f.write(f"  Denormalized tables: {denormalized_count}\n\n")

                # Performance comparison
                f.write("Performance Comparison:\n")
                for result in comparison_results:
                    f.write(f"\nMethod: {result['method']}\n")
                    f.write(f"  Normalized Recall@{self.k}: {result['normalized_recall']:.4f}\n")
                    f.write(f"  Denormalized Recall@{self.k}: {result['denormalized_recall']:.4f}\n")
                    f.write(f"  Difference: {result['recall_difference']:.4f}\n")
                    if result['p_value'] is not None:
                        f.write(f"  Statistical Significance: {'Yes' if result['significant'] else 'No'} (p={result['p_value']:.4f})\n")

                # Key findings
                f.write(f"\nKey Findings:\n")

                # Find best performing schema type
                if comparison_results:
                    best_result = max(comparison_results, key=lambda x: abs(x['recall_difference']))
                    if best_result['recall_difference'] > 0:
                        f.write(f"  - Normalized tables generally perform better\n")
                        f.write(f"  - Largest advantage: {best_result['recall_difference']:.4f} for {best_result['method']}\n")
                    elif best_result['recall_difference'] < 0:
                        f.write(f"  - Denormalized tables generally perform better\n")
                        f.write(f"  - Largest advantage: {abs(best_result['recall_difference']):.4f} for {best_result['method']}\n")
                    else:
                        f.write(f"  - Performance is similar between schema types\n")

                f.write(f"\nFiles Generated:\n")
                f.write(f"  - Normalized results: {os.path.basename(self.normalized_results_file)}\n")
                f.write(f"  - Denormalized results: {os.path.basename(self.denormalized_results_file)}\n")
                f.write(f"  - Comparison results: {os.path.basename(self.comparison_results_file)}\n")
                f.write(f"  - This summary: {os.path.basename(summary_file)}\n")

            logging.info(f"Analysis summary saved to: {summary_file}")

        except Exception as e:
            logging.exception(f"Error generating analysis summary: {e}")

    def __del__(self):
        """Restore original database name if needed"""
        if hasattr(self, 'use_schema_db') and self.use_schema_db and hasattr(self, 'original_db_name'):
            os.environ['EVAL_DB_NAME'] = self.original_db_name


if __name__ == "__main__":
    # Force logging config
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    evaluator = SchemaTypeEvaluator(
        data_split="eval_data_test",
        embed_col="example_2rows_table_name_embed",
        k=10,
        # limit=10,
        num_embed=2,
        use_schema_db=True,
        filter_should_include=None
    )

    # Run schema type analysis
    results = evaluator.run_schema_type_analysis()

    if results:
        print("\nSchema Type Analysis Results:")
        print("=" * 50)

        for result in results['comparison_results']:
            print(f"\nMethod: {result['method']}")
            print(f"  Normalized Recall@{evaluator.k}: {result['normalized_recall']:.4f}")
            print(f"  Denormalized Recall@{evaluator.k}: {result['denormalized_recall']:.4f}")
            print(f"  Difference: {result['recall_difference']:.4f}")
            if result['p_value'] is not None:
                print(f"  Significant: {'Yes' if result['significant'] else 'No'} (p={result['p_value']:.4f})")

        print(f"\nDetailed results saved to: {evaluator.schema_results_dir}")
    else:
        print("Analysis failed. Check logs for details.")