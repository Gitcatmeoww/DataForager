"""
Table Count Evaluator for analyzing HySE performance on single-table vs multi-table databases.

A "single-table" database has exactly one table in the eval set; a "multi-table" database
has more than one. The classification is stored in the `table_count_type` column of
`eval_data_test`, populated by `add_table_count_column.py`.
"""

import logging
import os
import csv
import numpy as np
from scipy import stats
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv

from backend.app.evals.evaluator import Evaluator
from backend.app.db.connect_db import DatabaseConnection

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TableCountEvaluator(Evaluator):
    def __init__(
        self,
        data_split="eval_data_test",
        embed_col="example_2rows_table_name_embed",
        k=10,
        limit=None,
        num_embed=2,
        use_schema_db=True,
        filter_should_include=None
    ):
        self.use_schema_db = use_schema_db

        if use_schema_db:
            self.original_db_name = os.getenv('EVAL_DB_NAME')
            os.environ['EVAL_DB_NAME'] = 'HITS-eval-data-corpus-exp-opt-schema'

        super().__init__(data_split, embed_col, k, limit, num_embed, filter_should_include)

        # table_count_type-specific state
        self.table_count_data = {}          # table_name -> 'single_table' | 'multi_table'
        self.table_count_results = {
            'single_table': defaultdict(list),
            'multi_table': defaultdict(list)
        }

        self.table_count_results_dir = os.path.join(self.results_dir, "table_count_results")
        os.makedirs(self.table_count_results_dir, exist_ok=True)

        self.load_table_count_data()
        self.initialize_table_count_result_files()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def check_column_exists(self, table_name, column_name):
        """Check if a column exists in a table"""
        try:
            with self.db_connection as db:
                db.cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = %s AND column_name = %s
                    )
                """, (table_name, column_name))
                result = db.cursor.fetchone()
                return result['exists'] if result else False
        except Exception as e:
            logging.warning(f"Error checking column existence: {e}")
            return False

    def load_table_count_data(self):
        """Load table_count_type from the database into a dict keyed by table_name"""
        try:
            has_column = self.check_column_exists(self.data_split, 'table_count_type')
            if not has_column:
                logging.error(
                    f"Column 'table_count_type' not found in {self.data_split}. "
                    "Run add_table_count_column.py first."
                )
                self.table_count_data = {}
                return

            with self.db_connection as db:
                db.cursor.execute(f"""
                    SELECT table_name, table_count_type
                    FROM {self.data_split}
                    WHERE table_count_type IS NOT NULL
                """)
                rows = db.cursor.fetchall()

            for row in rows:
                self.table_count_data[row['table_name']] = row['table_count_type']

            single_count = sum(1 for v in self.table_count_data.values() if v == 'single_table')
            multi_count = sum(1 for v in self.table_count_data.values() if v == 'multi_table')

            logging.info(f"Loaded table_count_type for {len(self.table_count_data)} rows")
            logging.info(f"  single_table: {single_count}")
            logging.info(f"  multi_table:  {multi_count}")

        except Exception as e:
            logging.exception(f"Error loading table_count_data: {e}")
            self.table_count_data = {}

    # ------------------------------------------------------------------
    # Result files
    # ------------------------------------------------------------------

    def initialize_table_count_result_files(self):
        """Initialize per-category and comparison result CSV files"""
        self.single_table_results_file = os.path.join(self.table_count_results_dir, "single_table_results.csv")
        self.multi_table_results_file = os.path.join(self.table_count_results_dir, "multi_table_results.csv")
        self.comparison_results_file = os.path.join(self.table_count_results_dir, "table_count_comparison.csv")

        header = ['Index', 'Table Name', 'Method', 'Query Type', 'Query',
                  'Recall', 'Table Count Type', 'Ground Truth Header', 'Hypothetical Schema']

        for path in (self.single_table_results_file, self.multi_table_results_file):
            if not os.path.exists(path):
                with open(path, 'w', newline='') as f:
                    csv.writer(f).writerow(header)

        if not os.path.exists(self.comparison_results_file):
            with open(self.comparison_results_file, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'Method', 'Single Table Recall', 'Multi Table Recall', 'Recall Difference',
                    'Single Table Count', 'Multi Table Count', 'P Value', 'Significant'
                ])

    def save_table_count_row_result(
        self, idx, table_name, method_name, query_type, query,
        recall, table_count_type, ground_truth_header='', hypothetical_schema=''
    ):
        """Append a result row to the appropriate category file"""
        try:
            result_file = (
                self.single_table_results_file
                if table_count_type == 'single_table'
                else self.multi_table_results_file
            )
            with open(result_file, 'a', newline='') as f:
                csv.writer(f).writerow([
                    idx, table_name, method_name, query_type, query,
                    recall, table_count_type, ground_truth_header, hypothetical_schema
                ])
        except Exception as e:
            logging.exception(f"Error saving table count result for index {idx}: {e}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_by_table_count(self):
        """Run evaluation and collect recalls grouped by table_count_type"""
        logging.info("Starting table count evaluation...")

        # Best configuration determined on validation set:
        # normalized schema generation (relational), N=2, corpus embed = example_2rows_table_name_embed
        methods = [
            # {
            #     'name': 'Multi-Component HySE (Relational)',
            #     'function': self.eval_methods.multi_component_hyse_search,
            #     'query_type': 'task',
            #     'schema_approach': 'relational'
            # },
            {
                'name': 'Semantic Task Search',
                'function': self.eval_methods.semantic_search,
                'query_type': 'task'
            },
        ]

        recalls_by_count = {
            'single_table': {m['name']: [] for m in methods},
            'multi_table': {m['name']: [] for m in methods}
        }
        retrieval_times_by_count = {
            'single_table': {m['name']: [] for m in methods},
            'multi_table': {m['name']: [] for m in methods}
        }

        for idx, ground_truth_table in tqdm(
            enumerate(self.ground_truths),
            total=len(self.ground_truths),
            desc="Evaluating by Table Count",
            unit="entry"
        ):
            try:
                if ground_truth_table not in self.table_count_data:
                    logging.warning(f"No table_count_type for '{ground_truth_table}', skipping")
                    continue

                table_count_type = self.table_count_data[ground_truth_table]
                task_queries = self.task_queries[idx]

                for method in methods:
                    method_name = method['name']
                    search_function = method['function']
                    query_type = method['query_type']
                    queries = task_queries if query_type == 'task' else self.keywords[idx]

                    for query in queries:
                        try:
                            retrieval_time = 0
                            if method_name.startswith('Multi-Component HySE'):
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

                            recalls_by_count[table_count_type][method_name].append(recall)
                            retrieval_times_by_count[table_count_type][method_name].append(retrieval_time)

                            self.save_row_result(
                                idx, ground_truth_table, method_name, query_type, query, recall, '', ''
                            )
                            self.save_table_count_row_result(
                                idx, ground_truth_table, method_name, query_type, query,
                                recall, table_count_type, '', ''
                            )

                        except Exception as e:
                            logging.exception(
                                f"Error in {method_name} with query '{query}' at index {idx}: {e}"
                            )
                            recalls_by_count[table_count_type][method_name].append(0)
                            retrieval_times_by_count[table_count_type][method_name].append(0)

            except Exception as e:
                logging.exception(f"Error processing row {idx} (table: {ground_truth_table}): {e}")

        return recalls_by_count, retrieval_times_by_count

    # ------------------------------------------------------------------
    # Comparison & reporting
    # ------------------------------------------------------------------

    def compare_table_count_performance(self, recalls_by_count):
        """Statistical comparison between single_table and multi_table recalls"""
        logging.info("Comparing table count performance...")

        comparison_results = []

        for method_name in recalls_by_count['single_table'].keys():
            single_recalls = recalls_by_count['single_table'][method_name]
            multi_recalls = recalls_by_count['multi_table'][method_name]

            if not single_recalls or not multi_recalls:
                logging.warning(f"Insufficient data for {method_name}")
                continue

            single_mean = np.mean(single_recalls)
            multi_mean = np.mean(multi_recalls)
            recall_diff = single_mean - multi_mean

            try:
                _, p_value = stats.mannwhitneyu(
                    single_recalls, multi_recalls, alternative='two-sided'
                )
                significant = p_value < 0.05
            except Exception as e:
                logging.warning(f"Could not perform statistical test for {method_name}: {e}")
                p_value = None
                significant = False

            result = {
                'method': method_name,
                'single_table_recall': single_mean,
                'multi_table_recall': multi_mean,
                'recall_difference': recall_diff,
                'single_table_count': len(single_recalls),
                'multi_table_count': len(multi_recalls),
                'p_value': p_value,
                'significant': significant
            }
            comparison_results.append(result)

            logging.info(f"\nResults for {method_name}:")
            logging.info(f"  Single Table Recall@{self.k}: {single_mean:.4f} (n={len(single_recalls)})")
            logging.info(f"  Multi Table Recall@{self.k}:  {multi_mean:.4f}  (n={len(multi_recalls)})")
            logging.info(f"  Difference: {recall_diff:.4f}")
            if p_value is not None:
                logging.info(f"  P-value: {p_value:.4f} ({'Significant' if significant else 'Not significant'})")

        self.save_comparison_results(comparison_results)
        return comparison_results

    def save_comparison_results(self, comparison_results):
        """Write comparison results to CSV"""
        try:
            with open(self.comparison_results_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Method', 'Single Table Recall', 'Multi Table Recall', 'Recall Difference',
                    'Single Table Count', 'Multi Table Count', 'P Value', 'Significant'
                ])
                for r in comparison_results:
                    writer.writerow([
                        r['method'],
                        f"{r['single_table_recall']:.4f}",
                        f"{r['multi_table_recall']:.4f}",
                        f"{r['recall_difference']:.4f}",
                        r['single_table_count'],
                        r['multi_table_count'],
                        f"{r['p_value']:.4f}" if r['p_value'] is not None else 'N/A',
                        r['significant']
                    ])
            logging.info(f"Comparison results saved to: {self.comparison_results_file}")
        except Exception as e:
            logging.exception(f"Error saving comparison results: {e}")

    def generate_analysis_summary(self, recalls_by_count, retrieval_times_by_count, comparison_results):
        """Write a human-readable summary file"""
        summary_file = os.path.join(self.table_count_results_dir, "analysis_summary.txt")
        try:
            with open(summary_file, 'w') as f:
                f.write("Table Count Type Analysis Summary\n")
                f.write("=" * 50 + "\n\n")

                single_count = sum(1 for v in self.table_count_data.values() if v == 'single_table')
                multi_count = sum(1 for v in self.table_count_data.values() if v == 'multi_table')

                f.write("Dataset Statistics:\n")
                f.write(f"  Total tables:        {len(self.table_count_data)}\n")
                f.write(f"  Single-table DBs:    {single_count}\n")
                f.write(f"  Multi-table DBs:     {multi_count}\n\n")

                f.write("Performance Comparison:\n")
                for r in comparison_results:
                    f.write(f"\nMethod: {r['method']}\n")
                    f.write(f"  Single Table Recall@{self.k}: {r['single_table_recall']:.4f} (n={r['single_table_count']})\n")
                    f.write(f"  Multi Table Recall@{self.k}:  {r['multi_table_recall']:.4f}  (n={r['multi_table_count']})\n")
                    f.write(f"  Difference: {r['recall_difference']:.4f}\n")
                    if r['p_value'] is not None:
                        f.write(f"  Statistical Significance: {'Yes' if r['significant'] else 'No'} (p={r['p_value']:.4f})\n")

                f.write("\nKey Findings:\n")
                if comparison_results:
                    best = max(comparison_results, key=lambda x: abs(x['recall_difference']))
                    if best['recall_difference'] > 0:
                        f.write("  - Single-table databases generally perform better\n")
                        f.write(f"  - Largest advantage: {best['recall_difference']:.4f} for {best['method']}\n")
                    elif best['recall_difference'] < 0:
                        f.write("  - Multi-table databases generally perform better\n")
                        f.write(f"  - Largest advantage: {abs(best['recall_difference']):.4f} for {best['method']}\n")
                    else:
                        f.write("  - Performance is similar between single- and multi-table databases\n")

                f.write("\nFiles Generated:\n")
                f.write(f"  - {os.path.basename(self.single_table_results_file)}\n")
                f.write(f"  - {os.path.basename(self.multi_table_results_file)}\n")
                f.write(f"  - {os.path.basename(self.comparison_results_file)}\n")
                f.write(f"  - {os.path.basename(summary_file)}\n")

            logging.info(f"Analysis summary saved to: {summary_file}")
        except Exception as e:
            logging.exception(f"Error generating analysis summary: {e}")

    # ------------------------------------------------------------------
    # Top-level orchestrator
    # ------------------------------------------------------------------

    def run_table_count_analysis(self):
        """Run the full single-table vs multi-table analysis"""
        logging.info("Starting table count type analysis...")

        if not self.table_count_data:
            logging.error(
                "No table_count_type data available. "
                "Run add_table_count_column.py first."
            )
            return None

        recalls_by_count, retrieval_times_by_count = self.evaluate_by_table_count()
        comparison_results = self.compare_table_count_performance(recalls_by_count)
        self.generate_analysis_summary(recalls_by_count, retrieval_times_by_count, comparison_results)

        return {
            'recalls_by_count': recalls_by_count,
            'retrieval_times_by_count': retrieval_times_by_count,
            'comparison_results': comparison_results
        }

    def __del__(self):
        """Restore original EVAL_DB_NAME if we overrode it"""
        if hasattr(self, 'use_schema_db') and self.use_schema_db and hasattr(self, 'original_db_name'):
            os.environ['EVAL_DB_NAME'] = self.original_db_name


if __name__ == "__main__":
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    evaluator = TableCountEvaluator(
        data_split="eval_data_test",
        embed_col="example_2rows_table_name_embed",
        k=10,
        limit=None,
        num_embed=2,
        use_schema_db=True,
        filter_should_include=None
    )

    results = evaluator.run_table_count_analysis()

    if results:
        print("\nTable Count Analysis Results:")
        print("=" * 50)
        for r in results['comparison_results']:
            print(f"\nMethod: {r['method']}")
            print(f"  Single Table Recall@{evaluator.k}: {r['single_table_recall']:.4f}")
            print(f"  Multi Table Recall@{evaluator.k}:  {r['multi_table_recall']:.4f}")
            print(f"  Difference: {r['recall_difference']:.4f}")
            if r['p_value'] is not None:
                print(f"  Significant: {'Yes' if r['significant'] else 'No'} (p={r['p_value']:.4f})")
        print(f"\nDetailed results saved to: {evaluator.table_count_results_dir}")
    else:
        print("Analysis failed. Check logs for details.")
