import logging
import os
import csv
import random
from tqdm import tqdm
from dotenv import load_dotenv
from dataforager.db.connect_db import DatabaseConnection
from evaluation.eval_methods import EvalMethods
from evaluation.eval_utils import get_ground_truth_header, get_hypo_schema
from dataforager.hyse.hypo_schema_search import cos_sim_search

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Evaluator:
    def __init__(self, data_split="eval_data_validation", embed_col="example_rows_embed", k=10, limit=None, num_embed=1, filter_should_include=None):
        self.data_split = data_split
        self.embed_col = embed_col
        self.k = k
        self.limit = limit
        self.num_embed = num_embed
        self.filter_should_include = filter_should_include  # None, 'Y', or 'N'
        self.eval_methods = EvalMethods(data_split=data_split, embed_col=embed_col, k=k)
        self.db_connection = DatabaseConnection()
        self.ground_truths = []
        self.task_queries = []
        self.metadata_queries = []
        self.keywords = []
        self.results_dir = "evaluation/results"
        self.results_file = os.path.join(self.results_dir, "per_row_results.csv")
        self.failed_rows_file = os.path.join(self.results_dir, "failed_rows.csv")
        self.failed_queries_file = os.path.join(self.results_dir, "failed_queries.csv")
        self.load_data()
        self.initialize_results_files()

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

    def load_data(self):
        try:
            # Check if should_include column exists
            has_should_include = self.check_column_exists(self.data_split, 'should_include')

            with self.db_connection as db:
                query = f"SELECT table_name, task_queries, keywords, metadata_queries FROM {self.data_split}"

                # Add filter for should_include if specified and column exists
                if self.filter_should_include is not None and has_should_include:
                    query += f" WHERE should_include = '{self.filter_should_include}'"
                elif self.filter_should_include is not None and not has_should_include:
                    logging.warning(f"Column 'should_include' does not exist in {self.data_split}, skipping filter")

                if self.limit:
                    query += f" LIMIT {self.limit}"

                query += ";"

                logging.info(f"Loading data with query: {query}")
                db.cursor.execute(query)
                rows = db.cursor.fetchall()

                # Each row has:
                #  row['table_name'] -> string
                #  row['task_queries'] -> text[] (3 queries)
                #  row['keywords'] -> text[]
                #  row['metadata_queries'] -> jsonb array of arrays
                for row in rows:
                    self.ground_truths.append(row['table_name'])
                    self.task_queries.append(row['task_queries'])
                    self.keywords.append(row['keywords'])
                    self.metadata_queries.append(row['metadata_queries'])

                logging.info(f"Loaded {len(self.ground_truths)} records from database")
                if self.filter_should_include is not None:
                    logging.info(f"  Filtered by should_include = '{self.filter_should_include}'")
        except Exception as e:
            logging.exception(f"Error loading data from the database: {e}")

    # Initialize the results files by writing the headers if they don't exist
    def initialize_results_files(self):
        # Ensure the results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize per_row_results.csv
        if not os.path.exists(self.results_file):
            with open(self.results_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = [
                    'Index', 'Table Name', 'Method', 'Query Type', 'Query',
                    'Recall', 'Ground Truth Header', 'Hypothetical Schema'
                ]
                writer.writerow(header)
            logging.info(f"Initialized results file '{self.results_file}'")

        # Initialize failed_rows.csv
        if not os.path.exists(self.failed_rows_file):
            with open(self.failed_rows_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['Index', 'Table Name', 'Error Message']
                writer.writerow(header)
            logging.info(f"Initialized failed rows file '{self.failed_rows_file}'")

        # Initialize failed_queries.csv
        if not os.path.exists(self.failed_queries_file):
            with open(self.failed_queries_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['Index', 'Table Name', 'Method', 'Query Type', 'Query', 'Error Message']
                writer.writerow(header)
            logging.info(f"Initialized failed queries file '{self.failed_queries_file}'")

    # Compute recall @k for a single query
    def compute_recall_at_k(self, results_list, ground_truth):
        return 1 if ground_truth in results_list else 0

    # Run the evaluation across all methods & compute average recall @k
    def evaluate(self):
        # Initialize performance metrics
        methods = [
            # {'name': 'HySE Search (Relational)', 'function': self.eval_methods.single_hyse_search, 'query_type': 'task', 'schema_approach': 'relational'},
            # {'name': 'HySE Search (Non-Relational)', 'function': self.eval_methods.single_hyse_search, 'query_type': 'task', 'schema_approach': 'non_relational'},
            # {'name': 'HySE Search (Dual_Avg)', 'function': self.eval_methods.single_hyse_search, 'query_type': 'task', 'schema_approach': 'dual_avg'},
            # {'name': 'HySE Dual Seperate Search', 'function': self.eval_methods.single_hyse_dual_separate_search, 'query_type': 'task'},
            {'name': 'Multi-Component HySE (Relational)', 'function': self.eval_methods.multi_component_hyse_search, 'query_type': 'task', 'schema_approach': 'relational'},
            # {'name': 'Multi-Component HySE (Non-Relational)', 'function': self.eval_methods.multi_component_hyse_search, 'query_type': 'task', 'schema_approach': 'non_relational'},
            # {'name': 'Semantic Task Search', 'function': self.eval_methods.semantic_search, 'query_type': 'task'},
            # {'name': 'Semantic Keyword Search', 'function': self.eval_methods.semantic_search, 'query_type': 'keyword'},
            # {'name': 'Syntactic Keyword Search', 'function': self.eval_methods.syntactic_search, 'query_type': 'keyword'}
        ]
        recalls = {method['name']: [] for method in methods}
        retrieval_times = {method['name']: [] for method in methods}

        # Iterate over each entry
        for idx, ground_truth_table in tqdm(enumerate(self.ground_truths), total=len(self.ground_truths), desc="Evaluating", unit="entry"):
            try:
                task_queries = self.task_queries[idx]
                keywords = self.keywords[idx]

                # Combine queries and methods
                for method in methods:
                    method_name = method['name']
                    search_function = method['function']
                    query_type = method['query_type']
                    queries = task_queries if query_type == 'task' else keywords

                    for query in queries:
                        try:
                            retrieval_time = 0
                            if method_name.startswith('HySE Search') or method_name.startswith('Multi-Component HySE'):
                                # Pass `num_embed` parameter, and also pass the approach
                                schema_approach = method.get('schema_approach', 'relational')
                                results, retrieval_time = search_function(
                                    query=query,
                                    num_embed=self.num_embed,
                                    schema_approach=schema_approach,
                                    return_timing=True
                                )
                            elif method_name == "HySE Dual Seperate Search":
                                results = search_function(query=query)
                            else:        
                                results = search_function(query=query, query_type=query_type)
                            recall = self.compute_recall_at_k(results, ground_truth_table)
                            recalls[method_name].append(recall)
                            retrieval_times[method_name].append(retrieval_time)

                            # Retrieve ground truth header & hypothetical schema for HySE search
                            ground_truth_header = ''
                            hypo_schema = ''

                            if method_name == 'HySE Search':
                                # Retrieve ground truth header from example_rows_md
                                ground_truth_header = get_ground_truth_header(ground_truth_table, self.data_split)
                                # Retrieve hypothetical schema from eval_hyse_schemas
                                hypo_schema = get_hypo_schema(query)

                            self.save_row_result(
                                idx, ground_truth_table, method_name, query_type, query, recall,
                                ground_truth_header, hypo_schema
                            )
                        except Exception as e:
                            logging.exception(f"Error in {method_name} with query '{query}' at index {idx}: {e}")
                            recalls[method_name].append(0)
                            retrieval_times[method_name].append(0)
                            self.save_failed_query(idx, ground_truth_table, method_name, query_type, query, str(e))
            except Exception as e:
                logging.exception(f"Error processing row {idx} (table: {ground_truth_table}): {e}")
                self.save_failed_row(idx, ground_truth_table, str(e))

        # Compute average recall @k for each method
        avg_recalls = {
            method_name: sum(scores) / len(scores) if scores else 0
            for method_name, scores in recalls.items()
        }
        
        # Compute average retrieval time for each method
        avg_retrieval_times = {
            method_name: sum(times) / len(times) if times else 0
            for method_name, times in retrieval_times.items()
        }

        # Report the results
        for method_name, avg_recall in avg_recalls.items():
            avg_time = avg_retrieval_times.get(method_name, 0)
            logging.info(f"Average Recall @{self.k} for {method_name}: {avg_recall}")
            if avg_time > 0:
                logging.info(f"Average Retrieval Time for {method_name} (N={self.num_embed}): {avg_time:.4f} seconds")

        # Return the results
        return avg_recalls, avg_retrieval_times
    
    # Evaluate retrieval recall across different weight combinations for HySE
    def evaluate_with_weights(self, weight_step=0.1):
        # Generate weight combinations: query_weight + hypo_weight = 1.0
        weight_combinations = [
            (round(i * weight_step, 2), round(1.0 - i * weight_step, 2))
            for i in range(int(1 / weight_step) + 1)
        ]

        # Initialize results storage
        recall_results = {f"HySE ({q_w}, {h_w})": [] for q_w, h_w in weight_combinations}

        # Iterate over each query and corresponding ground truth
        for idx, ground_truth_table in tqdm(enumerate(self.ground_truths), total=len(self.ground_truths), desc="Evaluating Weights", unit="entry"):
            try:
                task_queries = self.task_queries[idx]

                for query in task_queries:
                    for query_weight, hypo_weight in weight_combinations:
                        try:
                            # Perform HySE search with specified weights
                            results = self.eval_methods.single_hyse_search(
                                query=query,
                                num_embed=self.num_embed,
                                include_query_embed=True,
                                query_weight=query_weight,
                                hypo_weight=hypo_weight
                            )

                            # Compute recall @k
                            recall = self.compute_recall_at_k(results, ground_truth_table)
                            recall_results[f"HySE ({query_weight}, {hypo_weight})"].append(recall)

                        except Exception as e:
                            logging.exception(f"Error in weight evaluation (query: '{query}', weights: {query_weight}, {hypo_weight}): {e}")
                            recall_results[f"HySE ({query_weight}, {hypo_weight})"].append(0)

            except Exception as e:
                logging.exception(f"Error processing row {idx} (table: {ground_truth_table}): {e}")

        # Compute average recall @k for each weight combination
        avg_recalls = {
            weight_combo: sum(scores) / len(scores) if scores else 0
            for weight_combo, scores in recall_results.items()
        }

        # Report the results
        for weight_combo, avg_recall in avg_recalls.items():
            logging.info(f"Average Recall @{self.k} for {weight_combo}: {avg_recall}")

        return avg_recalls
    
    # Evaluate HITS iterative pipeline: HySE + Metadata Refinement
    def evaluate_multi_stage_retriever(
        self,
        stage1_method="HySE",  # or "Semantic Task", etc.
        stage2_mode="first",  # "first", "concat", "random", "union", or "best_k"
        num_top_tables=50,
        enable_diagnostics=False
    ):
        """
        1. Stage 1: Use 'stage1_method' to retrieve top-N tables for each query
        2. Stage 2: Use 'metadata' filter with one of these modes:
           - "first": use the first metadata query (single query)
           - "concat": concatenate all metadata queries into one (AND logic via LLM)
           - "random": randomly pick one metadata query (single query)
           - "union": run all metadata queries separately, take union (OR logic - less restrictive)
           - "best_k": automatically select the metadata query with best discrimination
        3. Compute recall@k for each query, then average
        """

        # Initialize results storage
        recall_scores = []

        # Diagnostic metrics
        if enable_diagnostics:
            diagnostics = {
                'ground_truth_in_stage1': 0,
                'ground_truth_in_stage2': 0,
                'ground_truth_removed_by_filter': 0,
                'avg_stage1_size': [],
                'avg_stage2_size': [],
                'metadata_filter_empty': 0,
                'total_queries': 0
            }

        # Iterate over each query and corresponding ground truth
        for idx, ground_truth_table in tqdm(
            enumerate(self.ground_truths),
            total=len(self.ground_truths),
            desc="Evaluating Multi-Stage Retriever",
            unit="entry"
        ):
            try:
                task_queries = self.task_queries[idx]
                metadata_sublists = self.get_metadata_sublists_for_index(idx)

                for tq_idx, query in enumerate(task_queries):
                    try:
                        if enable_diagnostics:
                            diagnostics['total_queries'] += 1

                        # Stage 1: Top-N retrieval
                        top_n_tables = self._run_stage1_retrieval(stage1_method, query, num_top_tables)

                        # Track stage 1 diagnostics
                        if enable_diagnostics:
                            diagnostics['avg_stage1_size'].append(len(top_n_tables))
                            if ground_truth_table in top_n_tables:
                                diagnostics['ground_truth_in_stage1'] += 1

                        # Stage 2: Metadata filtering
                        final_candidates = top_n_tables
                        if metadata_sublists and tq_idx < len(metadata_sublists):
                            meta_queries_for_this_task = metadata_sublists[tq_idx]

                            if stage2_mode == "first":
                                # Only the first metadata query
                                if meta_queries_for_this_task:
                                    meta_query = meta_queries_for_this_task[0]
                                    # Metadata_search returns a list of matching table names (unordered)
                                    meta_result = self.eval_methods.metadata_search(meta_query, final_candidates)
                                    # Now intersect meta_result with top_n_tables in the original order
                                    meta_set = set(meta_result)
                                    final_candidates = [t for t in final_candidates if t in meta_set]

                            elif stage2_mode == "concat":
                                # Combine all metadata queries for this sublist into one big string
                                if meta_queries_for_this_task:
                                    concatenated = ". ".join(meta_queries_for_this_task)
                                    meta_result = self.eval_methods.metadata_search(concatenated, final_candidates)
                                    meta_set = set(meta_result)
                                    final_candidates = [t for t in final_candidates if t in meta_set]

                            elif stage2_mode == "random":
                                # Randomly pick one metadata query
                                if meta_queries_for_this_task:
                                    meta_query = random.choice(meta_queries_for_this_task)
                                    meta_result = self.eval_methods.metadata_search(meta_query, final_candidates)
                                    meta_set = set(meta_result)
                                    final_candidates = [t for t in final_candidates if t in meta_set]

                            elif stage2_mode == "union":
                                # Take union of all metadata queries (less restrictive - OR logic)
                                if meta_queries_for_this_task:
                                    meta_set = set()
                                    for meta_query in meta_queries_for_this_task:
                                        meta_result = self.eval_methods.metadata_search(meta_query, final_candidates)
                                        meta_set.update(meta_result)
                                    final_candidates = [t for t in final_candidates if t in meta_set]

                            elif stage2_mode == "best_k":
                                # Try each metadata query, pick the one that keeps most tables while filtering some
                                if meta_queries_for_this_task:
                                    best_result = final_candidates
                                    best_size = len(final_candidates)

                                    for meta_query in meta_queries_for_this_task:
                                        meta_result = self.eval_methods.metadata_search(meta_query, final_candidates)
                                        # Prefer queries that filter but not too aggressively
                                        if len(meta_result) > 0 and len(meta_result) < best_size:
                                            best_result = meta_result
                                            best_size = len(meta_result)

                                    if best_size < len(final_candidates):
                                        meta_set = set(best_result)
                                        final_candidates = [t for t in final_candidates if t in meta_set]

                        # Track stage 2 diagnostics
                        if enable_diagnostics:
                            diagnostics['avg_stage2_size'].append(len(final_candidates))
                            if len(final_candidates) == 0:
                                diagnostics['metadata_filter_empty'] += 1
                            if ground_truth_table in final_candidates:
                                diagnostics['ground_truth_in_stage2'] += 1
                            elif ground_truth_table in top_n_tables:
                                # Ground truth was in stage 1 but removed by metadata filter
                                diagnostics['ground_truth_removed_by_filter'] += 1

                        # Final top-k
                        final_top_k = final_candidates[: self.k]

                        # Compute recall
                        recall_value = self.compute_recall_at_k(final_top_k, ground_truth_table)
                        recall_scores.append(recall_value)

                        # Save row-level results
                        self.save_row_result(
                            idx=idx,
                            table_name=ground_truth_table,
                            method_name=f"{stage1_method} + {stage2_mode}",
                            query_type="task",  # or "keyword" if appropriate
                            query=query,
                            recall=recall_value,
                            ground_truth_header="",
                            hypothetical_schema=""
                        )
                    except Exception as e:
                        logging.exception(f"Error in multi-stage retrieval for query '{query}' at index {idx}: {e}")
                        recall_scores.append(0)
                        self.save_failed_query(
                            idx=idx,
                            table_name=ground_truth_table,
                            method_name=f"{stage1_method} + {stage2_mode}",
                            query_type="task",
                            query=query,
                            error_message=str(e)
                        )
            except Exception as e:
                logging.exception(f"Error processing row {idx} (table: {ground_truth_table}): {e}")
                self.save_failed_row(idx, ground_truth_table, str(e))

        # Compute average recall
        if recall_scores:
            avg_recall = sum(recall_scores) / len(recall_scores)
        else:
            avg_recall = 0

        logging.info(f"Average Recall @{self.k} for Multi-Stage: Stage1={stage1_method}, Stage2={stage2_mode} => {avg_recall}")

        # Print diagnostics if enabled
        if enable_diagnostics:
            total = diagnostics['total_queries']
            if total > 0:
                logging.info(f"\n{'='*60}")
                logging.info(f"DIAGNOSTIC REPORT: Multi-Stage Retrieval")
                logging.info(f"{'='*60}")
                logging.info(f"Total Queries: {total}")
                logging.info(f"Stage 1 (Top-{num_top_tables}):")
                logging.info(f"  - Ground truth in Stage 1: {diagnostics['ground_truth_in_stage1']}/{total} ({diagnostics['ground_truth_in_stage1']/total*100:.2f}%)")
                logging.info(f"  - Avg stage 1 set size: {sum(diagnostics['avg_stage1_size'])/len(diagnostics['avg_stage1_size']):.2f}")
                logging.info(f"\nStage 2 (Metadata Filtering - {stage2_mode}):")
                logging.info(f"  - Ground truth in Stage 2: {diagnostics['ground_truth_in_stage2']}/{total} ({diagnostics['ground_truth_in_stage2']/total*100:.2f}%)")
                logging.info(f"  - Ground truth REMOVED by filter: {diagnostics['ground_truth_removed_by_filter']}/{total} ({diagnostics['ground_truth_removed_by_filter']/total*100:.2f}%)")
                logging.info(f"  - Empty results after filtering: {diagnostics['metadata_filter_empty']}/{total} ({diagnostics['metadata_filter_empty']/total*100:.2f}%)")
                logging.info(f"  - Avg stage 2 set size: {sum(diagnostics['avg_stage2_size'])/len(diagnostics['avg_stage2_size']):.2f}")
                logging.info(f"\nFinal Recall@{self.k}: {avg_recall:.4f}")
                logging.info(f"{'='*60}\n")

            return avg_recall, diagnostics

        return avg_recall

    def evaluate_metadata_refinement(self, mode="first", search_space=None):
        """
        Evaluate the effectiveness of metadata filtering modes in narrowing down the table set

        Args:
            mode: "first", "concat", "random", "union", or "best_k"
            search_space: Optional list of table names to restrict search to (default: None = all tables)

        Returns:
            dict with statistics about set sizes and ground truth coverage
        """
        result_sizes = []
        ground_truth_in_results = 0
        empty_results = 0
        total_sublists = 0

        logging.info(f"Evaluating metadata refinement with mode={mode}")
        logging.info(f"Total ground truths to process: {len(self.ground_truths)}")

        for idx, ground_truth_table in tqdm(
            enumerate(self.ground_truths),
            total=len(self.ground_truths),
            desc=f"Evaluating Metadata ({mode})",
            unit="entry"
        ):
            try:
                meta_sublists = self.metadata_queries[idx]

                if not meta_sublists:
                    continue

                for sublist in meta_sublists:
                    if not sublist:
                        continue

                    total_sublists += 1
                    meta_result = []

                    # Baseline: First metadata query ('tags' field)
                    if mode == "first":
                        meta_query = sublist[0]
                        meta_result = self.eval_methods.metadata_search(
                            metadata_query=meta_query,
                            search_space=search_space
                        )

                    # Concatenate all metadata queries
                    elif mode == "concat":
                        concatenated = ". ".join(sublist)
                        meta_result = self.eval_methods.metadata_search(
                            metadata_query=concatenated,
                            search_space=search_space
                        )

                    # Randomly pick one metadata query
                    elif mode == "random":
                        meta_query = random.choice(sublist)
                        meta_result = self.eval_methods.metadata_search(
                            metadata_query=meta_query,
                            search_space=search_space
                        )

                    # Take union of all metadata queries (OR logic)
                    elif mode == "union":
                        meta_set = set()
                        for meta_query in sublist:
                            result = self.eval_methods.metadata_search(
                                metadata_query=meta_query,
                                search_space=search_space
                            )
                            meta_set.update(result)
                        meta_result = list(meta_set)

                    # Select metadata query with best discrimination
                    elif mode == "best_k":
                        best_result = []
                        best_size = float('inf')

                        for meta_query in sublist:
                            result = self.eval_methods.metadata_search(
                                metadata_query=meta_query,
                                search_space=search_space
                            )
                            # Prefer queries that return results but not too many
                            if 0 < len(result) < best_size:
                                best_result = result
                                best_size = len(result)

                        meta_result = best_result if best_result else []

                    # Track statistics
                    result_sizes.append(len(meta_result))

                    if len(meta_result) == 0:
                        empty_results += 1

                    if ground_truth_table in meta_result:
                        ground_truth_in_results += 1

            except Exception as e:
                logging.exception(f"Error in evaluate_metadata_refinement at index {idx} (table: {self.ground_truths[idx]}): {e}")

        # Compute statistics
        avg_size = sum(result_sizes) / len(result_sizes) if result_sizes else 0
        ground_truth_coverage = ground_truth_in_results / total_sublists if total_sublists > 0 else 0
        empty_rate = empty_results / total_sublists if total_sublists > 0 else 0

        logging.info(f"\n{'='*60}")
        logging.info(f"METADATA REFINEMENT REPORT: mode={mode}")
        logging.info(f"{'='*60}")
        logging.info(f"Total metadata sublists: {total_sublists}")
        logging.info(f"Avg result set size: {avg_size:.2f}")
        logging.info(f"Ground truth coverage: {ground_truth_in_results}/{total_sublists} ({ground_truth_coverage*100:.2f}%)")
        logging.info(f"Empty results: {empty_results}/{total_sublists} ({empty_rate*100:.2f}%)")
        logging.info(f"{'='*60}\n")

        return {
            "mode": mode,
            "avg_size": avg_size,
            "ground_truth_coverage": ground_truth_coverage,
            "ground_truth_count": ground_truth_in_results,
            "total_sublists": total_sublists,
            "empty_results": empty_results,
            "empty_rate": empty_rate,
            "result_sizes": result_sizes
        }

    # Run the stage 1 method (HySE/Semantic) & returns top-N tables
    def _run_stage1_retrieval(self, stage1_method, query, num_top_tables):
        if stage1_method.lower().startswith("hyse"):
            # e.g. 'HySE Search'
            results = self.eval_methods.multi_component_hyse_search(query=query, num_embed=self.num_embed, top_k=num_top_tables, schema_approach="relational")
        elif stage1_method.lower().startswith("semantic"):
            # e.g. 'Semantic Task Search'
            results = self.eval_methods.semantic_search(query=query, query_type="task", top_k=num_top_tables)
        else:
            raise ValueError(f"Unknown stage1_method: {stage1_method}")
        return results

    # Retrieve the metadata queries for the given row index
    def get_metadata_sublists_for_index(self, idx):
        try:
            if hasattr(self, 'metadata_queries'):
                return self.metadata_queries[idx]
        except Exception as e:
            logging.exception(f"Error getting metadata query sublists for index {idx}: {e}")
    
    def save_row_result(self, idx, table_name, method_name, query_type, query, recall, ground_truth_header='', hypothetical_schema=''):
        try:
            with open(self.results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx, table_name, method_name, query_type, query, recall,
                    ground_truth_header, hypothetical_schema
                ])
        except Exception as e:
            logging.exception(f"Error saving result for index {idx}: {e}")

    def save_failed_row(self, idx, table_name, error_message):
        try:
            with open(self.failed_rows_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([idx, table_name, error_message])
        except Exception as e:
            logging.exception(f"Error saving failed row for index {idx}: {e}")

    def save_failed_query(self, idx, table_name, method_name, query_type, query, error_message):
        try:
            with open(self.failed_queries_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([idx, table_name, method_name, query_type, query, error_message])
        except Exception as e:
            logging.exception(f"Error saving failed query for index {idx}: {e}")


if __name__ == "__main__":
    # Force logging config
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    evaluator = Evaluator(
        data_split="eval_data_test",
        embed_col="example_2rows_table_name_embed",
        k=10,
        # limit=10,
        num_embed=2,
        filter_should_include=None  # set to 'Y' to filter semantically meaningful tables, 'N' for non-meaningful, or None for all
    )

    # avg_recalls, avg_retrieval_times = evaluator.evaluate()
    # print("Evaluation Results:")
    # for method in avg_recalls:
    #     recall = avg_recalls[method]
    #     avg_time = avg_retrieval_times.get(method, 0)
    #     print(f"{method}: Recall@{evaluator.k} = {recall:.4f}, Avg Time = {avg_time:.4f}s")
    
    # weight_evaluation_results = evaluator.evaluate_with_weights(weight_step=0.1)
    # for weight_combo, avg_recall in weight_evaluation_results.items():
    #     print(f"{weight_combo}: Average Recall = {avg_recall}")

    # Evaluate using Multi-Stage Retrieval
    # Try different stage2_modes to compare performance
    # for mode in ["first", "concat", "random", "union", "best_k"]:
    for mode in ["concat"]:
        logging.info(f"\n{'='*60}")
        logging.info(f"Testing stage2_mode={mode}")
        logging.info(f"{'='*60}")
        result = evaluator.evaluate_multi_stage_retriever(
                                    stage1_method = "HySE",  # or "Semantic Task"
                                    stage2_mode = mode,
                                    num_top_tables=50,
                                    enable_diagnostics=True
                                )
        if isinstance(result, tuple):
            multi_stage_recall, diagnostics = result
            print(f"\n[{mode.upper()}] Final Recall@10 = {multi_stage_recall:.4f}\n")

    # Evaluate metadata refinement with different modes
    # logging.info("\n" + "="*80)
    # logging.info("COMPARING METADATA REFINEMENT MODES")
    # logging.info("="*80)
    
    # results_summary = []
    # # for mode in ["first", "concat", "random", "union", "best_k"]:
    # for mode in ["concat"]:
    #     result = evaluator.evaluate_metadata_refinement(mode=mode)
    #     results_summary.append(result)
    
    # # Print comparison table
    # logging.info("\n" + "="*80)
    # logging.info("METADATA REFINEMENT COMPARISON")
    # logging.info("="*80)
    # logging.info(f"{'Mode':<10} {'Avg Size':<12} {'GT Coverage':<15} {'Empty Rate':<12}")
    # logging.info("-"*80)
    # for r in results_summary:
    #     logging.info(f"{r['mode']:<10} {r['avg_size']:<12.2f} {r['ground_truth_coverage']*100:<14.2f}% {r['empty_rate']*100:<11.2f}%")
    # logging.info("="*80)
