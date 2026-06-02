from backend.app.table_representation.openai_client import OpenAIClient
from backend.app.hyse.hypo_schema_search import cos_sim_search
import time
import logging
from copy import deepcopy
from dotenv import load_dotenv
from backend.app.evals.elastic_search.es_client import es_client
from backend.app.hyse.hypo_schema_search import hyse_search
from backend.app.actions.infer_action import infer_mentioned_metadata_fields, text_to_sql, execute_sql, TextToSQL, SQLClause
from backend.app.evals.eval_utils import generate_embeddings, average_embeddings, average_embeddings_with_weights, get_query_embedding_from_db, get_keyword_embedding_from_db, save_query_embedding_to_db, save_keyword_embedding_to_db, retrieve_or_generate_schemas, get_cached_metadata_sqlclauses, save_cached_metadata_sqlclauses, llm_rerank_tables, retrieve_or_generate_hyse_components

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client
openai_client = OpenAIClient()

class EvalMethods:
    def __init__(self, data_split, embed_col, k):
        self.openai_client = openai_client
        self.es_client = es_client.client
        self.data_split = data_split
        self.embed_col = embed_col
        self.k = k

    # Evaluate semantic keywords search & semantic task search against HySE  
    def semantic_search(self, query, query_type="task", top_k=None):
        try:
            if top_k is None:
                top_k = self.k      
            query_embedding = None

            # Step 1: Get or generate embedding for the query (keyword / task)
            if query_type == "task":
                # Check if the query embedding is cached
                query_embedding = get_query_embedding_from_db(query)
                if query_embedding is None:
                    # Generate & cache the query embedding
                    query_embedding = generate_embeddings([query])[0]
                    save_query_embedding_to_db(query, query_embedding)
            elif query_type == "keyword":
                # Check if the keyword embedding is cached
                query_embedding = get_keyword_embedding_from_db(query)
                if query_embedding is None:
                    # Generate & cache the keyword embedding
                    query_embedding = generate_embeddings([query])[0]
                    save_keyword_embedding_to_db(query, query_embedding)
            else:
                raise ValueError(f"Invalid query_type: {query_type}. Must be 'task' or 'keyword'.")

            # Step 2: Cosine similarity search between e(query_embed) and e(existing_scheme_embed)
            semantic_results = cos_sim_search(query_embedding, search_space=None, table_name=self.data_split, column_name=self.embed_col)

            # Extract and return only the table names of the top k results
            top_k_results = [result['table_name'] for result in semantic_results[:top_k]]
            return top_k_results
        except Exception as e:
            logging.error(f"Error during semantic search: {e}")
            return []
    
    # Evaluate syntactic keywords search against HySE
    # Perform syntactic keyword search against table_name & example_rows_md fields
    def syntactic_search(self, query, query_type=None):
        try:
            # Validate data_split
            valid_splits = ['eval_data_all', 'eval_data_test', 'eval_data_train', 'eval_data_validation']
            if self.data_split not in valid_splits:
                logging.error(f"Invalid data_split: {self.data_split}. Must be one of {valid_splits}")
                return []
            
            # Define the Elasticsearch index based on data_split
            index_name = self.data_split

            # Check if the index exists
            if not self.es_client.indices.exists(index=index_name):
                logging.error(f"Elasticsearch index '{index_name}' does not exist")
                return []

            # Define the search query
            es_query = {
                "size": self.k,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "table_name": {
                                        "query": query,
                                        "fuzziness": "AUTO"
                                    }
                                }
                            },
                            {
                                "match": {
                                    "example_2rows_md": {
                                    # "table_header": {
                                        "query": query,
                                        "fuzziness": "AUTO"
                                    }
                                }
                            }
                        ]
                    }
                }
            }

            # Execute the search query
            response = self.es_client.search(index=index_name, body=es_query)

            # Extract table names from the search results
            top_k_results = [hit['_source']['table_name'] for hit in response['hits']['hits']]
            return top_k_results
        except Exception as e:
            logging.error(f"Error during syntactic search: {e}")
            return []
      
    def single_hyse_search(
        self,
        query,
        num_embed=1,
        include_query_embed=True,
        query_weight=0.5,
        hypo_weight=0.5,
        return_embedding=False,
        search_space=None,
        top_k=None,
        schema_approach="relational",
        return_timing=False,
    ):
        start_time = time.time()
        
        try:
            if top_k is None:
                top_k = self.k

            # Step 1: Retrieve cached query embedding if needed
            query_embedding = None
            if include_query_embed:
                query_embedding = get_query_embedding_from_db(query)
                if query_embedding is None:
                    # Generate & cache query embedding if not already cached
                    query_embedding = generate_embeddings([query])[0]
                    save_query_embedding_to_db(query, query_embedding)

            # Step 2: Retrieve or generate schema embeddings
            all_schema_embeddings = []

            if schema_approach == "relational":
                rel_embeds = retrieve_or_generate_schemas(
                    query=query,
                    num_embed=num_embed,
                    table_name="eval_hyse_schemas",
                    schema_approach="relational"
                )
                all_schema_embeddings.extend(rel_embeds)

            elif schema_approach == "non_relational":
                non_rel_embeds = retrieve_or_generate_schemas(
                    query=query,
                    num_embed=num_embed,
                    table_name="eval_hyse_schemas_non_relational",
                    schema_approach="non_relational"
                )
                all_schema_embeddings.extend(non_rel_embeds)

            elif schema_approach == "dual_avg":
                rel_embeds = retrieve_or_generate_schemas(
                    query=query,
                    num_embed=num_embed,
                    table_name="eval_hyse_schemas",
                    schema_approach="relational"
                )
                non_rel_embeds = retrieve_or_generate_schemas(
                    query=query,
                    num_embed=num_embed,
                    table_name="eval_hyse_schemas_non_relational",
                    schema_approach="non_relational"
                )
                # Combine relational & non-relational embeddings
                all_schema_embeddings.extend(rel_embeds)
                all_schema_embeddings.extend(non_rel_embeds)

            else:
                raise ValueError(f"Unknown schema_approach: {schema_approach}")

            # Step 3: Combine & average embeddings based on the include_query_embedding flag
            combined_list = []
            if query_embedding is not None and include_query_embed:
                combined_list = [query_embedding] + all_schema_embeddings
                final_embedding = average_embeddings_with_weights(
                    combined_list,
                    query_weight=query_weight,
                    hypo_weight=hypo_weight
                )
            else:
                final_embedding = average_embeddings(all_schema_embeddings)

            if return_embedding:
                # Just return the HySE embedding, do not perform retrieval
                end_time = time.time()
                retrieval_time = end_time - start_time
                if return_timing:
                    return final_embedding, retrieval_time
                return final_embedding

            # Step 4: Perform similarity search between the final averaged embedding and e(existing_scheme_embed)
            results = cos_sim_search(final_embedding, search_space=search_space, table_name=self.data_split, column_name=self.embed_col)
            
            # Step 5: Extract and return only the table names of the top-k results
            top_k_results = [result['table_name'] for result in results[:top_k]]
            
            end_time = time.time()
            retrieval_time = end_time - start_time
            
            if return_timing:
                return top_k_results, retrieval_time
            return top_k_results

        except Exception as e:
            logging.exception(f"Error during single_hyse_search: {e}")
            if return_timing:
                return [], 0
            return []

    def single_hyse_dual_separate_search(
        self,
        query,
        num_embed=1,
        include_query_embed=True,
        query_weight=0.5,
        hypo_weight=0.5,
        search_space=None,
        top_k=None,
        use_rerank=True
    ):
        if top_k is None:
            top_k = self.k

        # Step 1: Perform single HySE search w/ RELATIONAL template
        rel_results = self.single_hyse_search(
            query=query,
            num_embed=num_embed,
            include_query_embed=include_query_embed,
            query_weight=query_weight,
            hypo_weight=hypo_weight,
            return_embedding=False,
            search_space=search_space,
            top_k=top_k,
            schema_approach="non_relational"
        )

        # Step 2: Perform single HySE search w/ NON-RELATIONAL template
        non_rel_results = self.single_hyse_search(
            query=query,
            num_embed=num_embed,
            include_query_embed=include_query_embed,
            query_weight=query_weight,
            hypo_weight=hypo_weight,
            return_embedding=False,
            search_space=search_space,
            top_k=top_k,
            schema_approach="non_relational"
        )

        # Step 3: Union both results & remove duplicates
        combined_list = rel_results + non_rel_results
        final_results = []
        seen = set()
        for tbl in combined_list:
            if tbl not in seen:
                final_results.append(tbl)
                seen.add(tbl)

        # Step 4: Optional LLM rerank to the final top‑k
        if use_rerank:
            return llm_rerank_tables(query, final_results, top_k)
        else:
            return final_results[:top_k]

    def multi_component_hyse_search(
        self,
        query,
        num_embed=1,
        include_query_embed=True,
        query_weight=0.5,
        hypo_weight=0.5,
        return_embedding=False,
        search_space=None,
        top_k=None,
        schema_approach="relational",
        return_timing=False,
    ):
        """
        Multi-component HySE search that matches HySE embedding type with corpus embedding type
        based on embed_col configuration
        """
        start_time = time.time()
        
        try:
            if top_k is None:
                top_k = self.k

            # Step 1: Map corpus embed_col to corresponding HySE component embedding type
            # The mapping handles cases where corpus has different naming conventions
            embed_col_mapping = {
                'table_header_embed': 'table_header_embed',
                'table_header_name_embed': 'table_header_name_embed',
                'example_rows_embed': 'example_2rows_embed',  # Map general 'example_rows' to 2-row version
                'example_2rows_embed': 'example_2rows_embed',
                'example_3rows_embed': 'example_3rows_embed',
                'example_2rows_table_name_embed': 'example_2rows_table_name_embed',
                'example_3rows_table_name_embed': 'example_3rows_table_name_embed'
            }
            
            hyse_embed_type = embed_col_mapping.get(self.embed_col)
            if not hyse_embed_type:
                raise ValueError(f"Unsupported embed_col: {self.embed_col}")
            
            logging.info(f"[MultiHySE] Query: '{query}' | Schema: {schema_approach} | Corpus embed_col: {self.embed_col} | HySE embed_type: {hyse_embed_type}")

            # Step 2: Retrieve cached query embedding if needed
            query_embedding = None
            if include_query_embed:
                query_embedding = get_query_embedding_from_db(query)
                if query_embedding is None:
                    # Generate & cache query embedding if not already cached
                    logging.info(f"[MultiHySE] Generating new query embedding for: '{query}'")
                    query_embedding = generate_embeddings([query])[0]
                    save_query_embedding_to_db(query, query_embedding)
                else:
                    logging.info(f"[MultiHySE] Using cached query embedding for: '{query}'")

            # Step 3: Retrieve or generate HySE components
            logging.info(f"[MultiHySE] Retrieving/generating {num_embed} HySE components for schema_approach: {schema_approach}")
            hyse_components = retrieve_or_generate_hyse_components(
                query=query,
                num_embed=num_embed,
                schema_approach=schema_approach
            )
            logging.info(f"[MultiHySE] Retrieved {len(hyse_components)} HySE components")

            # Step 4: Extract embeddings of the specified type
            hyse_embeddings = []
            for i, component in enumerate(hyse_components):
                if hyse_embed_type in component['embeddings']:
                    hyse_embeddings.append(component['embeddings'][hyse_embed_type])
                    logging.debug(f"[MultiHySE] Component {i}: Found {hyse_embed_type} embedding")
                else:
                    logging.warning(f"[MultiHySE] Component {i}: Missing {hyse_embed_type} embedding")

            if not hyse_embeddings:
                logging.error(f"[MultiHySE] No HySE embeddings found for type: {hyse_embed_type}")
                if return_timing:
                    return [], 0
                return []

            logging.info(f"[MultiHySE] Collected {len(hyse_embeddings)} embeddings of type: {hyse_embed_type}")

            # Step 5: Combine & average embeddings based on the include_query_embedding flag
            combined_list = []
            if query_embedding is not None and include_query_embed:
                combined_list = [query_embedding] + hyse_embeddings
                logging.info(f"[MultiHySE] Combining query + HySE embeddings with weights (query: {query_weight}, hypo: {hypo_weight})")
                final_embedding = average_embeddings_with_weights(
                    combined_list,
                    query_weight=query_weight,
                    hypo_weight=hypo_weight
                )
            else:
                logging.info(f"[MultiHySE] Averaging {len(hyse_embeddings)} HySE embeddings only")
                final_embedding = average_embeddings(hyse_embeddings)

            if return_embedding:
                # Just return the HySE embedding, do not perform retrieval
                end_time = time.time()
                retrieval_time = end_time - start_time
                if return_timing:
                    return final_embedding, retrieval_time
                return final_embedding

            # Step 6: Perform similarity search using the corpus embed_col
            logging.info(f"[MultiHySE] Performing similarity search against corpus column: {self.embed_col}")
            results = cos_sim_search(final_embedding, search_space=search_space, table_name=self.data_split, column_name=self.embed_col)
            
            # Step 7: Extract and return only the table names of the top-k results
            top_k_results = [result['table_name'] for result in results[:top_k]]
            
            end_time = time.time()
            retrieval_time = end_time - start_time
            
            logging.info(f"[MultiHySE] Retrieved {len(top_k_results)} results in {retrieval_time:.4f}s | Top 3: {top_k_results[:3]}")
            
            if return_timing:
                return top_k_results, retrieval_time
            return top_k_results

        except Exception as e:
            logging.exception(f"Error during multi_component_hyse_search: {e}")
            if return_timing:
                return [], 0
            return []

    # TODO: Multi-hyse implementation
    def multi_hyse_search(self, query):
        pass

    def metadata_search(self, metadata_query, search_space=None, relax=True):
        try:
            # Step 1: Check if we have a cached SQL translation for the metadata query
            cached = get_cached_metadata_sqlclauses(metadata_query)
            if cached:
                # Convert the cached dict back to a TextToSQL model
                text_to_sql_instance = TextToSQL(**cached)
                logging.info(f"[MetaSearch] Cache hit for: '{metadata_query}'")
            else:
                # Step 2: If No cache entry, call the LLM pipeline
                # Step 2.1: Infer which fields the query is referencing (tags, col_num, row_num, time_granu, geo_granu)
                inferred_fields = infer_mentioned_metadata_fields(
                    cur_query=metadata_query,
                    semantic_metadata=False
                ).get_true_fields()

                # Step 2.2: Generate structured SQL clauses
                text_to_sql_instance = text_to_sql(
                    cur_query=metadata_query,
                    identified_fields=inferred_fields
                )

                # Step 2.3: Save the final clauses to the cache DB
                save_cached_metadata_sqlclauses(
                    metadata_query,
                    text_to_sql_instance.model_dump()
                )
                logging.info(f"[MetaSearch] New translation cached for: '{metadata_query}'")

            # Step 3: Execute strict query without relaxations
            if not relax:
                logging.info(f"[MetaSearch-Strict] {metadata_query} → {text_to_sql_instance.sql_clauses}")
                results = execute_sql(
                    text_to_sql_instance=text_to_sql_instance,
                    search_space=search_space,
                    table_name=self.data_split
                )
                return [row["table_name"] for row in results]

            # Step 4: Relaxation #1 - fuzz numeric equality & split tag conjunctions
            relaxed = deepcopy(text_to_sql_instance)
            new_clauses = []

            for clause in relaxed.sql_clauses:
                if clause.field == "column_numbers" and clause.clause.startswith("= "):
                    try:
                        n = int(clause.clause.split()[1])
                        band = max(1, round(n * 0.15))  # ±15%
                        new_clauses.append(SQLClause(field="column_numbers", clause=f">= {n - band}"))
                        new_clauses.append(SQLClause(field="column_numbers", clause=f"<= {n + band}"))
                    except Exception:
                        logging.warning(f"Could not parse numeric band from clause: {clause.clause}")
                        new_clauses.append(clause)

                elif clause.field == "row_numbers" and clause.clause.startswith("= "):
                    try:
                        n = int(clause.clause.split()[1])
                        band = max(1, round(n * 0.15))
                        new_clauses.append(SQLClause(field="row_numbers", clause=f">= {n - band}"))
                        new_clauses.append(SQLClause(field="row_numbers", clause=f"<= {n + band}"))
                    except Exception:
                        logging.warning(f"Could not parse numeric band from clause: {clause.clause}")
                        new_clauses.append(clause)

                elif clause.field == "table_tags":
                    # Split `"= 'tag1 and tag2'"` into two separate clauses
                    tokens = [t.strip() for t in clause.clause[3:].strip("'").split(" and ")]
                    for tok in tokens:
                        new_clauses.append(SQLClause(field="table_tags", clause=f"= '{tok}'"))

                else:
                    new_clauses.append(clause)

            relaxed.sql_clauses = new_clauses
            logging.info(f"[MetaSearch-Loosen] Relaxed clauses: {relaxed.sql_clauses}")
            
            # Run the relaxed query
            results = execute_sql(
                text_to_sql_instance=relaxed,
                search_space=search_space,
                table_name=self.data_split
            )
            if results:
                return [row["table_name"] for row in results]

            # Step 5: Relaxation #2 — drop geo/temporal constraints
            drop = deepcopy(relaxed)
            drop.sql_clauses = [
                c for c in drop.sql_clauses
                if c.field not in {"temporal_granularity", "geographic_granularity"}
            ]
            logging.info(f"[MetaSearch-DropGran] Dropped granularity filters → {drop.sql_clauses}")
            
            # Run final relaxed query
            results = execute_sql(
                text_to_sql_instance=drop,
                search_space=search_space,
                table_name=self.data_split
            )
            return [row["table_name"] for row in results]

        except Exception as e:
            logging.exception(f"[MetaSearch-Error] {e}")
            return []


if __name__ == "__main__":
    # Create an instance of HyseEval
    hyse_eval = EvalMethods(
        data_split="eval_data_validation",
        embed_col="example_rows_embed",
        k=10
    )

    # Test for semantic search
    query = "medicine demand forecast"
    top_k_semantic = hyse_eval.semantic_search(query)
    print("Semantic Search Results:", top_k_semantic)

    # Test for syntactic search
    syntactic_query = "demand forecast"
    top_k_syntactic = hyse_eval.syntactic_search(syntactic_query)
    print("Syntactic Search Results:", top_k_syntactic)

    # Test for hyse search
    semantic_query = "What data is needed to train a machine learning model to forecast demand for medicines?"
    top_k_hyse = hyse_eval.hyse_search(semantic_query)
    print("HySE Search Results:", top_k_hyse)