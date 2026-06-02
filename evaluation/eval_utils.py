import logging
from dataforager.db.connect_db import DatabaseConnection
from dataforager.hyse.hypo_schema_search import infer_single_hypothetical_schema, infer_single_hypothetical_schema_with_examples, PROMPT_SINGLE_SCHEMA_RELATIONAL, PROMPT_SCHEMA_WITH_EXAMPLES_RELATIONAL, PROMPT_SINGLE_SCHEMA_NON_RELATIONAL, PROMPT_SCHEMA_WITH_EXAMPLES_NON_RELATIONAL
from dataforager.table_representation.openai_client import OpenAIClient
from pydantic import BaseModel
from typing import List
import numpy as np
import json
from dotenv import load_dotenv

load_dotenv()

# OpenAI client instantiation
openai_client = OpenAIClient()

class RerankResult(BaseModel):
    ranked_tables: List[str]


def get_hypo_schema_from_db(query):
    try:
        with DatabaseConnection() as db:
            select_query = """
            SELECT hypo_schema, hypo_schema_embed FROM eval_hyse_schemas_non_relational
            WHERE query = %s;
            """
            db.cursor.execute(select_query, (query,))
            result = db.cursor.fetchone()
            if result:
                hypo_schema = result['hypo_schema']
                hypo_schema_embed = result['hypo_schema_embed']
                return hypo_schema, hypo_schema_embed
            else:
                return None, None
    except Exception as e:
        logging.exception(f"Error retrieving hypothetical schema from DB: {e}")
        return None, None
    
def get_hypo_schemas_from_db(query, num_embed, table_name="eval_hyse_schemas"):
    try:
        with DatabaseConnection() as db:
            select_query = f"""
            SELECT hypo_schema, hypo_schema_embed
            FROM {table_name}
            WHERE query = %s
            ORDER BY hypo_schema_id ASC
            LIMIT %s;
            """
            db.cursor.execute(select_query, (query, num_embed))
            results = db.cursor.fetchall()

            schemas = []
            embeddings = []
            for result in results:
                schemas.append(result['hypo_schema'])

                if isinstance(result['hypo_schema_embed'], str):
                    embedding_list = json.loads(result['hypo_schema_embed'])
                elif isinstance(result['hypo_schema_embed'], list):
                    embedding_list = result['hypo_schema_embed']
                else:
                    embedding_list = result['hypo_schema_embed']

                # Convert to NumPy array
                embedding = np.array(embedding_list, dtype=float)
                embeddings.append(embedding)
            return schemas, embeddings
    except Exception as e:
        logging.exception(f"Error retrieving hypothetical schemas from DB: {e}")
        return [], []

def get_query_embedding_from_db(query):
    try:
        with DatabaseConnection() as db:
            select_query = """
            SELECT query_embed FROM eval_query_embeds
            WHERE query = %s;
            """
            db.cursor.execute(select_query, (query,))
            result = db.cursor.fetchone()
            if result:
                if isinstance(result['query_embed'], str):
                    query_embedding = json.loads(result['query_embed'])
                elif isinstance(result['query_embed'], list):
                    query_embedding = result['query_embed']
                else:
                    query_embedding = result['query_embed']
                
                # Convert to NumPy array
                return np.array(query_embedding, dtype=float)
            return None
    except Exception as e:
        logging.exception(f"Error retrieving query embedding from DB: {e}")
        return None
    
def get_keyword_embedding_from_db(keyword):
    try:
        with DatabaseConnection() as db:
            select_query = """
            SELECT keyword_embed FROM eval_keyword_embeds
            WHERE keyword = %s;
            """
            db.cursor.execute(select_query, (keyword,))
            result = db.cursor.fetchone()
            if result:
                if isinstance(result['keyword_embed'], str):
                    keyword_embedding = json.loads(result['keyword_embed'])
                elif isinstance(result['keyword_embed'], list):
                    keyword_embedding = result['keyword_embed']
                else:
                    keyword_embedding = result['keyword_embed']
                
                # Convert to NumPy array
                return np.array(keyword_embedding, dtype=float)
            return None
    except Exception as e:
        logging.exception(f"Error retrieving keyword embedding from DB: {e}")
        return None

def save_hypo_schema_to_db(query, hypo_schema, hypo_schema_embed):
    try:
        with DatabaseConnection() as db:
            insert_query = """
            INSERT INTO eval_hyse_schemas_non_relational (query, hypo_schema, hypo_schema_embed)
            VALUES (%s, %s, %s)
            ON CONFLICT (query) DO UPDATE SET
            hypo_schema = EXCLUDED.hypo_schema,
            hypo_schema_embed = EXCLUDED.hypo_schema_embed;
            """
            db.cursor.execute(insert_query, (query, hypo_schema, hypo_schema_embed))
            db.conn.commit()
    except Exception as e:
        logging.exception(f"Error saving hypothetical schema to DB: {e}")

def save_hypo_schemas_to_db(query, schemas, embeddings, table_name="eval_hyse_schemas"):
    try:
        with DatabaseConnection() as db:
            insert_query = f"""
            INSERT INTO {table_name} (query, hypo_schema, hypo_schema_embed)
            VALUES (%s, %s, %s);
            """
            data = []
            for schema, embedding in zip(schemas, embeddings):
                # Convert embedding to list
                embedding_list = embedding.tolist()
                data.append((query, schema, embedding_list))
            db.cursor.executemany(insert_query, data)
            db.conn.commit()
    except Exception as e:
        logging.exception(f"Error saving hypothetical schemas to DB: {e}")
    
def save_query_embedding_to_db(query, query_embedding):
    try:
        with DatabaseConnection() as db:
            insert_query = """
            INSERT INTO eval_query_embeds (query, query_embed)
            VALUES (%s, %s)
            ON CONFLICT (query) DO NOTHING;
            """
            # Ensure query_embedding is a list
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            db.cursor.execute(insert_query, (query, query_embedding))
            db.conn.commit()
    except Exception as e:
        logging.exception(f"Error saving query embedding to DB: {e}")

def save_keyword_embedding_to_db(keyword, keyword_embedding):
    try:
        with DatabaseConnection() as db:
            insert_query = """
            INSERT INTO eval_keyword_embeds (keyword, keyword_embed)
            VALUES (%s, %s)
            ON CONFLICT (keyword) DO NOTHING;
            """
            # Ensure keyword_embedding is a list
            if isinstance(keyword_embedding, np.ndarray):
                keyword_embedding = keyword_embedding.tolist()
            db.cursor.execute(insert_query, (keyword, keyword_embedding))
            db.conn.commit()
    except Exception as e:
        logging.exception(f"Error saving keyword embeddings to DB: {e}")

# def get_ground_truth_header(table_name, data_split):
#     try:
#         with DatabaseConnection() as db:
#             query = f"SELECT example_rows_md FROM {data_split} WHERE table_name = %s;"
#             db.cursor.execute(query, (table_name,))
#             result = db.cursor.fetchone()
#             if result and result['example_rows_md']:
#                 return result['example_rows_md']
#             else:
#                 return ''
#     except Exception as e:
#         logging.exception(f"Error retrieving ground truth header for table '{table_name}': {e}")
#         return ''

def get_ground_truth_header(table_name, data_split):
    try:
        with DatabaseConnection() as db:
            query = f"SELECT table_header FROM {data_split} WHERE table_name = %s;"
            db.cursor.execute(query, (table_name,))
            result = db.cursor.fetchone()
            if result and result['table_header']:
                return result['table_header']
            else:
                return ''
    except Exception as e:
        logging.exception(f"Error retrieving ground truth header for table '{table_name}': {e}")
        return ''

def get_hypo_schema(query):
    try:
        with DatabaseConnection() as db:
            select_query = "SELECT hypo_schema FROM eval_hyse_schemas_non_relational WHERE query = %s;"
            db.cursor.execute(select_query, (query,))
            result = db.cursor.fetchone()
            if result and result['hypo_schema']:
                return result['hypo_schema']
            else:
                return ''
    except Exception as e:
        logging.exception(f"Error retrieving hypothetical schema for query '{query}': {e}")
        return ''

def generate_hypothetical_schemas(query, num_to_generate, schema_approach="relational"):
    schemas = []
    try:
        # Pick the template based on schema_approach
        if schema_approach == "relational":
            prompt_template = PROMPT_SINGLE_SCHEMA_RELATIONAL
        else:
            prompt_template = PROMPT_SINGLE_SCHEMA_NON_RELATIONAL

        # Generate the requested number of schemas
        for _ in range(num_to_generate):
            schema = infer_single_hypothetical_schema(initial_query=query, prompt_template=prompt_template).json()
            schemas.append(schema)
        return schemas
    except Exception as e:
        logging.exception(f"Error generating hypothetical schemas for query '{query}': {e}")
        return []

def generate_embeddings(schemas):
    embeddings = []
    try:
        for schema in schemas:
            embedding = openai_client.generate_embeddings(text=schema)

            # Convert embedding to NumPy array if not already
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=float)

            embeddings.append(embedding)
        return embeddings
    except Exception as e:
        logging.exception(f"Error generating embeddings for schemas '{schemas}': {e}")
        return []

def retrieve_or_generate_schemas(query, num_embed, table_name, schema_approach): 
    # Fetch up to `num_embed` schemas + embeddings from `table_name`
    cached_schemas, cached_embeddings = get_hypo_schemas_from_db(query, num_embed, table_name=table_name)
    num_cached = len(cached_schemas)

    # If not enough cached, generate more using `generate_prompt_func`
    if num_cached < num_embed:
        num_to_generate = num_embed - num_cached
        new_schemas = generate_hypothetical_schemas(
            query,
            num_to_generate,
            schema_approach=schema_approach
        )
        new_embeddings = generate_embeddings(new_schemas)

        # Save new schemas & embeddings to DB
        save_hypo_schemas_to_db(query, new_schemas, new_embeddings, table_name=table_name)

        # Combine cached & new embeddings
        cached_embeddings.extend(new_embeddings)

    return cached_embeddings

def get_cached_metadata_sqlclauses(meta_query):
    try:
        with DatabaseConnection() as db:
            select_query = """
            SELECT clauses_json
            FROM eval_metadata_sqlclauses
            WHERE meta_query = %s
            """
            db.cursor.execute(select_query, (meta_query,))
            row = db.cursor.fetchone()
            if row:
                clauses_dict = row['clauses_json']
                return clauses_dict
            return None
    except Exception as e:
        logging.error(f"Error retrieving cached metadata sqlclauses: {e}")
        return None

def save_cached_metadata_sqlclauses(meta_query, clauses_dict):
    try:
        with DatabaseConnection() as db:
            insert_query = """
            INSERT INTO eval_metadata_sqlclauses (meta_query, clauses_json)
            VALUES (%s, %s)
            ON CONFLICT (meta_query)
            DO UPDATE SET clauses_json = EXCLUDED.clauses_json;
            """
            # Store `clauses_dict` as JSONB
            db.cursor.execute(insert_query, (meta_query, json.dumps(clauses_dict)))
            db.conn.commit()
    except Exception as e:
        logging.error(f"Error saving cached metadata sqlclauses: {e}")

def average_embeddings(embeddings):
    # Ensure all embeddings are NumPy arrays of floats
    embeddings = [np.array(embed, dtype=float) if not isinstance(embed, np.ndarray) else embed.astype(float) for embed in embeddings]
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding

def average_embeddings_with_weights(embeddings, query_weight, hypo_weight):
    query_embedding = embeddings[0]
    hypo_embeddings = embeddings[1:]
    avg_embedding = (query_weight * query_embedding + hypo_weight * np.mean(hypo_embeddings, axis=0))
    return avg_embedding

# Re‑order `candidate_tables` via a one‑shot LLM call & return the top‑k unique table names
def llm_rerank_tables(user_query, candidate_tables, k):
    try:
        if not candidate_tables:
            return []

        # Compose a compact, structured prompt
        prompt = (
            "You are an expert dataset search assistant. "
            "Given the user's task and a list of candidate table names, "
            "return the *best‑to‑worst* ordering of those names as JSON.\n\n"
            f"Task: {user_query}\n\n"
            f"Candidates: {json.dumps(candidate_tables)}\n\n"
            f"Respond ONLY with a JSON object of the form: "
            '{{"ranked_tables": ["tbl1.csv", "tbl2.csv", ...]}}'
        )

        response_model = RerankResult

        messages = [
            {"role": "system", "content": "You are an expert dataset search assistant."},
            {"role": "user",   "content": prompt}
        ]

        result = openai_client.infer_metadata(messages, response_model)
        ranked = [t for t in result.ranked_tables if t in candidate_tables]

        # Preserve any items the model might have skipped
        for t in candidate_tables:
            if t not in ranked:
                ranked.append(t)
        return ranked[:k]

    except Exception as e:
        logging.error(f"LLM re‑ranker failed – falling back to original order. Error: {e}")
        return candidate_tables[:k]

def get_hyse_components_from_db(query, num_embed, schema_approach="relational"):
    """Retrieve HySE components and their embeddings from the database"""
    try:
        with DatabaseConnection() as db:
            select_query = """
            SELECT table_name_comp, table_header_comp, example_2rows_comp, example_3rows_comp,
                   table_header_embed, table_header_name_embed, 
                   example_2rows_embed, example_2rows_table_name_embed,
                   example_3rows_embed, example_3rows_table_name_embed
            FROM eval_hyse_components
            WHERE query = %s AND schema_approach = %s
            ORDER BY hypo_schema_id ASC
            LIMIT %s;
            """
            db.cursor.execute(select_query, (query, schema_approach, num_embed))
            results = db.cursor.fetchall()

            components = []
            for result in results:
                component = {
                    'table_name_comp': result['table_name_comp'],
                    'table_header_comp': result['table_header_comp'],
                    'example_2rows_comp': result['example_2rows_comp'],
                    'example_3rows_comp': result['example_3rows_comp'],
                    'embeddings': {}
                }
                
                # Process all embedding columns
                for embed_col in ['table_header_embed', 'table_header_name_embed', 
                                'example_2rows_embed', 'example_2rows_table_name_embed',
                                'example_3rows_embed', 'example_3rows_table_name_embed']:
                    embed_data = result[embed_col]
                    if isinstance(embed_data, str):
                        embedding_list = json.loads(embed_data)
                    elif isinstance(embed_data, list):
                        embedding_list = embed_data
                    else:
                        embedding_list = embed_data
                    
                    component['embeddings'][embed_col] = np.array(embedding_list, dtype=float)
                
                components.append(component)           
            return components
    except Exception as e:
        logging.exception(f"Error retrieving HySE components from DB: {e}")
        return []

def generate_hyse_components(query, schema_approach="relational"):
    """Generate all HySE components (table name, header, 2-row and 3-row examples) in one go using structured output"""
    try:
        logging.info(f"[HySE-Gen] Generating {schema_approach} schema for query: '{query}'")
        
        # Generate a single complete hypothetical schema with examples
        schema_result = infer_single_hypothetical_schema_with_examples(
            initial_query=query, 
            schema_approach=schema_approach
        )
        
        if not schema_result:
            logging.error(f"[HySE-Gen] Failed to generate schema for query: {query}")
            return None
        
        logging.info(f"[HySE-Gen] Generated schema with table_name: '{schema_result.table_name}', columns: {len(schema_result.column_names)}")
        logging.info(f"[HySE-Gen] Column names: {schema_result.column_names}")
        
        # Extract components using structured output
        components = {
            'table_name_comp': schema_result.table_name,
            'table_header_comp': schema_result.get_table_header(),
            'example_2rows_comp': schema_result.get_example_2rows_markdown(),
            'example_3rows_comp': schema_result.get_example_3rows_markdown()
        }
        
        logging.info(f"[HySE-Gen] Components generated:")
        logging.info(f"  Table name: {components['table_name_comp']}")
        logging.info(f"  Header: {components['table_header_comp']}")
        
        # Show first few characters of examples to verify diversity without cluttering logs
        example_preview = components['example_2rows_comp'].replace('\n', ' | ')[:150]
        logging.info(f"  Example preview: {example_preview}...")
        
        return components
    except Exception as e:
        logging.exception(f"[HySE-Gen] Error generating HySE components for query '{query}': {e}")
        return None

def generate_component_embeddings(components):
    """Generate embeddings for all HySE components"""
    try:
        logging.info(f"[HySE-Embed] Generating embeddings for all 6 component types")
        embeddings = {}
        
        # Generate embeddings for each component and its variations
        table_header_text = components['table_header_comp']
        table_name_text = components['table_name_comp']
        example_2rows_text = components['example_2rows_comp']  
        example_3rows_text = components['example_3rows_comp']
        
        logging.info(f"[HySE-Embed] Table name: '{table_name_text}'")
        logging.debug(f"[HySE-Embed] Header text length: {len(table_header_text)} chars")
        logging.debug(f"[HySE-Embed] 2-rows text length: {len(example_2rows_text)} chars")
        logging.debug(f"[HySE-Embed] 3-rows text length: {len(example_3rows_text)} chars")
        
        # Generate base embeddings
        logging.info(f"[HySE-Embed] Generating base embeddings...")
        embeddings['table_header_embed'] = openai_client.generate_embeddings(table_header_text)
        embeddings['example_2rows_embed'] = openai_client.generate_embeddings(example_2rows_text)
        embeddings['example_3rows_embed'] = openai_client.generate_embeddings(example_3rows_text)
        
        # Generate embeddings with table name included
        logging.info(f"[HySE-Embed] Generating table name + content embeddings...")
        embeddings['table_header_name_embed'] = openai_client.generate_embeddings(f"{table_name_text} {table_header_text}")
        embeddings['example_2rows_table_name_embed'] = openai_client.generate_embeddings(f"{table_name_text} {example_2rows_text}")
        embeddings['example_3rows_table_name_embed'] = openai_client.generate_embeddings(f"{table_name_text} {example_3rows_text}")
        
        # Convert to numpy arrays
        for key in embeddings:
            if not isinstance(embeddings[key], np.ndarray):
                embeddings[key] = np.array(embeddings[key], dtype=float)
        
        logging.info(f"[HySE-Embed] Successfully generated all 6 embedding types: {list(embeddings.keys())}")
        return embeddings
    except Exception as e:
        logging.exception(f"[HySE-Embed] Error generating component embeddings: {e}")
        return {}

def save_hyse_components_to_db(query, components, embeddings, schema_approach="relational"):
    """Save HySE components and their embeddings to the database"""
    try:
        with DatabaseConnection() as db:
            insert_query = """
            INSERT INTO eval_hyse_components (
                query, schema_approach, table_name_comp, table_header_comp, example_2rows_comp, example_3rows_comp,
                table_header_embed, table_header_name_embed, 
                example_2rows_embed, example_2rows_table_name_embed,
                example_3rows_embed, example_3rows_table_name_embed
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            
            # Convert embeddings to lists for database storage
            embed_lists = {}
            for key, embedding in embeddings.items():
                embed_lists[key] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
            db.cursor.execute(insert_query, (
                query,
                schema_approach,
                components['table_name_comp'],
                components['table_header_comp'], 
                components['example_2rows_comp'],
                components['example_3rows_comp'],
                embed_lists['table_header_embed'],
                embed_lists['table_header_name_embed'],
                embed_lists['example_2rows_embed'], 
                embed_lists['example_2rows_table_name_embed'],
                embed_lists['example_3rows_embed'],
                embed_lists['example_3rows_table_name_embed']
            ))
            db.conn.commit()
    except Exception as e:
        logging.exception(f"Error saving HySE components to DB: {e}")

def retrieve_or_generate_hyse_components(query, num_embed, schema_approach="relational"):
    """Retrieve cached HySE components or generate new ones if needed"""
    logging.info(f"[HySE-Utils] Fetching {num_embed} components for query: '{query}' (schema: {schema_approach})")
    
    # Fetch cached components for the specific schema approach
    cached_components = get_hyse_components_from_db(query, num_embed, schema_approach=schema_approach)
    num_cached = len(cached_components)
    
    logging.info(f"[HySE-Utils] Found {num_cached} cached components, need {num_embed}")
    
    # If not enough cached, generate more
    if num_cached < num_embed:
        num_to_generate = num_embed - num_cached
        logging.info(f"[HySE-Utils] Generating {num_to_generate} new components")
        
        for i in range(num_to_generate):
            logging.info(f"[HySE-Utils] Generating component {i+1}/{num_to_generate}")
            
            # Generate new components
            new_components = generate_hyse_components(query, schema_approach=schema_approach)
            if new_components:
                logging.info(f"[HySE-Utils] Generated components: table_name='{new_components['table_name_comp']}'")
                logging.debug(f"[HySE-Utils] Table header: {new_components['table_header_comp']}")
                
                # Generate embeddings for new components
                new_embeddings = generate_component_embeddings(new_components)
                if new_embeddings:
                    logging.info(f"[HySE-Utils] Generated {len(new_embeddings)} embeddings: {list(new_embeddings.keys())}")
                    
                    # Save to database with schema_approach
                    save_hyse_components_to_db(query, new_components, new_embeddings, schema_approach=schema_approach)
                    logging.info(f"[HySE-Utils] Saved component to database")
                    
                    # Add to cached list
                    component_with_embeddings = new_components.copy()
                    component_with_embeddings['embeddings'] = new_embeddings
                    cached_components.append(component_with_embeddings)
                else:
                    logging.error(f"[HySE-Utils] Failed to generate embeddings for component {i+1}")
            else:
                logging.error(f"[HySE-Utils] Failed to generate component {i+1}")
    else:
        logging.info(f"[HySE-Utils] Using all cached components")
    
    logging.info(f"[HySE-Utils] Final result: {len(cached_components)} components ready")
    return cached_components

def clear_hyse_cache_for_testing(query=None, schema_approach=None):
    """
    Clear HySE component cache for testing new prompts - USE WITH CAUTION
    """
    try:
        with DatabaseConnection() as db:
            if query and schema_approach:
                delete_query = "DELETE FROM eval_hyse_components WHERE query = %s AND schema_approach = %s;"
                db.cursor.execute(delete_query, (query, schema_approach))
                logging.info(f"[HySE-Cache] Cleared cache for query: '{query}' (schema: {schema_approach})")
            elif query:
                delete_query = "DELETE FROM eval_hyse_components WHERE query = %s;"
                db.cursor.execute(delete_query, (query,))
                logging.info(f"[HySE-Cache] Cleared all cache for query: '{query}'")
            else:
                delete_query = "DELETE FROM eval_hyse_components;"
                db.cursor.execute(delete_query)
                logging.warning(f"[HySE-Cache] Cleared ALL HySE component cache")
            
            db.conn.commit()
            rows_deleted = db.cursor.rowcount
            logging.info(f"[HySE-Cache] Deleted {rows_deleted} cached components")
            
    except Exception as e:
        logging.exception(f"[HySE-Cache] Error clearing cache: {e}")