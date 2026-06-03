from dataforager.table_representation.openai_client import OpenAIClient
from dataforager.utils import format_prompt, format_cos_sim_results
from dataforager.db.connect_db import DatabaseConnection
from pydantic import BaseModel
from typing import List, Any
import random
import logging
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
openai_client = OpenAIClient()

# Craft schema inference prompt
# PROMPT_SINGLE_SCHEMA = """
# Given the task of {query}, help me generate a database schema to to implement the task.
# Only generate one table schema, excluding any introductory phrases and focusing exclusively on the tasks themselves.
# Generate a JSON with keys as table names and values as column names, data types, and example rows. For example:

# Task:
# What data is needed to train a machine learning model to forecast demand for medicines across suppliers?

# Output: 
# {{
#     "table_name": "Sales",
#     "column_names": ["sale_id", "medicine_id", "supplier_id", "quantity_sold", "sale_date", "price", "region"],
#     "data_types": ["INT", "INT", "INT", "INT", "DATE", "DECIMAL", "VARCHAR"],
#     "example_row": [1, 101, 201, 50, "2024-06-01", 19.99, "North America"]
# }}
# """

PROMPT_SINGLE_SCHEMA_RELATIONAL = """
Given the task of {query}, help me generate a relational database schema to implement the task.
Only generate one table schema, excluding any introductory phrases and focusing exclusively on the tasks themselves.
Generate a JSON with keys as table names and values as column names. For example:

Task:
What data is needed to train a machine learning model to forecast demand for medicines across suppliers?

Output: 
{{
    "table_name": "Sales",
    "column_names": ["sale_id", "medicine_id", "supplier_id", "quantity_sold", "sale_date", "price", "region"]
}}
"""

PROMPT_SINGLE_SCHEMA_NON_RELATIONAL = """
Given the task of {query}, help me generate a non-relational database schema to implement the task.
Only generate one table schema, excluding any introductory phrases and focusing exclusively on the tasks themselves.
Generate a JSON with keys as table names and values as column names. For example:

Task:
What data is needed to train a machine learning model to forecast demand for medicines across suppliers?

Output:
{{
    "table_name": "Medicine_Sales",
    "column_names": ["medicine_name", "supplier_name", "quantity_sold", "sale_date", "price", "region"]
}}
"""

PROMPT_SCHEMA_WITH_EXAMPLES_RELATIONAL = """
Given the task: "{query}"

Generate a **relational database schema** specifically designed for this task. Think carefully about what data would be needed to accomplish this specific goal.

Requirements:
1. Create ONE table that directly supports the given task
2. Use normalized relational design principles (separate entities, use IDs for relationships)
3. Column names should reflect the specific domain and requirements of the task
4. Table name should be descriptive and task-relevant
5. Provide realistic example data that matches the task context

Generate a JSON with this exact structure:
{{
    "table_name": "your_task_specific_table_name",
    "column_names": ["col1", "col2", "col3", ...],
    "example_2rows": [
        [val1, val2, val3, ...],
        [val1, val2, val3, ...]
    ],
    "example_3rows": [
        [val1, val2, val3, ...],
        [val1, val2, val3, ...],
        [val1, val2, val3, ...]
    ]
}}

Focus on the specific requirements of "{query}" - what entities, attributes, and relationships would be needed?
"""

PROMPT_SCHEMA_WITH_EXAMPLES_NON_RELATIONAL = """
Given the task: "{query}"

Generate a **non-relational database schema** specifically designed for this task. Think carefully about what data would be needed to accomplish this specific goal.

Requirements:
1. Create ONE denormalized table that captures all necessary information for the task
2. Use descriptive, human-readable column names (avoid IDs, prefer natural keys)
3. Include all relevant attributes directly in the table (no foreign key relationships)
4. Column names should reflect the specific domain and requirements of the task
5. Table name should be descriptive and task-relevant
6. Provide realistic example data that matches the task context

Generate a JSON with this exact structure:
{{
    "table_name": "your_task_specific_table_name",
    "column_names": ["descriptive_col1", "descriptive_col2", "descriptive_col3", ...],
    "example_2rows": [
        [val1, val2, val3, ...],
        [val1, val2, val3, ...]
    ],
    "example_3rows": [
        [val1, val2, val3, ...],
        [val1, val2, val3, ...],
        [val1, val2, val3, ...]
    ]
}}

Focus on the specific requirements of "{query}" - what information would be needed in a single, comprehensive table?
"""

def get_diverse_schema_examples():
    """
    Generate diverse schema examples to reduce overfitting to any specific domain
    """
    import random
    
    examples = [
        {
            "domain": "Education",
            "relational": {
                "table_name": "student_grades",
                "columns": ["student_id", "course_id", "grade", "semester", "credits"]
            },
            "non_relational": {
                "table_name": "student_performance", 
                "columns": ["student_name", "course_name", "grade", "semester", "instructor", "credits"]
            }
        },
        {
            "domain": "Transportation",
            "relational": {
                "table_name": "vehicle_trips",
                "columns": ["trip_id", "vehicle_id", "start_location", "end_location", "duration", "distance"]
            },
            "non_relational": {
                "table_name": "transportation_logs",
                "columns": ["vehicle_type", "route", "start_time", "end_time", "passenger_count", "weather"]
            }
        },
        {
            "domain": "Environmental",
            "relational": {
                "table_name": "weather_measurements",
                "columns": ["measurement_id", "station_id", "temperature", "humidity", "timestamp"]
            },
            "non_relational": {
                "table_name": "climate_data",
                "columns": ["location", "temperature", "humidity", "wind_speed", "date", "season"]
            }
        },
        {
            "domain": "Finance",
            "relational": {
                "table_name": "account_transactions", 
                "columns": ["transaction_id", "account_id", "amount", "transaction_type", "timestamp"]
            },
            "non_relational": {
                "table_name": "financial_records",
                "columns": ["account_holder", "transaction_amount", "category", "description", "date", "balance"]
            }
        },
        {
            "domain": "Social Media",
            "relational": {
                "table_name": "user_posts",
                "columns": ["post_id", "user_id", "content", "likes", "timestamp"]
            },
            "non_relational": {
                "table_name": "social_activity",
                "columns": ["username", "post_content", "likes", "shares", "hashtags", "platform"]
            }
        }
    ]
    
    return random.choice(examples)

PROMPT_MULTI_SCHEMA = """
Given the task of {query}, generate a database schema of at least 1, and at most {num_left} normalized table headers that are needed to implement the task.
Generate a list in which each element is a JSON with keys as table names and values as column names, data types, and example rows. For example:

Task:
What data is needed to train a machine learning model to forecast demand for medicines across suppliers?

Output: 
[
  {{
    "table_name": "Sales",
    "column_names": ["sale_id", "medicine_id", "supplier_id", "quantity_sold", "sale_date", "price", "region"],
    "data_types": ["INT", "INT", "INT", "INT", "DATE", "DECIMAL", "VARCHAR"],
    "example_row": [1, 101, 201, 50, "2024-06-01", 19.99, "North America"]
  }},
  {{
    "table_name": "Medicine",
    "column_names": ["medicine_id", "name", "type", "dosage", "packaging", "shelf_life", "supplier_id"],
    "data_types": ["INT", "VARCHAR", "VARCHAR", "VARCHAR", "VARCHAR", "INT", "INT"],
    "example_row": [101, "Aspirin", "Pain Reliever", "500mg", "Bottle", 24, 201]
  }}
]
"""

# TODO: refactor pydantic models
# Define desired output structure
# class TableSchema(BaseModel):
#     table_name: str
#     column_names: List[str]
#     data_types: List[str]
#     example_row: List[Any]

class TableSchema(BaseModel):
    table_name: str
    column_names: List[str]

class TableSchemaWithExamples(BaseModel):
    table_name: str
    column_names: List[str]
    example_2rows: List[List[Any]]  # First 2 data rows (excluding header)
    example_3rows: List[List[Any]]  # First 3 data rows (excluding header)
    
    def get_table_header(self) -> str:
        """Get formatted table header as string"""
        return " | ".join(self.column_names)
    
    def get_example_2rows_markdown(self) -> str:
        """Get header + 2 example rows formatted as markdown table"""
        header = " | ".join(self.column_names)
        separator = " | ".join(["---"] * len(self.column_names))
        rows = []
        for row in self.example_2rows[:2]:  # Ensure max 2 rows
            rows.append(" | ".join(str(cell) for cell in row))
        return "\n".join([header, separator] + rows)
    
    def get_example_3rows_markdown(self) -> str:
        """Get header + 3 example rows formatted as markdown table"""
        header = " | ".join(self.column_names)
        separator = " | ".join(["---"] * len(self.column_names))
        rows = []
        for row in self.example_3rows[:3]:  # Ensure max 3 rows
            rows.append(" | ".join(str(cell) for cell in row))
        return "\n".join([header, separator] + rows)

def hyse_search(initial_query, search_space=None, num_schema=1, k=10, table_name="corpus_raw_metadata_with_embedding", column_name="comb_embed"):
    # Step 0: Initialize the results list and num_left
    results = []
    num_left = num_schema

    # Step 1: Single HySE search
    # Step 1.1: Infer a single relational (normalized) hypothetical schema
    single_hypo_schema_json = infer_single_hypothetical_schema(initial_query, PROMPT_SINGLE_SCHEMA_RELATIONAL).json()
    logging.info(f"Single hypothetical schema JSON: {single_hypo_schema_json}")

    # Step 1.2: Generate embedding for the single hypothetical schema
    single_hypo_schema_embedding = openai_client.generate_embeddings(text=single_hypo_schema_json)

    # Step 1.3: Cosine similarity search between e(hypo_schema_embed) and e(existing_scheme_embed)
    single_hyse_results = cos_sim_search(single_hypo_schema_embedding, search_space, table_name, column_name)
    results.append(single_hyse_results)

    # Step 1.4: Update num_left by decrementing it by 1
    num_left -= 1
    
    # Step 2: Multiple HySE search
    while num_left > 0:
        # Step 2.1: Randomly generate the number of normalized schemas m ranging from 1 to num_left
        m = random.randint(1, num_left)

        # Step 2.2: Infer multiple normalized hypothetical schemas
        multi_hypo_schemas, m = infer_multiple_hypothetical_schema(initial_query, m)
        multi_hypo_schemas_json = [schema.json() for schema in multi_hypo_schemas]
        logging.info(f"Multiple hypothetical schemas JSON: {multi_hypo_schemas_json}")

        # Step 2.3: Generate embeddings for the multiple hypothetical schemas
        multi_hypo_schemas_embeddings = [openai_client.generate_embeddings(text=schema_json) for schema_json in multi_hypo_schemas_json]

        # Step 2.4: Cosine similarity search for each multiple hypothetical schema embedding
        for hypo_schema_embedding in multi_hypo_schemas_embeddings:
            results.append(cos_sim_search(hypo_schema_embedding, search_space, table_name, column_name))
        
        # Step 2.5: Update num_left by decrementing it by m
        num_left -= m

    # Step 3: Aggregate results from single & multiple HySE searches
    aggregated_results = aggregate_hyse_search_results(results)

    # Sort aggregated results by cosine similarity and keep top k
    aggregated_results.sort(key=lambda x: x['cosine_similarity'], reverse=True)
    top_k_results = aggregated_results[:k]

    return top_k_results, single_hypo_schema_json, single_hypo_schema_embedding

def infer_single_hypothetical_schema_with_examples(initial_query, schema_approach="relational"):
    """
    Generate a single hypothetical schema with example data using structured output
    """
    try:
        # Get a diverse example to inspire different designs
        diverse_example = get_diverse_schema_examples()
        example_schema = diverse_example["relational"] if schema_approach == "relational" else diverse_example["non_relational"]
        
        # Pick the template based on schema_approach
        if schema_approach == "relational":
            prompt_template = PROMPT_SCHEMA_WITH_EXAMPLES_RELATIONAL
        else:
            prompt_template = PROMPT_SCHEMA_WITH_EXAMPLES_NON_RELATIONAL

        # Format the prompt
        formatted_prompt = format_prompt(prompt_template, query=initial_query)
        
        response_model = TableSchemaWithExamples
        
        # Enhanced system prompt with dynamic example for inspiration
        system_prompt = f"""
            You are an expert database designer. For each task, you must:

            1. CAREFULLY analyze the specific task requirements
            2. Design a schema that directly supports THAT SPECIFIC TASK (not generic examples)
            3. Use column names and table names that reflect the ACTUAL DOMAIN of the task
            4. Generate realistic example data that matches the task context
            5. Avoid generic or repetitive schemas - be creative and domain-specific
            6. Think creatively about what data structures would best support the given objective

            For inspiration (but do NOT copy), here's an example from the {diverse_example["domain"]} domain:
            Table: {example_schema["table_name"]}
            Columns: {example_schema["columns"]}

            Your schema should be completely different and tailored to the specific task provided.
            Be diverse and task-specific in your designs. Different tasks should produce different schemas.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_prompt}
        ]
        
        # Add some randomness to encourage diversity (temperature)
        result = openai_client.infer_metadata(messages, response_model, temperature=0.8)
        return result
    except Exception as e:
        logging.exception(f"Error inferring single hypothetical schema with examples: {e}")
        return None

def infer_single_hypothetical_schema(initial_query, prompt_template):
    prompt = format_prompt(prompt_template=prompt_template, query=initial_query)

    response_model = TableSchema

    messages = [
        {"role": "system", "content": "You are an assistant skilled in generating database schemas."},
        {"role": "user", "content": prompt}
    ]

    return openai_client.infer_metadata(messages, response_model)

def infer_multiple_hypothetical_schema(initial_query, num_left):
    prompt = format_prompt(PROMPT_MULTI_SCHEMA, query=initial_query, num_left=num_left)

    response_model = List[TableSchema]

    messages = [
        {"role": "system", "content": "You are an assistant skilled in generating normalized database schemas."},
        {"role": "user", "content": prompt}
    ]

    response = openai_client.infer_metadata(messages, response_model)
    m = len(response)
    return response, m

def cos_sim_search(input_embedding, search_space, table_name="corpus_raw_metadata_with_embedding", column_name="comb_embed"):  
    # Ensure input_embedding is a list before passing to execute
    if isinstance(input_embedding, np.ndarray):
        input_embedding = input_embedding.tolist()
    elif isinstance(input_embedding, list):
        input_embedding = input_embedding
    else:
        input_embedding = list(input_embedding)
    
    with DatabaseConnection() as db:
        if search_space:
            # Filter by specific table names
            query = f"""
                SELECT table_name, 1 - ({column_name} <=> %s::VECTOR(1536)) AS cosine_similarity
                FROM {table_name}
                WHERE table_name = ANY(%s)
                ORDER BY cosine_similarity DESC;
            """
            db.cursor.execute(query, (input_embedding, search_space))
        else:
            # No specific search space, search through all table names
            query = f"""
                SELECT table_name, 1 - ({column_name} <=> %s::VECTOR(1536)) AS cosine_similarity
                FROM {table_name}
                ORDER BY cosine_similarity DESC;
            """
            db.cursor.execute(query, (input_embedding,))
        
        results = db.cursor.fetchall()
    
    # Ensure results are correctly structured
    formatted_results = [{'table_name': row['table_name'], 'cosine_similarity': float(row['cosine_similarity'])} for row in results]
    return formatted_results

def aggregate_hyse_search_results(results):
    # Flatten the list of results
    flat_results = [item for sublist in results for item in sublist]
    
    # Aggregate by table name and calculate mean cosine similarity
    aggregated_results = {}
    for result in flat_results:
        table_name = result['table_name']
        cosine_similarity = result['cosine_similarity']
        
        if not isinstance(cosine_similarity, (int, float)):
            logging.error(f"Unexpected type for cosine_similarity: {type(cosine_similarity)} with value {cosine_similarity}")
            raise ValueError(f"Unexpected type for cosine_similarity: {type(cosine_similarity)}")
        
        if table_name in aggregated_results:
            aggregated_results[table_name].append(cosine_similarity)
        else:
            aggregated_results[table_name] = [cosine_similarity]
    
    # Calculate the mean cosine similarity for each table
    final_results = [{'table_name': table_name, 'cosine_similarity': sum(similarities) / len(similarities)} for table_name, similarities in aggregated_results.items()]
    
    # Sort the final results by mean cosine similarity in descending order
    final_results.sort(key=lambda x: x['cosine_similarity'], reverse=True)
    return final_results
