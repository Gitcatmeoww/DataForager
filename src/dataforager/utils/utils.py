import json
import re
from dataforager.db.connect_db import DatabaseConnection
import pandas as pd
import ast

def format_prompt(prompt_template, **kwargs):
    """ Formats a given prompt template with provided keyword arguments """
    return prompt_template.format(**kwargs)

def load_json_file(file_path):
    """ Load JSON data from a file """ 
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_time_geo_granularity(granularity_str):
    """ Extract time and geo granularity lists from a JSON string """
    granularity = json.loads(granularity_str)

    # Initialize empty sets for time and geographic granularities
    time_granu = set()
    geo_granu = set()

    # Iterate through each item in the granularity dictionary
    for value in granularity.values():
        # Check if 'temporal' key exists and has a non-null value
        if 'temporal' in value and value['temporal']:
            time_granu.add(value['temporal'])
        
        # Check if 'geographic' key exists and has a non-null value
        if 'geographic' in value and value['geographic']:
            geo_granu.add(value['geographic'])

    # Convert sets to lists for JSON serialization
    return list(time_granu), list(geo_granu)


def run_sql_file(filename):
    """ Executes the SQL statements contained within a file """
    with open(filename, 'r') as file:
        sql_script = file.read()

   # Use the DatabaseConnection context manager to get a cursor
    with DatabaseConnection() as db:
        try:
            # Split the script into individual statements if necessary
            sql_statements = sql_script.split(';')
            
            # Execute each statement individually
            for statement in sql_statements:
                # Strip whitespace and skip empty statements
                if statement.strip():
                    db.cursor.execute(statement.strip())
            # Commit the transaction
            db.cursor.connection.commit()
        except Exception as e:
            # Rollback the transaction on error
            db.cursor.connection.rollback()
            print(f"An error occurred: {e}")
            raise

def format_cos_sim_results(results):
    """ Format results returned from pgvector operations """
    formatted_results = []

    for row in results:
        formatted_result = {
            "table_name": row["table_name"],
            "cosine_similarity": row["cosine_similarity"]
        }
        formatted_results.append(formatted_result)

    return formatted_results

def clean_json_string(json_str):
    """ Clean a JSON string by fixing common formatting issues """
    # Remove special characters like newlines and tabs
    json_str = json_str.replace('\n', '\\n').replace('\t', '\\t')
    
    # Replace single quotes with double quotes
    json_str = json_str.replace("'", '"')
    
    # Add double quotes around any keys that are not properly quoted
    json_str = re.sub(r'(?<!")(\b[^":\n\r]+?\b)(?=\s*:)', r'"\1"', json_str)
    
    # Remove trailing commas before closing brackets or braces
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    return json_str

def validate_and_load_json(json_str):
    """ Validate and parse a JSON string into a Python dictionary using the cleaned JSON string """
    cleaned_json_str = clean_json_string(json_str)
    try:
        return json.loads(cleaned_json_str), None
    except json.JSONDecodeError as e:
        return None, str(e)

def extract_granularities(json_data):
    time_granu = []
    geo_granu = []
    
    # Load JSON data
    data = json.loads(json_data) if isinstance(json_data, str) else json_data
    
    # Iterate over each table in the JSON array
    for table in data:
        # Extract temporal and geographic granularities
        if "Temporal granularity" in table:
            time_granu.append(table["Temporal granularity"])
        if "Geographic granularity" in table:
            geo_granu.append(table["Geographic granularity"])
    
    return time_granu, geo_granu

def get_finer_granularities(granularity, order_list):
        """ Return the finer granularities including the specified one """
        if granularity.lower() in order_list:
            index = order_list.index(granularity.lower())
            return order_list[:index + 1]
        return []

def parse_list_column(column_value):
    """ Parse a string representation of a list into a Python list of strings """
    if pd.isna(column_value):
        return []

    # Check if it's a "Postgres-like" curly-brace format: { ... }
    if (isinstance(column_value, str) 
        and column_value.startswith('{') 
        and column_value.endswith('}')):
        
        # Strip the surrounding braces
        content = column_value[1:-1].strip()
        
        # Split on commas that are NOT inside double quotes
        items = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', content)

        parsed_items = []
        for item in items:
            item = item.strip()
            # Remove surrounding double quotes if present
            if len(item) >= 2 and item.startswith('"') and item.endswith('"'):
                item = item[1:-1]
            parsed_items.append(item)
        return parsed_items
    
    # Otherwise, handle Python/JSON-like formats (e.g. ["item1","item2"] or just "item1")
    try:
        parsed_value = ast.literal_eval(column_value)
        if isinstance(parsed_value, list):
            return [str(elem).strip() for elem in parsed_value]
        else:
            # If it's just a single value, wrap it in a list
            return [str(parsed_value).strip()]
    except Exception as e:
        # Fallback: treat the entire value as a single string item
        print(f"Error parsing list column '{column_value}': {e}")
        return [str(column_value).strip()]

def parse_json_column(column_value):
    """ Parse a string representation of a JSON object into a Python dictionary """
    if pd.isna(column_value):
        return None
    try:
        # Use `json.loads` to handle JSON parsing
        if isinstance(column_value, str):
            return json.loads(column_value)
        return column_value
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON column '{column_value}': {e}")
        return None
    except Exception as e:
        print(f"Unhandled error parsing JSON column '{column_value}': {e}")
        return None
    
def parse_vector_column(column_value):
    """ Parses a stringified vector into a Python list of floats """
    if column_value is None or column_value == '':
        return None
    try:
        return [float(x) for x in eval(column_value)] if isinstance(column_value, str) else column_value
    except Exception as e:
        print(f"Error parsing vector: {e}")
        return None