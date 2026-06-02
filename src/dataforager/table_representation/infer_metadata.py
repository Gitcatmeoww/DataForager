"""
- This script utilizes pre-trained GPT models to infer missing metadata crucial for enhancing semantic searches within datasets.
- Specifically, for our mock dataset corpus sourced from data.gov, the following metadata are missing, but we can infer them from `table_name`, `table_description`, and `example_records`:
    * Table schema: Outlines the structure and types of data within the table.
    * Previous queries: Suggests potential queries users might have used to find this data, aiding in understanding user search behavior.
    * Granularity: Determines the level of detail or precision of the data contained within the dataset.
- Upon successfully inferring missing metadata, the script integrates the new information back into the dataset corpus.
- The enriched corpus is then prepared for further processing, including embedding, to facilitate more efficient and effective dataset retrieval and utilization.
"""

import json
from tqdm import tqdm
from .openai_client import OpenAIClient
from dataforager.utils import format_prompt
from pydantic import BaseModel

# Initialize OpenAI client
openai_client = OpenAIClient()

# Craft metadata inference prompts
# With the help of instructor and pydantic, we can simplify the prompts
PROMPTS = {
    "schema_prompt": "Given the dataset titled '{table_name}' which includes data on {table_description}, with example records like {example_records}, directly list the likely table schema.",
    "query_prompt": "Please provide some data analytics tasks (e.g. data analysis, machine learning, business intelligence, etc.) that can be performed for the table titled '{table_name}' which includes data on {table_description}, with example records like {example_records}? Specify the analytics tasks specific to the semantics of the table, and provide all tasks (without categorization) in a flat list, excluding any introductory phrases and focusing exclusively on the tasks themselves.",
    "granularity_prompt": "Given a dataset titled '{table_name}' with data on {table_description} and example records like {example_records}, determine the most likely geographic or temporal granularity present in the dataset, if any. Please select the temporal granularity from the following options: Year, Quarter, Month, Week, Day, Hour, Minute, or Second. For the geographic granularity, choose from: Continent, Country, State/Province, County/District, City, or Zip Code/Postal Code. Identify the granularity if present, based on the data provided."
}

# PROMPTS = {
#     "schema_prompt": "Given the dataset titled '{table_name}' which includes data on {table_description}, with example records like {example_records}, directly list the likely table schema. Please provide this schema as a concise list of column names followed by their data types, without any introductory text, commentary, or conclusion. Format the schema details in a straightforward manner, with each column and data type on a new line.",
#     "query_prompt": "Please provide some data analytics tasks (e.g. data analysis, machine learning, business intelligence, etc.) that can be performed for the table titled '{table_name}' which includes data on {table_description}, with example records like {example_records}? Specify the analytics tasks specific to the semantics of the table, and provide all tasks (without categorization) in a flat list, excluding any introductory phrases and focusing exclusively on the tasks themselves.",
#     "granularity_prompt": "Given a dataset titled '{table_name}' with data on {table_description} and example records like {example_records}, identify columns that express some granularity. For each identified column, determine if it relates to geographic or temporal attributes. Provide the results in a compact, single-line JSON format. Do not include markdown or additional annotations, also make sure property name enclosed in double quotes. The expected JSON structure is {{column_name: {{temporal: 'temporal_granularity', geographic: 'geographic_granularity'}}}}. Include only columns that have granularity attributes, and minimize whitespace in the output."
# }

# Define desired output structure
class TableSchema(BaseModel):
    column_names: list[str]
    data_types: list[str]

class PreviousQueries(BaseModel):
    queries: list[str]

class Granularity(BaseModel):
    time_granu: list[str]
    geo_granu: list[str]
    
def infer_table_schema(table_name, table_description, example_records):
    prompt = format_prompt(
        PROMPTS['schema_prompt'], 
        table_name=table_name, 
        table_description=table_description, 
        example_records=json.dumps(example_records)
    )

    response_model = TableSchema

    messages = [
        {"role": "system", "content": "You are an assistant skilled in database schemas."},
        {"role": "user", "content": prompt}
    ]
    return openai_client.infer_metadata(messages, response_model)

def infer_previous_queries(table_name, table_description, example_records):
    prompt = format_prompt(
        PROMPTS['query_prompt'], 
        table_name=table_name, 
        table_description=table_description, 
        example_records=json.dumps(example_records)
    )

    response_model = PreviousQueries
    
    messages = [
        {"role": "system", "content": "You are an assistant providing possible user queries for datasets."},
        {"role": "user", "content": prompt}
    ]
    return openai_client.infer_metadata(messages, response_model)

def infer_granularity(table_name, table_description, example_records):
    prompt = format_prompt(
        PROMPTS['granularity_prompt'], 
        table_name=table_name, 
        table_description=table_description, 
        example_records=json.dumps(example_records)
    )

    response_model = Granularity

    messages = [
        {"role": "system", "content": "You are an assistant skilled in determining data granularity."},
        {"role": "user", "content": prompt}
    ]
    return openai_client.infer_metadata(messages, response_model)

# Load the mock data
with open('mock_data/data_gov_mock_data.json', 'r') as file:
    mock_data_corpus = json.load(file)

for dataset in tqdm(mock_data_corpus, desc="Processing datasets"):
    table_name = dataset['Table name']
    print(f"Current processing dataset: %s" % table_name)
    table_description = dataset['Table description']
    example_records = dataset['Example records']

    # Call OpenAI API to infer unspecified metadata
    table_schema = infer_table_schema(
        table_name=dataset['Table name'], 
        table_description=dataset['Table description'], 
        example_records=dataset['Example records']
    )
    dataset['Table schema'] = table_schema.json()

    previous_queries = infer_previous_queries(
        table_name=dataset['Table name'], 
        table_description=dataset['Table description'], 
        example_records=dataset['Example records']
    )
    dataset['Previous queries'] = previous_queries.json()

    granularity = infer_granularity(
        table_name=dataset['Table name'], 
        table_description=dataset['Table description'], 
        example_records=dataset['Example records']
    )
    dataset['Granularity'] = granularity.json()
    
    # Extract the time/geo granularity
    dataset['Temporal granularity'] = granularity.time_granu
    dataset['Geographic granularity'] = granularity.geo_granu

# Save the updated data back to a new JSON file
with open('mock_data/updated_data_gov_mock_data.json', 'w') as file:
    json.dump(mock_data_corpus, file, indent=2)