import json
from tqdm import tqdm
from .openai_client import OpenAIClient

# Initialize OpenAI client
openai_client = OpenAIClient()

# Function to generate a single embedding for multiple metadata fields
def generate_combined_embedding(dataset, metadata_fields):
    combined_text = []

    for field in metadata_fields:
        value = dataset.get(field, '')
        if isinstance(value, list):
            # For lists of dictionaries, convert each item to a string
            value_str = ', '.join([json.dumps(item) if isinstance(item, dict) else str(item) for item in value])
            combined_text.append(f"{field}: {value_str}")
        elif isinstance(value, dict):
            # Directly convert dictionaries to a string
            combined_text.append(f"{field}: {json.dumps(value)}")
        else:
            # For simple string fields
            combined_text.append(f"{field}: {value}")
    
    combined_text_str = '\n'.join(combined_text)
    return openai_client.generate_embeddings(combined_text_str)

# Load the mock data
with open('mock_data/updated_data_gov_mock_data.json', 'r') as file:
    mock_data_corpus = json.load(file)

# Define the metadata fields that we want to embed together
COMBINED_METADATA_FIELDS = [
    'Table name',
    'Table schema',
    'Example records',
    'Table description',
    'Table tags'
]

# TODO: Try differenet model embeddings to compare performance
for dataset in tqdm(mock_data_corpus, desc="Embedding datasets"):
    dataset['Combined embedding'] = generate_combined_embedding(dataset=dataset, metadata_fields=COMBINED_METADATA_FIELDS)
    
    query_text = json.dumps(dataset['Previous queries'])
    dataset['Query embedding'] = openai_client.generate_embeddings(text=query_text)

# Save the updated data w/ embeddings back to a new JSON file
with open('mock_data/mock_data_with_embedding.json', 'w') as file:
    json.dump(mock_data_corpus, file, indent=2)
