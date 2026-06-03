"""
- This script extracts example tabular data and metadata from datasets available on data.gov to create a mock dataset corpus for our system.
- Data.gov is chosen for its diverse and accessible tabular datasets accompanied by rich metadata, which provides a comprehensive foundation for our corpus. 
- The script retrieves datasets in CSV format, emphasizing those with notable metadata like titles, descriptions, and tags. 
- It aims to enrich our system's data corpus with realistic dataset examples and metadata for testing and development purposes.
- For structure and example of the metadata response, see `mock_data/data_gov_metadata_example.json`.
- Below is a mapping of the metadata available in response & our desired metadata:
    * Table name: title
    * Table schema: N/A
    * Example records: read from csv
    * Table description: notes
    * Table tags: tags['display_name']
    * Column numbers: count example record dimensions
    * Previous queries: N/A
    * Granularity: N/A
    * Popularity: randomly generated number between 0~10,000
"""

import requests
import json
from random import randint
from io import StringIO
import urllib.request
import csv
import ssl

# Base URL for the data.gov CKAN API
CKAN_BASE_URL = 'https://catalog.data.gov/api/3/action'
# Number of desired datasets
LIMIT_NUM = 200
# Number of desired rows in example records for each dataset
LIMIT_ROW = 2


def get_example_records(csv_url, limit_row=LIMIT_ROW):
    # Create an SSLContext that does not verify certificates
    context = ssl._create_unverified_context()
    records = []

    try:
        # Use urllib to handle the SSL certificate verification issue
        with urllib.request.urlopen(csv_url, context=context) as response:
            # Read only the first few lines to limit to `limit_row` rows
            lines = [next(response) for _ in range(limit_row + 1)]  # +1 for the header
            csv_content = ''.join([line.decode('utf-8', errors='replace') for line in lines])
            
            # Use csv.DictReader to parse the CSV content
            reader = csv.DictReader(StringIO(csv_content))
            records = [row for row in reader]

    except Exception as e:
        print(f"Error fetching or parsing CSV: {e}")

    return records

# Function to get datasets in CSV format (since we only want to deal with tabular data)
def get_csv_datasets_and_metadata(limit_num=LIMIT_NUM):
    # Set the API endpoint for package search
    package_search_url = f'{CKAN_BASE_URL}/package_search'
    # Set parameters for the search, including the CSV format filter and the row limit
    params = {
        'rows': limit_num,  # the maximum number of matching rows (datasets) to return
        'start': 0,  # the offset in the complete result for where the set of returned datasets should begin
        # 'fq': 'res_format=CSV',  # any filter queries to apply
        'sort': 'views_recent desc'  # sort by popularity
    }

    # Make the GET request
    response = requests.get(package_search_url, params=params)
    # Check for request errors
    response.raise_for_status()  
    # Parse the response JSON content
    search_results = response.json()
    
    datasets_info = []
    # Extract dataset details
    for dataset in search_results['result']['results']:
        records = []  # Ensure records are reset for each dataset

        for resource in dataset['resources']:
            if resource['format'].upper() == 'CSV':
                records = get_example_records(resource['url'])

        # Only add datasets with valid (non-empty) example records
        if records:
            # Add metadata for each dataset
            datasets_info.append({
                'Table name': dataset.get('title', 'N/A'),
                'Table schema': 'N/A',
                'Table description': dataset.get('notes', 'N/A'),
                'Table tags': [tag['display_name'] for tag in dataset.get('tags', [])],
                'Column numbers': len(records[0]),
                'Popularity': randint(0, 10000),
                'Previous queries': 'N/A',
                'Granularity': 'N/A',
                'Example records': records
            })

    return datasets_info

datasets_metadata = get_csv_datasets_and_metadata()
print(f"# of valid datasets in mock data corpus is {len(datasets_metadata)}")

# Store the data into a JSON file as mock dataset corpus
with open('mock_data/data_gov_mock_data.json', 'w') as f:
    json.dump(datasets_metadata, f, indent=2)
