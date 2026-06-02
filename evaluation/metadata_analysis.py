"""
Analyze metadata queries and suggest improvements for metadata refinement
"""
import logging
from dataforager.db.connect_db import DatabaseConnection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_metadata_effectiveness(data_split="eval_data_validation", limit=None):
    """
    Analyze metadata queries to understand their effectiveness
    """
    db_connection = DatabaseConnection()

    with db_connection as db:
        query = f"SELECT table_name, metadata_queries FROM {data_split}"
        if limit:
            query += f" LIMIT {limit};"
        else:
            query += ";"

        db.cursor.execute(query)
        rows = db.cursor.fetchall()

        # Statistics
        total_entries = len(rows)
        metadata_field_counts = {
            'total_sublists': 0,
            'avg_queries_per_sublist': [],
            'entries_with_empty_metadata': 0
        }

        for row in rows:
            metadata_sublists = row['metadata_queries']

            if not metadata_sublists:
                metadata_field_counts['entries_with_empty_metadata'] += 1
                continue

            for sublist in metadata_sublists:
                metadata_field_counts['total_sublists'] += 1
                metadata_field_counts['avg_queries_per_sublist'].append(len(sublist))

        # Report
        logging.info(f"\n{'='*60}")
        logging.info(f"METADATA QUERY ANALYSIS")
        logging.info(f"{'='*60}")
        logging.info(f"Total entries: {total_entries}")
        logging.info(f"Entries with empty metadata: {metadata_field_counts['entries_with_empty_metadata']}")
        logging.info(f"Total metadata sublists: {metadata_field_counts['total_sublists']}")

        if metadata_field_counts['avg_queries_per_sublist']:
            avg = sum(metadata_field_counts['avg_queries_per_sublist']) / len(metadata_field_counts['avg_queries_per_sublist'])
            logging.info(f"Avg metadata queries per sublist: {avg:.2f}")

        logging.info(f"{'='*60}\n")

if __name__ == "__main__":
    analyze_metadata_effectiveness()
