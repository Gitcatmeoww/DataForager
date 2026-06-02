"""Quick script to check if eval_data_test table exists in the schema database."""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to the schema database
conn = psycopg2.connect(
    dbname='HITS-eval-data-corpus-exp-opt-schema',
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT'),
    cursor_factory=RealDictCursor
)

cursor = conn.cursor()

# Check all tables
cursor.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name
""")
tables = [row['table_name'] for row in cursor.fetchall()]
print("Tables in database:")
for table in tables:
    print(f"  - {table}")

# Check if eval_data_test exists
if 'eval_data_test' in tables:
    print("\n✅ eval_data_test table exists")

    # Get column info
    cursor.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'eval_data_test'
        ORDER BY ordinal_position
    """)
    columns = cursor.fetchall()
    print("\nColumns in eval_data_test:")
    for col in columns:
        print(f"  - {col['column_name']}: {col['data_type']}")

    # Get row count
    cursor.execute("SELECT COUNT(*) as count FROM eval_data_test")
    count = cursor.fetchone()['count']
    print(f"\nRows in eval_data_test: {count}")
else:
    print("\n❌ eval_data_test table does NOT exist")

cursor.close()
conn.close()
