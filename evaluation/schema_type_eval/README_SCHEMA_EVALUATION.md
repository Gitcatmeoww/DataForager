# Schema Type Evaluation Pipeline

This directory contains a comprehensive pipeline for evaluating Hyse performance on different schema types (normalized vs denormalized tables).

## Overview

The schema type evaluation pipeline extends the existing evaluation framework to provide detailed analysis of how Hyse performs on tables with different schema structures. This enables researchers to understand whether table normalization affects retrieval performance.

## Features

- **Database Migration**: Creates a new database with schema type support
- **Data Integration**: Populates schema type information from CSV files
- **Schema-Aware Evaluation**: Runs separate evaluations for normalized and denormalized tables
- **Statistical Analysis**: Provides statistical comparison between schema types
- **Comprehensive Reporting**: Generates detailed result files and analysis summaries

## Quick Start

### 1. Prerequisites

Ensure you have:

- PostgreSQL database with existing evaluation data
- CSV file with schema type annotations
- Required environment variables set:
  ```bash
  DB_USER=your_username
  DB_PASSWORD=your_password
  DB_HOST=localhost
  DB_PORT=5432
  EVAL_DB_NAME=HITS-eval-data-corpus-exp-opt
  ```

### 2. Complete Setup

Run the setup script to automatically configure everything:

```bash
cd backend/app/evals/schema_type_eval
python setup_schema_evaluation.py
```

This will:

1. Create new database: `HITS-eval-data-corpus-exp-opt-schema`
2. Copy all data from original database
3. Add `schema_type` column
4. Populate schema types from CSV
5. Run sample evaluation to verify setup

### 3. Run Schema Type Evaluation

```python
from schema_type_evaluator import SchemaTypeEvaluator

# Create evaluator
evaluator = SchemaTypeEvaluator(
    data_split="eval_data_validation",
    embed_col="example_2rows_table_name_embed",
    k=10,
    limit=None,  # Use None for full dataset
    num_embed=2,
    use_schema_db=True
)

# Run complete analysis
results = evaluator.run_schema_type_analysis()
```

## Manual Setup (Step by Step)

### Step 1: Database Migration

```bash
python migrate_database.py
```

Creates a new database with:

- All tables and data from original database
- Added `schema_type` column to `eval_data_validation`
- Proper indexes and constraints

### Step 2: Populate Schema Types

```bash
python populate_schema_types.py
```

Updates the database with schema type information from CSV file.

### Step 3: Test Setup

```python
from backend.app.db.schema_db_connect import SchemaAwareDatabaseConnection

connection_manager = SchemaAwareDatabaseConnection()
test_results = connection_manager.test_connections()
print(test_results)
```

### Step 4: Run Evaluation

```python
from schema_type_evaluator import SchemaTypeEvaluator

evaluator = SchemaTypeEvaluator(use_schema_db=True)
results = evaluator.run_schema_type_analysis()
```

## API Reference

### SchemaTypeEvaluator

Main class for schema-aware evaluation.

#### Constructor

```python
SchemaTypeEvaluator(
    data_split="eval_data_validation",
    embed_col="example_rows_embed",
    k=10,
    limit=None,
    num_embed=1,
    use_schema_db=True
)
```

**Parameters:**

- `data_split`: Database table name for evaluation data
- `embed_col`: Column name for embeddings
- `k`: Number of top results to consider for recall calculation
- `limit`: Maximum number of records to evaluate (None for all)
- `num_embed`: Number of embedding components for Hyse
- `use_schema_db`: Whether to use schema-enhanced database

#### Methods

**`run_schema_type_analysis()`**
Runs complete schema type analysis including evaluation and comparison.

**`evaluate_by_schema_type()`**
Runs evaluation separately for normalized and denormalized tables.

**`compare_schema_performance(recalls_by_schema)`**
Performs statistical comparison between schema types.

### DatabaseMigrator

Handles database migration and setup.

```python
migrator = DatabaseMigrator()
success = migrator.migrate()  # Returns True/False
migrator.rollback()  # Rollback if needed
```

### SchemaTypePopulator

Populates schema type data from CSV.

```python
populator = SchemaTypePopulator(csv_path="path/to/csv")
success = populator.populate()  # Returns True/False
```

## Results Structure

### Per-Row Results

- `normalized_results.csv`: Detailed results for normalized tables
- `denormalized_results.csv`: Detailed results for denormalized tables

Columns:

- Index, Table Name, Method, Query Type, Query, Recall, Schema Type, Ground Truth Header, Hypothetical Schema

### Comparison Results

- `schema_comparison.csv`: Statistical comparison between schema types

Columns:

- Method, Normalized Recall, Denormalized Recall, Recall Difference, Normalized Count, Denormalized Count, P Value, Significant

### Analysis Summary

- `analysis_summary.txt`: Human-readable summary of key findings

## Configuration Options

### Database Selection

```python
# Use original database
evaluator = SchemaTypeEvaluator(use_schema_db=False)

# Use schema-enhanced database
evaluator = SchemaTypeEvaluator(use_schema_db=True)
```

### Custom CSV Path

```python
populator = SchemaTypePopulator(
    csv_path="/path/to/your/schema_types.csv"
)
```

### Evaluation Methods

The evaluator supports all methods from the base Evaluator class:

```python
methods = [
    'Multi-Component HySE (Relational)',
    'Multi-Component HySE (Non-Relational)',
    'Semantic Task Search',
    'Semantic Keyword Search',
    'Syntactic Keyword Search'
]
```

## Troubleshooting

### Common Issues

1. **Database Connection Error**

   - Check environment variables
   - Ensure PostgreSQL is running
   - Verify database exists

2. **CSV File Not Found**

   - Check file path in `populate_schema_types.py`
   - Ensure CSV has required columns: `table_name`, `schema_type`

3. **Schema Type Data Missing**

   - Run population script: `python populate_schema_types.py`
   - Verify CSV data matches database tables

4. **Permission Errors**
   - Ensure database user has CREATE DATABASE privileges
   - Check file system permissions for result directories

### Verification Commands

```bash
# Test database connections
python -c "from backend.app.db.schema_db_connect import SchemaAwareDatabaseConnection; print(SchemaAwareDatabaseConnection().test_connections())"

# Verify schema type population
python populate_schema_types.py --verify-only

# Run sample evaluation
python setup_schema_evaluation.py --sample-eval-only
```

## Data Requirements

### CSV File Format

The CSV file must contain:

- `table_name`: Exact table name as in database
- `schema_type`: Either "normalized" or "denormalized"

Example:

```csv
table_name,schema_type
adobe_stock_data,denormalized
user_profiles,normalized
sales_transactions,denormalized
```

### Database Schema

The evaluation database should contain:

- `eval_data_validation`: Main evaluation table
- `eval_hyse_components`: Hyse component data
- `eval_query_embeds`: Cached query embeddings
- `eval_keyword_embeds`: Cached keyword embeddings
- `eval_metadata_sqlclauses`: Metadata SQL clauses

## Performance Considerations

- **Memory Usage**: Large datasets may require significant memory for embeddings
- **Execution Time**: Full evaluation can take several hours depending on dataset size
- **Disk Space**: New database requires approximately same space as original
- **CPU Usage**: Statistical comparisons are computationally intensive

## Extending the Pipeline

### Adding New Schema Types

1. Update CSV file with new schema type values
2. Modify validation in `SchemaTypePopulator`
3. Update result file headers if needed

### Adding New Evaluation Methods

1. Extend the `methods` list in `SchemaTypeEvaluator.evaluate_by_schema_type()`
2. Add method-specific handling if required
3. Update result file headers if needed

### Custom Statistical Tests

Override `compare_schema_performance()` method:

```python
class CustomSchemaTypeEvaluator(SchemaTypeEvaluator):
    def compare_schema_performance(self, recalls_by_schema):
        # Custom statistical analysis
        pass
```
