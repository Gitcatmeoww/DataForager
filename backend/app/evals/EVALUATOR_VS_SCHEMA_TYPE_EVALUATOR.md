# Evaluator vs SchemaTypeEvaluator for Test Set

## Overview

Both evaluators work with `eval_data_test`, but they serve different purposes:

### Evaluator (Base Class)
- **Purpose**: General evaluation across all data
- **Output**: Overall performance metrics
- **Use when**: You want to know overall Recall@K for different methods

### SchemaTypeEvaluator (Extended Class)
- **Purpose**: Schema-type-aware evaluation (normalized vs denormalized)
- **Output**: Performance breakdown by schema type + statistical comparison
- **Use when**: You want to analyze how HySE performs on different table structures

## Quick Comparison

| Feature | Evaluator | SchemaTypeEvaluator |
|---------|-----------|---------------------|
| Database | Any eval table | Must have `schema_type` column |
| Output | Overall metrics | Schema-specific metrics |
| Statistical tests | No | Yes (Mann-Whitney U test) |
| Results files | 3 files | 6 files (3 general + 3 schema-specific) |
| Use case | General performance | Schema type analysis |

## Running on Test Set

### Option 1: Base Evaluator

```bash
cd "/Users/gitcat/Documents/Academic/UC Berkeley/Research/Semantic Dataset Search/Prototype"
python backend/app/evals/run_evaluator_on_test_set.py
```

**Output files** (in `eval/results/`):
- `per_row_results.csv` - Recall for each query
- `failed_rows.csv` - Tables that failed evaluation
- `failed_queries.csv` - Queries that failed

**Metrics**:
- Average Recall@K per method
- Average retrieval time per method

### Option 2: SchemaTypeEvaluator

```bash
cd "/Users/gitcat/Documents/Academic/UC Berkeley/Research/Semantic Dataset Search/Prototype"
python backend/app/evals/schema_type_eval/schema_type_evaluator.py
```

**Output files** (in `eval/results/schema_type_results/`):
- `normalized_results.csv` - Results for normalized tables
- `denormalized_results.csv` - Results for denormalized tables
- `schema_comparison.csv` - Statistical comparison
- `analysis_summary.txt` - Human-readable summary
- Plus the 3 general result files from base Evaluator

**Metrics**:
- Average Recall@K per method per schema type
- Recall difference (normalized - denormalized)
- P-value for statistical significance
- Effect size

## Configuration for Test Set

### Both evaluators need:
```python
data_split="eval_data_test"
embed_col="example_2rows_table_name_embed"
k=10  # or 50, depending on your needs
num_embed=2
filter_should_include=None  # Test set doesn't have this column
```

### Additional for SchemaTypeEvaluator:
```python
use_schema_db=True  # Use HITS-eval-data-corpus-exp-opt-schema
```

## Example Code

### Using Base Evaluator:
```python
from backend.app.evals.evaluator import Evaluator

evaluator = Evaluator(
    data_split="eval_data_test",
    embed_col="example_2rows_table_name_embed",
    k=10,
    num_embed=2,
    filter_should_include=None
)

results = evaluator.evaluate()
```

### Using SchemaTypeEvaluator:
```python
from backend.app.evals.schema_type_eval.schema_type_evaluator import SchemaTypeEvaluator

evaluator = SchemaTypeEvaluator(
    data_split="eval_data_test",
    embed_col="example_2rows_table_name_embed",
    k=10,
    num_embed=2,
    use_schema_db=True,
    filter_should_include=None
)

results = evaluator.run_schema_type_analysis()
```

## Recommendation

For your test set evaluation, I recommend:

1. **Run SchemaTypeEvaluator** - Since you have schema_type annotations, you can get deeper insights into how HySE performs on normalized vs denormalized tables.

2. **Use the full dataset** - Don't set `limit` parameter to get complete test set results.

3. **Check both recall and retrieval time** - Important for understanding both accuracy and efficiency.

## Key Differences in Results

### Base Evaluator Output:
```
Multi-Component HySE (Relational): 0.7234
Multi-Component HySE (Non-Relational): 0.6891
Semantic Task Search: 0.5123
```

### SchemaTypeEvaluator Output:
```
Multi-Component HySE (Relational):
  Normalized: 0.7856 (n=345)
  Denormalized: 0.7102 (n=1992)
  Difference: +0.0754 (p=0.0234, Significant)

Multi-Component HySE (Non-Relational):
  Normalized: 0.7234 (n=345)
  Denormalized: 0.6789 (n=1992)
  Difference: +0.0445 (p=0.0456, Significant)
```

The SchemaTypeEvaluator gives you much more insight into **where** your method performs better!
