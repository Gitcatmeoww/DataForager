"""Corpus adapters feeding the DTR trainer, index, and evaluator.

Everything downstream of this module works in terms of TableRecord and
RetrievalExample, so adding a second corpus later (a tabular-QA dataset, say)
means writing one new adapter rather than touching training or evaluation code.

Three identifiers hang off every table, and they are deliberately distinct:

- table_id is unique within a split and is what the index and the training
  pairs join on.
- recall_key is what recall is scored against.
- group_id marks tables that may legitimately answer the same query, and exists
  to suppress false negatives during training and hard-negative mining.
"""

import csv
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from evaluation.dtr.serialization import parse_markdown_table

# Corpus rows carry long markdown blobs and embedding vectors, well past the
# default field limit.
csv.field_size_limit(sys.maxsize)

# The train split stores nine example rows but test stores two. Pin both to two
# so that training and indexing serialize tables identically.
DEFAULT_MAX_ROWS = 2

# Markdown columns in preference order; splits disagree on which they carry.
_MARKDOWN_COLUMNS = ("example_2rows_md", "example_rows_md", "example_3rows_md")


@dataclass(frozen=True)
class TableRecord:
    """One retrievable table."""

    table_id: str
    recall_key: str
    title: str
    table: pd.DataFrame
    group_id: str


@dataclass(frozen=True)
class RetrievalExample:
    """One (query, gold table) training or evaluation pair."""

    query: str
    table_id: str


class CorpusAdapter(Protocol):
    """What the trainer and evaluator require of any corpus."""

    name: str

    def load(self, split: str) -> tuple[list[TableRecord], list[RetrievalExample]]:
        """Return the split's tables and its query/table pairs."""


def parse_pg_array(value) -> list[str]:
    """Parse a Postgres text[] literal such as {"a","b"} into a list.

    The corpus stores task_queries, keywords, and tags this way. Values arrive
    as strings from CSV but already as lists when read through the datasets
    library, so both are accepted.

    Args:
        value: A Postgres array literal, or an already-parsed sequence.

    Returns:
        The elements, empty for a null or empty {} array.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]

    text = str(value).strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    if not text.strip():
        return []

    # Elements are comma-separated and double-quoted, with no backslash or
    # doubled-quote escaping anywhere in this corpus, so csv reads them cleanly.
    return next(csv.reader(io.StringIO(text), quotechar='"', skipinitialspace=True))


class KaggleDSAdapter:
    """The KaggleDS corpus that backs the DataForager paper.

    Reads the published CSV splits when given a directory, and otherwise pulls
    them from the HuggingFace Hub through evaluation/data/load_kaggleds.py.
    """

    name = "kaggleds"

    def __init__(self, csv_dir: Path | None = None, max_rows: int = DEFAULT_MAX_ROWS):
        """
        Args:
            csv_dir: Directory of eval_data_{split}.csv files. When omitted the
                splits are fetched from the Hub instead.
            max_rows: Example rows to keep per table.
        """
        self.csv_dir = Path(csv_dir) if csv_dir else None
        self.max_rows = max_rows

    def _rows(self, split: str):
        if self.csv_dir is not None:
            path = self.csv_dir / f"eval_data_{split}.csv"
            with open(path, newline="", encoding="utf-8") as handle:
                yield from csv.DictReader(handle)
            return

        from evaluation.data.load_kaggleds import load_kaggleds

        yield from load_kaggleds(split)

    @staticmethod
    def _markdown(row) -> str:
        for column in _MARKDOWN_COLUMNS:
            if row.get(column):
                return row[column]
        return ""

    def load(self, split: str) -> tuple[list[TableRecord], list[RetrievalExample]]:
        """Load one split.

        Args:
            split: One of train, validation, or test.

        Returns:
            The split's TableRecords and its RetrievalExamples. Tables with no
            task queries still appear in the corpus, since they remain
            retrievable distractors, but contribute no pairs.
        """
        tables: list[TableRecord] = []
        examples: list[RetrievalExample] = []
        occurrences: dict[str, int] = {}

        for row in self._rows(split):
            table_name = (row["table_name"] or "").strip()
            database_name = (row["database_name"] or "").strip()

            table_id = f"{database_name}::{table_name}"
            count = occurrences.get(table_id, 0)
            occurrences[table_id] = count + 1
            if count:
                table_id = f"{table_id}#{count}"

            tables.append(
                TableRecord(
                    table_id=table_id,
                    recall_key=table_name,
                    # The paper's title(T) is the document title. Pairing the
                    # dataset name with the file name matches the table name
                    # plus header plus rows content the other methods index.
                    title=f"{database_name} {table_name}".strip(),
                    table=parse_markdown_table(self._markdown(row), max_rows=self.max_rows),
                    group_id=database_name,
                )
            )

            for query in parse_pg_array(row.get("task_queries")):
                if query.strip():
                    examples.append(RetrievalExample(query=query.strip(), table_id=table_id))

        return tables, examples