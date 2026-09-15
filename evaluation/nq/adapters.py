"""NQ-Tables corpus adapter.

Implements the CorpusAdapter protocol from evaluation/dtr/adapters.py, so the
DTR index, the evaluator, and the HySE fusion read NQ-Tables the same way they
read KaggleDS. The shared record types are imported from there rather than
moved: two corpora do not yet justify a separate module, and moving them would
touch training code that produced published numbers.

Two differences from KaggleDSAdapter are deliberate and matter when reading
results:

- The corpus is not split-scoped. KaggleDS gives each split its own tables;
  NQ-Tables has one 8,205-table corpus that every split retrieves against, and
  only the questions are split-scoped. load() therefore returns the whole corpus
  for any split. Scoping it to a split's gold tables would shrink the haystack
  to 896 tables for test and inflate recall.
- Recall is scored on the IBM did, a content hash, not on a name. NQ has no
  analogue of KaggleDS's table_name collisions.
"""

import json
from pathlib import Path

import pandas as pd

from evaluation.dtr.adapters import DEFAULT_MAX_ROWS, RetrievalExample, TableRecord
from evaluation.dtr.serialization import normalize_columns
from evaluation.nq.build_title_map import title_from_corpus_row, unquote_did

# Built by evaluation/nq/build_title_map.py and committed, so the corpus loads
# without re-downloading IBM's 860 MB corpus_structure.jsonl.
DEFAULT_TITLE_MAP = Path(__file__).parent / "data" / "openti_title_map.parquet"

# KaggleDS calls it validation, NQ-Tables calls it dev.
_SPLIT_ALIASES = {"validation": "dev", "valid": "dev"}

# A did is PageTitle_HASH16; the hash identifies the content, the prefix the
# Wikipedia page.
_DID_HASH_LENGTH = 17


class NQTablesAdapter:
    """The NQ-Tables corpus, assembled from the OpenTI and IBM releases.

    Tables and questions come from trl-lab/openti-nq-tables; page titles come
    from the join in evaluation/nq/build_title_map.py, because OpenTI drops them
    and they carry the entity NQ questions ask about.
    """

    name = "nq_tables"

    def __init__(self, data_dir, title_map: Path | None = None,
                 max_rows: int = DEFAULT_MAX_ROWS):
        """
        Args:
            data_dir: Directory holding the OpenTI release: instances.parquet,
                tables_index.parquet, and the unpacked tables/ tree.
            title_map: The parquet written by build_title_map. Defaults to the
                committed one.
            max_rows: Example rows to keep per table, pinned to match how the
                KaggleDS corpus is serialized.
        """
        self.data_dir = Path(data_dir)
        self.title_map = Path(title_map) if title_map else DEFAULT_TITLE_MAP
        self.max_rows = max_rows
        self._corpus: list[TableRecord] | None = None

    @staticmethod
    def _normalize_split(split: str) -> str:
        return _SPLIT_ALIASES.get(split, split)

    @staticmethod
    def _page(did: str) -> str:
        """Strip a did's content hash, leaving the Wikipedia page it came from."""
        return did[:-_DID_HASH_LENGTH] if len(did) > _DID_HASH_LENGTH else did

    def _read_table(self, table_idx: int) -> pd.DataFrame:
        """Read one table, shaped the way the TAPAS tokenizer needs it.

        Cells are cast to string and missing values blanked, since OpenTI wrote
        the tables through pandas and numeric columns come back typed with NaN
        for the gaps.
        """
        path = self.data_dir / "tables" / f"{table_idx // 1000:03d}" / f"{table_idx}.parquet"
        frame = pd.read_parquet(path).head(self.max_rows)
        frame = frame.fillna("").astype(str)
        frame.columns = normalize_columns(frame.columns)
        return frame

    def _load_corpus(self) -> list[TableRecord]:
        """Read all 8,205 tables once and cache them on the adapter."""
        if self._corpus is not None:
            return self._corpus

        titles = pd.read_parquet(self.title_map).set_index("table_id")
        index = pd.read_parquet(self.data_dir / "tables_index.parquet")

        missing = set(index.table_id) - set(titles.index)
        if missing:
            raise ValueError(
                f"{len(missing)} table(s) absent from {self.title_map.name}, "
                f"for example {sorted(missing)[:3]}. Rebuild it with "
                "python -m evaluation.nq.build_title_map."
            )

        records = []
        for table in index.itertuples():
            entry = titles.loc[table.table_id]
            did = entry.did
            records.append(
                TableRecord(
                    table_id=table.table_id,
                    # A did is a content hash, so unlike KaggleDS's table_name
                    # it cannot collide across unrelated tables.
                    recall_key=did,
                    title=entry.title,
                    table=self._read_table(table.table_idx),
                    # Tables from one Wikipedia page can answer the same
                    # question, so they must never be opposed as negatives.
                    group_id=self._page(did),
                )
            )
        self._corpus = records
        return records

    def load(self, split: str) -> tuple[list[TableRecord], list[RetrievalExample]]:
        """Load the corpus and one split's questions.

        Args:
            split: train, dev (or validation), or test.

        Returns:
            All 8,205 TableRecords, and the split's RetrievalExamples. Questions
            with two gold tables appear as two examples sharing a query_id, so
            the evaluator can score them once.
        """
        wanted = self._normalize_split(split)
        tables = self._load_corpus()
        known = {t.table_id for t in tables}

        instances = pd.read_parquet(self.data_dir / "instances.parquet")
        examples = []
        for row in instances.itertuples():
            meta = json.loads(row.original_metadata_json)
            if meta["split"] != wanted:
                continue
            query = (row.query or "").strip()
            if not query:
                continue
            for table_id in row.target_table_ids:
                if table_id not in known:
                    raise ValueError(
                        f"question {meta['original_question_id']!r} targets unknown "
                        f"table {table_id!r}"
                    )
                examples.append(
                    RetrievalExample(
                        query=query,
                        table_id=table_id,
                        query_id=meta["original_question_id"],
                    )
                )

        if not examples:
            raise ValueError(
                f"split {split!r} matched no questions; expected one of "
                "train, dev/validation, test"
            )
        return tables, examples


class NQTablesFullAdapter:
    """The full 169,898-table NQ-Tables corpus, read from the IBM release.

    The gold-only corpus that NQTablesAdapter serves turned out to be too easy
    to measure on: a frozen encoder reaches .981 R@10 there, because NQ
    questions name an entity and the gold table's page title names the same
    entity, and 95% of the distractors that would compete on title are absent.
    This adapter restores them.

    It needs no OpenTI data at all. Tables come from corpus_structure.jsonl,
    questions from {split}_queries.jsonl, and labels from {split}_qrels.jsonl,
    so the did is both the table id and the recall key.

    Iterating is the supported way to read the corpus. Materializing 169,898
    DataFrames costs gigabytes, so iter_records streams them and callers that
    only need text should convert as they go.
    """

    name = "nq_tables_full"

    def __init__(self, data_dir, max_rows: int = DEFAULT_MAX_ROWS):
        """
        Args:
            data_dir: Directory holding the IBM release files.
            max_rows: Example rows to keep per table.
        """
        self.data_dir = Path(data_dir)
        self.max_rows = max_rows

    def _frame(self, headers, cells) -> pd.DataFrame:
        """Rebuild a table from IBM's headers and flat row-major cells."""
        width = len(headers) or 1
        rows = [
            [str(c) for c in cells[start : start + width]]
            for start in range(0, len(cells), width)
        ][: self.max_rows]
        columns = normalize_columns(headers) if headers else ["col_0"]
        # A short final row would make the frame ragged.
        rows = [(row + [""] * width)[:width] for row in rows]
        return pd.DataFrame(rows, columns=columns, dtype=str)

    def iter_records(self):
        """Stream every corpus table as a TableRecord, in file order."""
        path = self.data_dir / "corpus_structure.jsonl"
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                did = row["_id"]
                yield TableRecord(
                    table_id=did,
                    recall_key=did,
                    title=title_from_corpus_row(row),
                    table=self._frame(row.get("headers") or [], row.get("cells") or []),
                    group_id=NQTablesAdapter._page(did),
                )

    def examples(self, split: str) -> list[RetrievalExample]:
        """Read one split's questions and their gold tables.

        Args:
            split: train, dev (or validation), or test.

        Returns:
            One RetrievalExample per (question, gold table) pair, with
            query_id set so the evaluator scores each question once.
        """
        wanted = _SPLIT_ALIASES.get(split, split)
        queries = {}
        with open(self.data_dir / f"{wanted}_queries.jsonl", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                queries[row["_id"]] = row["text"]

        examples = []
        with open(self.data_dir / f"{wanted}_qrels.jsonl", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                text = queries.get(row["qid"])
                if text is None:
                    raise ValueError(f"qrel references unknown question {row['qid']!r}")
                examples.append(
                    RetrievalExample(
                        query=text,
                        table_id=unquote_did(row["did"]),
                        query_id=row["qid"],
                    )
                )
        if not examples:
            raise ValueError(f"split {split!r} matched no questions")
        return examples

    def load(self, split: str) -> tuple[list[TableRecord], list[RetrievalExample]]:
        """Materialize the whole corpus and one split's questions.

        Holding 169,898 DataFrames is expensive; prefer iter_records plus
        examples when only the text is needed.
        """
        return list(self.iter_records()), self.examples(split)
