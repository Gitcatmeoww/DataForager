"""Recover NQ-Tables page titles for the OpenTI corpus.

trl-lab/openti-nq-tables ships the tables without their Wikipedia page title.
That title is the main identifying signal in NQ-Tables: the table behind "when
was the last time easter was in the month of march" has the header
Year | Western | Eastern and the word Easter appears nowhere in its cells.
DTR serializes title, header, and rows, so dropping the title both cripples
retrieval and puts the released TAPAS checkpoints off-distribution.

ibm-research/NQTablesRetrieval keeps the title. The two releases use unrelated
table ids, but OpenTI's original_question_id is IBM's qid, so tables join
through the questions that point at them:

    openti table -> its instances -> qid -> IBM qrels -> did -> corpus title

Every OpenTI table is gold for at least one question, so this covers the corpus.
Content hashing is used only to break the tie for the few questions IBM labels
with two gold tables, and afterwards as an independent check: the id path and
the content path share nothing, so their agreement is evidence the join is
right.

Usage:
    python -m evaluation.nq.build_title_map --data-dir <dir> --output <path>

The data directory holds the files fetched from both Hub releases:
    instances.parquet, tables_index.parquet, tables/          (OpenTI)
    {train,dev,test}_qrels.jsonl, corpus_structure.jsonl      (IBM)
"""

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Separates cells inside a content key, so that ["ab", "c"] and ["a", "bc"]
# cannot collide.
UNIT_SEPARATOR = "\x1f"

# Trailing 16 hex characters of an IBM did, as in Brazos_River_8F7B4BA175AC5E8F.
_DID_HASH = re.compile(r"_[0-9A-F]{16}$")


def unquote_did(did: str) -> str:
    """Undo the CSV quoting IBM's qrels apply to ids containing double quotes.

    Three of the 8,254 gold dids ship as

        "Whenever_I_Call_You_""Friend""_5DE9FEBACB12C224"

    and match no corpus row until unescaped. This is a defect in the published
    qrels, not a convention, and it affects the .jsonl and .tsv forms alike: 6
    qrel rows across the three splits, one of them in test.

    Requiring both a leading and a trailing quote is what makes this safe. 56
    corpus ids genuinely begin with a quote, as in

        "Heroes"_(David_Bowie_song)_19471739CF2A662

    but no corpus id ends with one, since every id ends in a 16 character hex
    hash. So a trailing quote only ever means CSV wrapping.

    Args:
        did: A did as read from a qrels file.

    Returns:
        The did as spelled in the corpus.
    """
    if len(did) >= 2 and did.startswith('"') and did.endswith('"'):
        return did[1:-1].replace('""', '"')
    return did


def normalize_cell(value) -> str:
    """Render a cell comparably across the two releases.

    OpenTI's cells went through pandas, so numbers are typed and missing values
    are NaN; IBM's stayed strings carrying non-breaking spaces. Both sides are
    reduced to the same casefolded, whitespace-collapsed text.

    Args:
        value: One cell, from either release.

    Returns:
        The normalized text, empty for null.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return repr(value)
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip().lower()


def content_key(cells, n_cols: int) -> str:
    """Hash a table's content, row-major, including its width.

    Args:
        cells: Normalized cells in row-major order.
        n_cols: Column count, so that reshapes of the same cells differ.

    Returns:
        A hex digest.
    """
    payload = f"{n_cols}{UNIT_SEPARATOR}" + UNIT_SEPARATOR.join(cells)
    return hashlib.sha256(payload.encode()).hexdigest()


def title_from_corpus_row(row: dict) -> str:
    """Read a table's page title from an IBM corpus row.

    Args:
        row: One parsed line of corpus_structure.jsonl.

    Returns:
        The page title, recovered from the did when meta_data is empty.
    """
    meta = (row.get("meta_data") or "").strip()
    if meta:
        return meta
    return _DID_HASH.sub("", row["_id"]).replace("_", " ")


def load_qrels(data_dir: Path) -> dict[str, set[str]]:
    """Map every question id to its gold dids, across all three splits.

    Args:
        data_dir: Directory holding {train,dev,test}_qrels.jsonl.

    Returns:
        qid to the set of dids IBM marks relevant. A handful of questions carry
        two.
    """
    qrels: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "dev", "test"):
        with open(data_dir / f"{split}_qrels.jsonl", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                qrels[record["qid"]].add(unquote_did(record["did"]))
    return dict(qrels)


def candidate_dids(instances: pd.DataFrame, qrels: dict[str, set[str]]) -> dict[str, set[str]]:
    """Collect, per OpenTI table, the dids its questions point at.

    Args:
        instances: OpenTI instances.parquet, with a qid column already derived.
        qrels: Output of load_qrels.

    Returns:
        OpenTI table_id to candidate dids.
    """
    candidates: dict[str, set[str]] = defaultdict(set)
    for row in instances.itertuples():
        gold = qrels.get(row.qid, ())
        for table_id in row.target_table_ids:
            candidates[table_id].update(gold)
    return dict(candidates)


def resolve_did(candidates: set[str], key: str, corpus: dict[str, dict]) -> tuple[str | None, str]:
    """Pick the did for one table.

    Args:
        candidates: Dids reachable from the table's questions.
        key: The table's content key.
        corpus: did to {"title", "key"}, from read_corpus.

    Returns:
        The chosen did and how it was chosen, or (None, "none") when the table
        has no candidate at all.
    """
    ordered = sorted(candidates)
    if not ordered:
        return None, "none"
    if len(ordered) == 1:
        return ordered[0], "qid"
    matching = [did for did in ordered if did in corpus and corpus[did]["key"] == key]
    if matching:
        return matching[0], "qid+content"
    return ordered[0], "qid_ambiguous"


def read_corpus(path: Path, wanted: set[str]) -> dict[str, dict]:
    """Read titles and content keys for the requested dids.

    Streams the file, which holds 169,898 rows and is far larger than the subset
    the join needs.

    Args:
        path: corpus_structure.jsonl.
        wanted: Dids to keep.

    Returns:
        did to {"title", "key"}.
    """
    corpus: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["_id"] not in wanted:
                continue
            cells = [normalize_cell(cell) for cell in row["cells"]]
            corpus[row["_id"]] = {
                "title": title_from_corpus_row(row),
                "key": content_key(cells, len(row["headers"]) or 1),
            }
    return corpus


def require_all_resolved(wanted: set[str], found: set[str]) -> None:
    """Fail loudly when a gold did does not appear in the corpus.

    A did that resolves to nothing costs its questions silently: the table drops
    out of the corpus and every query pointing at it becomes an automatic miss,
    which reads as a slightly worse retrieval number rather than as a bug. The
    CSV quoting defect presented exactly this way, so unresolved dids are
    treated as fatal rather than skipped.

    Args:
        wanted: Dids the join needs.
        found: Dids actually read from the corpus.

    Raises:
        ValueError: If any wanted did is missing.
    """
    missing = sorted(wanted - found)
    if not missing:
        return
    shown = "\n".join(f"    {did!r}" for did in missing[:5])
    more = f"\n    ... and {len(missing) - 5} more" if len(missing) > 5 else ""
    raise ValueError(
        f"{len(missing)} gold did(s) missing from corpus_structure.jsonl:\n"
        f"{shown}{more}\n"
        "  If these carry double quotes, the qrels may have introduced a "
        "quoting form unquote_did does not handle."
    )


def table_path(data_dir: Path, table_idx: int) -> Path:
    """Locate one OpenTI table parquet, which are sharded a thousand per directory."""
    return data_dir / "tables" / f"{table_idx // 1000:03d}" / f"{table_idx}.parquet"


def build(data_dir: Path) -> pd.DataFrame:
    """Join the two releases and return one row per OpenTI table.

    Args:
        data_dir: Directory holding both releases' files.

    Returns:
        table_idx, table_id, did, title, resolved_by, and content_agrees, the
        last recording whether the independent content path confirms the did.
    """
    instances = pd.read_parquet(data_dir / "instances.parquet")
    instances["qid"] = instances.original_metadata_json.map(
        lambda blob: json.loads(blob)["original_question_id"]
    )

    qrels = load_qrels(data_dir)
    candidates = candidate_dids(instances, qrels)
    wanted = {did for dids in candidates.values() for did in dids}
    corpus = read_corpus(data_dir / "corpus_structure.jsonl", wanted)
    require_all_resolved(wanted, set(corpus))

    rows = []
    for table in pd.read_parquet(data_dir / "tables_index.parquet").itertuples():
        frame = pd.read_parquet(table_path(data_dir, table.table_idx))
        key = content_key(
            [normalize_cell(value) for value in frame.to_numpy().ravel(order="C")],
            frame.shape[1],
        )
        did, how = resolve_did(candidates.get(table.table_id, set()), key, corpus)
        rows.append(
            {
                "table_idx": table.table_idx,
                "table_id": table.table_id,
                "did": did,
                "title": corpus.get(did, {}).get("title") if did else None,
                "resolved_by": how,
                "content_agrees": bool(did and did in corpus and corpus[did]["key"] == key),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory holding the OpenTI and IBM files.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write the title map parquet.")
    args = parser.parse_args()

    result = build(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)

    total = len(result)
    print(f"openti tables: {total}")
    for how, count in result.resolved_by.value_counts().items():
        print(f"  resolved_by {how:15} {count:6} ({100 * count / total:5.1f}%)")
    print(f"  with a title         {result.title.notna().sum():6} "
          f"({100 * result.title.notna().mean():5.1f}%)")
    print(f"  content cross-check  {result.content_agrees.sum():6} "
          f"({100 * result.content_agrees.mean():5.1f}%) agree")
    print(f"  distinct dids        {result.did.nunique()}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
