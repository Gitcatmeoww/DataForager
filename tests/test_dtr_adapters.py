"""Unit tests for the DTR corpus adapter (no model / network)."""

import csv

import pytest

from evaluation.dtr.adapters import KaggleDSAdapter, parse_pg_array

MARKDOWN = "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"


def write_split(directory, rows, split="test"):
    """Write a minimal split CSV the adapter can read."""
    path = directory / f"eval_data_{split}.csv"
    columns = ["table_name", "database_name", "example_2rows_md", "task_queries"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def row(table_name, database_name, queries=("q1", "q2")):
    return {
        "table_name": table_name,
        "database_name": database_name,
        "example_2rows_md": MARKDOWN,
        "task_queries": "{" + ",".join(f'"{q}"' for q in queries) + "}",
    }


def test_parses_quoted_array():
    assert parse_pg_array('{"one","two, with comma"}') == ["one", "two, with comma"]


def test_parses_empty_and_null_arrays():
    # One train row genuinely carries an empty {} array.
    assert parse_pg_array("{}") == []
    assert parse_pg_array(None) == []


def test_passes_through_already_parsed_lists():
    # The datasets library hands back real lists rather than literals.
    assert parse_pg_array(["a", "b"]) == ["a", "b"]


def test_loads_tables_and_examples(tmp_path):
    write_split(tmp_path, [row("t1.csv", "db one"), row("t2.csv", "db two")])

    tables, examples = KaggleDSAdapter(csv_dir=tmp_path).load("test")

    assert len(tables) == 2
    assert len(examples) == 4  # two queries per table
    assert tables[0].title == "db one t1.csv"
    assert tables[0].group_id == "db one"
    assert tables[0].table.shape == (2, 2)


def test_recall_key_is_the_bare_table_name(tmp_path):
    # Recall is scored on table_name to match the existing baselines, even
    # though names collide across unrelated datasets.
    write_split(tmp_path, [row("train.csv", "db one"), row("train.csv", "db two")])

    tables, _ = KaggleDSAdapter(csv_dir=tmp_path).load("test")

    assert [t.recall_key for t in tables] == ["train.csv", "train.csv"]
    assert tables[0].group_id != tables[1].group_id


def test_table_ids_stay_unique_for_repeated_rows(tmp_path):
    # A few corpus tables are repeated verbatim within one database, each copy
    # carrying different task queries.
    write_split(tmp_path, [row("dup.csv", "db", ("q1",)), row("dup.csv", "db", ("q2",))])

    tables, examples = KaggleDSAdapter(csv_dir=tmp_path).load("test")

    assert len({t.table_id for t in tables}) == 2
    assert {e.query for e in examples} == {"q1", "q2"}


def test_tables_without_queries_are_still_indexed(tmp_path):
    write_split(tmp_path, [row("empty.csv", "db", ())])

    tables, examples = KaggleDSAdapter(csv_dir=tmp_path).load("test")

    assert len(tables) == 1  # still a retrievable distractor
    assert examples == []


def test_max_rows_is_applied(tmp_path):
    write_split(tmp_path, [row("t.csv", "db")])

    tables, _ = KaggleDSAdapter(csv_dir=tmp_path, max_rows=1).load("test")

    assert tables[0].table.shape[0] == 1


def test_every_example_points_at_a_known_table(tmp_path):
    write_split(tmp_path, [row("a.csv", "db one"), row("b.csv", "db two")])

    tables, examples = KaggleDSAdapter(csv_dir=tmp_path).load("test")

    known = {t.table_id for t in tables}
    assert all(e.table_id in known for e in examples)


def test_missing_split_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        KaggleDSAdapter(csv_dir=tmp_path).load("nonexistent")
