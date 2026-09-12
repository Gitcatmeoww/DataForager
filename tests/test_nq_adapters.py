"""Unit tests for the NQ-Tables corpus adapter (no corpus download)."""

import json

import pandas as pd
import pytest

from evaluation.nq.adapters import NQTablesAdapter


def write_corpus(tmp_path, tables, instances, titles):
    """Write a stand-in OpenTI release plus its title map.

    Args:
        tmp_path: Directory to write into.
        tables: table_id to DataFrame, in corpus order.
        instances: (query, qid, split, [target table_ids]) tuples.
        titles: table_id to (did, title).
    """
    (tmp_path / "tables" / "000").mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, (table_id, frame) in enumerate(tables.items()):
        frame.to_parquet(tmp_path / "tables" / "000" / f"{idx}.parquet")
        rows.append({"table_idx": idx, "table_id": table_id})
    pd.DataFrame(rows).to_parquet(tmp_path / "tables_index.parquet")

    pd.DataFrame(
        [
            {
                "query": query,
                "target_table_ids": targets,
                "original_metadata_json": json.dumps(
                    {"split": split, "original_question_id": qid}
                ),
            }
            for query, qid, split, targets in instances
        ]
    ).to_parquet(tmp_path / "instances.parquet")

    title_map = tmp_path / "title_map.parquet"
    pd.DataFrame(
        [{"table_id": t, "did": d, "title": n} for t, (d, n) in titles.items()]
    ).to_parquet(title_map)
    return title_map


@pytest.fixture
def release(tmp_path):
    tables = {
        "t_easter": pd.DataFrame({"Year": [1998, 1999, 2000], "Western": ["Apr 12", "Apr 4", "Apr 23"]}),
        "t_water": pd.DataFrame({"Country": ["Brazil"], "Total": [8233]}),
        "t_heroes": pd.DataFrame({"Track": ["Heroes"]}),
    }
    instances = [
        ("when was easter in march", "q_easter", "test", ["t_easter"]),
        ("who has the most water", "q_water", "train", ["t_water"]),
        # One question, two gold tables, as a handful genuinely have.
        ("heroes bowie song", "q_heroes", "test", ["t_heroes"]),
        ("heroes bowie song", "q_heroes", "test", ["t_water"]),
    ]
    titles = {
        "t_easter": ("List_of_dates_for_Easter_1111111111111111", "List of dates for Easter"),
        "t_water": ("Water_resources_2222222222222222", "Water resources"),
        "t_heroes": ('"Heroes"_(David_Bowie_song)_3333333333333333', '"Heroes" (David Bowie song)'),
    }
    return tmp_path, write_corpus(tmp_path, tables, instances, titles)


def adapter(release, **kwargs):
    data_dir, title_map = release
    return NQTablesAdapter(data_dir, title_map=title_map, **kwargs)


# --- corpus ---------------------------------------------------------------

def test_corpus_is_not_scoped_to_the_split(release):
    """Every split retrieves against all tables; scoping would inflate recall."""
    tables_test, _ = adapter(release).load("test")
    tables_train, _ = adapter(release).load("train")
    assert {t.table_id for t in tables_test} == {"t_easter", "t_water", "t_heroes"}
    assert {t.table_id for t in tables_test} == {t.table_id for t in tables_train}


def test_recall_key_is_the_did_not_a_name(release):
    tables, _ = adapter(release).load("test")
    keys = {t.table_id: t.recall_key for t in tables}
    assert keys["t_easter"] == "List_of_dates_for_Easter_1111111111111111"


def test_titles_are_attached(release):
    tables, _ = adapter(release).load("test")
    titles = {t.table_id: t.title for t in tables}
    assert titles["t_easter"] == "List of dates for Easter"
    assert titles["t_heroes"] == '"Heroes" (David Bowie song)'


def test_group_id_strips_the_content_hash(release):
    """Tables from one page must never be opposed as negatives."""
    tables, _ = adapter(release).load("test")
    groups = {t.table_id: t.group_id for t in tables}
    assert groups["t_easter"] == "List_of_dates_for_Easter"
    assert groups["t_heroes"] == '"Heroes"_(David_Bowie_song)'


# --- table shaping --------------------------------------------------------

def test_rows_are_capped(release):
    tables, _ = adapter(release, max_rows=2).load("test")
    easter = next(t for t in tables if t.table_id == "t_easter")
    assert len(easter.table) == 2


def test_cells_are_strings_for_the_tokenizer(release):
    tables, _ = adapter(release).load("test")
    water = next(t for t in tables if t.table_id == "t_water")
    assert water.table["Total"].tolist() == ["8233"]


def test_missing_values_become_blank(tmp_path):
    title_map = write_corpus(
        tmp_path,
        {"t": pd.DataFrame({"a": [1.0, None]})},
        [("q", "q1", "test", ["t"])],
        {"t": ("Page_1111111111111111", "Page")},
    )
    tables, _ = NQTablesAdapter(tmp_path, title_map=title_map, max_rows=2).load("test")
    assert tables[0].table["a"].tolist() == ["1.0", ""]


def test_blank_headers_are_normalized(tmp_path):
    """61% of NQ tables carry a blank header cell, which the tokenizer rejects.

    Duplicates are not tested through a parquet round trip because pyarrow
    refuses to write them, which is why OpenTI's own headers arrive as _2 / _3.
    normalize_columns covers that case directly.
    """
    frame = pd.DataFrame([["x", "y"]], columns=["", "Born"])
    title_map = write_corpus(
        tmp_path,
        {"t": frame},
        [("q", "q1", "test", ["t"])],
        {"t": ("Page_1111111111111111", "Page")},
    )
    tables, _ = NQTablesAdapter(tmp_path, title_map=title_map).load("test")
    assert tables[0].table.columns.tolist() == ["col_0", "Born"]


# --- examples -------------------------------------------------------------

def test_examples_are_scoped_to_the_split(release):
    _, examples = adapter(release).load("train")
    assert [e.query for e in examples] == ["who has the most water"]


def test_validation_is_an_alias_for_dev(tmp_path):
    title_map = write_corpus(
        tmp_path,
        {"t": pd.DataFrame({"a": ["x"]})},
        [("q", "q1", "dev", ["t"])],
        {"t": ("Page_1111111111111111", "Page")},
    )
    _, examples = NQTablesAdapter(tmp_path, title_map=title_map).load("validation")
    assert len(examples) == 1


def test_multi_gold_question_shares_one_query_id(release):
    _, examples = adapter(release).load("test")
    heroes = [e for e in examples if e.query_id == "q_heroes"]
    assert len(heroes) == 2
    assert {e.table_id for e in heroes} == {"t_heroes", "t_water"}


def test_single_gold_questions_still_carry_their_query_id(release):
    _, examples = adapter(release).load("test")
    easter = [e for e in examples if e.table_id == "t_easter"]
    assert len(easter) == 1
    assert easter[0].query_id == "q_easter"


# --- failure modes --------------------------------------------------------

def test_unknown_split_is_rejected(release):
    with pytest.raises(ValueError, match="matched no questions"):
        adapter(release).load("nonexistent")


def test_table_absent_from_title_map_is_fatal(tmp_path):
    title_map = write_corpus(
        tmp_path,
        {"t_known": pd.DataFrame({"a": ["x"]}), "t_orphan": pd.DataFrame({"a": ["y"]})},
        [("q", "q1", "test", ["t_known"])],
        {"t_known": ("Page_1111111111111111", "Page")},
    )
    with pytest.raises(ValueError, match="absent from"):
        NQTablesAdapter(tmp_path, title_map=title_map).load("test")


def test_question_targeting_an_unknown_table_is_fatal(tmp_path):
    title_map = write_corpus(
        tmp_path,
        {"t": pd.DataFrame({"a": ["x"]})},
        [("q", "q1", "test", ["t_missing"])],
        {"t": ("Page_1111111111111111", "Page")},
    )
    with pytest.raises(ValueError, match="targets unknown table"):
        NQTablesAdapter(tmp_path, title_map=title_map).load("test")


def test_corpus_is_read_once_across_loads(release):
    """Reading 8,205 parquet files per split would be wasteful."""
    instance = adapter(release)
    first, _ = instance.load("test")
    second, _ = instance.load("train")
    assert first is second
