"""Unit tests for the NQ-Tables title join (no corpus download)."""

import json

import pandas as pd
import pytest

from evaluation.nq.build_title_map import (
    build,
    candidate_dids,
    content_key,
    load_qrels,
    normalize_cell,
    read_corpus,
    require_all_resolved,
    resolve_did,
    table_path,
    title_from_corpus_row,
    unquote_did,
)


# --- unquote_did ----------------------------------------------------------

def test_unquotes_csv_quoted_did():
    # Three real dids ship this way and match no corpus row until unescaped.
    assert (
        unquote_did('"Whenever_I_Call_You_""Friend""_5DE9FEBACB12C224"')
        == 'Whenever_I_Call_You_"Friend"_5DE9FEBACB12C224'
    )


def test_leaves_ordinary_dids_alone():
    assert unquote_did("Brazos_River_8F7B4BA175AC5E8F") == "Brazos_River_8F7B4BA175AC5E8F"


def test_does_not_strip_unpaired_quote():
    assert unquote_did('"leading_only') == '"leading_only'


@pytest.mark.parametrize(
    "did",
    [
        # 56 corpus ids genuinely begin with a quote. None end with one, since
        # every id ends in a 16 character hex hash, which is what makes
        # requiring both ends safe.
        '"Heroes"_(David_Bowie_song)_19471739CF2A662',
        '"Heroes"_(David_Bowie_song)_1C25194A9005312C',
        '"Weird_Al"_Yankovic_1CC9B516ED3183FA',
    ],
)
def test_leaves_ids_that_merely_start_with_a_quote_intact(did):
    assert unquote_did(did) == did


def test_unquotes_a_wrapped_id_that_also_starts_with_a_quote():
    # The two defects composed: a quote-initial title, CSV-wrapped.
    wrapped = '"""Heroes""_(David_Bowie_song)_19471739CF2A662"'
    assert unquote_did(wrapped) == '"Heroes"_(David_Bowie_song)_19471739CF2A662'


def test_handles_degenerate_short_values():
    assert unquote_did('"') == '"'
    assert unquote_did("") == ""


# --- normalize_cell -------------------------------------------------------

def test_normalizes_nbsp_and_case():
    # IBM cells carry non-breaking spaces; NFKC folds them to plain spaces.
    assert normalize_cell("5\xa0ft 2\xa0in") == "5 ft 2 in"
    assert normalize_cell("  Brazil  ") == "brazil"


def test_renders_integral_floats_without_decimal_point():
    # pandas types a column of years as float when it holds a missing value.
    assert normalize_cell(2011.0) == "2011"
    assert normalize_cell(2011) == "2011"


def test_treats_missing_as_empty():
    assert normalize_cell(float("nan")) == ""
    assert normalize_cell(None) == ""


def test_keeps_fractional_floats_distinct():
    assert normalize_cell(1.57) != normalize_cell(1.58)


# --- content_key ----------------------------------------------------------

def test_key_is_stable_and_content_sensitive():
    assert content_key(["a", "b"], 2) == content_key(["a", "b"], 2)
    assert content_key(["a", "b"], 2) != content_key(["a", "c"], 2)


def test_key_separates_differently_shaped_tables():
    # Same cells, different width, different table.
    assert content_key(["a", "b"], 2) != content_key(["a", "b"], 1)


def test_key_resists_cell_boundary_collisions():
    assert content_key(["ab", "c"], 2) != content_key(["a", "bc"], 2)


# --- title_from_corpus_row ------------------------------------------------

def test_prefers_meta_data_title():
    row = {"_id": "Lesley_Joseph_A1D55A57012E3362", "meta_data": "Lesley Joseph\n"}
    assert title_from_corpus_row(row) == "Lesley Joseph"


def test_falls_back_to_did_when_meta_data_blank():
    row = {"_id": "Brazos_River_8F7B4BA175AC5E8F", "meta_data": ""}
    assert title_from_corpus_row(row) == "Brazos River"


def test_fallback_keeps_underscores_that_are_not_the_hash():
    row = {"_id": "List_of_dates_for_Easter_8CCF64605CCDEB97", "meta_data": None}
    assert title_from_corpus_row(row) == "List of dates for Easter"


# --- resolve_did ----------------------------------------------------------

CORPUS = {
    "A_1111111111111111": {"title": "A", "key": content_key(["x"], 1)},
    "B_2222222222222222": {"title": "B", "key": content_key(["y"], 1)},
}


def test_single_candidate_resolves_by_id_alone():
    did, how = resolve_did({"A_1111111111111111"}, content_key(["x"], 1), CORPUS)
    assert (did, how) == ("A_1111111111111111", "qid")


def test_two_candidates_are_broken_by_content():
    # A question with two gold tables; only one matches the table at hand.
    did, how = resolve_did(
        {"A_1111111111111111", "B_2222222222222222"}, content_key(["y"], 1), CORPUS
    )
    assert (did, how) == ("B_2222222222222222", "qid+content")


def test_unbreakable_tie_is_flagged_not_hidden():
    did, how = resolve_did(
        {"A_1111111111111111", "B_2222222222222222"}, content_key(["zzz"], 1), CORPUS
    )
    assert how == "qid_ambiguous"
    assert did == "A_1111111111111111"


def test_no_candidate_yields_none():
    assert resolve_did(set(), content_key(["x"], 1), CORPUS) == (None, "none")


# --- require_all_resolved -------------------------------------------------

def test_passes_when_every_did_resolves():
    require_all_resolved({"A_1111111111111111"}, {"A_1111111111111111", "B_2222222222222222"})


def test_raises_rather_than_silently_dropping_a_gold_table():
    # An unresolved did would otherwise read as a worse recall number, not a bug.
    with pytest.raises(ValueError, match="1 gold did"):
        require_all_resolved({"A_1111111111111111"}, set())


def test_error_names_the_missing_dids_and_truncates():
    missing = {f"D{i}_1111111111111111" for i in range(8)}
    with pytest.raises(ValueError) as excinfo:
        require_all_resolved(missing, set())
    message = str(excinfo.value)
    assert "8 gold did" in message
    assert "and 3 more" in message


# --- table_path -----------------------------------------------------------

@pytest.mark.parametrize(
    "idx, expected",
    [(0, "000/0.parquet"), (999, "000/999.parquet"),
     (1000, "001/1000.parquet"), (8204, "008/8204.parquet")],
)
def test_shards_tables_a_thousand_per_directory(tmp_path, idx, expected):
    assert table_path(tmp_path, idx) == tmp_path / "tables" / expected


# --- file readers ---------------------------------------------------------

def test_load_qrels_unions_splits_and_unquotes(tmp_path):
    rows = {
        "train": [{"qid": "q1", "did": "A_1111111111111111", "score": 1}],
        "dev": [{"qid": "q2", "did": '"B_""x""_2222222222222222"', "score": 1}],
        # One question with two gold tables, as a handful genuinely have.
        "test": [{"qid": "q3", "did": "C_3333333333333333", "score": 1},
                 {"qid": "q3", "did": "D_4444444444444444", "score": 1}],
    }
    for split, records in rows.items():
        (tmp_path / f"{split}_qrels.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8"
        )

    qrels = load_qrels(tmp_path)
    assert qrels["q1"] == {"A_1111111111111111"}
    assert qrels["q2"] == {'B_"x"_2222222222222222'}
    assert qrels["q3"] == {"C_3333333333333333", "D_4444444444444444"}


def test_read_corpus_streams_only_wanted_rows(tmp_path):
    path = tmp_path / "corpus_structure.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"_id": "A_1111111111111111", "meta_data": "A\n",
                 "headers": ["h"], "cells": ["X"]},
                {"_id": "Z_9999999999999999", "meta_data": "Z\n",
                 "headers": ["h"], "cells": ["ignored"]},
            ]
        ),
        encoding="utf-8",
    )

    corpus = read_corpus(path, {"A_1111111111111111"})
    assert set(corpus) == {"A_1111111111111111"}
    assert corpus["A_1111111111111111"]["title"] == "A"
    # Keys are computed from normalized cells, so casing does not matter.
    assert corpus["A_1111111111111111"]["key"] == content_key(["x"], 1)


def test_candidate_dids_unions_every_question_targeting_a_table():
    instances = pd.DataFrame(
        {"qid": ["q1", "q2"], "target_table_ids": [["t1"], ["t1"]]}
    )
    qrels = {"q1": {"A_1111111111111111"}, "q2": {"A_1111111111111111", "B_2222222222222222"}}
    assert candidate_dids(instances, qrels) == {
        "t1": {"A_1111111111111111", "B_2222222222222222"}
    }


# --- build ----------------------------------------------------------------

def write_release(tmp_path):
    """Write a two-table stand-in for both releases."""
    (tmp_path / "tables" / "000").mkdir(parents=True)
    pd.DataFrame({"Year": [1998], "Western": ["April 12"]}).to_parquet(
        tmp_path / "tables" / "000" / "0.parquet"
    )
    pd.DataFrame({"Country": ["Brazil"]}).to_parquet(
        tmp_path / "tables" / "000" / "1.parquet"
    )

    pd.DataFrame(
        {"table_idx": [0, 1], "table_id": ["t_easter", "t_water"]}
    ).to_parquet(tmp_path / "tables_index.parquet")

    pd.DataFrame(
        {
            "target_table_ids": [["t_easter"], ["t_water"]],
            "original_metadata_json": [
                json.dumps({"original_question_id": "q_easter"}),
                json.dumps({"original_question_id": "q_water"}),
            ],
        }
    ).to_parquet(tmp_path / "instances.parquet")

    qrels = {
        "train": [{"qid": "q_easter", "did": "List_of_dates_for_Easter_1111111111111111"}],
        "dev": [],
        "test": [{"qid": "q_water", "did": "Water_resources_2222222222222222"}],
    }
    for split, records in qrels.items():
        (tmp_path / f"{split}_qrels.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8"
        )

    (tmp_path / "corpus_structure.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"_id": "List_of_dates_for_Easter_1111111111111111",
                 "meta_data": "List of dates for Easter\n",
                 "headers": ["Year", "Western"], "cells": ["1998", "April 12"]},
                {"_id": "Water_resources_2222222222222222",
                 "meta_data": "Water resources\n",
                 "headers": ["Country"], "cells": ["Brazil"]},
            ]
        ),
        encoding="utf-8",
    )


def test_build_recovers_titles_and_confirms_them_by_content(tmp_path):
    write_release(tmp_path)
    result = build(tmp_path).set_index("table_id")

    assert result.loc["t_easter", "title"] == "List of dates for Easter"
    assert result.loc["t_water", "title"] == "Water resources"
    assert result.resolved_by.tolist() == ["qid", "qid"]
    # The content path is independent of the id path; agreement is the check.
    assert result.content_agrees.all()


def write_csv_quoted_release(tmp_path):
    """A release whose single gold did ships CSV-quoted, as three real ones do."""
    (tmp_path / "tables" / "000").mkdir(parents=True)
    pd.DataFrame({"Rank": ["1"], "Film": ["Swearnet"]}).to_parquet(
        tmp_path / "tables" / "000" / "0.parquet"
    )
    pd.DataFrame({"table_idx": [0], "table_id": ["t_films"]}).to_parquet(
        tmp_path / "tables_index.parquet"
    )
    pd.DataFrame(
        {
            "target_table_ids": [["t_films"]],
            "original_metadata_json": [json.dumps({"original_question_id": "q_films"})],
        }
    ).to_parquet(tmp_path / "instances.parquet")

    wrapped = '"List_of_films_that_most_frequently_use_the_word_""fuck""_8CCF64605CCDEB97"'
    for split, records in {
        "train": [],
        "dev": [],
        "test": [{"qid": "q_films", "did": wrapped}],
    }.items():
        (tmp_path / f"{split}_qrels.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8"
        )

    # The corpus spells it unwrapped, which is the whole problem.
    (tmp_path / "corpus_structure.jsonl").write_text(
        json.dumps(
            {
                "_id": 'List_of_films_that_most_frequently_use_the_word_"fuck"_8CCF64605CCDEB97',
                "meta_data": 'List of films that most frequently use the word "fuck"\n',
                "headers": ["Rank", "Film"],
                "cells": ["1", "Swearnet"],
            }
        ),
        encoding="utf-8",
    )


def test_build_resolves_a_csv_quoted_did(tmp_path):
    write_csv_quoted_release(tmp_path)
    result = build(tmp_path).set_index("table_id")

    assert result.loc["t_films", "title"] == (
        'List of films that most frequently use the word "fuck"'
    )
    assert result.loc["t_films", "content_agrees"]


def test_build_raises_when_a_did_cannot_be_unquoted(tmp_path, monkeypatch):
    """Negative control: without the unquoting, the table is lost — loudly."""
    write_csv_quoted_release(tmp_path)
    monkeypatch.setattr("evaluation.nq.build_title_map.unquote_did", lambda did: did)

    with pytest.raises(ValueError, match="missing from corpus_structure"):
        build(tmp_path)


def test_build_flags_a_table_whose_content_contradicts_the_id_join(tmp_path):
    write_release(tmp_path)
    # Repoint the Easter table's cells so the two paths disagree.
    pd.DataFrame({"Year": [2024], "Western": ["March 31"]}).to_parquet(
        tmp_path / "tables" / "000" / "0.parquet"
    )
    result = build(tmp_path).set_index("table_id")

    assert result.loc["t_easter", "did"] == "List_of_dates_for_Easter_1111111111111111"
    assert not result.loc["t_easter", "content_agrees"]
    assert result.loc["t_water", "content_agrees"]
