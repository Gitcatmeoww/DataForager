"""Unit tests for HySE result aggregation (pure logic, no DB / network)."""

import pytest

from dataforager.hyse.hypo_schema_search import aggregate_hyse_search_results


def test_averages_cosine_similarity_per_table():
    # "a" appears in two sub-results and should be averaged; "b" appears once.
    results = [
        [
            {"table_name": "a", "cosine_similarity": 0.8},
            {"table_name": "b", "cosine_similarity": 0.4},
        ],
        [
            {"table_name": "a", "cosine_similarity": 0.6},
        ],
    ]

    by_name = {r["table_name"]: r["cosine_similarity"] for r in aggregate_hyse_search_results(results)}

    assert by_name["a"] == pytest.approx(0.7)  # mean of 0.8 and 0.6
    assert by_name["b"] == pytest.approx(0.4)


def test_sorted_descending_by_similarity():
    results = [[
        {"table_name": "low", "cosine_similarity": 0.1},
        {"table_name": "high", "cosine_similarity": 0.9},
        {"table_name": "mid", "cosine_similarity": 0.5},
    ]]

    ordered = [r["table_name"] for r in aggregate_hyse_search_results(results)]

    assert ordered == ["high", "mid", "low"]


def test_empty_input_returns_empty_list():
    assert aggregate_hyse_search_results([]) == []


def test_raises_on_non_numeric_similarity():
    with pytest.raises(ValueError):
        aggregate_hyse_search_results([[{"table_name": "a", "cosine_similarity": "oops"}]])
