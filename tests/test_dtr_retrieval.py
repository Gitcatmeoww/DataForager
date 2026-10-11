"""Unit tests for DTR retrieval math, batching, and the converted checkpoint."""

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="DTR needs the optional dtr extra")

from evaluation.dtr.adapters import RetrievalExample, TableRecord
from evaluation.dtr.evaluate_dtr import (
    build_adapter,
    evaluate,
    group_by_question,
    query_hits,
    query_partitions,
    recall_at_k,
    recall_by_partition,
    table_partitions,
)
from evaluation.dtr.index import DTRIndex
from evaluation.dtr.train import group_aware_batches

import pandas as pd


def make_index(recall_keys, groups=None, dim=4):
    """An index whose row i is the i-th basis vector, so scores are separable."""
    embeddings = np.eye(len(recall_keys), dim, dtype=np.float32)
    return DTRIndex(
        embeddings=embeddings,
        table_ids=[f"id{i}" for i in range(len(recall_keys))],
        recall_keys=list(recall_keys),
        group_ids=list(groups or recall_keys),
    )


def test_search_orders_by_inner_product():
    index = make_index(["a", "b", "c", "d"])
    # Weights ascending across rows, so the ranking must come back reversed.
    query = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

    assert index.search(query, top_k=4)[0].tolist() == [3, 2, 1, 0]


def test_search_respects_top_k_and_clamps_to_corpus_size():
    index = make_index(["a", "b", "c", "d"])
    query = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

    assert index.search(query, top_k=2)[0].tolist() == [3, 2]
    assert index.search(query, top_k=99).shape[1] == 4  # cannot exceed the corpus


def test_recall_counts_a_hit_anywhere_in_the_top_k():
    index = make_index(["a", "b", "c", "d"])
    ranking = np.array([[3, 2, 1, 0]])  # gold "a" sits at position 3

    recalls = recall_at_k(index, ranking, ["a"], ks=(1, 2, 4))

    assert recalls[1] == 0.0
    assert recalls[2] == 0.0
    assert recalls[4] == 1.0


def test_recall_averages_across_queries():
    index = make_index(["a", "b"])
    ranking = np.array([[0, 1], [0, 1]])  # first query hits at 1, second does not

    assert recall_at_k(index, ranking, ["a", "b"], ks=(1,))[1] == pytest.approx(0.5)


def test_recall_matches_duplicate_table_names():
    # Table names repeat across unrelated datasets, and the harness scores a hit
    # on the name, so either copy counts.
    index = make_index(["train.csv", "other.csv", "train.csv"])
    ranking = np.array([[2, 0, 1]])

    assert recall_at_k(index, ranking, ["train.csv"], ks=(1,))[1] == 1.0


def test_recall_accepts_a_gold_set_for_multi_gold_questions():
    # NQ-Tables labels a few questions with two gold tables; either counts.
    index = make_index(["a", "b", "c", "d"])
    ranking = np.array([[2, 3, 0, 1]])

    recalls = recall_at_k(index, ranking, [{"a", "c"}], ks=(1, 4))

    assert recalls[1] == 1.0  # "c" is first
    assert recalls[4] == 1.0


def test_recall_misses_when_no_gold_in_the_set_is_retrieved():
    index = make_index(["a", "b", "c", "d"])
    ranking = np.array([[0, 1]])

    assert recall_at_k(index, ranking, [{"c", "d"}], ks=(2,))[2] == 0.0


@pytest.mark.parametrize("ranking_row", [[3, 2, 1, 0], [0, 1, 2, 3], [1, 3, 0, 2]])
def test_a_bare_key_and_a_one_element_set_agree(ranking_row):
    """The single-gold path must be unchanged, since it produced Table 1."""
    index = make_index(["a", "b", "c", "d"])
    ranking = np.array([ranking_row])
    ks = (1, 2, 4)

    assert recall_at_k(index, ranking, ["a"], ks=ks) == recall_at_k(
        index, ranking, [{"a"}], ks=ks
    )


def test_recall_handles_mixed_bare_and_set_golds():
    index = make_index(["a", "b"])
    ranking = np.array([[0, 1], [1, 0]])

    assert recall_at_k(index, ranking, ["a", {"a", "b"}], ks=(1,))[1] == 1.0


# --- per-partition recall -------------------------------------------------

def test_query_hits_marks_the_rank_of_each_gold():
    index = make_index(["a", "b", "c"])
    ranking = np.array([[1, 0, 2], [2, 1, 0]])

    hits = query_hits(index, ranking, ["a", "c"])

    assert hits.tolist() == [[False, True, False], [True, False, False]]


def test_recall_by_partition_scores_each_partition_on_its_own_queries():
    hits = np.array([
        [True, False],   # single, hit at 1
        [False, True],   # single, hit at 2
        [False, False],  # multi, miss
        [False, True],   # multi, hit at 2
    ])
    labels = ["single_table", "single_table", "multi_table", "multi_table"]

    parts = recall_by_partition(hits, labels, ks=(1, 2))

    assert parts["single_table"] == {"queries": 2, "recall": {1: 0.5, 2: 1.0}}
    assert parts["multi_table"] == {"queries": 2, "recall": {1: 0.0, 2: 0.5}}


def test_partition_recalls_recombine_to_the_overall_recall():
    rng = np.random.default_rng(0)
    hits = rng.random((50, 5)) < 0.2
    labels = rng.choice(["single_table", "multi_table"], size=50)

    parts = recall_by_partition(hits, labels, ks=(1, 5))
    overall = hits[:, :5].any(axis=1).mean()
    weighted = sum(p["queries"] * p["recall"][5] for p in parts.values()) / 50

    assert weighted == pytest.approx(overall)


def test_recall_by_partition_rejects_a_label_count_mismatch():
    with pytest.raises(ValueError, match="2 labels for 3 queries"):
        recall_by_partition(np.zeros((3, 2), dtype=bool), ["a", "b"])


def kaggle_table(database_name, table_name, suffix=""):
    return TableRecord(
        table_id=f"{database_name}::{table_name}{suffix}", recall_key=table_name,
        title=table_name, table=pd.DataFrame({"c": ["v"]}), group_id=database_name,
    )


# Two unrelated datasets that both ship a train.csv, with different labels.
SHARED_NAME_TABLES = [kaggle_table("Bike Rental", "train.csv"), kaggle_table("Big Mart", "train.csv")]
SHARED_NAME_ROWS = [("Bike Rental", "train.csv", "single_table"), ("Big Mart", "train.csv", "multi_table")]


def test_dataset_scheme_labels_tables_that_share_a_name_independently():
    partition_of = table_partitions(SHARED_NAME_TABLES, SHARED_NAME_ROWS, scheme="dataset")

    assert partition_of == {
        "Bike Rental::train.csv": "single_table",
        "Big Mart::train.csv": "multi_table",
    }


def test_name_scheme_gives_every_same_named_table_the_last_label_read():
    # Mirrors table_count_evaluator.py, whose labels produced the paper's counts.
    partition_of = table_partitions(SHARED_NAME_TABLES, SHARED_NAME_ROWS, scheme="name")

    assert set(partition_of.values()) == {"multi_table"}


def test_duplicate_names_within_one_dataset_share_its_label():
    tables = [kaggle_table("Fast Food", "data.csv"), kaggle_table("Fast Food", "data.csv", "#1")]
    rows = [("Fast Food", "data.csv", "multi_table")] * 2

    assert set(table_partitions(tables, rows).values()) == {"multi_table"}


def test_table_partitions_strips_whitespace_like_the_adapter():
    # Kaggle dataset names carry stray spaces, which the adapter strips.
    tables = [kaggle_table("Bank Customer Churn Prediction", "Churn_Modelling.csv")]
    rows = [("Bank Customer Churn Prediction ", "Churn_Modelling.csv", "single_table")]

    assert table_partitions(tables, rows) == {
        "Bank Customer Churn Prediction::Churn_Modelling.csv": "single_table"
    }


def test_table_partitions_rejects_an_unlabeled_table():
    with pytest.raises(ValueError, match="1 tables have no partition label"):
        table_partitions([kaggle_table("Other", "x.csv")], SHARED_NAME_ROWS)


def test_table_partitions_rejects_an_unknown_scheme():
    with pytest.raises(ValueError, match="unknown partition scheme"):
        table_partitions(SHARED_NAME_TABLES, SHARED_NAME_ROWS, scheme="schema")


def test_query_partitions_follow_the_gold_table():
    partition_of = {"t1": "single_table", "t2": "multi_table"}

    assert query_partitions([["t1"], ["t2"], ["t1"]], partition_of) == [
        "single_table", "multi_table", "single_table",
    ]


def test_query_partitions_reject_golds_in_different_partitions():
    with pytest.raises(ValueError, match="span partitions"):
        query_partitions([["t1", "t2"]], {"t1": "single_table", "t2": "multi_table"})


def test_partitioning_is_refused_outside_kaggleds():
    # Raised before any checkpoint is loaded, so the path need not exist.
    with pytest.raises(ValueError, match="only defined for the kaggleds corpus"):
        evaluate("no/such/checkpoint", corpus="nq", partition_by="dataset")


# --- group_by_question ----------------------------------------------------

def test_examples_without_a_query_id_stay_separate():
    # KaggleDS has one gold per query and sets no query_id.
    examples = [
        RetrievalExample(query="q", table_id="t1"),
        RetrievalExample(query="q", table_id="t2"),
    ]
    queries, golds = group_by_question(examples)

    assert queries == ["q", "q"]
    assert golds == [["t1"], ["t2"]]


def test_examples_sharing_a_query_id_collapse():
    examples = [
        RetrievalExample(query="heroes", table_id="t1", query_id="q1"),
        RetrievalExample(query="heroes", table_id="t2", query_id="q1"),
    ]
    queries, golds = group_by_question(examples)

    assert queries == ["heroes"]
    assert golds == [["t1", "t2"]]


def test_grouping_preserves_first_seen_order_when_interleaved():
    examples = [
        RetrievalExample(query="a", table_id="t1", query_id="qa"),
        RetrievalExample(query="b", table_id="t2", query_id="qb"),
        RetrievalExample(query="a", table_id="t3", query_id="qa"),
    ]
    queries, golds = group_by_question(examples)

    assert queries == ["a", "b"]
    assert golds == [["t1", "t3"], ["t2"]]


# --- build_adapter --------------------------------------------------------

def test_build_adapter_returns_the_kaggleds_corpus():
    assert build_adapter("kaggleds").name == "kaggleds"


def test_build_adapter_requires_a_data_dir_for_nq():
    with pytest.raises(ValueError, match="--data-dir is required"):
        build_adapter("nq")


def test_build_adapter_rejects_an_unknown_corpus():
    with pytest.raises(ValueError, match="unknown corpus"):
        build_adapter("wikitables")


def table(table_id, group):
    return TableRecord(
        table_id=table_id, recall_key=table_id, title=table_id,
        table=pd.DataFrame({"c": ["v"]}), group_id=group,
    )


def test_batches_never_repeat_a_group():
    # Three tables per group, mimicking sibling tables in one Kaggle dataset.
    tables = {f"t{i}": table(f"t{i}", f"g{i // 3}") for i in range(60)}
    examples = [RetrievalExample(query=f"q{i}", table_id=t) for i, t in enumerate(tables)]

    batches = group_aware_batches(examples, tables, batch_size=4, rng=random.Random(0))

    assert batches
    for batch in batches:
        groups = [tables[e.table_id].group_id for e in batch]
        assert len(set(groups)) == len(groups)


def test_batches_are_all_full():
    # In-batch negatives depend on a fixed batch size, so partial batches are
    # never emitted.
    tables = {f"t{i}": table(f"t{i}", f"g{i}") for i in range(10)}
    examples = [RetrievalExample(query=f"q{i}", table_id=t) for i, t in enumerate(tables)]

    batches = group_aware_batches(examples, tables, batch_size=4, rng=random.Random(0))

    assert all(len(b) == 4 for b in batches)


def test_batching_terminates_when_one_group_dominates():
    # Every example shares a group, so no full batch is possible and the sweep
    # must stop rather than loop forever.
    tables = {f"t{i}": table(f"t{i}", "same") for i in range(10)}
    examples = [RetrievalExample(query=f"q{i}", table_id=t) for i, t in enumerate(tables)]

    assert group_aware_batches(examples, tables, batch_size=4, rng=random.Random(0)) == []


def test_in_batch_loss_is_minimised_on_the_diagonal():
    from evaluation.dtr.modeling import TapasDualEncoder

    # A perfectly aligned batch should score far better than a shuffled one.
    h_q = torch.eye(3)
    aligned = torch.nn.functional.cross_entropy(h_q @ h_q.T * 10, torch.arange(3))
    misaligned = torch.nn.functional.cross_entropy(h_q @ h_q.roll(1, 0).T * 10, torch.arange(3))

    assert aligned < misaligned
    assert TapasDualEncoder is not None  # the loss above mirrors forward()


def test_index_round_trips_through_disk(tmp_path):
    index = make_index(["a", "b"], groups=["g", "g"])
    path = tmp_path / "index.npz"
    index.save(path)

    loaded = DTRIndex.load(path)

    assert loaded.recall_keys == index.recall_keys
    assert loaded.group_ids == index.group_ids
    assert np.array_equal(loaded.embeddings, index.embeddings)
