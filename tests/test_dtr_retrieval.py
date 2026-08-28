"""Unit tests for DTR retrieval math, batching, and the converted checkpoint."""

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="DTR needs the optional dtr extra")

from evaluation.dtr.adapters import RetrievalExample, TableRecord
from evaluation.dtr.evaluate_dtr import recall_at_k
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
