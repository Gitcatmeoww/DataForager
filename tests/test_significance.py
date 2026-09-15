"""Unit tests for the recall confidence intervals and paired tests."""

import numpy as np
import pytest

from evaluation.significance import (
    bootstrap_recall,
    compare_recall,
    hits_at_k,
)


# --- bootstrap_recall -----------------------------------------------------

def test_point_estimate_is_the_observed_recall():
    assert bootstrap_recall([1, 1, 0, 0]).estimate == pytest.approx(0.5)


def test_interval_brackets_the_estimate():
    result = bootstrap_recall([1] * 70 + [0] * 30, seed=1)
    assert result.low < result.estimate < result.high


def test_a_unanimous_outcome_has_a_degenerate_interval():
    result = bootstrap_recall([1] * 50)
    assert (result.estimate, result.low, result.high) == (1.0, 1.0, 1.0)


def test_more_queries_narrow_the_interval():
    narrow = bootstrap_recall([1, 0] * 500, seed=2)
    wide = bootstrap_recall([1, 0] * 25, seed=2)
    assert (narrow.high - narrow.low) < (wide.high - wide.low)


def test_interval_is_reproducible_for_a_seed():
    assert bootstrap_recall([1, 0, 1, 1], seed=7) == bootstrap_recall([1, 0, 1, 1], seed=7)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="at least one query"):
        bootstrap_recall([])


# --- compare_recall -------------------------------------------------------

def test_identical_methods_are_not_significant():
    hits = [1, 0, 1, 1, 0] * 20
    result = compare_recall(hits, hits)

    assert result.difference.estimate == 0.0
    assert result.discordant_wins == 0 and result.discordant_losses == 0
    assert result.p_value == 1.0
    assert not result.significant


def test_a_uniform_win_is_significant():
    baseline = [0] * 40
    treatment = [1] * 40
    result = compare_recall(baseline, treatment)

    assert result.difference.estimate == pytest.approx(1.0)
    assert result.discordant_wins == 40 and result.discordant_losses == 0
    assert result.significant


def test_counts_discordant_pairs_in_both_directions():
    baseline = [1, 0, 1, 0]
    treatment = [0, 1, 1, 0]
    result = compare_recall(baseline, treatment)

    assert result.discordant_wins == 1  # query 1: baseline miss, treatment hit
    assert result.discordant_losses == 1  # query 0: baseline hit, treatment miss


def test_concordant_queries_do_not_affect_the_p_value():
    """McNemar conditions on disagreements only."""
    few = compare_recall([1, 0, 0], [0, 1, 0])
    padded = compare_recall([1, 0, 0] + [1] * 50, [0, 1, 0] + [1] * 50)
    assert few.p_value == pytest.approx(padded.p_value)


def test_a_small_gain_on_many_queries_can_be_significant():
    # 30 wins against 10 losses out of 1000 queries: +2pp, and detectable.
    baseline = np.array([True] * 500 + [False] * 500)
    treatment = baseline.copy()
    treatment[500:530] = True   # 30 wins
    treatment[:10] = False      # 10 losses
    result = compare_recall(baseline, treatment, seed=3)

    assert result.difference.estimate == pytest.approx(0.02)
    assert result.significant


def test_the_same_gain_on_few_queries_is_not():
    """The power problem, made concrete: same effect, too few queries."""
    baseline = np.array([True] * 25 + [False] * 25)
    treatment = baseline.copy()
    treatment[25:27] = True
    treatment[:1] = False
    result = compare_recall(baseline, treatment, seed=3)

    assert result.difference.estimate > 0
    assert not result.significant
    assert result.difference.low < 0 < result.difference.high


def test_misaligned_vectors_are_rejected():
    with pytest.raises(ValueError, match="aligned"):
        compare_recall([1, 0, 1], [1, 0])


def test_paired_interval_is_tighter_than_unpaired_intuition():
    """Pairing is the point: correlated methods give a narrow difference CI."""
    rng = np.random.default_rng(0)
    baseline = rng.random(400) < 0.6
    treatment = baseline.copy()
    flip = rng.random(400) < 0.05
    treatment[flip] = ~treatment[flip]

    paired = compare_recall(baseline, treatment, seed=0)
    spread = paired.difference.high - paired.difference.low
    base_spread = bootstrap_recall(baseline, seed=0).high - bootstrap_recall(baseline, seed=0).low
    assert spread < base_spread


# --- hits_at_k ------------------------------------------------------------

def test_hit_within_cutoff():
    assert hits_at_k([["x", "gold", "y"]], ["gold"], k=2).tolist() == [True]


def test_miss_outside_cutoff():
    assert hits_at_k([["x", "y", "gold"]], ["gold"], k=2).tolist() == [False]


def test_accepts_a_gold_set():
    # NQ-Tables labels a few questions with two gold tables.
    assert hits_at_k([["x", "b"]], [{"a", "b"}], k=2).tolist() == [True]


def test_handles_several_queries():
    retrieved = [["a", "z"], ["z", "z"]]
    assert hits_at_k(retrieved, ["a", "a"], k=2).tolist() == [True, False]
