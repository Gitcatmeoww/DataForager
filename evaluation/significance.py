"""Confidence intervals and paired significance tests for recall metrics.

Retrieval runs compare methods over the same queries, so the comparisons are
paired and the useful tests are paired ones: what matters is how often the two
methods disagree, not the variance of either alone. Two are provided.

- A bootstrap over queries, for a single method's recall and for a difference
  between two methods.
- McNemar's exact test, which is the right test for a paired binary outcome and
  conditions on exactly the queries where the methods disagree.

Both take per-query hit vectors, so they are independent of how retrieval was
run and are shared by the KaggleDS and NQ-Tables evaluations.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

DEFAULT_BOOTSTRAP = 10000
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval."""

    estimate: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.estimate:.4f} [{self.low:.4f}, {self.high:.4f}]"


@dataclass(frozen=True)
class Comparison:
    """A paired comparison between two methods on one metric."""

    baseline: float
    treatment: float
    difference: Interval
    p_value: float
    discordant_wins: int
    discordant_losses: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def __str__(self) -> str:
        return (
            f"{self.treatment:.4f} vs {self.baseline:.4f}  "
            f"diff {self.difference}  p={self.p_value:.4g}  "
            f"(+{self.discordant_wins}/-{self.discordant_losses})"
        )


def _percentile_interval(samples, confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2
    return (
        float(np.quantile(samples, tail)),
        float(np.quantile(samples, 1 - tail)),
    )


def bootstrap_recall(hits, n_resamples=DEFAULT_BOOTSTRAP, confidence=DEFAULT_CONFIDENCE,
                     seed=0) -> Interval:
    """Recall and its bootstrap confidence interval.

    Args:
        hits: Per-query booleans, one per query.
        n_resamples: Bootstrap resamples.
        confidence: Interval mass.
        seed: Seed, so a reported interval is reproducible.

    Returns:
        The observed recall with its interval.
    """
    hits = np.asarray(hits, dtype=float)
    if hits.size == 0:
        raise ValueError("need at least one query")

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, hits.size, size=(n_resamples, hits.size))
    means = hits[draws].mean(axis=1)
    low, high = _percentile_interval(means, confidence)
    return Interval(float(hits.mean()), low, high)


def compare_recall(baseline_hits, treatment_hits, n_resamples=DEFAULT_BOOTSTRAP,
                   confidence=DEFAULT_CONFIDENCE, seed=0) -> Comparison:
    """Compare two methods over the same queries.

    The bootstrap resamples queries jointly, which preserves the pairing and is
    why the interval on the difference is far tighter than the two methods'
    separate intervals would suggest.

    McNemar's exact test is reported alongside, conditioning on the queries the
    two methods disagree on. With few discordant pairs a visible difference in
    recall can still be indistinguishable from noise, which is the situation
    worth surfacing rather than hiding.

    Args:
        baseline_hits: Per-query booleans for the baseline.
        treatment_hits: Per-query booleans for the method under test, aligned
            query for query with the baseline.
        n_resamples: Bootstrap resamples.
        confidence: Interval mass.
        seed: Seed.

    Returns:
        Both recalls, the difference with its interval, McNemar's p value, and
        the discordant counts.

    Raises:
        ValueError: If the two vectors are not aligned.
    """
    baseline = np.asarray(baseline_hits, dtype=bool)
    treatment = np.asarray(treatment_hits, dtype=bool)
    if baseline.shape != treatment.shape:
        raise ValueError(
            f"hit vectors must be aligned; got {baseline.shape} and {treatment.shape}"
        )
    if baseline.size == 0:
        raise ValueError("need at least one query")

    wins = int((treatment & ~baseline).sum())
    losses = int((~treatment & baseline).sum())

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, baseline.size, size=(n_resamples, baseline.size))
    differences = treatment[draws].mean(axis=1) - baseline[draws].mean(axis=1)
    low, high = _percentile_interval(differences, confidence)

    # Exact binomial test on the discordant pairs. With none, the methods agree
    # everywhere and there is nothing to detect.
    if wins + losses == 0:
        p_value = 1.0
    else:
        p_value = float(stats.binomtest(wins, wins + losses, 0.5).pvalue)

    return Comparison(
        baseline=float(baseline.mean()),
        treatment=float(treatment.mean()),
        difference=Interval(float(treatment.mean() - baseline.mean()), low, high),
        p_value=p_value,
        discordant_wins=wins,
        discordant_losses=losses,
    )


def hits_at_k(retrieved_keys, gold_keys, k: int):
    """Per-query hit booleans at one cutoff.

    Args:
        retrieved_keys: Per query, the recall keys retrieved in rank order.
        gold_keys: Per query, the gold key or a collection of them.
        k: Cutoff.

    Returns:
        One boolean per query.
    """
    out = []
    for retrieved, gold in zip(retrieved_keys, gold_keys):
        wanted = {gold} if isinstance(gold, str) else set(gold)
        out.append(any(key in wanted for key in retrieved[:k]))
    return np.array(out, dtype=bool)
