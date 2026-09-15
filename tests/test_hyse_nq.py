"""Unit tests for the NQ HySE pipeline pieces that need no API calls."""

import pandas as pd
import pytest

from dataforager.table_representation.openai_client import resolve_provider
from evaluation.nq.hyse_nq import (
    EMBEDDING_TOKEN_LIMIT,
    Meter,
    Batch,
    TokenPacer,
    corpus_text,
    load_cached_embeddings,
    plan_batches,
    to_markdown,
    truncate_to_tokens,
)


# --- provider resolution --------------------------------------------------

def test_native_openai_is_the_default_when_a_key_is_present(monkeypatch):
    monkeypatch.delenv("DATAFORAGER_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert resolve_provider() == "openai"


def test_azure_is_the_fallback_without_a_native_key(monkeypatch):
    monkeypatch.delenv("DATAFORAGER_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resolve_provider() == "azure"


def test_environment_variable_overrides_the_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DATAFORAGER_LLM_PROVIDER", "azure")
    assert resolve_provider() == "azure"


def test_explicit_argument_wins_over_everything(monkeypatch):
    monkeypatch.setenv("DATAFORAGER_LLM_PROVIDER", "azure")
    assert resolve_provider("openai") == "openai"


def test_an_unrecognized_value_falls_through_to_detection(monkeypatch):
    monkeypatch.setenv("DATAFORAGER_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert resolve_provider() == "openai"


# --- text representation --------------------------------------------------

def test_markdown_matches_the_kaggleds_corpus_format():
    """Verified against a stored corpus vector at cosine 0.995."""
    frame = pd.DataFrame({"a": ["1", "3"], "b": ["2", "4"]})
    assert to_markdown(frame) == (
        "| a | b |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n"
        "| 3 | 4 |"
    )


def test_markdown_handles_a_single_column():
    assert to_markdown(pd.DataFrame({"only": ["x"]})) == "| only |\n| --- |\n| x |"


def test_markdown_handles_a_header_with_no_rows():
    assert to_markdown(pd.DataFrame(columns=["a", "b"])) == "| a | b |\n| --- | --- |"


def test_markdown_survives_unicode_that_collides_as_an_identifier():
    """One NQ table heads a column with U+212A KELVIN SIGN next to an ASCII K.

    The two are distinct strings, but Python NFKC-normalizes identifiers, so
    building a namedtuple from the columns (as DataFrame.itertuples does)
    rejects them as duplicates.
    """
    frame = pd.DataFrame([["a", "b"]], columns=["K", "\u212a"])
    assert to_markdown(frame) == "| K | \u212a |\n| --- | --- |\n| a | b |"


def test_corpus_text_leads_with_the_title():
    class Record:
        title = "List of dates for Easter"
        table = pd.DataFrame({"Year": ["1998"]})

    text = corpus_text(Record())
    assert text.startswith("List of dates for Easter | Year |")


# --- truncation -----------------------------------------------------------

def test_short_text_is_returned_untouched():
    text, trimmed = truncate_to_tokens("a short table")
    assert text == "a short table"
    assert not trimmed


def test_overlong_text_is_trimmed_and_flagged():
    # Two of the 8,205 NQ tables exceed the embedding limit.
    text, trimmed = truncate_to_tokens("word " * 20000)
    assert trimmed
    assert len(text) < len("word " * 20000)


def test_trimmed_text_fits_the_embedding_limit():
    import tiktoken

    text, _ = truncate_to_tokens("word " * 20000)
    assert len(tiktoken.get_encoding("cl100k_base").encode(text)) <= EMBEDDING_TOKEN_LIMIT


def test_truncation_is_deterministic():
    assert truncate_to_tokens("word " * 20000) == truncate_to_tokens("word " * 20000)


def test_a_custom_limit_is_respected():
    text, trimmed = truncate_to_tokens("word " * 100, limit=10)
    assert trimmed
    assert len(text) < len("word " * 100)


# --- cost accounting ------------------------------------------------------

def test_meter_prices_a_known_model():
    meter = Meter("gen", model="gpt-4o-mini")
    meter.record(1.0, prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert meter.cost == pytest.approx(0.75)  # 0.15 in + 0.60 out


def test_meter_prices_an_unknown_model_as_zero_rather_than_crashing():
    meter = Meter("gen", model="some-future-model")
    meter.record(1.0, prompt_tokens=1000, completion_tokens=1000)
    assert meter.cost == 0.0


def test_meter_reports_latency_percentiles():
    meter = Meter("embed", model="text-embedding-3-small")
    for seconds in [0.1, 0.2, 0.3, 10.0]:
        meter.record(seconds)
    summary = meter.summary()
    assert summary["calls"] == 4
    assert summary["p50_call_seconds"] == pytest.approx(0.25)
    assert summary["p95_call_seconds"] > summary["p50_call_seconds"]


def test_meter_summary_is_safe_with_no_calls():
    assert Meter("idle").summary()["calls"] == 0


# --- batching -------------------------------------------------------------

def test_batches_respect_the_item_cap():
    batches, _ = plan_batches(["x"] * 10, max_items=4, max_tokens=10**6)
    assert [len(b) for b in batches] == [4, 4, 2]


def test_batches_respect_the_token_budget():
    # Each text is a few tokens; a tight budget must split them up.
    batches, _ = plan_batches(["hello world"] * 6, max_items=100, max_tokens=4)
    assert len(batches) > 1


def test_batches_preserve_every_position_exactly_once():
    texts = [f"table {i}" for i in range(37)]
    batches, _ = plan_batches(texts, max_items=5, max_tokens=10**6)
    positions = [p for batch in batches for p, _ in batch.items]
    assert positions == list(range(37))


def test_batches_carry_the_text_alongside_the_position():
    batches, _ = plan_batches(["alpha", "beta"], max_items=1, max_tokens=10**6)
    assert [t for b in batches for _, t in b.items] == ["alpha", "beta"]


def test_batching_reports_truncated_texts():
    _, truncated = plan_batches(["short", "word " * 20000], max_items=10, max_tokens=10**9)
    assert truncated == 1


def test_an_oversized_single_text_still_forms_a_batch():
    # Truncation happens first, so one huge text cannot produce an empty plan.
    batches, truncated = plan_batches(["word " * 20000], max_items=256, max_tokens=1)
    assert truncated == 1
    assert sum(len(b) for b in batches) == 1


def test_empty_input_produces_no_batches():
    assert plan_batches([]) == ([], 0)


# --- token pacing ---------------------------------------------------------

def test_batches_carry_their_token_cost():
    batches, _ = plan_batches(["hello world", "another table"], max_items=1, max_tokens=10**6)
    assert all(b.tokens > 0 for b in batches)
    assert all(b.tokens >= len(b) for b in batches)


def test_pacer_allows_traffic_within_budget_without_waiting():
    import time as _time

    pacer = TokenPacer(tokens_per_minute=1_000_000)
    started = _time.monotonic()
    for _ in range(10):
        pacer.acquire(1000)
    assert _time.monotonic() - started < 0.5


def test_pacer_blocks_once_the_budget_is_spent():
    import time as _time

    # 6000 per minute is 100 per second, so a second 6000 costs about a second.
    pacer = TokenPacer(tokens_per_minute=6000)
    pacer.acquire(6000)
    started = _time.monotonic()
    pacer.acquire(100)
    assert _time.monotonic() - started >= 0.5


def test_pacer_lets_through_a_request_larger_than_the_whole_budget():
    """Otherwise such a request would block forever instead of being retried."""
    import time as _time

    pacer = TokenPacer(tokens_per_minute=1000)
    started = _time.monotonic()
    pacer.acquire(50_000)
    assert _time.monotonic() - started < 0.5


# --- checkpointing --------------------------------------------------------

def test_checkpoint_round_trips_through_a_path_without_npz_mangling(tmp_path):
    """np.savez appends .npz to a bare path, which broke the atomic replace."""
    import numpy as np

    target = tmp_path / "corpus.partial.npz"
    tmp = tmp_path / "corpus.partial.npz.tmp"
    vectors = np.arange(6, dtype=np.float32).reshape(3, 2)
    done = np.array([True, False, True])

    with open(tmp, "wb") as handle:
        np.savez(handle, vectors=vectors, done=done)
    tmp.replace(target)

    blob = np.load(target)
    assert np.array_equal(blob["vectors"], vectors)
    assert np.array_equal(blob["done"], done)


# --- cache validation -----------------------------------------------------

def test_cache_miss_returns_none(tmp_path):
    assert load_cached_embeddings(tmp_path / "absent.npy", 10) is None


def test_cache_of_the_right_shape_is_used(tmp_path):
    import numpy as np

    path = tmp_path / "vectors.npy"
    np.save(path, np.zeros((5, 3), dtype=np.float32))
    assert load_cached_embeddings(path, 5).shape == (5, 3)


def test_stale_cache_is_rejected(tmp_path):
    """A failed generation leaves one fewer component; the next run regenerates
    it, and the saved matrix is then a row short of what the caller indexes."""
    import numpy as np

    path = tmp_path / "vectors.npy"
    np.save(path, np.zeros((2133, 3), dtype=np.float32))
    assert load_cached_embeddings(path, 2134) is None
