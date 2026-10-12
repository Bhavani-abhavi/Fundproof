"""Tests for uwcopilot.llm.budget: prices from config, stop before the cap, refuse unpriced models."""

import pytest

from llm_test_helpers import PretendPaidClient
from uwcopilot.common.paths import CONFIG_DIR
from uwcopilot.llm.base import IncompleteLLMResponse, LLMMessage
from uwcopilot.llm.budget import (
    BudgetedLLMClient,
    BudgetExceeded,
    CostTracker,
    ModelPrice,
    PriceNotVerified,
    load_prices,
    worst_case_input_tokens,
)
from uwcopilot.llm.fake_client import FakeLLM

USER = [LLMMessage(role="user", content="hello")]


def haiku_price() -> ModelPrice:
    """A price entry with a long-context tier, like the one in config/llm_pricing.yaml."""
    return ModelPrice.model_validate(
        {
            "model": "claude-haiku-5-5",
            "provider": "anthropic",
            "input_usd_per_mtok": 0.10,
            "output_usd_per_mtok": 0.50,
            "long_context": {"above_input_tokens": 100000, "input_usd_per_mtok": 0.50, "output_usd_per_mtok": 2.50},
            "source": "https://example.invalid/pricing",
            "checked_on": "2026-10-10",
        }
    )


def test_repo_price_list_has_a_source_and_date_for_every_model() -> None:
    prices = load_prices(CONFIG_DIR / "llm_pricing.yaml")
    assert "claude-haiku-5-5" in prices
    for price in prices.values():
        assert price.checked_on
        if price.provider == "ollama":
            # Local models: no pricing page exists; the source says there is no API charge and the price is 0.0.
            assert "no API charge" in price.source
            assert (price.input_usd_per_mtok, price.output_usd_per_mtok) == (0.0, 0.0)
        else:
            assert price.source.startswith("https://")


def test_cost_uses_the_standard_and_long_context_tiers() -> None:
    price = haiku_price()
    # Up to 100,000 prompt tokens: standard prices.
    assert price.cost_usd(100_000, 0) == pytest.approx(0.01)
    assert price.cost_usd(0, 1_000_000) == pytest.approx(0.50)
    # Above 100,000 prompt tokens: the long-context prices apply to input and output.
    assert price.cost_usd(200_000, 1_000_000) == pytest.approx(200_000 * 0.50 / 1e6 + 2.50)


def test_null_price_means_cost_not_measured() -> None:
    price = ModelPrice(model="m", provider="openai", source="https://example.invalid", checked_on="2026-10-10")
    assert not price.is_verified
    assert price.cost_usd(10, 10) is None


def test_worst_case_input_counts_every_byte() -> None:
    assert worst_case_input_tokens("ab", [LLMMessage(role="user", content="é")]) == 2 + 2 + 2 * 50


def test_call_is_refused_before_it_could_exceed_the_cap() -> None:
    inner = PretendPaidClient()
    tracker = CostTracker(max_run_cost_usd=0.0001, prices={"claude-haiku-5-5": haiku_price()})
    client = BudgetedLLMClient(inner, tracker)
    # Worst case: max_tokens=1000 output tokens at $0.50/MTok = $0.0005 > $0.0001.
    with pytest.raises(BudgetExceeded):
        client.complete("sys", USER, max_tokens=1000)
    assert inner.calls == 0
    assert tracker.spent_usd == 0.0


def test_spending_accumulates_until_the_cap_stops_the_run() -> None:
    # The pretend API reports 5 input tokens, below the byte-count upper bound, as a real API would.
    inner = PretendPaidClient(input_tokens=5, output_tokens=500)
    tracker = CostTracker(max_run_cost_usd=0.001, prices={"claude-haiku-5-5": haiku_price()})
    client = BudgetedLLMClient(inner, tracker)
    response = client.complete("sys", USER, max_tokens=500)
    expected = (5 * 0.10 + 500 * 0.50) / 1e6
    assert response.cost_usd == pytest.approx(expected)
    assert tracker.spent_usd == pytest.approx(expected)
    assert tracker.paid_calls == 1
    calls_made = 1
    with pytest.raises(BudgetExceeded):
        for _ in range(100):
            client.complete("sys", USER, max_tokens=500)
            calls_made += 1
    assert inner.calls == calls_made
    assert tracker.spent_usd <= tracker.max_run_cost_usd


def test_incomplete_answers_are_still_paid_for() -> None:
    inner = PretendPaidClient(input_tokens=5, output_tokens=500, stop_reason="max_tokens")
    tracker = CostTracker(max_run_cost_usd=1.0, prices={"claude-haiku-5-5": haiku_price()})
    with pytest.raises(IncompleteLLMResponse) as raised:
        BudgetedLLMClient(inner, tracker).complete("sys", USER, max_tokens=500)
    expected = (5 * 0.10 + 500 * 0.50) / 1e6
    assert raised.value.response.cost_usd == pytest.approx(expected)
    assert tracker.spent_usd == pytest.approx(expected)
    assert tracker.paid_calls == 1


def test_unpriced_paid_model_is_refused() -> None:
    inner = PretendPaidClient(model="model-without-a-price")
    client = BudgetedLLMClient(inner, CostTracker(10.0, prices={}))
    with pytest.raises(PriceNotVerified):
        client.complete("sys", USER, max_tokens=10)
    assert inner.calls == 0


def test_fake_provider_is_free_and_never_counts_as_paid() -> None:
    tracker = CostTracker(0.000001, prices={})
    response = BudgetedLLMClient(FakeLLM(["x"]), tracker).complete("sys", USER, max_tokens=10_000)
    assert response.cost_usd == 0.0
    assert tracker.paid_calls == 0


def test_tracker_rejects_a_non_positive_cap() -> None:
    with pytest.raises(ValueError):
        CostTracker(0, prices={})
