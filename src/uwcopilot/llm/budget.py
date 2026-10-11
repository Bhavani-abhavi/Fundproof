"""Run-level cost cap for LLM calls (ground rule 4: stop before a run would exceed MAX_RUN_COST_USD).

How it works:
1. Prices come from config/llm_pricing.yaml (USD per million tokens, with the source URL and the date they were
   checked). A price that could not be verified is null; such a model's cost is None ("not measured").
2. Before each call, CostTracker.check_affordable() computes a worst-case cost: every byte of the prompt counted
   as one input token (a deliberately pessimistic upper bound) plus max_tokens output tokens. If that plus what
   was already spent would exceed the cap, BudgetExceeded is raised and the call is never made.
3. A paid model without a verified price raises PriceNotVerified (a BudgetExceeded): with no price the cap
   cannot be enforced, so the call is refused rather than guessed.
4. After the call, the actual cost is computed from the token counts the API returned and added to the total.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from uwcopilot.llm.base import FAKE_PROVIDER, IncompleteLLMResponse, LLMClient, LLMMessage, LLMResponse

TOKENS_PER_MILLION = 1_000_000
# Allowance for message framing (role markers and similar) per message, in tokens.
PER_MESSAGE_OVERHEAD_TOKENS = 50


class BudgetExceeded(RuntimeError):
    """Raised before an LLM call whose worst-case cost would push the run over MAX_RUN_COST_USD."""


class PriceNotVerified(BudgetExceeded):
    """Raised before a paid call to a model that has no verified price, because the cap cannot be enforced."""


class LongContextTier(BaseModel):
    """Higher prices that apply when the prompt is longer than above_input_tokens."""

    model_config = ConfigDict(extra="forbid")

    above_input_tokens: int = Field(gt=0)
    input_usd_per_mtok: float | None = Field(default=None, ge=0)
    output_usd_per_mtok: float | None = Field(default=None, ge=0)


class ModelPrice(BaseModel):
    """One model's price list entry from config/llm_pricing.yaml."""

    model_config = ConfigDict(extra="forbid")

    model: str
    provider: str
    input_usd_per_mtok: float | None = Field(default=None, ge=0)
    output_usd_per_mtok: float | None = Field(default=None, ge=0)
    long_context: LongContextTier | None = None
    source: str
    checked_on: str

    @property
    def is_verified(self) -> bool:
        """True when both the input and the output price are known."""
        return self.input_usd_per_mtok is not None and self.output_usd_per_mtok is not None

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float | None:
        """Cost of one call, or None if a needed price is not verified."""
        input_price = self.input_usd_per_mtok
        output_price = self.output_usd_per_mtok
        tier = self.long_context
        if tier is not None and input_tokens > tier.above_input_tokens:
            input_price = tier.input_usd_per_mtok
            output_price = tier.output_usd_per_mtok
        if input_price is None or output_price is None:
            return None
        return (input_tokens * input_price + output_tokens * output_price) / TOKENS_PER_MILLION


def load_prices(path: Path) -> dict[str, ModelPrice]:
    """Read config/llm_pricing.yaml into {model id: ModelPrice}."""
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    prices: dict[str, ModelPrice] = {}
    for model_id, entry in (document.get("models") or {}).items():
        prices[model_id] = ModelPrice.model_validate({"model": model_id, **entry})
    return prices


def worst_case_input_tokens(system: str, messages: list[LLMMessage]) -> int:
    """An upper bound on prompt tokens: one token per UTF-8 byte, plus a framing allowance per message.

    Byte-level tokenizers produce at most one token per byte, so this over-estimates (usually by 3-4x). That is
    the point: the check must never let a call through that could break the cap.
    """
    total_bytes = len(system.encode("utf-8"))
    for message in messages:
        total_bytes += len(message.content.encode("utf-8"))
    return total_bytes + PER_MESSAGE_OVERHEAD_TOKENS * (len(messages) + 1)


class CostTracker:
    """Adds up what a run has spent on LLM calls and refuses calls that could exceed the cap."""

    def __init__(self, max_run_cost_usd: float, prices: dict[str, ModelPrice]) -> None:
        """max_run_cost_usd is the hard cap for this run (MAX_RUN_COST_USD)."""
        if max_run_cost_usd <= 0:
            raise ValueError("max_run_cost_usd must be greater than zero.")
        self.max_run_cost_usd = max_run_cost_usd
        self.prices = prices
        self.spent_usd = 0.0
        self.paid_calls = 0

    @property
    def remaining_usd(self) -> float:
        """How much of the cap is left."""
        return self.max_run_cost_usd - self.spent_usd

    def price_for(self, model: str) -> ModelPrice | None:
        """The price entry for a model, or None if the model is not in the price list."""
        return self.prices.get(model)

    def cost_of(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float | None:
        """Actual cost of a finished call. The fake provider is free; an unpriced model gives None."""
        if provider == FAKE_PROVIDER:
            return 0.0
        price = self.price_for(model)
        if price is None:
            return None
        return price.cost_usd(input_tokens, output_tokens)

    def check_affordable(
        self, provider: str, model: str, system: str, messages: list[LLMMessage], max_tokens: int
    ) -> float:
        """Raise BudgetExceeded if the worst-case cost of this call could break the cap; else return that cost."""
        if provider == FAKE_PROVIDER:
            return 0.0
        price = self.price_for(model)
        if price is None or not price.is_verified:
            raise PriceNotVerified(
                f"No verified price for model {model!r} in the LLM price list, so MAX_RUN_COST_USD cannot be "
                "enforced. The call was not made. Add the official price to config/llm_pricing.yaml first."
            )
        worst_case = price.cost_usd(worst_case_input_tokens(system, messages), max_tokens)
        if worst_case is None:
            raise PriceNotVerified(
                f"Model {model!r} has no verified long-context price and this prompt may need it. "
                "The call was not made."
            )
        if self.spent_usd + worst_case > self.max_run_cost_usd:
            raise BudgetExceeded(
                f"Stopping before the call: worst-case cost ${worst_case:.4f} for {model!r} plus "
                f"${self.spent_usd:.4f} already spent would exceed MAX_RUN_COST_USD=${self.max_run_cost_usd:.2f}."
            )
        return worst_case

    def record(self, cost_usd: float | None) -> None:
        """Add the actual cost of a finished paid call to the running total."""
        self.paid_calls += 1
        if cost_usd is not None:
            self.spent_usd += cost_usd

    def summary(self) -> dict[str, float | int]:
        """A small dict for run reports."""
        return {
            "max_run_cost_usd": self.max_run_cost_usd,
            "spent_usd": round(self.spent_usd, 6),
            "remaining_usd": round(self.remaining_usd, 6),
            "paid_calls": self.paid_calls,
        }


class BudgetedLLMClient:
    """Wraps any LLMClient: checks the budget before each call and fills in cost_usd afterwards."""

    def __init__(self, inner: LLMClient, tracker: CostTracker) -> None:
        """inner does the real work; tracker holds the cap and the running total."""
        self.inner = inner
        self.tracker = tracker
        self.provider = inner.provider
        self.model = inner.model

    def complete(
        self,
        system: str,
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float = 0,
    ) -> LLMResponse:
        """Refuse the call if it could break the cap; otherwise make it and record what it cost."""
        self.tracker.check_affordable(self.provider, self.model, system, messages, max_tokens)
        try:
            response = self.inner.complete(system, messages, max_tokens, temperature)
        except IncompleteLLMResponse as error:
            # A cut-off or refused answer was still paid for: record it, then let the caller see the failure.
            raise IncompleteLLMResponse(self.record(error.response)) from error
        return self.record(response)

    def record(self, response: LLMResponse) -> LLMResponse:
        """Add the response's cost to the tracker and return the response with cost_usd filled in."""
        cost = self.tracker.cost_of(self.provider, self.model, response.input_tokens, response.output_tokens)
        if self.provider != FAKE_PROVIDER:
            self.tracker.record(cost)
        return response.model_copy(update={"cost_usd": cost})

    @property
    def request_fingerprint(self) -> dict[str, str]:
        """The wrapped client's fingerprint (used in the LLM cache key)."""
        return dict(getattr(self.inner, "request_fingerprint", {}))
