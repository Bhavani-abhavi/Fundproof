"""Next-best-evidence: a rules-based, scenario-based value-of-information score for evidence requests (Phase 4 D,
innovation S5).

For an open item (open_items.py) the resolution template lists its plausible resolutions r with documented prior
weights p(r) (config/resolution_templates.yaml; the priors are assumptions written by the project author). For each
resolution the engine applies its levers to the case state, runs the SAME reference policy, and reads the
recommendation and the decision metric. Then

    VOI(request) = sum_r p(r) x 1[recommendation(r) != current recommendation] x size(r) / cost_weight(request)
    size(r)      = 1 + min(|decision metric(r) - current decision metric|, 1.0) when both metrics exist, else 1.0

The decision metric is the decision DSCR for a loan (projected DSCR for a startup loan) and the highest 5-year loss
ratio of the quoted lines, as a fraction (55.0% -> 0.55), for insurance (scenarios.decision_metric). The cost weight
comes from config/evidence_requests.yaml (a relative planning assumption). VOI is exactly 0 when every resolution
leaves the recommendation unchanged: such a request has no decision value, however interesting the answer.

Items that share a request (the same request id and params, for example every missing checklist document, or two
problems on the same carrier's loss run) are valued together: one request reveals all of them, so the resolutions
are the combinations of each item's resolutions (priors multiplied), applied one item after the other.

What this is not: it is not optimal and not an economic model. It is myopic (it values one request at a time, so two
items that each block the same outcome, such as a missing statement month and another missing document, each score
0 alone), it trusts the priors, and it measures decision change with the reference policy only. A resolution whose
levers cannot be applied, or whose application does not validate, is listed with its error and counts as no change.
"""

import itertools
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from uwcopilot.agent.applications import canonical_json
from uwcopilot.agent.policy import PolicyResult
from uwcopilot.docintel.levers import LeverError
from uwcopilot.evidence.open_item_models import OpenItem
from uwcopilot.evidence.requests import ResolutionChoice, cost_weight, instantiate_options, request_catalog
from uwcopilot.evidence.scenarios import DecisionMetric, decision_metric
from uwcopilot.evidence.state import CaseState, IncompleteApplicationError, apply, evaluate

SIZE_CAP = 1.0
MAX_JOINT_OPTIONS = 729
PCT_TO_FRACTION = 100.0
SCORE_NAME = "a rules-based, scenario-based value-of-information score"


class Frozen(BaseModel):
    """Base for VOI models: unknown fields are an error and objects are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class OptionOutcome(Frozen):
    """One plausible resolution (or combination, for a merged request) and what the reference policy then says."""

    option_id: str
    description: str
    prior: float
    recommendation: str | None
    metric: float | None
    changes_recommendation: bool
    size: float
    weighted: float  # prior x 1[change] x size (before the cost weight)
    error: str | None = None


class RequestValue(Frozen):
    """The value of information of one request (one or more items that share it)."""

    request_id: str
    request_params: dict[str, str]
    item_ids: tuple[str, ...]
    cost_weight: float
    current_recommendation: str
    metric_name: str
    current_metric: float | None
    outcomes: tuple[OptionOutcome, ...]
    p_change: float
    voi: float
    dropped_option_ids: tuple[str, ...] = ()
    computable: bool = True


class RankedRequest(Frozen):
    """One recommended evidence request, its score and a one-sentence reason filled from the numbers."""

    rank: int
    request_id: str
    title: str
    params: dict[str, str]
    item_ids: tuple[str, ...]
    voi: float
    p_change: float
    cost_weight: float
    options_summary: tuple[dict, ...]
    reason: str


# --------------------------------------------------------------------------------------------------------------
# Options of one item
# --------------------------------------------------------------------------------------------------------------


def resolution_choices(item: OpenItem) -> tuple[list[ResolutionChoice], tuple[str, ...]]:
    """(the item's plausible resolutions, ids of template options left out for missing case values)."""
    if item.template_key is None:
        return [], ()
    filled = instantiate_options(item.template_key, item.placeholders, item.extra_substitutions)
    return list(filled.options), filled.dropped_option_ids


def metric_fraction(metric: DecisionMetric, value: float | None) -> float | None:
    """A metric value on the VOI scale: DSCR as is, a percent as a fraction."""
    if value is None:
        return None
    return value / PCT_TO_FRACTION if metric.unit == "pct" else value


def change_size(metric: DecisionMetric, current: float | None, resolved: float | None) -> float:
    """size(r) = 1 + min(|delta metric|, 1.0) when both metrics exist, else 1.0 (module docstring)."""
    before, after = metric_fraction(metric, current), metric_fraction(metric, resolved)
    if before is None or after is None:
        return 1.0
    return 1.0 + min(abs(after - before), SIZE_CAP)


def evaluate_choice(state: CaseState, steps: Sequence[ResolutionChoice]) -> tuple[PolicyResult | None, str | None]:
    """The policy result after applying each choice in order (None and the error name when that fails)."""
    resolved = state
    try:
        for step in steps:
            resolved = apply(resolved, step.substitutions)
        return evaluate(resolved), None
    except (LeverError, IncompleteApplicationError, ValueError) as error:
        return None, type(error).__name__


def outcome_of(
    option_id: str,
    description: str,
    prior: float,
    result: PolicyResult | None,
    error: str | None,
    current: PolicyResult,
) -> OptionOutcome:
    """One OptionOutcome compared with the current result."""
    baseline = decision_metric(current)
    if result is None:
        return OptionOutcome(
            option_id=option_id,
            description=description,
            prior=prior,
            recommendation=None,
            metric=None,
            changes_recommendation=False,
            size=0.0,
            weighted=0.0,
            error=error,
        )
    metric = decision_metric(result)
    changed = result.recommendation != current.recommendation
    size = change_size(baseline, baseline.value, metric.value) if changed else 0.0
    return OptionOutcome(
        option_id=option_id,
        description=description,
        prior=prior,
        recommendation=result.recommendation,
        metric=metric.value,
        changes_recommendation=changed,
        size=round(size, 6),
        weighted=round(prior * size, 6) if changed else 0.0,
    )


def options(item: OpenItem, state: CaseState, current: PolicyResult | None = None) -> list[OptionOutcome]:
    """Each plausible resolution of one item: apply its levers, evaluate, read recommendation and metric."""
    reference = current if current is not None else evaluate(state)
    choices, _ = resolution_choices(item)
    outcomes = []
    for choice in choices:
        result, error = evaluate_choice(state, [choice])
        outcomes.append(outcome_of(choice.option_id, choice.description, choice.prior, result, error, reference))
    return outcomes


# --------------------------------------------------------------------------------------------------------------
# Value of a request
# --------------------------------------------------------------------------------------------------------------


def joint_choices(choice_lists: list[list[ResolutionChoice]]) -> list[tuple[ResolutionChoice, ...]]:
    """Every combination of one choice per item (the cartesian product), capped at MAX_JOINT_OPTIONS."""
    total = 1
    for choices in choice_lists:
        total *= len(choices)
    if total > MAX_JOINT_OPTIONS:
        raise ValueError(f"{total} joint resolutions exceed the cap of {MAX_JOINT_OPTIONS}")
    return list(itertools.product(*choice_lists))


def request_value(
    state: CaseState,
    items: Sequence[OpenItem],
    current: PolicyResult | None = None,
    choices_by_item: dict[str, list[ResolutionChoice]] | None = None,
    cost_weights: dict[str, float] | None = None,
) -> RequestValue:
    """VOI of the request the items share (module docstring).

    choices_by_item replaces an item's template options (the simulator's robustness runs use it: priors scaled, or
    the true resolution removed); cost_weights overrides catalog weights.
    """
    reference = current if current is not None else evaluate(state)
    baseline = decision_metric(reference)
    request_id = str(items[0].request_id)
    weight = cost_weight(request_id, cost_weights)
    choice_lists: list[list[ResolutionChoice]] = []
    dropped: list[str] = []
    for item in items:
        if choices_by_item is not None and item.item_id in choices_by_item:
            choice_lists.append(list(choices_by_item[item.item_id]))
            continue
        choices, missing = resolution_choices(item)
        choice_lists.append(choices)
        dropped.extend(f"{item.item_id}:{option_id}" for option_id in missing)
    computable = all(choice_lists)
    outcomes: list[OptionOutcome] = []
    if computable:
        for combination in joint_choices(choice_lists):
            prior = 1.0
            for choice in combination:
                prior *= choice.prior
            result, error = evaluate_choice(state, combination)
            option_id = " + ".join(choice.option_id for choice in combination)
            description = " ".join(choice.description for choice in combination)
            outcomes.append(outcome_of(option_id, description, round(prior, 6), result, error, reference))
    p_change = round(sum(outcome.prior for outcome in outcomes if outcome.changes_recommendation), 6)
    voi = round(sum(outcome.weighted for outcome in outcomes) / weight, 6)
    return RequestValue(
        request_id=request_id,
        request_params=dict(items[0].request_params),
        item_ids=tuple(item.item_id for item in items),
        cost_weight=weight,
        current_recommendation=reference.recommendation,
        metric_name=baseline.name,
        current_metric=baseline.value,
        outcomes=tuple(outcomes),
        p_change=p_change,
        voi=voi,
        dropped_option_ids=tuple(dropped),
        computable=computable,
    )


def voi(item: OpenItem, state: CaseState, current: PolicyResult | None = None) -> float:
    """VOI of a single item's own request (VOI(item) in the spec)."""
    return request_value(state, [item], current).voi


# --------------------------------------------------------------------------------------------------------------
# Ranking and reasons
# --------------------------------------------------------------------------------------------------------------


def request_key(item: OpenItem) -> str:
    """Items with the same request id and params share one request."""
    return f"{item.request_id}|{canonical_json(item.request_params)}"


def group_items(items: Sequence[OpenItem]) -> list[list[OpenItem]]:
    """Items grouped by shared request, in first-seen order (items without a catalog request are left out)."""
    groups: dict[str, list[OpenItem]] = {}
    for item in items:
        if item.request_id is None:
            continue
        groups.setdefault(request_key(item), []).append(item)
    return list(groups.values())


def format_metric(name: str, value: float | None) -> str:
    """A decision metric for a sentence: a loss ratio as a percent with 1 decimal, a DSCR with 2 decimals."""
    if value is None:
        return "not available"
    if "loss ratio" in name:
        return f"{value:.1f}%"
    return f"{value:.2f}"


def request_phrase(value: RequestValue) -> str:
    """'the <catalog title>' plus what is asked for, from ids and codes only (never document text)."""
    title = request_catalog()[value.request_id].title
    phrase = f"the {title[0].lower()}{title[1:]}"
    request_params = value.request_params
    if "month" in request_params:
        phrase += f" for {request_params['month']}"
    if "document" in request_params:
        phrase += f" ({request_params['document']})"
    if "carrier" in request_params:
        phrase += " from the named carrier"
    return phrase


def representative(value: RequestValue) -> OptionOutcome | None:
    """The changing resolution with the highest prior (the first one on a tie)."""
    changing = [outcome for outcome in value.outcomes if outcome.changes_recommendation]
    if not changing:
        return None
    return max(changing, key=lambda outcome: outcome.prior)


def reason_sentence(value: RequestValue) -> str:
    """The one-sentence reason, with every number taken from the calculation."""
    phrase = request_phrase(value)
    now = value.current_recommendation
    if not value.computable:
        return f"No plausible resolution could be built from the case values, so {phrase} is not scored."
    example = representative(value)
    if example is None:
        return (
            f"Requesting {phrase} has zero decision value: under every plausible resolution the reference "
            f"recommendation stays {now}."
        )
    probability = f"(probability {value.p_change:.2f} under the documented priors)"
    if value.current_metric is not None and example.metric is not None and example.metric != value.current_metric:
        before = format_metric(value.metric_name, value.current_metric)
        after = format_metric(value.metric_name, example.metric)
        return (
            f"Requesting {phrase} could move the {value.metric_name} from {before} to {after}, which changes "
            f"{now} -> {example.recommendation} {probability}."
        )
    return f"Requesting {phrase} could change the recommendation {now} -> {example.recommendation} {probability}."


def options_summary(value: RequestValue) -> tuple[dict, ...]:
    """A compact list of the resolutions: id, prior, recommendation, metric, change, error."""
    return tuple(
        {
            "option_id": outcome.option_id,
            "prior": outcome.prior,
            "recommendation": outcome.recommendation,
            "metric": outcome.metric,
            "changes_recommendation": outcome.changes_recommendation,
            "error": outcome.error,
        }
        for outcome in value.outcomes
    )


def request_values(
    state: CaseState,
    items: Sequence[OpenItem],
    current: PolicyResult | None = None,
    choices_by_item: dict[str, list[ResolutionChoice]] | None = None,
    cost_weights: dict[str, float] | None = None,
) -> list[RequestValue]:
    """The value of every request the items need (one per group of items sharing a request)."""
    reference = current if current is not None else evaluate(state)
    return [request_value(state, group, reference, choices_by_item, cost_weights) for group in group_items(items)]


def rank_requests(
    state: CaseState,
    items: Sequence[OpenItem],
    current: PolicyResult | None = None,
    choices_by_item: dict[str, list[ResolutionChoice]] | None = None,
    cost_weights: dict[str, float] | None = None,
) -> list[RankedRequest]:
    """Requests ranked by VOI (highest first; ties: lower cost weight, then request id and params).

    Zero-value requests are listed too, after the others, with a reason that says so.
    """
    values = request_values(state, items, current, choices_by_item, cost_weights)
    values.sort(
        key=lambda value: (-value.voi, value.cost_weight, value.request_id, canonical_json(value.request_params))
    )
    return [
        RankedRequest(
            rank=rank,
            request_id=value.request_id,
            title=request_catalog()[value.request_id].title,
            params=value.request_params,
            item_ids=value.item_ids,
            voi=value.voi,
            p_change=value.p_change,
            cost_weight=value.cost_weight,
            options_summary=options_summary(value),
            reason=reason_sentence(value),
        )
        for rank, value in enumerate(values, start=1)
    ]


def nonzero_requests(ranked: Sequence[RankedRequest]) -> list[RankedRequest]:
    """The ranked requests with a VOI above zero (the stopping rule: none left means stop asking)."""
    return [request for request in ranked if request.voi > 0]
