"""Plain-language text for the path-to-approval result (Studio S5).

Every sentence is filled from numbers the search computed; nothing here adds a number of its own. The text never
says a loan will be approved: Studio prepares and explains, and lenders make their own decisions (rule C).
Probabilities are printed with three decimals (1,000 paths give steps of 0.001) and always carry the label
"simulated, based on stated assumptions" (rule E).
"""

from uwcopilot.agent.policy_startup import STARTUP_MAX_TERM_MONTHS, STARTUP_MIN_TERM_MONTHS
from uwcopilot.studio import SIMULATION_LABEL
from uwcopilot.studio.path_to_approval.models import AppliedChange, Outcome, PlanAmounts, PlanCase, Target
from uwcopilot.studio.path_to_approval.space import LeverSpace

NOT_AN_APPROVAL = (
    "Reaching these simulated targets does not mean a lender will approve the loan. Lenders make their own decisions."
)
# Plain words for the target names in Outcome.targets_missed (targets.TARGET_NAMES).
TARGET_WORDS: dict[str, str] = {
    "dscr_year2": "the year-2 DSCR target",
    "stressed_dscr_year2": "the lender's revenue-stress target",
    "cash_negative": "the cash target",
}


def dollars(value: float) -> str:
    """$1,234 for whole dollars, $1,234.56 otherwise."""
    if abs(value - round(value)) < 0.005:
        return f"${value:,.0f}"
    return f"${value:,.2f}"


def number(value: float) -> str:
    """A plain number without trailing zeros (6, 6.5, 1012.5)."""
    return f"{value:g}"


def amount_sentence(change: AppliedChange) -> str:
    """One dollar lever's change in words."""
    start, end = float(change.from_value), float(change.to_value)
    delta = dollars(abs(end - start))
    up = end > start
    if change.lever == "owner_equity":
        if up:
            return (
                f"Put in {delta} more of your own cash (owner equity {dollars(start)} -> {dollars(end)}) and borrow "
                "that much less. Only if you really have this cash: the lender checks equity against bank "
                "statements (NSP-3.1, SYNTHETIC program)."
            )
        return (
            f"Put in {delta} less of your own cash (owner equity {dollars(start)} -> {dollars(end)}) and borrow "
            "that much more."
        )
    if change.lever == "loan_amount":
        if up:
            return f"Borrow {delta} more (loan {dollars(start)} -> {dollars(end)}) and keep it as cash at opening."
        return (
            f"Borrow {delta} less (loan {dollars(start)} -> {dollars(end)}); your cash reserve at opening is {delta} "
            "smaller."
        )
    if change.lever == "build_out_cost":
        if up:
            return (
                f"Raise the build-out budget by {delta} ({dollars(start)} -> {dollars(end)}) and borrow that much more."
            )
        return (
            f"Bring the build-out and equipment cost down by {delta} ({dollars(start)} -> {dollars(end)}) and borrow "
            "that much less. Only if the same business can be built for less (a lower quote, used equipment); a "
            "smaller build-out that also cuts capacity is not modeled."
        )
    if up:
        return (
            f"Keep {delta} more cash in reserve at opening ({dollars(start)} -> {dollars(end)}), funded by your own "
            "cash."
        )
    return (
        f"Keep {delta} less cash in reserve at opening ({dollars(start)} -> {dollars(end)}) and put in that much "
        "less of your own cash."
    )


def site_sentence(change: AppliedChange, space: LeverSpace) -> str:
    """The site switch in words, with both demand estimates and what happens to the rent."""
    candidate = space.candidates[str(change.to_value)]
    current = space.case.site
    text = (
        f"Use your candidate site '{candidate.label}' instead of '{change.from_value}' (its most likely demand is "
        f"{number(round(candidate.site.mode, 1))} {space.demand_unit}, against {number(round(current.mode, 1))}; "
        f"status {candidate.site.status})."
    )
    if candidate.monthly_rent is not None:
        return text + f" Its rent quote is {dollars(candidate.monthly_rent)} a month."
    return text + " It has no rent quote, so the metro's rent range (ASSUMPTION) is used until you get one."


def change_sentence(change: AppliedChange, space: LeverSpace, after: PlanCase) -> str:
    """One change in words."""
    if change.lever in ("owner_equity", "loan_amount", "build_out_cost", "working_capital_reserve"):
        return amount_sentence(change)
    if change.lever == "price":
        typed = space.case.plan.price is not None
        start = dollars(float(change.from_value))
        if not typed:
            start = f"the profile's most likely {start}"
        return (
            f"Charge {dollars(float(change.to_value))} per unit ({space.demand_unit}) instead of {start}. Demand "
            "responds to price through the price elasticity, an ASSUMPTION range; test the price with customers "
            "before relying on it."
        )
    if change.lever == "days_open_per_week":
        hours_before = space.case.plan.paid_hours_per_month or 0.0
        hours_after = after.plan.paid_hours_per_month or 0.0
        return (
            f"Open {number(float(change.to_value))} days a week instead of {number(float(change.from_value))}; paid "
            f"staff hours change from {number(hours_before)} to {number(hours_after)} a month in proportion (a design "
            "choice)."
        )
    if change.lever == "loan_term_months":
        return (
            f"Ask the lender for a {int(float(change.to_value))}-month term instead of "
            f"{int(float(change.from_value))} months. The lender decides the term (the SYNTHETIC program allows "
            f"{STARTUP_MIN_TERM_MONTHS} to {STARTUP_MAX_TERM_MONTHS} months, NSP-8.1)."
        )
    return site_sentence(change, space)


def probability_lines(before: Outcome, after: Outcome, target: Target) -> list[str]:
    """Each target's probability before -> after, with the target."""
    return [
        f"chance that year-2 DSCR is at least {target.dscr_min:.2f}: {before.p_dscr_at_least_target_year2:.3f} -> "
        f"{after.p_dscr_at_least_target_year2:.3f} (target: at least {target.p_dscr_min:.2f})",
        f"chance that year-2 DSCR stays at least {target.stressed_dscr_min:.2f} under the lender's "
        f"{target.stress_revenue_share:.0%} revenue stress: {before.p_stressed_dscr_at_least_min_year2:.3f} -> "
        f"{after.p_stressed_dscr_at_least_min_year2:.3f} (target: at least {target.p_stressed_min:.2f})",
        f"chance that cash falls below zero within {target.cash_months} months: "
        f"{before.p_cash_negative_within_months:.3f} -> {after.p_cash_negative_within_months:.3f} "
        f"(target: at most {target.p_cash_negative_max:.2f})",
    ]


def ratio_text(value: float | None) -> str:
    """A ratio with two decimals, or 'none' when it has no denominator."""
    return "none" if value is None else f"{value:.2f}"


def base_case_line(before: Outcome, after: Outcome) -> str:
    """The lender's single-number checks on the year-2 median, before -> after."""
    return (
        "The lender's own check on the year-2 median (NSP-4.1 and NSP-4.2): projected DSCR "
        f"{ratio_text(before.projected_dscr_base_case)} -> {ratio_text(after.projected_dscr_base_case)}, "
        f"revenue-stressed DSCR {ratio_text(before.stressed_dscr_base_case)} -> "
        f"{ratio_text(after.stressed_dscr_base_case)}."
    )


def plan_line(amounts: PlanAmounts) -> str:
    """The plan's money after the change (sources = uses), with the owner's share of the project (NSP-3.1)."""
    checks = amounts.checks
    return (
        f"Plan after the change: owner equity {dollars(amounts.owner_equity)}, loan {dollars(amounts.loan_amount)} "
        f"over {amounts.loan_term_months} months, build-out {dollars(amounts.build_out_cost)}, cash reserve at "
        f"opening {dollars(amounts.cash_reserve_at_opening)}; equity injection {checks.equity_injection_pct:.1f}% of "
        f"the project (the SYNTHETIC program asks for at least {checks.equity_minimum_pct:.1f}%, NSP-3.1)."
    )


def explain_change_set(
    changes: list[AppliedChange],
    space: LeverSpace,
    after_case: PlanCase,
    before: Outcome,
    after: Outcome,
    plan_after: PlanAmounts,
    target: Target,
) -> str:
    """The whole explanation of one change-set."""
    parts = [change_sentence(change, space, after_case) for change in changes]
    parts.append(plan_line(plan_after))
    lines = "; ".join(probability_lines(before, after, target))
    parts.append(f"Result ({SIMULATION_LABEL}; {after.n_paths:,} simulated futures, seed {after.seed}): {lines}.")
    if after.meets_target:
        parts.append("All three targets are met.")
    else:
        parts.append(f"Still missed: {', '.join(TARGET_WORDS[name] for name in after.targets_missed)}.")
    parts.append(base_case_line(before, after))
    parts.append(NOT_AN_APPROVAL)
    return " ".join(parts)


def unchanged_text(case: PlanCase) -> str:
    """What every change-set leaves alone."""
    facts = case.facts
    debt = (
        f"your other debts ({len(facts.other_debt)} listed, {dollars(facts.other_debt_monthly_payment)} a month)"
        if facts.other_debt
        else "your other debts (none listed)"
    )
    verified = ""
    if facts.verified_fields:
        verified = f", and the figures you marked as verified ({', '.join(facts.verified_fields)})"
    return (
        f"Unchanged: your credit score, your experience, {debt}, the site's demand estimate, every assumption range "
        f"and the note rate{verified}."
    )


def status_message(status: str, n_found: int) -> str:
    """The one-line answer at the top of the result."""
    if status == "already_meets_target":
        return (
            "Your plan already meets the three targets (simulated, based on stated assumptions); no change is needed "
            "to reach them. " + NOT_AN_APPROVAL
        )
    if status == "paths_found":
        ways = "one way" if n_found == 1 else f"{n_found} ways"
        return (
            f"Found {ways} to reach the targets by changing your plan (simulated, based on stated assumptions), "
            "smallest first. " + NOT_AN_APPROVAL
        )
    return (
        "No path within your bounds: no change of one or two plan choices inside your bounds reaches all three "
        "targets (simulated, based on stated assumptions). The closest result is shown. Next steps are to widen a "
        "bound you can really widen, or to rethink the plan itself (site, size, concept)."
    )
