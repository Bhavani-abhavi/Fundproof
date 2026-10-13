"""The default targets of the path-to-approval search, each with where it comes from. All three are editable.

1. P(DSCR >= 1.25 in year 2) >= 0.80. The 1.25 is the NSP-4.1 projected-DSCR minimum of the SYNTHETIC startup
   program, which is the NBC-4.2 term-loan minimum of the SYNTHETIC Northwind guideline; it is imported from
   agent/policy_credit.py (through feasibility.assumptions.default_dscr_target), never retyped.
2. P(stressed DSCR >= 1.00 in year 2) >= 0.80, with the NSP-4.2 lender revenue stress applied to every simulated
   path: revenue 20% lower and NOI lower by the same dollars. The share and the 1.00 are imported from
   agent/policy_credit.py (NBC-4.5, applied to startup loans by NSP-4.2).
3. P(cash < 0 within 24 months) <= 0.10.

Why the stress target exists: the lender's revenue stress removes 20% of revenue from NOI dollar for dollar, so a
plan passes it only when NOI >= debt service + 20% of revenue. For a business that keeps, say, 30 cents of NOI per
dollar of revenue, that means a DSCR of about 3, far above 1.25. The startup program says the stress, not the
1.25 minimum, is the binding test (corpus/programs/northwind_startup_program.md NSP-4.2). A path that only reached
1.25 would show the founder a plan the lender's own stress test still flags.

The probability levels (0.80, 0.80, 0.10) are design choices written in the build plan (Prompt S5 and the lead
engineer's decision of 2026-10-13), not lender rules: the SYNTHETIC program tests one base case, not a share of
simulated futures (NSP-4.4 records simulated probabilities as information only).
"""

from uwcopilot.agent.policy_credit import REVENUE_STRESS_MIN_DSCR, REVENUE_STRESS_SHARE
from uwcopilot.studio.feasibility.assumptions import default_dscr_target
from uwcopilot.studio.path_to_approval.models import Target

P_DSCR_MIN = 0.80
P_STRESSED_MIN = 0.80
P_CASH_NEGATIVE_MAX = 0.10
CASH_MONTHS = 24
PLAN_SOURCE = (
    "design choice of the build plan (docs/studio/founder_studio_build_prompts.md Prompt S5 and the lead engineer's "
    "decision of 2026-10-13); editable; not a lender rule"
)
STRESS_SOURCE = (
    "NSP-4.2 of the SYNTHETIC startup program (corpus/programs/northwind_startup_program.md), which applies the "
    "NBC-4.5 revenue stress of the SYNTHETIC Northwind guideline; constants imported from agent/policy_credit.py"
)
TARGET_NAMES: tuple[str, ...] = ("dscr_year2", "stressed_dscr_year2", "cash_negative")


def default_target() -> Target:
    """The three default targets with their sources (see the module docstring)."""
    dscr = default_dscr_target()
    return Target(
        dscr_min=dscr.value,
        p_dscr_min=P_DSCR_MIN,
        stress_revenue_share=REVENUE_STRESS_SHARE,
        stressed_dscr_min=REVENUE_STRESS_MIN_DSCR,
        p_stressed_min=P_STRESSED_MIN,
        p_cash_negative_max=P_CASH_NEGATIVE_MAX,
        cash_months=CASH_MONTHS,
        sources={
            "dscr_min": f"NSP-4.1 of the SYNTHETIC startup program = {dscr.source}",
            "stress_revenue_share": STRESS_SOURCE,
            "stressed_dscr_min": STRESS_SOURCE,
            "p_dscr_min": PLAN_SOURCE,
            "p_stressed_min": PLAN_SOURCE,
            "p_cash_negative_max": PLAN_SOURCE,
            "cash_months": "the S3 simulator's cash check (P(cash < 0 within 24 months)); " + PLAN_SOURCE,
        },
    )


def custom_target(**changes: float | int) -> Target:
    """The default target with some fields edited by the user (each edited source says so)."""
    base = default_target()
    editable = set(Target.model_fields) - {"sources", "year"}
    unknown = sorted(set(changes) - editable)
    if unknown:
        raise ValueError(f"cannot edit {unknown}; editable: {sorted(editable)}")
    sources = dict(base.sources)
    for name in changes:
        sources[name] = "edited by the user"
    return Target.model_validate({**base.model_dump(), **changes, "sources": sources})


def describe(target: Target) -> list[str]:
    """The three targets in plain words, in TARGET_NAMES order."""
    return [
        f"P(year-2 DSCR >= {target.dscr_min:.2f}) at least {target.p_dscr_min:.2f}",
        f"P(year-2 DSCR >= {target.stressed_dscr_min:.2f} after the lender's {target.stress_revenue_share:.0%} revenue "
        f"stress) at least {target.p_stressed_min:.2f}",
        f"P(cash < 0 within {target.cash_months} months) at most {target.p_cash_negative_max:.2f}",
    ]
