"""The human approval checkpoint: what the reviewer sees, and which decisions are accepted.

The graph always stops here (LangGraph interrupt) and only a human's resume value can finish the run. There is no
code path that approves or declines without one: not a clean file, not a "pre-approved" note, not a memo.

validate_decision() refuses, with reasons, a decision that the guidelines do not allow:
- schema: action, reviewer_id, role and a rationale of at least 10 characters are required; unknown fields are
  rejected;
- only a senior_underwriter may decline (NBC-7.2, HM-9.2), and a decline-for-review file must be reviewed by a
  senior_underwriter whatever the action;
- nothing can be approved or declined while a blocking referral is open (sanctions potential match, embedded
  instruction, document integrity, ownership/KYC): the file can only be referred or sent back for information
  (NBC-6.3, NBC-7.3, HM-8.2, HM-9.3);
- a file in request-more-information status cannot be approved (NBC-3.1, HM-3.1);
- approval authority by role: underwriter up to $150,000 exposure / $5,000,000 TIV, senior underwriter up to
  $500,000 / $15,000,000 (NBC-7.2, HM-9.2); above that the file must be referred;
- approving against a refer or decline-for-review recommendation is a policy exception: senior_underwriter only;
- an approval or decline that differs from the recommendation is an override and needs an override_reason from the
  approved list (NBC-8.2, HM-10.2). Referring or requesting information is never an override.
A refused decision is recorded and the checkpoint asks again.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from uwcopilot.agent.applications import CreditApplication, InsuranceApplication
from uwcopilot.agent.insurance_calculations import schedule_tiv
from uwcopilot.agent.policy import PolicyResult, blocking_functions_in, money
from uwcopilot.agent.vocab import FAKE_LLM_STAMP, MATCHING_ACTION, REVIEWER_ACTIONS, ReviewerAction, ReviewerRole

# Approval limits by role: credit uses the requested exposure (NBC-7.2), insurance the account TIV (HM-9.2).
APPROVAL_LIMITS: dict[str, dict[str, float]] = {
    "credit": {"underwriter": 150_000, "senior_underwriter": 500_000},
    "insurance": {"underwriter": 5_000_000, "senior_underwriter": 15_000_000},
}
APPROVED_OVERRIDE_REASONS: tuple[str, ...] = (
    "documented_compensating_factor",
    "data_error_in_model_input",
    "information_not_available_to_model",
    "policy_exception_approved",
    "referral_resolved_in_writing",
)


class HumanDecision(BaseModel):
    """What a reviewer submits at the checkpoint."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: ReviewerAction
    reviewer_id: str = Field(pattern=r"^[A-Za-z0-9_.@-]{1,64}$")
    role: ReviewerRole
    rationale: str = Field(min_length=10, max_length=2000)
    override_reason: str | None = None

    @field_validator("override_reason")
    @classmethod
    def override_reason_known(cls, value: str | None) -> str | None:
        """Only reasons from the approved override list are accepted."""
        if value is not None and value not in APPROVED_OVERRIDE_REASONS:
            raise ValueError(f"override_reason must be one of {list(APPROVED_OVERRIDE_REASONS)}")
        return value


def authority_exposure(application: CreditApplication | InsuranceApplication) -> float:
    """The amount approval authority is measured against: requested amount (credit) or schedule TIV (insurance)."""
    if isinstance(application, CreditApplication):
        return application.requested_amount
    return schedule_tiv(application)


def is_override(action: str, recommendation: str) -> bool:
    """True when a final decision (approve or decline) differs from the recommendation (NBC-8.2, HM-10.2).

    Referring a file or asking for more information is not a final decision, so it is never an override.
    """
    return action in ("approve", "decline") and MATCHING_ACTION[recommendation] != action


def validate_decision(
    raw: Any, policy: PolicyResult, application: CreditApplication | InsuranceApplication
) -> tuple[HumanDecision | None, list[str]]:
    """Return (decision, []) when the decision is allowed, or (None, reasons) when it is refused."""
    try:
        decision = HumanDecision.model_validate(raw)
    except ValidationError as error:
        messages = [
            f"{'.'.join(str(part) for part in item['loc']) or 'decision'}: {item['msg']}" for item in error.errors()
        ]
        return None, messages

    errors: list[str] = []
    recommendation = policy.recommendation
    senior = decision.role == "senior_underwriter"
    blocking = blocking_functions_in(policy.referral_targets)
    if recommendation == "decline_for_review" and not senior:
        errors.append("a decline-for-review file must be reviewed by a senior_underwriter")
    if decision.action == "decline" and not senior:
        errors.append("only a senior_underwriter may decline (NBC-7.2 / HM-9.2)")
    if decision.action in ("approve", "decline") and policy.blocking_open:
        errors.append(
            f"cannot {decision.action} while a referral to {', '.join(blocking) or 'a blocking function'} is open; "
            "refer the file or request more information"
        )
    if decision.action == "approve":
        if recommendation == "request_more_information":
            errors.append("cannot approve a file that is missing required documents or data")
        limit = APPROVAL_LIMITS[policy.line][decision.role]
        exposure = authority_exposure(application)
        if exposure > limit:
            errors.append(
                f"approval authority exceeded: {decision.role} may approve up to {money(limit)}, this file is "
                f"{money(exposure)}; refer it to a higher authority"
            )
        if recommendation in ("refer", "decline_for_review") and not senior:
            errors.append("approving against a refer or decline-for-review recommendation needs a senior_underwriter")
    if is_override(decision.action, recommendation) and decision.override_reason is None:
        errors.append(f"this action overrides the recommendation {recommendation!r}; give an override_reason")
    if errors:
        return None, errors
    return decision, []


def allowed_actions(policy: PolicyResult) -> list[str]:
    """Actions that some reviewer role could take on this file (the UI can grey out the rest)."""
    actions = []
    for action in REVIEWER_ACTIONS:
        if action in ("approve", "decline") and policy.blocking_open:
            continue
        if action == "approve" and policy.recommendation == "request_more_information":
            continue
        actions.append(action)
    return actions


def final_status(action: str) -> str:
    """The file status a reviewer's action leads to."""
    return {
        "approve": "approved_by_human",
        "decline": "declined_by_human",
        "refer": "referred_by_human",
        "request_more_information": "more_information_requested_by_human",
    }[action]


RESUME_INSTRUCTIONS = (
    "Nothing is approved or declined until a human decides. Resume with "
    '{"action": "approve" | "decline" | "refer" | "request_more_information", "reviewer_id": "...", '
    '"role": "underwriter" | "senior_underwriter", "rationale": "...", "override_reason": null}.'
)


def build_review_request(
    state: dict[str, Any], policy: PolicyResult, application: CreditApplication | InsuranceApplication
) -> dict[str, Any]:
    """The payload shown to the reviewer at the interrupt: recommendation, memo, escalations and limits."""
    memo = state.get("memo")
    llm_calls = state.get("llm_calls") or []
    fake = any(call.get("fake") for call in llm_calls)
    return {
        "application_id": policy.application_id,
        "line": policy.line,
        "data_label": "SYNTHETIC",
        "reference_recommendation": policy.recommendation,
        "memo_recommendation": memo.get("recommendation") if memo else None,
        "memo": memo,
        "memo_status": state.get("memo_status"),
        "memo_check": state.get("memo_check"),
        "escalations": [finding.model_dump() for finding in policy.findings],
        "referral_targets": policy.referral_targets,
        "review_queue": policy.review_queue,
        "blocking_open": policy.blocking_open,
        "required_reviewer_role": policy.required_reviewer_role,
        "authority_exposure": authority_exposure(application),
        "approval_limits": APPROVAL_LIMITS[policy.line],
        "allowed_actions": allowed_actions(policy),
        "tool_results": [
            {key: result.get(key) for key in ("result_id", "tool", "purpose", "party_id", "status", "error")}
            for result in state.get("tool_results") or []
        ],
        "tool_rejections": state.get("tool_rejections") or [],
        "injection_hits": state.get("injection_hits") or [],
        "not_evaluated": [item.model_dump() for item in policy.not_evaluated],
        "llm": {"calls": len(llm_calls), "fake": fake},
        "stamp": FAKE_LLM_STAMP if fake else None,
        "previous_rejections": state.get("rejected_decisions") or [],
        "instructions": RESUME_INSTRUCTIONS,
    }
