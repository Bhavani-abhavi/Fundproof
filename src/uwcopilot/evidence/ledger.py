"""The evidence ledger (Phase 4, SIG step 3): every claim a recommendation or a founder plan depends on.

One ledger serves both sides of the merged project (lender copilot and Founder Studio). It is not duplicated.

A Claim is one value with its status and provenance. Six statuses:
- VERIFIED: extracted from a document (or a public data row) and consistent with every other source, or a
  deterministic calculation over such values;
- MODEL_ESTIMATE: produced by a risk model or simulator, with the model name, version and its uncertainty;
- CONFLICTING: two or more sources disagree; conflict_ids links the contradictions (CT-xx) or claims involved;
- UNKNOWN: required but not present; reason says why. A value filled in for a model is marked imputed, with the
  imputation method; nothing is imputed silently;
- USER_STATED (Founder Studio, and any applicant figure no document supports): the applicant's own figure;
- ASSUMPTION (Founder Studio): a labeled planning assumption, with its range when it is a number and its source or
  the word "unsourced".
Every claim has id, label, value, unit, display (a renderer format), status, provenance (kind, ref, document
citations, model name and version, source, metro and snapshot date for Studio data) and inputs_sha256 (SHA-256 of
agent.applications.canonical_json(inputs)).

An Assertion is one rule outcome: result True (passed), False (finding) or None (not evaluated), with the rule's
trigger id, guideline section and the three phrases the memo renderer prints.

Ledger satisfies the renderer protocol of uwcopilot.provenance.renderer (LedgerLike: claim(), assertion(),
claim_values(); ClaimLike.provenance_note(); AssertionLike phrases), so {{claim:ID}} / {{assert:ID}} slots fill
from it. Two ways to build one:
- build_ledger(state, policy_result, extraction, provenance, contradictions, estimator) for a lender case;
- Ledger.builder().add_claim(...).add_assertion(...).build() for anything else (Studio packages), with the helper
  constructors user_stated_claim, assumption_claim and unknown_claim.

The Phase 1 models do not apply to these SYNTHETIC commercial cases, so model PD and expected loss are UNKNOWN with
the reason state.MODEL_NOT_EVALUATED_REASON. A real model can be plugged in through the ModelEstimator protocol;
tests use a fake estimator labeled fake.

Calculations stay deterministic: SR 26-2 / OCC Bulletin 2026-13 (April 17, 2026) excludes "simple arithmetic
calculations" and "deterministic rule-based processes" from its model definition. Applicability depends on the
institution and its risk profile; this is a demonstration, not a compliance program.
"""

import re
from typing import Any, Protocol

from pydantic import Field

from uwcopilot.agent.applications import AnyApplication, CreditApplication, InsuranceApplication
from uwcopilot.agent.policy import DOCUMENT_RULE_PREFIX, Calculation, PolicyResult
from uwcopilot.agent.policy_credit import required_documents as credit_required_documents
from uwcopilot.agent.policy_credit import stress_applies
from uwcopilot.agent.policy_insurance import coverage_lines
from uwcopilot.agent.policy_insurance import required_documents as insurance_required_documents
from uwcopilot.agent.policy_startup import required_startup_documents
from uwcopilot.docintel.check_models import CHECK_TITLES
from uwcopilot.docintel.contradictions import Contradiction, ContradictionReport
from uwcopilot.docintel.levers import LeverError, parse_path, walk
from uwcopilot.docintel.schemas import PacketExtraction
from uwcopilot.evidence import ledger_tables as tables
from uwcopilot.evidence.ledger_models import (
    CLAIM_STATUSES,
    ID_PATTERN,
    KNOWN_UNITS,
    OPEN_STATUSES,
    SYNTHETIC,
    UNIT_PATTERN,
    UNSOURCED,
    Assertion,
    Citation,
    Claim,
    ClaimStatus,
    ClaimValue,
    InputValue,
    Ledger,
    LedgerBuilder,
    LedgerModel,
    Provenance,
    ProvenanceKind,
    ValueRange,
    assumption_claim,
    inputs_hash,
    unknown_claim,
    user_stated_claim,
)
from uwcopilot.evidence.state import (
    CLEAN_SCREENING_ASSUMPTION,
    MODEL_NOT_EVALUATED_REASON,
    CaseState,
    application_of,
    evaluate,
    screening_assumed,
)

MODEL_PD_CLAIM = "MODEL-pd"
MODEL_EXPECTED_LOSS_CLAIM = "MODEL-expected_loss"
SCREENING_ASSUMPTION_CLAIM = "ASSUME-screening_clear"
NOT_COMPUTABLE_REASON = "not computable from its inputs (for example a zero denominator)"
IMPUTED_DEFAULT_METHOD = "default written by docintel.to_application (the application model has no room for None)"
CITATION_NOT_RECORDED = "not recorded by docintel.to_application"
FEATURE_ID_UNSAFE = re.compile(r"[^A-Za-z0-9_.\-]")

# --------------------------------------------------------------------------------------------------------------
# Model estimates (a clean path for a real model; none applies to these SYNTHETIC commercial cases)
# --------------------------------------------------------------------------------------------------------------


class ModelEstimate(LedgerModel):
    """One model output for the ledger, with the features the model lacked or had imputed."""

    claim_id: str = Field(pattern=ID_PATTERN)
    label: str
    value: float
    unit: str = Field(pattern=UNIT_PATTERN)
    display: str
    model_name: str
    model_version: str
    uncertainty: str
    inputs: dict[str, InputValue] = Field(default_factory=dict)
    unknown_features: tuple[str, ...] = ()
    imputed_features: dict[str, str] = Field(default_factory=dict)
    is_fake: bool = False


class ModelEstimator(Protocol):
    """Anything that can score a validated application for the ledger."""

    def estimate(self, application: AnyApplication) -> list[ModelEstimate]:
        """Model outputs for the application (an empty list when the model does not apply)."""
        ...


def feature_claim_id(name: str) -> str:
    """'FEATURE-<name>' with characters outside the slot grammar replaced by '_'."""
    return "FEATURE-" + FEATURE_ID_UNSAFE.sub("_", name)


def estimate_claims(estimate: ModelEstimate) -> list[Claim]:
    """The MODEL_ESTIMATE claim, plus one UNKNOWN claim per missing or imputed model feature."""
    label = estimate.label + (" (FAKE estimator, test only)" if estimate.is_fake else "")
    details: dict[str, InputValue] = {"fake": True} if estimate.is_fake else {}
    claims = [
        Claim(
            id=estimate.claim_id,
            label=label,
            value=estimate.value,
            unit=estimate.unit,
            display=estimate.display,
            status="MODEL_ESTIMATE",
            provenance=Provenance(
                kind="model",
                ref=estimate.model_name,
                model_name=estimate.model_name,
                model_version=estimate.model_version,
                details=details,
            ),
            inputs=dict(estimate.inputs),
            uncertainty=estimate.uncertainty,
        )
    ]
    for name in estimate.unknown_features:
        claims.append(
            unknown_claim(
                feature_claim_id(name),
                f"Model feature {name}",
                "NONE",
                "text",
                f"needed by {estimate.model_name} but not available in this case; not imputed",
                kind="model",
                ref=estimate.model_name,
            )
        )
    for name, method in estimate.imputed_features.items():
        claims.append(
            Claim(
                id=feature_claim_id(name),
                label=f"Model feature {name}",
                value=None,
                unit="NONE",
                display="text",
                status="UNKNOWN",
                provenance=Provenance(kind="model", ref=estimate.model_name),
                reason=f"needed by {estimate.model_name} but not available in this case",
                imputed=True,
                imputation_method=method,
            )
        )
    return claims


def default_model_claims(line: str) -> list[Claim]:
    """The model claims of a line, UNKNOWN with the honesty reason (no Phase 1 model applies)."""
    if line == "insurance":
        return [
            unknown_claim(
                MODEL_EXPECTED_LOSS_CLAIM,
                "Model expected loss per policy year",
                "USD",
                "usd2",
                MODEL_NOT_EVALUATED_REASON,
                kind="model",
            )
        ]
    return [
        unknown_claim(
            MODEL_PD_CLAIM, "Model probability of default", "PROBABILITY", "ratio2", MODEL_NOT_EVALUATED_REASON, "model"
        )
    ]


# --------------------------------------------------------------------------------------------------------------
# build_ledger: a lender case
# --------------------------------------------------------------------------------------------------------------


def citations_for(path: str, provenance: dict[str, list[dict[str, Any]]]) -> tuple[Citation, ...]:
    """Citations of an application path, and of every path below it ("bank_statements.months" covers months[i])."""
    found: list[Citation] = []
    for key, sources in provenance.items():
        if key == path or key.startswith(path + ".") or key.startswith(path + "["):
            for source in sources:
                citation = Citation(doc_id=source.get("doc_id"), page=source.get("page"), line=source.get("line"))
                if citation not in found:
                    found.append(citation)
    return tuple(found)


def path_value(application: dict[str, Any], path: str) -> Any:
    """The value at an application path (None when the path does not exist)."""
    try:
        return walk(application, parse_path(path), path)
    except LeverError:
        return None


def open_contradictions(contradictions: list[Contradiction], state: CaseState) -> list[Contradiction]:
    """Contradictions whose check a resolution has not cleared (state.resolved_checks)."""
    return [item for item in contradictions if item.check_id not in state.resolved_checks]


def conflicts_by_claim(contradictions: list[Contradiction]) -> dict[str, list[str]]:
    """Claim id -> contradiction ids that put it in doubt (the check's own claim plus CONFLICT_CLAIMS_BY_CHECK)."""
    result: dict[str, list[str]] = {}
    for item in contradictions:
        claim_ids = [f"CHK-{item.check_id}", *tables.CONFLICT_CLAIMS_BY_CHECK.get(item.check_id, ())]
        for claim_id in claim_ids:
            result.setdefault(claim_id, [])
            if item.contradiction_id not in result[claim_id]:
                result[claim_id].append(item.contradiction_id)
    return result


def unread_fact_claim(spec: tables.FactSpec, value: Any, unread: list[str], citations: tuple[Citation, ...]) -> Claim:
    """UNKNOWN: the extraction could not read the value. A default the application model needed (docintel
    to_application writes one where the agent's model has no room for None) is kept and marked imputed."""
    imputed = value is not None
    return Claim(
        id=spec.claim_id,
        label=spec.label,
        value=value,
        unit=spec.unit,
        display=spec.display,
        status="UNKNOWN",
        provenance=Provenance(kind="document", ref=spec.path, citations=citations),
        inputs={"application_path": spec.path},
        reason=f"the extraction could not read {', '.join(unread)}",
        imputed=imputed,
        imputation_method=IMPUTED_DEFAULT_METHOD if imputed else None,
    )


def fact_claim(
    spec: tables.FactSpec,
    value: Any,
    state: CaseState,
    provenance: dict[str, list[dict[str, Any]]],
    conflicts: dict[str, list[str]],
) -> Claim | None:
    """One application fact (None for a value that is legitimately absent, e.g. GL limits on a property quote).

    - UNKNOWN when the extraction could not read it (state.unknown_paths), imputed when a default was written;
    - CONFLICTING when an open contradiction involves it;
    - VERIFIED when it was read from the packet (with its citations; a packet value whose citation the mapping did
      not record is still VERIFIED, and its provenance says the citation was not recorded);
    - USER_STATED when the state has no document provenance at all (an application typed by the applicant).
    """
    citations = citations_for(spec.path, provenance)
    unread = [path for path in state.unknown_paths if path == spec.path]
    if unread:
        return unread_fact_claim(spec, value, unread, citations)
    if value is None:
        return None
    details: dict[str, InputValue] = {}
    if spec.claim_id in conflicts:
        status = "CONFLICTING"
    elif citations or provenance:
        status = "VERIFIED"
    else:
        status = "USER_STATED"
    if provenance and not citations:
        details["citations"] = CITATION_NOT_RECORDED
    kind = "document" if provenance else "application"
    return Claim(
        id=spec.claim_id,
        label=spec.label,
        value=value,
        unit=spec.unit,
        display=spec.display,
        status=status,
        provenance=Provenance(kind=kind, ref=spec.path, citations=citations, details=details),
        inputs={"application_path": spec.path},
        conflict_ids=tuple(conflicts.get(spec.claim_id, ())),
    )


def fact_claims(
    state: CaseState, provenance: dict[str, list[dict[str, Any]]], conflicts: dict[str, list[str]], line: str
) -> list[Claim]:
    """Every application fact of the line, then the list counts (statement months, loss-run years, locations)."""
    claims = []
    for spec in tables.FACTS_BY_LINE.get(line, ()):
        claim = fact_claim(spec, path_value(state.application, spec.path), state, provenance, conflicts)
        if claim is not None:
            claims.append(claim)
    for spec in tables.COUNT_FACTS_BY_LINE.get(line, ()):
        items = path_value(state.application, spec.path)
        count = len(items) if isinstance(items, list) else None
        claim = fact_claim(spec, count, state, provenance, conflicts)
        if claim is not None:
            claims.append(claim)
    return claims


def required_documents_for(application: AnyApplication) -> list[str]:
    """The checklist the reference policy applies to this application."""
    if isinstance(application, CreditApplication):
        return credit_required_documents(application)
    if isinstance(application, InsuranceApplication):
        return insurance_required_documents(application)
    return required_startup_documents(application)


def missing_document_claims(application: AnyApplication) -> list[Claim]:
    """One UNKNOWN claim "DOC-<code>" per required document that is not in the file."""
    provided = set(application.documents_provided)
    return [
        unknown_claim(
            f"DOC-{code}",
            f"Required document {code}",
            "TEXT",
            "text",
            "required by the reference policy's document checklist but not in the file",
            kind="policy",
            ref=code,
        )
        for code in required_documents_for(application)
        if code not in provided
    ]


def check_claims(state: CaseState, conflicts: dict[str, list[str]]) -> list[Claim]:
    """One claim "CHK-<check_id>" per document check that ran (state.check_status, plus any check with a conflict).

    The claim is the check's outcome. CONFLICTING with an open contradiction; UNKNOWN when a gap check failed (a
    missing document, page, period or out-of-date loss run: GAP_CHECKS); otherwise VERIFIED with the value "pass",
    "fail" (a deterministic result the policy also evaluates from the application) or "resolved" (a resolution
    cleared it). A check that could not run (not_evaluated) creates no claim: the values it compares appear once or
    not at all, and a missing document or value already shows as a DOC-* or UNKNOWN fact claim.
    """
    check_ids = list(state.check_status)
    for claim_id in conflicts:
        check_id = claim_id.removeprefix("CHK-")
        if claim_id.startswith("CHK-") and check_id not in check_ids:
            check_ids.append(check_id)
    claims = []
    for check_id in check_ids:
        claim_id = f"CHK-{check_id}"
        status_text = state.check_status.get(check_id, "fail")
        reason = None
        if claim_id in conflicts:
            status = "CONFLICTING"
        elif check_id in state.resolved_checks:
            status, status_text = "VERIFIED", "resolved"
        elif status_text == "not_evaluated":
            continue
        elif status_text == "fail" and check_id in tables.GAP_CHECKS:
            status = "UNKNOWN"
            reason = "the check found a gap in the documents (a missing document, page or period)"
        else:
            status = "VERIFIED"
        claims.append(
            Claim(
                id=claim_id,
                label=CHECK_TITLES.get(check_id, check_id),
                value=None if status == "UNKNOWN" else status_text,
                unit="TEXT",
                display="text",
                status=status,
                provenance=Provenance(kind="document", ref=check_id),
                inputs={"check_id": check_id, "status": status_text},
                reason=reason,
                conflict_ids=tuple(conflicts.get(claim_id, ())),
            )
        )
    return claims


def calc_claim(calculation: Calculation) -> Claim:
    """A policy calculation: VERIFIED arithmetic, or UNKNOWN when its value could not be computed."""
    unit, display = tables.CALC_UNITS[calculation.unit]
    status = "VERIFIED" if calculation.value is not None else "UNKNOWN"
    return Claim(
        id=calculation.calc_id,
        label=calculation.label,
        value=calculation.value,
        unit=unit,
        display=display,
        status=status,
        provenance=Provenance(
            kind="calculation", ref=calculation.calc_id, details={"sections": ", ".join(calculation.section_ids)}
        ),
        inputs=dict(calculation.inputs),
        reason=None if calculation.value is not None else NOT_COMPUTABLE_REASON,
    )


def direct_dependencies(calculation: Calculation) -> list[str]:
    """Claim ids a calculation reads: through its input names, plus CALC_EXTRA_DEPENDENCIES."""
    found: list[str] = []
    for name in calculation.inputs:
        for claim_id in tables.CALC_INPUT_CLAIMS.get(name, ()):
            if claim_id not in found:
                found.append(claim_id)
    for claim_id in tables.CALC_EXTRA_DEPENDENCIES.get(calculation.calc_id, ()):
        if claim_id not in found:
            found.append(claim_id)
    return found


def conflicting_roots(
    claim_id: str, dependencies: dict[str, list[str]], conflicting: set[str], seen: set[str] | None = None
) -> list[str]:
    """The CONFLICTING claims a claim depends on, directly or through other calculations (sorted)."""
    visited = seen if seen is not None else set()
    roots: set[str] = set()
    for dependency in dependencies.get(claim_id, []):
        if dependency in visited:
            continue
        visited.add(dependency)
        if dependency in conflicting:
            roots.add(dependency)
        roots.update(conflicting_roots(dependency, dependencies, conflicting, visited))
    return sorted(roots)


def calculation_claims(policy_result: PolicyResult, conflicting: set[str]) -> list[Claim]:
    """Every policy calculation as a claim; a calculation over a CONFLICTING claim lists it in
    depends_on_conflicting (the arithmetic itself stays VERIFIED)."""
    dependencies = {item.calc_id: direct_dependencies(item) for item in policy_result.calculations}
    claims = []
    for calculation in policy_result.calculations:
        claim = calc_claim(calculation)
        roots = conflicting_roots(calculation.calc_id, dependencies, conflicting)
        if roots:
            claim = claim.model_copy(update={"depends_on_conflicting": tuple(roots)})
        claims.append(claim)
    return claims


def not_evaluated_specs(rule: str, application: AnyApplication) -> list[tuple[str, str, str]]:
    """(calc id, label, CalcUnit) of the calculations a not-evaluated rule left out.

    Stress tests count only when NBC-4.5 applies to the request (the rate stress only for a variable rate), and
    loss ratios only for the lines being quoted.
    """
    if rule == "stress_tests" and isinstance(application, CreditApplication):
        if not stress_applies(application):
            return []
        specs = list(tables.NOT_EVALUATED_CLAIMS[rule])
        if application.rate_type == "variable":
            specs.append(tables.RATE_STRESS_CALC)
        return specs
    if rule in tables.NOT_EVALUATED_CLAIMS:
        return list(tables.NOT_EVALUATED_CLAIMS[rule])
    if not isinstance(application, InsuranceApplication):
        return []
    if rule == "loss_ratios_and_frequency":
        specs = [
            (tables.LOSS_RATIO_CALC.format(line=line), f"5-year loss ratio, {line.replace('_', ' ')}", "pct")
            for line in coverage_lines(application)
        ]
        return [*specs, tables.CLAIM_FREQUENCY_CALC]
    if rule.startswith("loss_ratio_"):
        line = rule.removeprefix("loss_ratio_")
        return [(tables.LOSS_RATIO_CALC.format(line=line), f"5-year loss ratio, {line.replace('_', ' ')}", "pct")]
    return []


def not_evaluated_claims(policy_result: PolicyResult, application: AnyApplication, existing: set[str]) -> list[Claim]:
    """UNKNOWN claims for the calculations the policy could not run (reason: the policy's own reason)."""
    claims: list[Claim] = []
    for item in policy_result.not_evaluated:
        for calc_id, label, calc_unit in not_evaluated_specs(item.rule, application):
            if calc_id in existing or any(claim.id == calc_id for claim in claims):
                continue
            unit, display = tables.CALC_UNITS[calc_unit]
            claims.append(unknown_claim(calc_id, label, unit, display, item.reason, kind="calculation"))
    return claims


def screening_assumption_claim() -> Claim:
    """The clean screening assumption as an ASSUMPTION claim (no screening outcomes were given)."""
    return Claim(
        id=SCREENING_ASSUMPTION_CLAIM,
        label="Sanctions screening outcome (assumed)",
        value="clear",
        unit="TEXT",
        display="text",
        status="ASSUMPTION",
        provenance=Provenance(
            kind="assumption", ref="screening", source=UNSOURCED, details={"note": CLEAN_SCREENING_ASSUMPTION}
        ),
        inputs={"assumption": CLEAN_SCREENING_ASSUMPTION},
    )


def rule_phrases(rule: str) -> tuple[str, str, str, str]:
    """(expression, phrase_true, phrase_false, phrase_unknown) for a rule (TOPIC_PHRASES, else the defaults)."""
    if rule.startswith(DOCUMENT_RULE_PREFIX):
        topic = tables.DOCUMENT_RULE_TOPIC
    else:
        topic = tables.RULE_TOPIC.get(rule, "")
    if topic:
        return tables.TOPIC_PHRASES[topic]
    return (f"rule {rule}", *tables.DEFAULT_PHRASES)


def assertion_id(rule: str, used: set[str]) -> str:
    """'A-<rule>', with '-2', '-3' ... when the rule occurs more than once."""
    base = f"A-{rule}"
    candidate = base
    number = 2
    while candidate in used:
        candidate = f"{base}-{number}"
        number += 1
    used.add(candidate)
    return candidate


def policy_assertions(policy_result: PolicyResult) -> list[Assertion]:
    """Findings (result False), passed checks (True) and not-evaluated rules (None), in that order."""
    used: set[str] = set()
    assertions = []
    for finding in policy_result.findings:
        expression, phrase_true, phrase_false, phrase_unknown = rule_phrases(finding.rule)
        assertions.append(
            Assertion(
                id=assertion_id(finding.rule, used),
                rule=finding.rule,
                rule_ref=finding.trigger_ids[0] if finding.trigger_ids else None,
                section_id=finding.section_ids[0] if finding.section_ids else None,
                section_ids=tuple(finding.section_ids),
                expression=expression,
                result=False,
                phrase_true=phrase_true,
                phrase_false=phrase_false,
                phrase_unknown=phrase_unknown,
                detail=finding.detail,
                trigger_ids=tuple(finding.trigger_ids),
                evidence_ids=tuple(finding.evidence_ids),
                outcome=finding.outcome,
                blocking=finding.blocking,
            )
        )
    for check in policy_result.checks_passed:
        expression, phrase_true, phrase_false, phrase_unknown = rule_phrases(check.rule)
        assertions.append(
            Assertion(
                id=assertion_id(check.rule, used),
                rule=check.rule,
                section_id=check.section_ids[0] if check.section_ids else None,
                section_ids=tuple(check.section_ids),
                expression=expression,
                result=True,
                phrase_true=phrase_true,
                phrase_false=phrase_false,
                phrase_unknown=phrase_unknown,
                detail=check.detail,
                evidence_ids=tuple(check.evidence_ids),
            )
        )
    for item in policy_result.not_evaluated:
        expression, phrase_true, phrase_false, phrase_unknown = rule_phrases(item.rule)
        assertions.append(
            Assertion(
                id=assertion_id(item.rule, used),
                rule=item.rule,
                section_id=item.section_ids[0] if item.section_ids else None,
                section_ids=tuple(item.section_ids),
                expression=expression,
                result=None,
                phrase_true=phrase_true,
                phrase_false=phrase_false,
                phrase_unknown=phrase_unknown,
                detail=item.reason,
            )
        )
    return assertions


def model_claims(application: AnyApplication, estimator: ModelEstimator | None) -> list[Claim]:
    """The line's model claims: UNKNOWN with the honesty reason, replaced by an estimator's output when given."""
    defaults = default_model_claims(application.line)
    if estimator is None:
        return defaults
    estimated: list[Claim] = []
    for estimate in estimator.estimate(application):
        estimated.extend(estimate_claims(estimate))
    estimated_ids = {claim.id for claim in estimated}
    return [claim for claim in defaults if claim.id not in estimated_ids] + estimated


def contradictions_of(contradictions: ContradictionReport | list[Contradiction] | None) -> list[Contradiction]:
    """A contradiction list from a report, a list or None."""
    if contradictions is None:
        return []
    if isinstance(contradictions, ContradictionReport):
        return list(contradictions.contradictions)
    return list(contradictions)


def build_ledger(
    state: CaseState,
    policy_result: PolicyResult | None = None,
    extraction: PacketExtraction | None = None,
    provenance: dict[str, list[dict[str, Any]]] | None = None,
    contradictions: ContradictionReport | list[Contradiction] | None = None,
    estimator: ModelEstimator | None = None,
) -> Ledger:
    """The evidence ledger of a lender case.

    policy_result defaults to evaluate(state); provenance to state.field_provenance; extraction is optional and only
    names the subject when the state has no case id (citations come from provenance and the contradictions).
    Claims, in order: application facts, missing required documents (DOC-*), document checks (CHK-*), policy
    calculations (CALC-*), calculations the policy could not run (UNKNOWN), model claims (MODEL-*, FEATURE-*) and
    the clean screening assumption when it is used. Assertions come from the policy result.
    """
    application = application_of(state)
    result = policy_result if policy_result is not None else evaluate(state)
    field_provenance = provenance if provenance is not None else state.field_provenance
    found = open_contradictions(contradictions_of(contradictions), state)
    conflicts = conflicts_by_claim(found)
    subject = state.case_id or (extraction.packet_id if extraction is not None else application.application_id)
    builder = Ledger.builder(subject_id=subject, line=application.line, data_label=state.data_label)
    claims = fact_claims(state, field_provenance, conflicts, application.line)
    claims.extend(missing_document_claims(application))
    claims.extend(check_claims(state, conflicts))
    conflicting = {claim.id for claim in claims if claim.status == "CONFLICTING"}
    claims.extend(calculation_claims(result, conflicting))
    claims.extend(not_evaluated_claims(result, application, {claim.id for claim in claims}))
    claims.extend(model_claims(application, estimator))
    if screening_assumed(state):
        claims.append(screening_assumption_claim())
    for claim in claims:
        builder.add_claim(claim)
    for assertion in policy_assertions(result):
        builder.add_assertion(assertion)
    return builder.build()


__all__ = [
    "CLAIM_STATUSES",
    "ID_PATTERN",
    "KNOWN_UNITS",
    "MODEL_EXPECTED_LOSS_CLAIM",
    "MODEL_PD_CLAIM",
    "OPEN_STATUSES",
    "SCREENING_ASSUMPTION_CLAIM",
    "SYNTHETIC",
    "UNIT_PATTERN",
    "UNSOURCED",
    "Assertion",
    "Citation",
    "Claim",
    "ClaimStatus",
    "ClaimValue",
    "InputValue",
    "Ledger",
    "LedgerBuilder",
    "LedgerModel",
    "ModelEstimate",
    "ModelEstimator",
    "Provenance",
    "ProvenanceKind",
    "ValueRange",
    "assumption_claim",
    "build_ledger",
    "inputs_hash",
    "unknown_claim",
    "user_stated_claim",
]
