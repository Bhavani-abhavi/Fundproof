"""Ledger data models: Claim, Assertion, Ledger and the builder (Phase 4 evidence ledger, SIG step 3).

This module holds the shapes and their validation rules only; uwcopilot.evidence.ledger re-exports everything here
and adds build_ledger (a lender case). See ledger.py for the six claim statuses and how the ledger is used.

Validation rules (Claim.status_rules):
- inputs_sha256 is SHA-256 of agent.applications.canonical_json(inputs), computed when not given and checked;
- UNKNOWN needs a reason; it may carry a value only when imputed, and an imputed claim names its method;
- VERIFIED, USER_STATED and MODEL_ESTIMATE need a value;
- CONFLICTING links its conflicts (contradiction ids such as CT-01, or other claim ids);
- MODEL_ESTIMATE names the model and version and states its uncertainty ("not measured" is allowed);
- ASSUMPTION names its source or "unsourced", and a numeric assumption carries its range;
- display is one of the renderer's formats; ids fit the memo slot grammar {{claim:ID}}.
Ledger checks that ids are unique and that every depends_on_conflicting id is a CONFLICTING claim of the ledger.
"""

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from uwcopilot.agent.applications import canonical_json
from uwcopilot.provenance.renderer import DISPLAY_FORMATS

ClaimStatus = Literal["VERIFIED", "MODEL_ESTIMATE", "CONFLICTING", "UNKNOWN", "USER_STATED", "ASSUMPTION"]
CLAIM_STATUSES: tuple[str, ...] = ("VERIFIED", "MODEL_ESTIMATE", "CONFLICTING", "UNKNOWN", "USER_STATED", "ASSUMPTION")
OPEN_STATUSES: frozenset[str] = frozenset({"CONFLICTING", "UNKNOWN"})
ProvenanceKind = Literal[
    "document", "calculation", "model", "policy", "application", "user_input", "assumption", "public_data"
]
ClaimValue = float | int | str | bool | None
InputValue = float | int | str | bool | None
# Ids must fit the memo slot grammar {{claim:ID}} (provenance/memo_grammar.SLOT).
ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_.\-]*$"
UNIT_PATTERN = r"^[A-Z][A-Z0-9_]*$"
# Units used so far; unit is free text (UNIT_PATTERN) so Studio can add its own.
KNOWN_UNITS: tuple[str, ...] = (
    "USD",
    "USD_PER_SQFT",
    "PERCENT",
    "BPS",
    "RATIO",
    "MONTHS",
    "YEARS",
    "DAYS",
    "COUNT",
    "PROBABILITY",
    "DATE",
    "TEXT",
    "NONE",
)
UNSOURCED = "unsourced"
SYNTHETIC = "SYNTHETIC"
HASH_PREFIX_CHARS = 12


def inputs_hash(inputs: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a claim's inputs (agent.applications.canonical_json)."""
    return hashlib.sha256(canonical_json(inputs).encode("utf-8")).hexdigest()


class LedgerModel(BaseModel):
    """Base for ledger models: unknown fields are an error and objects are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Citation(LedgerModel):
    """Where a value is printed: document id, page and line (any part may be unknown)."""

    doc_id: str | None = None
    page: int | None = None
    line: int | None = None

    def text(self) -> str:
        """'D01 p3 L14' (unknown parts left out)."""
        parts = [self.doc_id or "document"]
        if self.page is not None:
            parts.append(f"p{self.page}")
        if self.line is not None:
            parts.append(f"L{self.line}")
        return " ".join(parts)


class ValueRange(LedgerModel):
    """The low and high end of an ASSUMPTION's range (same unit as the claim)."""

    low: float
    high: float

    @model_validator(mode="after")
    def ordered(self) -> "ValueRange":
        """low must not exceed high."""
        if self.low > self.high:
            raise ValueError(f"range low {self.low} is above high {self.high}")
        return self


class Provenance(LedgerModel):
    """Where a claim's value came from.

    kind: document (cited pages), calculation (a CALC-* formula), model (a risk model or simulator), policy (the
    reference policy's own state), application (an applicant-stated field), user_input (a founder's typed input),
    assumption (a profile or metro range, or a stated default), public_data (a public dataset row).
    ref: the calculation id, model name, application path, intake field or dataset id.
    source: publisher, document, date, URL and page/table for public data and sourced assumptions, or "unsourced".
    details: anything else a caller must keep (request date, response SHA-256, geography id, recorded_at ...).
    """

    kind: ProvenanceKind
    ref: str
    citations: tuple[Citation, ...] = ()
    model_name: str | None = None
    model_version: str | None = None
    source: str | None = None
    metro_id: str | None = None
    snapshot_date: str | None = None
    details: dict[str, InputValue] = Field(default_factory=dict)


class Claim(LedgerModel):
    """One value the recommendation (or a founder plan) depends on. See the module docstring for the statuses."""

    id: str = Field(pattern=ID_PATTERN)
    label: str = Field(min_length=1)
    value: ClaimValue
    unit: str = Field(pattern=UNIT_PATTERN)
    display: str
    status: ClaimStatus
    provenance: Provenance
    inputs: dict[str, InputValue] = Field(default_factory=dict)
    inputs_sha256: str = ""
    reason: str | None = None
    imputed: bool = False
    imputation_method: str | None = None
    conflict_ids: tuple[str, ...] = ()
    depends_on_conflicting: tuple[str, ...] = ()
    uncertainty: str | None = None
    value_range: ValueRange | None = None

    @field_validator("display")
    @classmethod
    def known_display(cls, display: str) -> str:
        """display must be one of the renderer's formats (provenance/renderer.DISPLAY_FORMATS)."""
        if display not in DISPLAY_FORMATS:
            raise ValueError(f"display {display!r} is not one of {DISPLAY_FORMATS}")
        return display

    @model_validator(mode="before")
    @classmethod
    def fill_inputs_hash(cls, data: Any) -> Any:
        """inputs_sha256 is computed from inputs when it is not given."""
        if isinstance(data, dict) and not data.get("inputs_sha256"):
            data = {**data, "inputs_sha256": inputs_hash(dict(data.get("inputs") or {}))}
        return data

    @model_validator(mode="after")
    def status_rules(self) -> "Claim":
        """Each status carries what it needs (documented in the module docstring)."""
        if self.inputs_sha256 != inputs_hash(self.inputs):
            raise ValueError(f"{self.id}: inputs_sha256 does not match the inputs")
        if self.imputed and not self.imputation_method:
            raise ValueError(f"{self.id}: an imputed claim must name its imputation method")
        if self.status == "UNKNOWN":
            if not self.reason:
                raise ValueError(f"{self.id}: an UNKNOWN claim needs a reason")
            if self.value is not None and not self.imputed:
                raise ValueError(f"{self.id}: an UNKNOWN claim with a value must be marked imputed")
        elif self.imputed:
            raise ValueError(f"{self.id}: only an UNKNOWN claim can be imputed")
        if self.status in ("VERIFIED", "USER_STATED", "MODEL_ESTIMATE") and self.value is None:
            raise ValueError(f"{self.id}: a {self.status} claim needs a value")
        if self.status == "CONFLICTING" and not self.conflict_ids:
            raise ValueError(f"{self.id}: a CONFLICTING claim must link its conflicts")
        if self.status == "MODEL_ESTIMATE":
            if not (self.provenance.model_name and self.provenance.model_version):
                raise ValueError(f"{self.id}: a MODEL_ESTIMATE needs the model name and version")
            if not self.uncertainty:
                raise ValueError(f"{self.id}: a MODEL_ESTIMATE needs its uncertainty (or 'not measured')")
        if self.status == "ASSUMPTION":
            if not self.provenance.source:
                raise ValueError(f"{self.id}: an ASSUMPTION needs its source or the word 'unsourced'")
            if isinstance(self.value, int | float) and not isinstance(self.value, bool) and self.value_range is None:
                raise ValueError(f"{self.id}: a numeric ASSUMPTION needs its range")
        return self

    def provenance_note(self) -> str:
        """One footnote line: where the value came from, the inputs hash, and any conflict or reason."""
        source = self.provenance
        parts = [f"{source.kind} {source.ref}"]
        if source.citations:
            parts.append("; ".join(citation.text() for citation in source.citations))
        if source.model_name:
            parts.append(f"model {source.model_name} {source.model_version or ''}".rstrip())
        if self.uncertainty:
            parts.append(f"uncertainty: {self.uncertainty}")
        if self.value_range is not None:
            parts.append(f"range {self.value_range.low:g} to {self.value_range.high:g}")
        if source.source:
            parts.append(f"source: {source.source}")
        if source.metro_id:
            parts.append(f"metro {source.metro_id}")
        if source.snapshot_date:
            parts.append(f"snapshot {source.snapshot_date}")
        parts.append(f"inputs sha256 {self.inputs_sha256[:HASH_PREFIX_CHARS]}")
        if self.conflict_ids:
            parts.append(f"conflicts: {', '.join(self.conflict_ids)}")
        if self.depends_on_conflicting:
            parts.append(f"depends on conflicting: {', '.join(self.depends_on_conflicting)}")
        if self.imputed:
            parts.append(f"imputed ({self.imputation_method})")
        if self.reason:
            parts.append(f"reason: {self.reason}")
        return ", ".join(parts)


class Assertion(LedgerModel):
    """One rule outcome: True (passed), False (a finding) or None (not evaluated), with its memo phrases."""

    id: str = Field(pattern=ID_PATTERN)
    rule: str
    rule_ref: str | None = None
    section_id: str | None = None
    section_ids: tuple[str, ...] = ()
    expression: str
    result: bool | None
    phrase_true: str
    phrase_false: str
    phrase_unknown: str
    detail: str = ""
    trigger_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    outcome: str | None = None
    blocking: bool = False


class Ledger(LedgerModel):
    """Claims and assertions of one case or package, by id (implements the renderer's LedgerLike)."""

    subject_id: str = ""
    line: str | None = None
    data_label: str = SYNTHETIC
    claims: tuple[Claim, ...] = ()
    assertions: tuple[Assertion, ...] = ()

    @model_validator(mode="after")
    def consistent(self) -> "Ledger":
        """Ids are unique, and every depends_on_conflicting id is a CONFLICTING claim of this ledger."""
        claim_ids = [claim.id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim ids must be unique")
        assertion_ids = [assertion.id for assertion in self.assertions]
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("assertion ids must be unique")
        conflicting = {claim.id for claim in self.claims if claim.status == "CONFLICTING"}
        for claim in self.claims:
            missing = [item for item in claim.depends_on_conflicting if item not in conflicting]
            if missing:
                raise ValueError(f"{claim.id}: depends_on_conflicting names non-CONFLICTING claims {missing}")
        return self

    @classmethod
    def builder(cls, subject_id: str = "", line: str | None = None, data_label: str = SYNTHETIC) -> "LedgerBuilder":
        """A LedgerBuilder: Ledger.builder().add_claim(...).add_assertion(...).build()."""
        return LedgerBuilder(subject_id=subject_id, line=line, data_label=data_label)

    def claim(self, claim_id: str) -> Claim | None:
        """The claim with this id, or None."""
        for claim in self.claims:
            if claim.id == claim_id:
                return claim
        return None

    def assertion(self, assertion_id: str) -> Assertion | None:
        """The assertion with this id, or None."""
        for assertion in self.assertions:
            if assertion.id == assertion_id:
                return assertion
        return None

    def claim_values(self) -> list[Claim]:
        """Every claim, in ledger order."""
        return list(self.claims)

    def claims_with_status(self, status: str) -> list[Claim]:
        """The claims with one status."""
        return [claim for claim in self.claims if claim.status == status]

    def open_claims(self) -> list[Claim]:
        """CONFLICTING and UNKNOWN claims: the open problems the evidence requests work on."""
        return [claim for claim in self.claims if claim.status in OPEN_STATUSES]

    def status_counts(self) -> dict[str, int]:
        """Number of claims per status (every status listed, zero included)."""
        return {status: len(self.claims_with_status(status)) for status in CLAIM_STATUSES}


class LedgerBuilder:
    """Collects claims and assertions, then builds a validated Ledger. Duplicate ids are refused when added."""

    def __init__(self, subject_id: str = "", line: str | None = None, data_label: str = SYNTHETIC) -> None:
        self.subject_id = subject_id
        self.line = line
        self.data_label = data_label
        self.claims: list[Claim] = []
        self.assertions: list[Assertion] = []

    def has_claim(self, claim_id: str) -> bool:
        """True when a claim with this id was added."""
        return any(claim.id == claim_id for claim in self.claims)

    def add_claim(self, claim: Claim | None = None, **fields: Any) -> "LedgerBuilder":
        """Add a Claim, or build one from keyword fields (the Claim field names). Returns the builder."""
        item = claim if claim is not None else Claim.model_validate(fields)
        if self.has_claim(item.id):
            raise ValueError(f"duplicate claim id {item.id}")
        self.claims.append(item)
        return self

    def add_assertion(self, assertion: Assertion | None = None, **fields: Any) -> "LedgerBuilder":
        """Add an Assertion, or build one from keyword fields. Returns the builder."""
        item = assertion if assertion is not None else Assertion.model_validate(fields)
        if any(existing.id == item.id for existing in self.assertions):
            raise ValueError(f"duplicate assertion id {item.id}")
        self.assertions.append(item)
        return self

    def build(self) -> Ledger:
        """The validated, immutable Ledger."""
        return Ledger(
            subject_id=self.subject_id,
            line=self.line,
            data_label=self.data_label,
            claims=tuple(self.claims),
            assertions=tuple(self.assertions),
        )


# --------------------------------------------------------------------------------------------------------------
# Claim helpers for callers outside the lender case (Founder Studio)
# --------------------------------------------------------------------------------------------------------------


def user_stated_claim(
    claim_id: str,
    label: str,
    value: ClaimValue,
    unit: str,
    display: str,
    field: str,
    recorded_at: str | None = None,
    metro_id: str | None = None,
) -> Claim:
    """USER_STATED: a figure the applicant or founder typed (intake field and, when known, the time it was typed)."""
    details: dict[str, InputValue] = {"recorded_at": recorded_at} if recorded_at else {}
    return Claim(
        id=claim_id,
        label=label,
        value=value,
        unit=unit,
        display=display,
        status="USER_STATED",
        provenance=Provenance(kind="user_input", ref=field, metro_id=metro_id, details=details),
        inputs={"field": field},
    )


def assumption_claim(
    claim_id: str,
    label: str,
    value: ClaimValue,
    unit: str,
    display: str,
    source: str,
    ref: str,
    low: float | None = None,
    high: float | None = None,
    metro_id: str | None = None,
    snapshot_date: str | None = None,
) -> Claim:
    """ASSUMPTION: a labeled planning assumption with its range (low, high) and its source or "unsourced"."""
    value_range = ValueRange(low=low, high=high) if low is not None and high is not None else None
    return Claim(
        id=claim_id,
        label=label,
        value=value,
        unit=unit,
        display=display,
        status="ASSUMPTION",
        provenance=Provenance(
            kind="assumption", ref=ref, source=source, metro_id=metro_id, snapshot_date=snapshot_date
        ),
        inputs={"ref": ref, "low": low, "high": high},
        value_range=value_range,
    )


def unknown_claim(
    claim_id: str, label: str, unit: str, display: str, reason: str, kind: ProvenanceKind = "policy", ref: str = ""
) -> Claim:
    """UNKNOWN: a value that is needed but absent, with the reason."""
    return Claim(
        id=claim_id,
        label=label,
        value=None,
        unit=unit,
        display=display,
        status="UNKNOWN",
        provenance=Provenance(kind=kind, ref=ref or claim_id),
        reason=reason,
    )
