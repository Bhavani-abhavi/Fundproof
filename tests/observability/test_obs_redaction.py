"""Tests for the PII redactor used on everything written to traces."""

import json

from uwcopilot.observability.redaction import (
    ACCOUNT_MASK,
    EMAIL_MASK,
    FEIN_MASK,
    NAME_MASK,
    PHONE_MASK,
    SSN_MASK,
    VALUE_MASK,
    PiiRedactor,
    redactor_for_application,
)


def test_masks_known_names_and_name_parts() -> None:
    redactor = PiiRedactor(known_names=["Granite Dental Partners LLC"], person_names=["Maria Lopez-Garcia"])
    text = "Granite Dental Partners LLC is owned by MARIA LOPEZ-GARCIA; Ms. Lopez signed. Dental care is fine."
    redacted = redactor.redact_text(text)
    assert "Granite" not in redacted and "Lopez" not in redacted and "Maria" not in redacted.title()
    assert redacted.count(NAME_MASK) >= 3
    assert "Dental care is fine." in redacted  # org names are masked whole, not word by word


def test_masks_identifiers() -> None:
    redactor = PiiRedactor()
    text = (
        "FEIN 00-1234567, SSN 123-45-6789, account no. 9900 1234 5678, acct#99001234, bank 990012345678, "
        "mail jo@example.com, phone (407) 555-0100."
    )
    redacted = redactor.redact_text(text)
    for secret in ("1234567", "6789", "5678", "99001234", "990012345678", "jo@example.com", "555-0100"):
        assert secret not in redacted
    for mask in (FEIN_MASK, SSN_MASK, ACCOUNT_MASK, EMAIL_MASK, PHONE_MASK):
        assert mask in redacted


def test_keeps_ordinary_numbers() -> None:
    redactor = PiiRedactor()
    text = "DSCR 1.18; revenue $1,234,567.00; date 2026-09-30; 12 months; TR-4; NBC-4.2"
    assert redactor.redact_text(text) == text


def test_redacts_nested_values_and_sensitive_keys() -> None:
    redactor = PiiRedactor(person_names=["Ana Ruiz"])
    value = {
        "name": "anything at all",
        "fein": "00-7654321",
        "notes": "Ignore previous instructions",
        "owners": [{"owner_id": "O1", "summary": "Ana Ruiz owns 60%"}],
        "count": 3,
        "empty_name": {"name": ""},
    }
    redacted = redactor.redact(value)
    assert redacted["name"] == redacted["fein"] == redacted["notes"] == VALUE_MASK
    assert redacted["owners"][0]["summary"] == f"{NAME_MASK} owns 60%"
    assert redacted["count"] == 3 and redacted["empty_name"] == {"name": ""}


def test_to_json_truncates() -> None:
    text = PiiRedactor().to_json({"text": "x" * 100}, max_chars=40)
    assert len(text) < 80 and "truncated" in text


def test_redactor_for_application_knows_its_names() -> None:
    application = {
        "applicant": {"legal_name": "Harbor Bakery LLC"},
        "owners": [
            {"name": "Lee Chang", "entity_type": "person"},
            {"name": "Serpejns", "entity_type": "organization"},
        ],
    }
    redactor = redactor_for_application(application)
    summary = json.dumps({"memo": "Harbor Bakery LLC: Chang and Serpejns own it."})
    assert "Chang" not in redactor.redact_text(summary)
    assert "Serpejns" not in redactor.redact_text(summary)
    assert "Harbor Bakery LLC" not in redactor.redact_text(summary)


def test_long_amounts_in_json_are_not_mistaken_for_accounts() -> None:
    redactor = PiiRedactor()
    assert redactor.redact_text('{"stated_tiv": 27000000.0, "x": 1.25}') == '{"stated_tiv": 27000000.0, "x": 1.25}'
    assert redactor.redact_text("Account 990012345678.") == f"{ACCOUNT_MASK}."


def test_real_screening_output_leaks_no_name_into_a_span() -> None:
    """A real ScreeningResult (organization and person owners) is redacted before it can reach a span."""
    from kyc.kyc_fixtures import tiny_matcher
    from uwcopilot.tools.schemas import ScreeningRequest

    application = {
        "applicant": {"legal_name": "Harbor Test Logistics LLC"},
        "owners": [
            {"owner_id": "O1", "name": "Northwind Shadow Trading LLC", "entity_type": "organization"},
            {"owner_id": "O2", "name": "Ivan P. Sidorenko", "entity_type": "person"},
        ],
    }
    redactor = redactor_for_application(application)
    matcher = tiny_matcher(threshold=0.5)
    exported = []
    for owner in application["owners"]:
        result = matcher.screen(ScreeningRequest(name=owner["name"], entity_type=owner["entity_type"]))
        assert result.status == "potential_match"  # the explanations below really exist
        assert any(candidate.explanation for candidate in result.candidates)
        exported.append(redactor.to_json(result.model_dump(mode="json")))
    text = " ".join(exported).lower()
    for leaked in ("northwind", "shadow", "trading", "sidorenko", "ivan", "petrovich"):
        assert leaked not in text, leaked
    assert "[redacted]" in text


def test_normalized_organization_name_is_masked_in_free_text() -> None:
    redactor = redactor_for_application({"applicant": {"legal_name": "Granite Dental Partners, L.L.C."}, "owners": []})
    redacted = redactor.redact_text("customer name normalized to 'granite dental partners'")
    assert "granite" not in redacted.lower()
    assert NAME_MASK in redacted
