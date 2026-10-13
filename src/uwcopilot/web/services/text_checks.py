"""Checks on free text a person types into the web app (an idea, a site label, a reviewer's rationale).

Two different checks:

1. Data minimization (Studio addendum rule B, docs/studio/privacy.md): v1 does not accept SSNs, tax returns or bank
   account numbers. Text with an SSN-shaped number (123-45-6789), a labeled account number ("acct 12345678") or a
   run of 8 to 17 digits is refused with a plain message, so it is never stored. The patterns are the ones the trace
   redactor uses (uwcopilot.observability.redaction), so "what we refuse" and "what we would mask" agree.
2. Injection screen: the founder's idea text is untrusted data. It is normalized (invisible characters, homoglyphs
   and tag-character smuggling are handled by uwcopilot.security.normalize.detection_text) and then read by the
   agent's tripwire (uwcopilot.agent.injection.detect). Flagged text is still stored (it is the founder's own
   text) but it is marked "flagged" and is never given to any language model. v0.1 gives no idea text to any
   model at all.
"""

from uwcopilot.agent.injection import detect
from uwcopilot.observability.redaction import LABELED_ACCOUNT, LONG_DIGITS, SSN
from uwcopilot.web.services.errors import InputError

PRIVACY_DOC = "docs/studio/privacy.md"


def forbidden_identifiers(text: str) -> list[str]:
    """Which kinds of forbidden identifiers the text contains (empty when none)."""
    found = []
    if SSN.search(text):
        found.append("an SSN-shaped number")
    if LABELED_ACCOUNT.search(text) or LONG_DIGITS.search(text):
        found.append("an account-number-like run of digits")
    return found


def refuse_identifiers(text: str | None, field: str) -> None:
    """Raise InputError when text holds an SSN or an account-number-like number (data minimization)."""
    if not text:
        return
    found = forbidden_identifiers(text)
    if found:
        raise InputError(
            f"{field} contains {' and '.join(found)}. Please remove it: this prototype does not accept SSNs, tax "
            f"returns or bank account numbers ({PRIVACY_DOC})."
        )


def injection_flags(text: str | None) -> list[str]:
    """Names of the instruction-like patterns found in the normalized text (empty when none or no text)."""
    if not text:
        return []
    from uwcopilot.security.normalize import detection_text  # imported late: the module is still being finished

    return detect(detection_text(text))
