"""Document-level permissions: what a principal may see of a file's review payload and of a packet's PDF.

    review = review_for_principal(review, principal, packet_id="PKT-C-024", audit=log)  # needs view_file
    pdf = read_packet_for(principal, "PKT-C-024", store, audit=log)                     # needs view_documents

Two permissions (security/rbac.py):
- view_file: the review payload at the human checkpoint (recommendation, findings, limits, tool statuses).
  Without it, nothing is shown: AccessDenied.
- view_documents: the packet's documents and extraction (page text, extracted fields, document checks,
  contradictions, field provenance, quoted excerpts). It needs a document-viewing role AND the packet access list
  (config/security.yaml). Without it the review payload is still shown, with every document-content part replaced
  by WITHHELD, and document_access says why.

What counts as document content is listed here, in one place:
- DOCUMENT_CONTENT_KEYS: top-level review keys that hold document content (spec Phase 5 A names: extraction,
  document_checks, contradictions, field_provenance, plus page_texts and documents). The packet intake that adds
  them to the review payload (Phase 5 A, not built yet) must add any NEW document key here;
- DOCUMENT_TEXT_KEYS: keys that quote document or applicant text wherever they appear (for example the excerpt of
  an injection hit). Withheld anywhere in the payload.
Everything else (ids, rule codes, numbers, statuses, the memo) stays visible: the reviewer needs it to act.

Every call writes an audit event when an AuditLog is given: view, or access_denied.
"""

from typing import Any

from uwcopilot.security.audit import AuditLog, record_access
from uwcopilot.security.config import SecurityConfig
from uwcopilot.security.packet_store import EncryptedPacketStore
from uwcopilot.security.rbac import Decision, Principal, Resource, authorize

WITHHELD = "[WITHHELD: no document permission for this packet]"
DOCUMENT_CONTENT_KEYS: frozenset[str] = frozenset(
    {"extraction", "document_checks", "contradictions", "field_provenance", "page_texts", "documents"}
)
DOCUMENT_TEXT_KEYS: frozenset[str] = frozenset({"excerpt", "quote", "snippet", "page_text", "raw_text", "source_text"})


class AccessDenied(PermissionError):
    """authorize() refused; .decision holds the reason."""

    def __init__(self, decision: Decision) -> None:
        """Keep the decision so callers can show its reason."""
        super().__init__(decision.reason)
        self.decision = decision


def withhold_text(value: Any) -> Any:
    """A copy of value with every DOCUMENT_TEXT_KEYS entry replaced by WITHHELD, at any depth."""
    if isinstance(value, dict):
        return {
            key: (WITHHELD if key in DOCUMENT_TEXT_KEYS and item not in (None, "") else withhold_text(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [withhold_text(item) for item in value]
    return value


def hide_document_content(review: dict[str, Any], decision: Decision, packet_id: str | None) -> dict[str, Any]:
    """The review payload with document content withheld unless decision allows it; adds document_access."""
    access = {"packet_id": packet_id, "allowed": decision.allowed, "reason": decision.reason}
    if decision.allowed:
        return {**review, "document_access": access}
    hidden = {key: (WITHHELD if key in DOCUMENT_CONTENT_KEYS else value) for key, value in review.items()}
    return {**withhold_text(hidden), "document_access": access}


def review_for_principal(
    review: dict[str, Any],
    principal: Principal,
    packet_id: str | None = None,
    audit: AuditLog | None = None,
    config: SecurityConfig | None = None,
) -> dict[str, Any]:
    """The review payload this principal may see. AccessDenied without view_file.

    packet_id: the packet the file came from, or None for a structured application. A structured application has
    no packet documents, so nothing is withheld and document_access.allowed is None (not applicable).
    """
    file_id = str(review.get("application_id") or "unknown")
    file_resource = Resource("file", file_id)
    file_decision = authorize(principal, "view_file", file_resource, config)
    if audit is not None:
        record_access(audit, principal, "view_file", file_resource, file_decision)
    if not file_decision.allowed:
        raise AccessDenied(file_decision)
    if packet_id is None:
        not_applicable = {"packet_id": None, "allowed": None, "reason": "structured application: no packet documents"}
        return {**review, "document_access": not_applicable}
    packet_resource = Resource("packet", packet_id)
    document_decision = authorize(principal, "view_documents", packet_resource, config)
    if audit is not None:
        record_access(audit, principal, "view_documents", packet_resource, document_decision)
    return hide_document_content(review, document_decision, packet_id)


def read_packet_for(
    principal: Principal,
    packet_id: str,
    store: EncryptedPacketStore,
    audit: AuditLog | None = None,
    config: SecurityConfig | None = None,
) -> bytes:
    """The decrypted packet PDF, only for a principal allowed to view its documents (AccessDenied otherwise)."""
    resource = Resource("packet", packet_id)
    decision = authorize(principal, "view_documents", resource, config)
    if audit is not None:
        record_access(audit, principal, "view_documents", resource, decision, {"what": "packet_pdf"})
    if not decision.allowed:
        raise AccessDenied(decision)
    return store.get(packet_id)


def audit_events_for(principal: Principal, audit: AuditLog, config: SecurityConfig | None = None) -> list[dict]:
    """The audit events, only for a principal allowed to view the audit log; reading them is itself logged."""
    resource = Resource("audit_log", audit.path.name)
    decision = authorize(principal, "view_audit_log", resource, config)
    record_access(audit, principal, "view_audit_log", resource, decision)
    if not decision.allowed:
        raise AccessDenied(decision)
    return audit.events()
