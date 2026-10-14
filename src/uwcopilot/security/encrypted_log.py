"""EncryptedDecisionLog: the append-only decision log with every record's content encrypted (Fernet).

    log = EncryptedDecisionLog(Path("data/processed/agent/decision_records.enc.jsonl"), encryptor_from_env())
    stored = log.append({"record_id": "DR-...", ...})     # same call as DecisionLog.append
    log.records()                                          # decrypted records (needs the key)
    log.verify_chain()                                     # hash chain check: needs NO key
    log.verify_payloads()                                  # every payload decrypts and belongs to its line (key)

It is a drop-in for agent.decision_record.DecisionLog (a subclass with the same methods), so AgentDeps and
evidence.record.append_record accept it unchanged.

One JSONL line per record, the "envelope":
    {"record_id": "DR-...", "payload_encryption": "fernet", "payload": "<Fernet token>",
     "schema_version": 1, "previous_record_sha256": "...", "record_sha256": "..."}
The envelope is appended with a plain DecisionLog, so the chain fields are computed over the envelope (record id
plus ciphertext) exactly as DecisionLog computes them over a plaintext record. The record itself (names, tool
outputs, memo, the reviewer's rationale) is only inside the encrypted payload.

Trade-off of keeping the chain fields in clear (docs/security.md):
- gained: anyone can check the chain WITHOUT the key (an auditor, a CI job, scripts/verify_audit.py), and append
  stays idempotent by record_id without decrypting the file;
- given up: the clear fields show how many records there are, their order, their opaque record ids (DR-/EV- plus
  24 hex characters of a SHA-256) and their approximate size; a Fernet token also carries its creation time in
  clear (the Fernet format puts the timestamp outside the ciphertext);
- the envelope hash covers the ciphertext, and Fernet encryption is randomized, so the same record written twice
  (in two logs) gets two different record_sha256 values. Replay checks must compare decrypted records, not hashes;
- like DecisionLog, the chain alone does not detect a truncated tail, and someone who can rewrite the whole file
  can rebuild a consistent chain (but cannot forge payloads without the key). Anchor the latest hash elsewhere.
"""

import json
from pathlib import Path
from typing import Any

from uwcopilot.agent.applications import canonical_json
from uwcopilot.agent.decision_record import DecisionLog
from uwcopilot.security.crypto import CryptoError, Encryptor

PAYLOAD_ENCRYPTION = "fernet"
CHAIN_FIELDS: tuple[str, ...] = ("schema_version", "previous_record_sha256", "record_sha256")


class EncryptedDecisionLog(DecisionLog):
    """A DecisionLog whose lines hold encrypted payloads; the hash chain stays verifiable without the key."""

    def __init__(self, path: Path, encryptor: Encryptor) -> None:
        """path: the JSONL file (created on the first append); encryptor: from security.crypto.encryptor_from_env."""
        super().__init__(path)
        self._encryptor = encryptor
        self._envelopes = DecisionLog(path)

    def envelopes(self) -> list[dict[str, Any]]:
        """The lines as stored (record id, ciphertext and chain fields); no key needed."""
        return self._envelopes.records()

    def open_envelope(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """The decrypted record plus the envelope's chain fields; CryptoError if the payload does not fit the line."""
        if envelope.get("payload_encryption") != PAYLOAD_ENCRYPTION or not isinstance(envelope.get("payload"), str):
            raise CryptoError(f"line for {envelope.get('record_id')!r} has no Fernet payload")
        plaintext = self._encryptor.decrypt(envelope["payload"].encode("ascii"))
        record = json.loads(plaintext.decode("utf-8"))
        if not isinstance(record, dict) or record.get("record_id") != envelope.get("record_id"):
            raise CryptoError(f"the payload of line {envelope.get('record_id')!r} belongs to another record")
        for field in CHAIN_FIELDS:
            record[field] = envelope.get(field)
        return record

    def records(self) -> list[dict[str, Any]]:
        """Every record, decrypted, in file order (needs the key)."""
        return [self.open_envelope(envelope) for envelope in self.envelopes()]

    def find(self, record_id: str) -> dict[str, Any] | None:
        """The decrypted record with this id, or None (only that line is decrypted)."""
        for envelope in self.envelopes():
            if envelope.get("record_id") == record_id:
                return self.open_envelope(envelope)
        return None

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        """Encrypt and append one record (idempotent by record_id); return it decrypted with its chain fields."""
        if "record_id" not in record:
            raise ValueError("a decision record needs a record_id")
        token = self._encryptor.encrypt(canonical_json(record).encode("utf-8"))
        envelope = {
            "record_id": record["record_id"],
            "payload_encryption": PAYLOAD_ENCRYPTION,
            "payload": token.decode("ascii"),
        }
        stored = self._envelopes.append(envelope)
        return self.open_envelope(stored)

    def verify_chain(self) -> list[str]:
        """Hash-chain problems over the stored lines; needs no key (an empty list means the chain is intact)."""
        return self._envelopes.verify_chain()

    def verify_payloads(self) -> list[str]:
        """Lines whose payload cannot be decrypted or names another record (needs the key)."""
        problems = []
        for index, envelope in enumerate(self.envelopes()):
            try:
                self.open_envelope(envelope)
            except (CryptoError, ValueError) as error:
                problems.append(f"record {index}: {error}")
        return problems
