"""Share links: a founder lets someone read one plan, without signing in, until a date (1 to 30 days ahead).

    links = ShareLinks(session_secret)            # create_app() keeps one on app.state.share_links
    link = links.create(store, owner, project_id, expires_in_days=7, plan_expires_at=meta.expires_at)
    link = links.resolve(store, token)            # ShareLinkInvalid for every kind of bad link
    revoked, pending = links.revoke(store, owner, project_id)   # links that stopped working; backup still pending?

The token. A Fernet token (AES-128-CBC + HMAC-SHA256 from `cryptography`, as web/crypto.py uses for data at rest)
of the JSON {"v": 1, "owner": owner id, "project_id", "link_id", "expires_at": Unix seconds}. Its key is DERIVED
from the session secret: HMAC-SHA256(key = UWC_SESSION_SECRET, message = "fundproof-share-link-v1"), base64url.
- It is never the store key (UWC_FERNET_KEY): a token cannot be decrypted with the key of stored data or the other
  way round, and the token key opens no stored object.
- The fixed label keeps this key apart from every other use of the session secret (the cookie signer hashes the
  secret with its own salts), and "-v1" lets a later token format get a new key.
- Encrypted, not only signed, so the owner id inside never shows to whoever holds the link (the session cookie, by
  contrast, is signed only). Fernet also authenticates: a changed token does not decrypt.
- A token has exactly one spelling. base64 decoding ignores some changes (characters after the padding, the unused
  low bits of the last character), so a token that is not the base64 text of its own bytes is refused first.
- Fernet writes the time the token was made, in clear, at its start. That time is not secret.
- Changing UWC_SESSION_SECRET ends every link (and every session). Without the setting, outside a Space, a random
  secret is made per process (settings.py), so links then stop working when the process restarts.

The record. Making a link also stores {link_id, project_id, created_at, expires_at} in the OWNER's own store
namespace "shares" (key = the link id; encrypted like every object; retention = until the link expires). A token
works only while its record exists, so:
- revoking deletes the plan's records (the tokens still decrypt, but no longer work). With the dataset store the
  backup copy of a record may not be deletable yet (DeletionPending); revoke() then says "pending" so the owner is
  told (see revoke());
- deleting the plan revokes its links first, and "delete all my data" deletes this namespace with the rest;
- links are known only inside their owner's store folder; nothing lists the links of all users.
Expiry is checked twice, from the token and from the record, and the plan must still exist (an expired or deleted
plan ends its links too). A link never outlives the plan's retention date as it was when the link was made.

Every failure (malformed, tampered, expired, revoked, record or plan gone) is the same ShareLinkInvalid, and the
router answers each one with the same 404 body, so a caller cannot learn which check failed.

Link ids are 20 lowercase letters (about 94 bits): letters only, so the audit log's redactor, which masks runs of 8
or more digits as account numbers, never changes one. A plan keeps at most MAX_LINKS_PER_PLAN working links (a
design choice that bounds storage; one more is a 409).
"""

import base64
import hashlib
import hmac
import json
import math
import secrets
import string
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, ConfigDict

from uwcopilot.web.services.errors import Conflict
from uwcopilot.web.store import DeletionPending, ObjectNotFound, Store

SHARES_NAMESPACE = "shares"
TOKEN_KEY_LABEL = b"fundproof-share-link-v1"
TOKEN_VERSION = 1
LINK_ID_LETTERS = 20
DEFAULT_EXPIRES_IN_DAYS = 7
MAX_EXPIRES_IN_DAYS = 30
# Design choices: enough links for a few lenders per plan; a length far above any token made here (its payload is
# a few short ids), so an oversized path is refused before any decoding.
MAX_LINKS_PER_PLAN = 10
MAX_TOKEN_CHARS = 1024


class ShareLinkInvalid(Exception):
    """A share token that must show nothing. The message is for tests and debugging; it never reaches a client."""


class TokenPayload(BaseModel):
    """What a token carries, encrypted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    v: Literal[1]
    owner: str
    project_id: str
    link_id: str
    expires_at: int  # Unix seconds


class ShareRecord(BaseModel):
    """The stored record of one link, in the owner's "shares" namespace."""

    model_config = ConfigDict(extra="forbid")

    link_id: str
    project_id: str
    created_at: datetime
    expires_at: datetime


@dataclass(frozen=True)
class SharedLink:
    """A link that was just made, or that passed every check."""

    token: str
    owner: str
    project_id: str
    link_id: str
    expires_at: datetime


def utc_now() -> datetime:
    """The current time in UTC."""
    return datetime.now(UTC)


def share_key(session_secret: str) -> bytes:
    """The Fernet key of share tokens: HMAC-SHA256 of the fixed label under the session secret, base64url."""
    digest = hmac.new(session_secret.encode("utf-8"), TOKEN_KEY_LABEL, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest)


def new_link_id() -> str:
    """LINK_ID_LETTERS random lowercase letters."""
    return "".join(secrets.choice(string.ascii_lowercase) for _ in range(LINK_ID_LETTERS))


class ShareLinks:
    """Makes, checks and revokes share links (see the module docstring)."""

    def __init__(self, session_secret: str, clock: Callable[[], datetime] = utc_now) -> None:
        """session_secret: UWC_SESSION_SECRET; clock: the current time (tests move it)."""
        self._fernet = Fernet(share_key(session_secret))
        self.clock = clock

    def _token(self, owner: str, project_id: str, link_id: str, expires_at: datetime) -> str:
        """The encrypted token for one link."""
        payload = TokenPayload(
            v=TOKEN_VERSION, owner=owner, project_id=project_id, link_id=link_id, expires_at=int(expires_at.timestamp())
        )
        return self._fernet.encrypt(payload.model_dump_json().encode("utf-8")).decode("ascii")

    def _payload(self, token: str) -> TokenPayload:
        """Decrypt and parse a token; ShareLinkInvalid when it is malformed, tampered with or made with another key."""
        if not token or len(token) > MAX_TOKEN_CHARS:
            raise ShareLinkInvalid("the token is empty or too long")
        try:
            raw = base64.urlsafe_b64decode(token.encode("ascii"))
        except ValueError as error:  # not ASCII, or not base64 (binascii.Error is a ValueError)
            raise ShareLinkInvalid("the token is not base64") from error
        if base64.urlsafe_b64encode(raw).decode("ascii") != token:
            raise ShareLinkInvalid("the token is not spelled as the base64 text of its own bytes")
        try:
            plaintext = self._fernet.decrypt(token.encode("ascii"))
            return TokenPayload.model_validate(json.loads(plaintext))
        except (InvalidToken, ValueError) as error:  # ValueError: not JSON, or not a share payload
            raise ShareLinkInvalid("the token does not decrypt to a share payload") from error

    def stored_records_for(self, store: Store, owner: str, project_id: str) -> list[ShareRecord]:
        """Every stored link record of the plan, including records whose link has already expired."""
        records = []
        for meta in store.list(owner, SHARES_NAMESPACE):
            try:
                record = ShareRecord.model_validate(store.get_json(owner, SHARES_NAMESPACE, meta.key))
            except ObjectNotFound:
                continue  # deleted or expired since the listing
            if record.project_id == project_id:
                records.append(record)
        return records

    def links_for(self, store: Store, owner: str, project_id: str) -> list[ShareRecord]:
        """The plan's WORKING link records.

        The store hides records past their retention, but a record's retention is whole days rounded up, so it can
        outlive its link by up to a day: a record whose link has expired (expires_at <= now) is skipped too, so it is
        neither counted against MAX_LINKS_PER_PLAN nor reported as revoked. revoke() still deletes it.
        """
        now = self.clock()
        return [record for record in self.stored_records_for(store, owner, project_id) if record.expires_at > now]

    def create(
        self,
        store: Store,
        owner: str,
        project_id: str,
        expires_in_days: int,
        plan_expires_at: datetime | None,
        link_id: str | None = None,
    ) -> SharedLink:
        """Store a record and return the new link. The caller has checked that the plan is the owner's.

        The link expires expires_in_days from now, or when the plan's retention ends if that is sooner. link_id lets
        the caller name the link in its access event first; None picks a new one. Conflict when the plan already
        has MAX_LINKS_PER_PLAN working links.
        """
        if not 1 <= expires_in_days <= MAX_EXPIRES_IN_DAYS:
            raise ValueError(f"expires_in_days must be 1 to {MAX_EXPIRES_IN_DAYS}")
        if len(self.links_for(store, owner, project_id)) >= MAX_LINKS_PER_PLAN:
            raise Conflict(
                f"this plan already has {MAX_LINKS_PER_PLAN} working share links; revoke them to make a new one"
            )
        now = self.clock()
        expires_at = now + timedelta(days=expires_in_days)
        if plan_expires_at is not None:
            expires_at = min(expires_at, plan_expires_at)
        expires_at = expires_at.replace(microsecond=0)  # the token holds whole seconds
        link_id = link_id or new_link_id()
        record = ShareRecord(link_id=link_id, project_id=project_id, created_at=now, expires_at=expires_at)
        retention_days = max(1, math.ceil((expires_at - now) / timedelta(days=1)))
        store.put_json(owner, SHARES_NAMESPACE, link_id, record.model_dump(mode="json"), retention_days=retention_days)
        token = self._token(owner, project_id, link_id, expires_at)
        return SharedLink(token=token, owner=owner, project_id=project_id, link_id=link_id, expires_at=expires_at)

    def resolve(self, store: Store, token: str) -> SharedLink:
        """The link a token stands for, after every check; ShareLinkInvalid otherwise.

        The caller still loads the plan (a deleted or expired plan is also an invalid link).
        """
        payload = self._payload(token)
        now = self.clock()
        token_expires_at = datetime.fromtimestamp(payload.expires_at, UTC)
        if now >= token_expires_at:
            raise ShareLinkInvalid("the token has expired")
        try:
            data = store.get_json(payload.owner, SHARES_NAMESPACE, payload.link_id)
            record = ShareRecord.model_validate(data)
        except (ObjectNotFound, ValueError) as error:  # ValueError: a bad name or record (StoreError, CryptoError)
            raise ShareLinkInvalid("no record for this link: revoked, deleted or expired") from error
        if record.project_id != payload.project_id or record.link_id != payload.link_id:
            raise ShareLinkInvalid("the record belongs to another link")
        if now >= record.expires_at:
            raise ShareLinkInvalid("the record has expired")
        return SharedLink(
            token=token,
            owner=payload.owner,
            project_id=payload.project_id,
            link_id=payload.link_id,
            expires_at=min(token_expires_at, record.expires_at),
        )

    def revoke(self, store: Store, owner: str, project_id: str) -> tuple[int, bool]:
        """Delete every link record of the plan; returns (how many WORKING links stopped working, pending).

        Records of links that have already expired are deleted too (so a plan deletion leaves none behind), but they
        are not counted.

        pending is True when the dataset store could not delete the backup copy of a record yet (DeletionPending).
        The link still stops working at once: the record is gone from the working copy that resolve() reads, and the
        backup deletion is retried. But that retry is kept on the container disk, which in a Space does not survive a
        restart: if the app restarts before the backup deletion goes through, the record comes back from the dataset
        and the link works again until it expires. So the caller must tell the owner, who should revoke again after a
        restart.
        """
        now = self.clock()
        revoked = 0
        pending = False
        for record in self.stored_records_for(store, owner, project_id):
            try:
                store.delete(owner, SHARES_NAMESPACE, record.link_id)
            except DeletionPending:
                pending = True
            if record.expires_at > now:
                revoked += 1
        return revoked, pending
