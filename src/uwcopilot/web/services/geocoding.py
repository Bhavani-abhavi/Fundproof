"""A thin adapter over the Studio Census Geocoder connector (uwcopilot.studio.data.geocode) for the web app.

Only REAL metros are geocoded (the SYNTHETIC metro has no addresses: its sites are picked on its map). The Census
Geocoder needs no key.

Privacy (docs/studio/privacy.md, Studio addendum rule B): a founder's address is personal data, so this adapter
does NOT use the Studio on-disk response cache (which would keep the address in plain text under data/studio/cache).
Instead:
- the result is cached in memory for the life of the process, keyed by the SHA-256 of the normalized address;
- the match (coordinates, tract, matched address and the SourceRecord with the request date and the response
  checksum) is stored inside the founder's encrypted project, so the plan is never geocoded twice.
The PoliteClient still rate-limits and retries requests to the host (Studio addendum rule F).
"""

import hashlib
import threading
from collections import OrderedDict
from typing import Protocol

from uwcopilot.studio.data.geocode import GeocodeMatch, geocode_address
from uwcopilot.studio.data.http import PoliteClient

MEMORY_CACHE_ENTRIES = 512


class Geocoder(Protocol):
    """Turns a one-line US address into a GeocodeMatch (raises the Studio data errors on failure)."""

    def geocode(self, address: str) -> GeocodeMatch:
        """Look up one address."""
        ...


def address_key(address: str) -> str:
    """The cache key of an address: SHA-256 of the address with case and spacing normalized."""
    normalized = " ".join(address.split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class CensusGeocoder:
    """The Census Geocoder through one shared PoliteClient without the on-disk cache, plus a memory cache."""

    def __init__(self, client: PoliteClient | None = None) -> None:
        """client: a PoliteClient (tests pass one with an httpx.MockTransport); None builds one without a cache."""
        self.client = client if client is not None else PoliteClient(cache=None)
        self._lock = threading.Lock()
        self._matches: OrderedDict[str, GeocodeMatch] = OrderedDict()

    def geocode(self, address: str) -> GeocodeMatch:
        """The match for an address. NotAvailable when the geocoder finds none; HostUnreachable / UpstreamError
        when it cannot be asked. One request at a time (the client keeps per-host timing state)."""
        key = address_key(address)
        with self._lock:
            if key in self._matches:
                self._matches.move_to_end(key)
                return self._matches[key]
            match = geocode_address(self.client, address)
            self._matches[key] = match
            while len(self._matches) > MEMORY_CACHE_ENTRIES:
                self._matches.popitem(last=False)
            return match
