"""When each account last used "Delete all my data", kept in memory, so a slow request from before it saves nothing.

    deletions = DataDeletions()                  # on app.state.services.deletions
    started = deletions.now()                    # a create request notes when it started ...
    deletions.record(owner)                      # DELETE /api/me/data notes the deletion ...
    deletions.deleted_since(owner, started)      # ... and the create checks after its slow part: True -> 409

Why. Creating a plan can wait seconds for the Census Geocoder (lookups are spaced server-wide). If the founder clicks
"Delete everything" in another tab meanwhile, the create used to finish afterwards and store a new plan in a fresh
folder, after the deletion had answered that everything was deleted (review finding S8-DEL-3).

Limits (honest):
- In memory, one process. After a restart the note is gone, but so is every request that started before it.
- Keyed by the owner digest (store.owner_digest), not the account id; one entry per account that deleted its data
  while this process runs (not pruned: a few dozen bytes each).
- The create route checks once, after the geocoder wait and before its access event and save. A deletion that lands
  in the few milliseconds between that check and the save is not caught. Updates (PUT) re-load the plan before
  saving, so one that resumes after the deletion already gets 404; creating a share link has no slow step.
"""

import threading
import time
from collections.abc import Callable

from uwcopilot.web.store import owner_digest


class DataDeletions:
    """When each owner last deleted all their data, on a monotonic clock (seconds)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        """clock: a monotonic clock (tests may pass their own)."""
        self.clock = clock
        self._lock = threading.Lock()
        self._deleted_at: dict[str, float] = {}

    def now(self) -> float:
        """The current time on this clock: a request notes it when it starts."""
        return self.clock()

    def record(self, owner: str) -> None:
        """Note that the owner deleted all their data now."""
        with self._lock:
            self._deleted_at[owner_digest(owner)] = self.clock()

    def deleted_since(self, owner: str, started: float) -> bool:
        """True when the owner deleted all their data at or after `started`."""
        with self._lock:
            deleted_at = self._deleted_at.get(owner_digest(owner))
        return deleted_at is not None and deleted_at >= started
