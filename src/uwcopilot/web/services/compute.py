"""Heavy work (simulations, path searches, agent runs) in worker threads, with a timeout and a result cache.

    runner = ComputeRunner(settings)
    result = await runner.run(work, key=("feasibility", owner, project_id, version, site_index))

Why: a Monte Carlo run or an agent run takes from a fraction of a second to several seconds. Running it inside the
event loop would block every other request, so it runs in a small thread pool, and the request waits at most
timeout_s seconds for it:

- on time: the result is returned (and cached under `key`, so a page reload does not recompute it);
- too slow: ComputeTimeout (the router answers 503 with a clear message). Python cannot stop a running thread, so
  the job keeps running in the background; when it finishes its result is cached, and the next try is instant;
- too many jobs running or queued at once (max_pending): ComputeBusy (503 "server busy, try again").

Two requests for the same key while it is running share one job (no double work). A job that raises is not
cached: the error goes to every waiting request and the next request runs it again.

The cache is in memory, per process, bounded to cache_entries results (least recently used first out). Keys are
tuples; forget(prefix) drops every cached result whose key starts with prefix (used when a plan is deleted).
"""

import asyncio
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar

from uwcopilot.web.services.settings import ServiceSettings

T = TypeVar("T")


class ComputeTimeout(Exception):
    """The job did not finish within the timeout (it keeps running; its result is cached when it ends)."""


class ComputeBusy(Exception):
    """Too many heavy jobs are running or queued; the request should be retried later."""


def consume_outcome(future: "asyncio.Future[Any]") -> None:
    """Read a finished asyncio future's exception so nobody is warned that it was never retrieved."""
    if not future.cancelled():
        future.exception()


class ComputeRunner:
    """A small thread pool with per-key de-duplication, a timeout per request and an LRU result cache."""

    def __init__(self, settings: ServiceSettings) -> None:
        """Threads are started lazily by the executor on the first job."""
        self.timeout_s = settings.timeout_s
        self.max_pending = settings.max_pending
        self.cache_entries = settings.cache_entries
        self._executor = ThreadPoolExecutor(max_workers=settings.workers, thread_name_prefix="uwc-compute")
        self._lock = threading.Lock()
        self._results: OrderedDict[Hashable, Any] = OrderedDict()
        self._running: dict[Hashable, Future[Any]] = {}
        self._pending: set[Future[Any]] = set()

    def cached(self, key: Hashable) -> tuple[bool, Any]:
        """(True, result) when key has a cached result, else (False, None)."""
        with self._lock:
            if key in self._results:
                self._results.move_to_end(key)
                return True, self._results[key]
        return False, None

    def _remember(self, key: Hashable, future: Future[Any]) -> None:
        """Done-callback of a keyed job: stop tracking it and cache its result if it succeeded."""
        with self._lock:
            if self._running.get(key) is future:
                del self._running[key]
            if future.cancelled() or future.exception() is not None:
                return
            self._results[key] = future.result()
            self._results.move_to_end(key)
            while len(self._results) > self.cache_entries:
                self._results.popitem(last=False)

    def _release(self, future: Future[Any]) -> None:
        """Done-callback of every job: it no longer counts as pending."""
        with self._lock:
            self._pending.discard(future)

    def _submit(self, work: Callable[[], Any], key: Hashable | None) -> Future[Any]:
        """The running job for key, or a new job (ComputeBusy when max_pending jobs are already in flight)."""
        with self._lock:
            if key is not None and key in self._running:
                return self._running[key]
            if len(self._pending) >= self.max_pending:
                raise ComputeBusy(f"{len(self._pending)} heavy jobs are already running or queued")
            future = self._executor.submit(work)
            self._pending.add(future)
            if key is not None:
                self._running[key] = future
        future.add_done_callback(self._release)
        if key is not None:
            future.add_done_callback(lambda done: self._remember(key, done))
        return future

    async def run(self, work: Callable[[], T], key: Hashable | None = None, timeout_s: float | None = None) -> T:
        """work()'s result: from the cache when key has one, else from a worker thread within the timeout.

        key=None runs the job every time and never caches it (for example a decision, which changes state).
        """
        if key is not None:
            found, result = self.cached(key)
            if found:
                return result
        future = self._submit(work, key)
        waiting = asyncio.wrap_future(future)
        waiting.add_done_callback(consume_outcome)
        limit = self.timeout_s if timeout_s is None else timeout_s
        done, _ = await asyncio.wait({waiting}, timeout=limit)
        if not done:
            raise ComputeTimeout(f"the job did not finish within {limit:g} seconds")
        return waiting.result()

    def pending_jobs(self) -> int:
        """How many jobs are running or queued right now."""
        with self._lock:
            return len(self._pending)

    def forget(self, prefix: tuple[Any, ...]) -> int:
        """Drop every cached result whose (tuple) key starts with prefix; returns how many were dropped."""
        with self._lock:
            stale = [key for key in self._results if isinstance(key, tuple) and key[: len(prefix)] == prefix]
            for key in stale:
                del self._results[key]
        return len(stale)

    def shutdown(self) -> None:
        """Stop accepting jobs; queued jobs are cancelled, running ones finish in the background."""
        self._executor.shutdown(wait=False, cancel_futures=True)
