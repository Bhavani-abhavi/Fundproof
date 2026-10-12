"""Tests for uwcopilot.llm.ollama_queue: the file lock lets one Ollama job run at a time, across processes."""

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from uwcopilot.llm.ollama_queue import (
    DEFAULT_LOCK_PATH,
    OllamaQueueBusy,
    ollama_queue_lock,
    read_lock_holder,
)

HOLD_SECONDS = 0.6

# A child process: take the lock, write when it got it and when it let go.
CHILD_HOLDS_LOCK = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from uwcopilot.llm.ollama_queue import ollama_queue_lock

    lock_path, out_path, hold = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
    with ollama_queue_lock(f"child {out_path.name}", lock_path=lock_path, poll_interval_s=0.05):
        start = time.time()
        time.sleep(hold)
        end = time.time()
    out_path.write_text(f"{start} {end}")
    """
)

# A child process that takes the lock and dies without releasing it.
CHILD_CRASHES_WITH_LOCK = textwrap.dedent(
    """
    import os, sys
    from pathlib import Path
    from uwcopilot.llm.ollama_queue import ollama_queue_lock

    with ollama_queue_lock("crashing child", lock_path=Path(sys.argv[1])):
        Path(sys.argv[2]).write_text("locked")
        os._exit(1)
    """
)


def run_child(code: str, *args: str) -> subprocess.Popen[bytes]:
    """Start a Python child process running code with the given arguments."""
    return subprocess.Popen([sys.executable, "-c", code, *args])


def test_default_lock_path_is_the_shared_contract() -> None:
    assert DEFAULT_LOCK_PATH.parts[-3:] == ("data", "cache", "ollama.lock")


def test_lock_serializes_two_processes(tmp_path: Path) -> None:
    lock_path = tmp_path / "ollama.lock"
    outputs = [tmp_path / "first.txt", tmp_path / "second.txt"]
    children = [run_child(CHILD_HOLDS_LOCK, str(lock_path), str(out), str(HOLD_SECONDS)) for out in outputs]
    for child in children:
        assert child.wait(timeout=60) == 0
    intervals = sorted(tuple(float(value) for value in out.read_text().split()) for out in outputs)
    (first_start, first_end), (second_start, second_end) = intervals
    assert first_end - first_start >= HOLD_SECONDS
    assert second_start >= first_end, f"the two jobs overlapped: {intervals}"


def test_holder_is_recorded_while_held_and_cleared_after(tmp_path: Path) -> None:
    lock_path = tmp_path / "ollama.lock"
    with ollama_queue_lock("eval job", lock_path=lock_path) as holder:
        record = read_lock_holder(lock_path)
        assert record is not None
        assert record["label"] == "eval job"
        assert record["pid"] == holder.pid
    assert read_lock_holder(lock_path) is None


def test_waiting_longer_than_the_timeout_raises_busy(tmp_path: Path) -> None:
    lock_path = tmp_path / "ollama.lock"
    ran = []
    with ollama_queue_lock("first job", lock_path=lock_path):
        started = time.monotonic()
        with pytest.raises(OllamaQueueBusy, match="first job"):
            with ollama_queue_lock("second job", lock_path=lock_path, timeout_s=0.3, poll_interval_s=0.05):
                ran.append("second job")
        assert time.monotonic() - started >= 0.3
    assert ran == []
    with ollama_queue_lock("third job", lock_path=lock_path, timeout_s=0.3) as holder:
        assert holder.label == "third job"


def test_lock_is_released_when_the_holder_crashes(tmp_path: Path) -> None:
    lock_path = tmp_path / "ollama.lock"
    marker = tmp_path / "locked.txt"
    child = run_child(CHILD_CRASHES_WITH_LOCK, str(lock_path), str(marker))
    assert child.wait(timeout=60) == 1
    assert marker.read_text() == "locked"
    with ollama_queue_lock("after crash", lock_path=lock_path, timeout_s=2) as holder:
        assert holder.waited_s < 2
