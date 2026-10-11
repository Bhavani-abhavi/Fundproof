"""Kill-and-resume: a process that is killed at the human checkpoint loses nothing.

A child process runs the agent with a SQLite checkpointer until the interrupt and is then killed with SIGKILL (no
cleanup code runs). This test rebuilds the graph from the same SQLite file, resumes with a human decision, and
checks that the final state and the decision record equal those of a run that was never interrupted.
"""

import os
import select
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from agent_helpers import SENIOR_REFER, committed_application, fake_handlers, make_deps, memory_agent

from uwcopilot.agent.decision_record import DecisionLog
from uwcopilot.agent.runner import (
    close_checkpointer,
    compile_agent,
    current_outcome,
    open_sqlite_checkpointer,
    resume_run,
    start_run,
)

CHILD = Path(__file__).with_name("agent_kill_child.py")
APPLICATION_ID = "APP-I-032"  # watchlist-derived owner: a blocking Compliance referral
STARTUP_TIMEOUT_S = 120


def planted_name() -> str:
    """The watchlist-derived owner's name, so the fake screening flags it."""
    application = committed_application(APPLICATION_ID)
    planted = application.ground_truth.watchlist_derived_owners[0].owner_id
    return next(owner.name for owner in application.owners if owner.owner_id == planted)


def wait_for_checkpoint(child: subprocess.Popen) -> str:
    """Read the child's output until it reports the checkpoint (or fail after a timeout)."""
    assert child.stdout is not None
    ready, _, _ = select.select([child.stdout], [], [], STARTUP_TIMEOUT_S)
    if not ready:
        child.kill()
        pytest.fail("child process did not reach the checkpoint in time")
    return child.stdout.readline()


def comparable(state: dict) -> dict:
    """State without wall-clock latencies and the log path (which differ between runs)."""
    copy = dict(state)
    copy["decision_record"] = {key: value for key, value in state["decision_record"].items() if key != "path"}
    copy["tool_results"] = sorted(
        ({key: value for key, value in result.items() if key != "latency_ms"} for result in state["tool_results"]),
        key=lambda result: result["result_id"],
    )
    return copy


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="needs POSIX signals")
def test_killed_process_resumes_from_sqlite_to_the_same_final_state(tmp_path: Path) -> None:
    application = committed_application(APPLICATION_ID)
    name = planted_name()
    db = tmp_path / "durable" / "checkpoints.sqlite"
    durable_logs = tmp_path / "durable"
    env = {**os.environ, "PYTHONPATH": str(CHILD.parent)}
    child = subprocess.Popen(
        [sys.executable, str(CHILD), str(db), str(durable_logs), APPLICATION_ID, "kill-thread", name],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        line = wait_for_checkpoint(child)
        assert line.startswith("AT_CHECKPOINT awaiting_review"), line
    finally:
        child.send_signal(signal.SIGKILL)
        child.wait(timeout=30)
        if child.stdout is not None:
            child.stdout.close()
    assert child.returncode == -signal.SIGKILL

    saver = open_sqlite_checkpointer(db)
    try:
        agent = compile_agent(make_deps(durable_logs, handlers=fake_handlers({name})), saver)
        waiting = current_outcome(agent, "kill-thread")
        assert waiting.status == "awaiting_review"
        assert waiting.review_request["review_queue"] == "Compliance"
        resumed = resume_run(agent, "kill-thread", SENIOR_REFER)
    finally:
        close_checkpointer(saver)

    reference_agent = memory_agent(make_deps(tmp_path / "reference", handlers=fake_handlers({name})))
    start_run(reference_agent, application.model_dump(mode="json"), "kill-thread")
    reference = resume_run(reference_agent, "kill-thread", SENIOR_REFER)

    assert resumed.status == reference.status == "completed"
    assert comparable(resumed.state) == comparable(reference.state)
    [durable_record] = DecisionLog(durable_logs / "decision_records.jsonl").records()
    [reference_record] = DecisionLog(tmp_path / "reference" / "decision_records.jsonl").records()
    assert durable_record == reference_record
