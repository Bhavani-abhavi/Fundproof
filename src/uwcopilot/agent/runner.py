"""Start, pause, resume: running the agent graph with a durable SQLite checkpointer.

A run goes:
1. start_run(): invoke the graph with an application on a thread id. It stops at the human checkpoint and the
   state is saved in the SQLite file after every step.
2. The process may exit or be killed. Nothing is lost: the checkpoint holds the full state and the pending
   interrupt.
3. resume_run(): any later process opens the same SQLite file, rebuilds the graph with the same dependencies,
   and resumes the thread with the human's decision. A refused decision leaves the run waiting again.

The SQLite file is local state (gitignored, *.sqlite*). One writer at a time is assumed.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from uwcopilot.agent.graph import AgentDeps, build_graph
from uwcopilot.observability.tracing import AgentTracer


@dataclass
class RunOutcome:
    """Where a thread stands after start_run() or resume_run()."""

    thread_id: str
    status: Literal["awaiting_review", "completed"]
    review_request: dict[str, Any] | None
    state: dict[str, Any]


def open_sqlite_checkpointer(path: Path) -> SqliteSaver:
    """A SqliteSaver on path (folders are created). Close it with close_checkpointer() when done."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(connection)


def close_checkpointer(saver: SqliteSaver) -> None:
    """Close the SQLite connection behind a saver."""
    saver.conn.close()


def compile_agent(deps: AgentDeps, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """The runnable agent: the graph compiled with a checkpointer."""
    return build_graph(deps).compile(checkpointer=checkpointer)


def thread_config(thread_id: str) -> dict[str, Any]:
    """LangGraph run config for one thread."""
    return {"configurable": {"thread_id": thread_id}}


def current_outcome(agent: CompiledStateGraph, thread_id: str) -> RunOutcome:
    """Read the thread's saved state: waiting at the checkpoint, or finished."""
    snapshot = agent.get_state(thread_config(thread_id))
    if snapshot.interrupts:
        return RunOutcome(thread_id, "awaiting_review", snapshot.interrupts[0].value, dict(snapshot.values))
    return RunOutcome(thread_id, "completed", None, dict(snapshot.values))


def thread_exists(agent: CompiledStateGraph, thread_id: str) -> bool:
    """True when the checkpointer already holds state for this thread."""
    return bool(agent.get_state(thread_config(thread_id)).values)


def start_run(
    agent: CompiledStateGraph,
    application: dict[str, Any],
    thread_id: str,
    tracer: AgentTracer | None = None,
) -> RunOutcome:
    """Run a new thread until the human checkpoint (or the end)."""
    if thread_exists(agent, thread_id):
        raise ValueError(f"thread {thread_id!r} already exists; resume it or use a new thread id")
    tracer = tracer or AgentTracer()
    with tracer.run_span(thread_id, str(application.get("application_id", "")), "start"):
        agent.invoke({"application": application}, thread_config(thread_id))
    return current_outcome(agent, thread_id)


def resume_run(
    agent: CompiledStateGraph,
    thread_id: str,
    decision: dict[str, Any],
    tracer: AgentTracer | None = None,
) -> RunOutcome:
    """Resume a waiting thread with a human decision."""
    outcome = current_outcome(agent, thread_id)
    if outcome.status != "awaiting_review":
        raise ValueError(f"thread {thread_id!r} is not waiting for a human decision")
    tracer = tracer or AgentTracer()
    with tracer.run_span(thread_id, str(outcome.state.get("application_id", "")), "resume"):
        agent.invoke(Command(resume=decision), thread_config(thread_id))
    return current_outcome(agent, thread_id)
