"""Child process for the kill-and-resume test (not a test module).

    python agent_kill_child.py <checkpoint.sqlite> <log dir> <application id> <thread id> <flagged name>

Runs the agent until the human checkpoint with a SQLite checkpointer, prints AT_CHECKPOINT, then waits to be
killed. The parent test kills it with SIGKILL and resumes the run from the same SQLite file.
"""

import sys
import time
from pathlib import Path

from agent_helpers import committed_application, fake_handlers, make_deps

from uwcopilot.agent.runner import compile_agent, open_sqlite_checkpointer, start_run

WAIT_SECONDS = 120


def main() -> None:
    """Run to the checkpoint, report it, and wait."""
    db, log_dir, application_id, thread_id, flagged = sys.argv[1:6]
    deps = make_deps(Path(log_dir), handlers=fake_handlers({flagged}))
    agent = compile_agent(deps, open_sqlite_checkpointer(Path(db)))
    outcome = start_run(agent, committed_application(application_id).model_dump(mode="json"), thread_id)
    print(f"AT_CHECKPOINT {outcome.status}", flush=True)
    time.sleep(WAIT_SECONDS)


if __name__ == "__main__":
    main()
