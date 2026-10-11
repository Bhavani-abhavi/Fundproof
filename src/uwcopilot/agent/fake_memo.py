"""A deterministic memo writer for the fake LLM provider (tests, `make agent-demo`, the deterministic evaluation).

template_memo_responder(system, messages) reads the reference policy JSON back out of the memo prompt and returns
a memo that restates each finding (or, for a clean file, each passed check) with that rule's own citations. No
language model is involved, and the summary says so: every such memo is stamped "FAKE LLM — test mode".

It exists so the graph, the claim checker and the human checkpoint can be exercised end to end without an API key.
Its memos say nothing about how a real model would write; LLM memo quality stays "not measured".
"""

import json
from typing import Any

from uwcopilot.agent.memo import reference_policy_from_prompt
from uwcopilot.agent.vocab import FAKE_LLM_STAMP
from uwcopilot.llm.base import LLMMessage


def template_memo(policy: dict[str, Any]) -> dict[str, Any]:
    """A memo built only from the policy result: one reason per finding, or per passed check if none."""
    entries = policy["findings"] or policy["checks_passed"]
    reasons = [
        {"text": entry["detail"], "citations": list(dict.fromkeys(entry["section_ids"] + entry["evidence_ids"]))}
        for entry in entries
    ]
    summary = (
        f"{FAKE_LLM_STAMP}: template memo restating the reference policy result "
        f"(recommendation {policy['recommendation']}); no language model wrote this text. A human decides."
    )
    return {"recommendation": policy["recommendation"], "summary": summary, "reasons": reasons, "tool_requests": []}


def template_memo_responder(system: str, messages: list[LLMMessage]) -> str:
    """FakeLLM responder: the template memo as JSON text."""
    del system
    return json.dumps(template_memo(reference_policy_from_prompt(messages[0].content)))
