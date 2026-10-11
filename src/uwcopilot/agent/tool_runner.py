"""The tool gate: every tool request passes the allow-list, schema validation and a loop guard before it runs.

vet_requests() decides which requests may run:
- the tool name must be in the registry's ALLOWED_TOOLS (anything else is rejected as "not_allowed");
- a request that came from the LLM may only use LLM_REQUESTABLE_TOOLS (read-only guideline search), so a memo can
  ask for more context but can never screen, score or do anything else;
- the arguments must validate against the tool's input schema (unknown fields are rejected);
- loop guard: a call identical to one already made (same tool, same validated arguments) is rejected as
  "repeated_call", and so is a duplicate inside the same batch;
- a run makes at most MAX_TOOL_CALLS calls in total.

execute_call() then runs one approved call through uwcopilot.tools.registry.call_tool (which validates the
output too) and turns any failure into a recorded result instead of an exception, so one broken tool cannot
crash the run: the failure is shown to the human reviewer and to the policy.
"""

import hashlib
import time
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from uwcopilot.agent.applications import canonical_json
from uwcopilot.agent.tool_plan import ToolRequest
from uwcopilot.tools.registry import ALLOWED_TOOLS, ToolHandler, ToolUnavailable, call_tool

LLM_REQUESTABLE_TOOLS: frozenset[str] = frozenset({"search_guidelines"})
MAX_TOOL_CALLS = 25
RejectionReason = Literal[
    "not_allowed", "not_permitted_for_llm", "invalid_arguments", "repeated_call", "call_budget_exhausted"
]


class ApprovedCall(BaseModel):
    """A request that passed the gate, with its result id and validated arguments."""

    model_config = ConfigDict(extra="forbid")

    result_id: str
    request: ToolRequest
    arguments: dict[str, Any]
    signature: str


class ToolRejection(BaseModel):
    """A request that the gate refused, and why. It is recorded and shown to the reviewer; it never runs."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    tool: str
    origin: str
    reason: RejectionReason
    detail: str


class ToolResult(BaseModel):
    """What one executed call returned. output is the validated tool output (None when the call failed)."""

    model_config = ConfigDict(extra="forbid")

    result_id: str
    request_id: str
    tool: str
    origin: str
    purpose: str
    party_id: str | None
    arguments: dict[str, Any]
    signature: str
    status: Literal["ok", "error", "unavailable"]
    output: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: float


def call_signature(tool: str, arguments: Mapping[str, Any]) -> str:
    """SHA-256 of the tool name and its validated arguments: two calls with the same signature are identical."""
    return hashlib.sha256(f"{tool}:{canonical_json(dict(arguments))}".encode()).hexdigest()


def vet_requests(
    requests: list[ToolRequest],
    executed_signatures: set[str],
    executed_count: int,
    max_calls: int = MAX_TOOL_CALLS,
) -> tuple[list[ApprovedCall], list[ToolRejection]]:
    """Split requests into approved calls (with result ids TR-n) and rejections, in request order."""
    approved: list[ApprovedCall] = []
    rejections: list[ToolRejection] = []
    seen = set(executed_signatures)

    def reject(request: ToolRequest, reason: RejectionReason, detail: str) -> None:
        """Record one refused request."""
        rejections.append(
            ToolRejection(
                request_id=request.request_id, tool=request.tool, origin=request.origin, reason=reason, detail=detail
            )
        )

    for request in requests:
        spec = ALLOWED_TOOLS.get(request.tool)
        if spec is None:
            reject(request, "not_allowed", f"tool {request.tool!r} is not on the allow-list {sorted(ALLOWED_TOOLS)}")
            continue
        if request.origin == "llm" and request.tool not in LLM_REQUESTABLE_TOOLS:
            reject(
                request,
                "not_permitted_for_llm",
                f"the LLM may only request {sorted(LLM_REQUESTABLE_TOOLS)}, not {request.tool!r}",
            )
            continue
        try:
            validated = spec.input_model.model_validate(request.arguments).model_dump(mode="json")
        except ValidationError as error:
            reject(
                request, "invalid_arguments", f"{error.error_count()} validation error(s): {error.errors()[0]['msg']}"
            )
            continue
        signature = call_signature(request.tool, validated)
        if signature in seen:
            reject(request, "repeated_call", "an identical call was already made in this run (loop guard)")
            continue
        if executed_count + len(approved) >= max_calls:
            reject(request, "call_budget_exhausted", f"the run already used its {max_calls} tool calls")
            continue
        seen.add(signature)
        approved.append(
            ApprovedCall(
                result_id=f"TR-{executed_count + len(approved) + 1}",
                request=request,
                arguments=validated,
                signature=signature,
            )
        )
    return approved, rejections


def execute_call(call: ApprovedCall, handlers: Mapping[str, ToolHandler]) -> ToolResult:
    """Run one approved call and record its output or its failure."""
    started = time.perf_counter()
    status: Literal["ok", "error", "unavailable"] = "ok"
    output: dict[str, Any] | None = None
    error_text: str | None = None
    try:
        result = call_tool(call.request.tool, call.arguments, handler_overrides=handlers)
        output = result.model_dump(mode="json")
    except ToolUnavailable as error:
        status, error_text = "unavailable", str(error)
    except ValidationError as error:
        status, error_text = "error", f"tool output failed validation: {error.error_count()} error(s)"
    except Exception as error:  # noqa: BLE001 - a failing tool must be recorded and shown to a human, not crash the run
        status, error_text = "error", f"{type(error).__name__}: {error}"
    return ToolResult(
        result_id=call.result_id,
        request_id=call.request.request_id,
        tool=call.request.tool,
        origin=call.request.origin,
        purpose=call.request.purpose,
        party_id=call.request.party_id,
        arguments=call.arguments,
        signature=call.signature,
        status=status,
        output=output,
        error=error_text,
        latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
    )
