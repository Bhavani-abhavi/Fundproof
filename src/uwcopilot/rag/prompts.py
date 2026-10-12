"""The prompt registry: every prompt that produces a measured answer has an id and a content hash.

Results files, the migration certificate and the release gate (docs/innovations.md, S1 and S3) refer to a prompt by
its id (for example "rag_answer_v1") and record prompt_sha256(id). If anyone edits the prompt text, the hash
changes, so a number measured with the old text can be recognised as stale instead of silently describing the new
prompt.

The registry only points at the existing constants in rag/prompt.py; it never copies or changes them:

    PROMPTS = {"rag_answer_v1": (SYSTEM_PROMPT, ANSWER_INSTRUCTION)}

prompt_sha256(id) is the SHA-256 of the canonical JSON of {"id", "system", "instruction"} (sorted keys, no extra
whitespace, UTF-8), the same canonical form the LLM cache uses for its keys.
"""

import hashlib
import json

from uwcopilot.rag.prompt import ANSWER_INSTRUCTION, SYSTEM_PROMPT

PROMPTS: dict[str, tuple[str, str]] = {
    "rag_answer_v1": (SYSTEM_PROMPT, ANSWER_INSTRUCTION),
}


class UnknownPromptError(KeyError):
    """The prompt id is not in the registry."""


def get_prompt(prompt_id: str) -> tuple[str, str]:
    """(system prompt, answer instruction) for a registered prompt id."""
    if prompt_id not in PROMPTS:
        raise UnknownPromptError(f"Unknown prompt id {prompt_id!r}; registered: {sorted(PROMPTS)}.")
    return PROMPTS[prompt_id]


def canonical_prompt_json(prompt_id: str) -> str:
    """The canonical JSON text that is hashed for a prompt id."""
    system, instruction = get_prompt(prompt_id)
    document = {"id": prompt_id, "system": system, "instruction": instruction}
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def prompt_sha256(prompt_id: str) -> str:
    """SHA-256 (hex) of the canonical JSON of a registered prompt."""
    return hashlib.sha256(canonical_prompt_json(prompt_id).encode("utf-8")).hexdigest()


def all_prompt_hashes() -> dict[str, str]:
    """{prompt id: sha256} for every registered prompt (for provenance blocks and the release gate)."""
    hashes: dict[str, str] = {}
    for prompt_id in sorted(PROMPTS):
        hashes[prompt_id] = prompt_sha256(prompt_id)
    return hashes
