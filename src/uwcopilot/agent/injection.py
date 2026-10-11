"""Injection screen for the application's free-text fields (NBC-9.2, HM-11.2).

The notes field and the names on an application are untrusted data. This module looks for text that tries to
instruct a reviewer, a system or an AI assistant ("ignore previous instructions", "already approved",
"set the recommendation to approve"). It is a tripwire, not the defense itself:

- a flag sends the file to Fraud Operations / the SIU as a blocking referral (the guidelines treat embedded
  instructions as a document integrity indicator);
- the real protection is structural: free text never chooses a tool call (tool_plan.py reads only structured
  fields), never changes a calculation or the policy recommendation, and reaches the LLM only escaped and quoted
  as data (memo.py). An injection the detector misses still cannot do any of those things.

The patterns were fixed before the synthetic applications were generated and were not tuned on them. They reuse
the retrieval-side patterns from uwcopilot.rag.prompt and add three that target decisions on an application.
"""

import re

from pydantic import BaseModel, ConfigDict

from uwcopilot.agent.applications import CreditApplication, InsuranceApplication
from uwcopilot.rag.prompt import INSTRUCTION_PATTERNS, InstructionPattern

MAX_EXCERPT_CHARS = 200

AGENT_PATTERNS: tuple[InstructionPattern, ...] = (
    InstructionPattern(
        "decision_directive",
        re.compile(
            r"\b(?:approve|accept|quote|bind|fund)\s+(?:this|the)\s+"
            r"(?:application|file|loan|account|submission|request|deal)\b",
            re.IGNORECASE,
        ),
    ),
    InstructionPattern(
        "recommendation_directive",
        re.compile(
            r"\b(?:set|change|make|record)\s+(?:the\s+|your\s+)?"
            r"(?:recommendation|decision|outcome|status|risk band)\s+(?:to|as)\b",
            re.IGNORECASE,
        ),
    ),
    InstructionPattern(
        "addressed_to_ai",
        re.compile(
            r"\b(?:to|for|dear|attention)\s+(?:the\s+)?(?:ai|a\.i\.|assistant|language model|llm|chatbot|bot|"
            r"automated (?:reviewer|system|underwriter))\b",
            re.IGNORECASE,
        ),
    ),
)
ALL_PATTERNS: tuple[InstructionPattern, ...] = INSTRUCTION_PATTERNS + AGENT_PATTERNS


class InjectionHit(BaseModel):
    """One free-text field with instruction-like text. excerpt is untrusted text, kept for the audit record."""

    model_config = ConfigDict(extra="forbid")

    field: str
    patterns: list[str]
    excerpt: str


def detect(text: str) -> list[str]:
    """Names of the instruction-like patterns found in text (empty when none)."""
    return [item.name for item in ALL_PATTERNS if item.pattern.search(text)]


def free_text_fields(application: CreditApplication | InsuranceApplication) -> dict[str, str]:
    """Every free-text field an applicant controls: the notes, the legal name and each owner's name."""
    fields = {"notes": application.notes, "applicant.legal_name": application.applicant.legal_name}
    for owner in application.owners:
        fields[f"owners.{owner.owner_id}.name"] = owner.name
    return fields


def screen_application(application: CreditApplication | InsuranceApplication) -> list[InjectionHit]:
    """Check every free-text field; return one hit per field that contains instruction-like text."""
    hits = []
    for field, text in free_text_fields(application).items():
        patterns = detect(text)
        if patterns:
            hits.append(InjectionHit(field=field, patterns=patterns, excerpt=text[:MAX_EXCERPT_CHARS]))
    return hits
