"""Grounded answering: retrieve, prompt with delimited documents, call the LLM, check the citations.

answer() is the whole loop for one question. answer_from_hits() does the last three steps when the hits are
already known (for example when an agent called the search tool itself).

If retrieval returns nothing, no LLM call is made and the answer is the not-found sentence. The result always
says which provider and model produced the text, and whether it was the fake test client (is_fake), so fake
output can never be presented as a model's work.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from uwcopilot.llm.base import LLMClient
from uwcopilot.rag.citations import CitationCheck, check_citations
from uwcopilot.rag.prompt import NOT_FOUND_ANSWER, build_answer_prompt
from uwcopilot.rag.retriever import RetrievalMode, Retriever, to_guideline_hits
from uwcopilot.rag.search import DEFAULT_MODE, get_default_retriever
from uwcopilot.tools.schemas import GuidelineHit

# Headroom for adaptive thinking, which counts toward max_tokens on current Claude models; a cut-off answer raises
# IncompleteLLMResponse instead of being returned (llm/base.py).
DEFAULT_ANSWER_MAX_TOKENS = 4000


@dataclass(frozen=True)
class GroundedAnswer:
    """An answer with everything needed to audit it."""

    question: str
    text: str
    hits: list[GuidelineHit]
    citation_check: CitationCheck
    instruction_flags: dict[str, list[str]] = field(default_factory=dict)
    llm_called: bool = False
    provider: str | None = None
    model: str | None = None
    is_fake: bool = False

    @property
    def retrieved_ids(self) -> list[str]:
        """Ids of the documents that were given to the model."""
        return [hit.section_id for hit in self.hits]


def answer_from_hits(
    question: str,
    hits: Sequence[GuidelineHit],
    llm_client: LLMClient,
    max_tokens: int = DEFAULT_ANSWER_MAX_TOKENS,
) -> GroundedAnswer:
    """Prompt the LLM with the hits as delimited documents and check the citations in its answer."""
    hit_list = list(hits)
    if not hit_list:
        return GroundedAnswer(
            question=question,
            text=NOT_FOUND_ANSWER,
            hits=[],
            citation_check=check_citations(NOT_FOUND_ANSWER, []),
        )
    system, messages, flags = build_answer_prompt(question, hit_list)
    response = llm_client.complete(system=system, messages=messages, max_tokens=max_tokens, temperature=0)
    return GroundedAnswer(
        question=question,
        text=response.text,
        hits=hit_list,
        citation_check=check_citations(response.text, [hit.section_id for hit in hit_list]),
        instruction_flags=flags,
        llm_called=True,
        provider=response.provider,
        model=response.model,
        is_fake=response.is_fake,
    )


def answer(
    question: str,
    llm_client: LLMClient,
    k: int = 5,
    *,
    retriever: Retriever | None = None,
    mode: RetrievalMode = DEFAULT_MODE,
    source_docs: Sequence[str] | None = None,
    max_tokens: int = DEFAULT_ANSWER_MAX_TOKENS,
) -> GroundedAnswer:
    """Retrieve the top k chunks for question and answer from them with citations."""
    active = retriever or get_default_retriever()
    results = active.search(question, mode=mode, top_k=k, source_docs=source_docs)
    return answer_from_hits(question, to_guideline_hits(results), llm_client, max_tokens=max_tokens)
