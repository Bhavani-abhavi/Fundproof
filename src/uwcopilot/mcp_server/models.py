"""Output models that exist only on the MCP side: a full guideline section, and the score tool's result envelope.

The search and screening tools return the shared tool models from uwcopilot.tools.schemas unchanged
(GuidelineSearchResult and ScreeningResult). Two MCP tools need their own shape:

- get_guideline_section returns one whole section (GuidelineSection), not a ranked search hit.
- score_application must be able to say "the risk service could not be reached" without inventing a score, so it
  returns ScoreApplicationResult: either status "scored" with the service's RiskScoreResponse, or status "error"
  with a RiskApiErrorInfo (defined with the shared risk client, uwcopilot.risk.client) and no score at all.
"""

from typing import Literal

from pydantic import Field, model_validator

from uwcopilot.risk.client import RiskApiErrorInfo
from uwcopilot.tools.schemas import SECTION_ID_PATTERN, SOURCE_DOC_PATTERN, RiskScoreResponse, StrictOutput


class GuidelineSection(StrictOutput):
    """One complete SYNTHETIC guideline section, with the citation and the MCP resource URI that point to it."""

    section_id: str = Field(pattern=SECTION_ID_PATTERN)
    title: str = Field(min_length=1)
    source_doc: str = Field(pattern=SOURCE_DOC_PATTERN)
    doc_title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    citation: str
    resource_uri: str
    source_file: str
    synthetic: bool

    @model_validator(mode="after")
    def citation_matches_section(self) -> "GuidelineSection":
        """The citation is always "[<section_id>]"."""
        if self.citation != f"[{self.section_id}]":
            raise ValueError(f"citation {self.citation!r} does not match section_id {self.section_id!r}")
        return self


class ScoreApplicationResult(StrictOutput):
    """Either a score from the risk service (status "scored") or the reason there is none (status "error").

    There is no third state: an error result can never carry a score, so a caller cannot mistake a failed call for
    a low-risk applicant.
    """

    status: Literal["scored", "error"]
    score: RiskScoreResponse | None = None
    error: RiskApiErrorInfo | None = None

    @model_validator(mode="after")
    def exactly_one_of_score_or_error(self) -> "ScoreApplicationResult":
        """status "scored" needs a score and no error; status "error" needs an error and no score."""
        if self.status == "scored" and (self.score is None or self.error is not None):
            raise ValueError("a 'scored' result must have a score and no error")
        if self.status == "error" and (self.error is None or self.score is not None):
            raise ValueError("an 'error' result must have an error and no score")
        return self
