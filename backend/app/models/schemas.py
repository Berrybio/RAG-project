from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class SourceDoc(BaseModel):
    nct_id: str
    title: str
    score: float
    # Status & design
    phases: str
    status: str
    study_type: str = ""
    allocation: str = ""
    masking: str = ""
    primary_purpose: str = ""
    # Conditions
    conditions: str
    keywords: str = ""
    # Interventions
    intervention: str
    intervention_type: str = ""
    drug_aliases: str = ""
    # Enrollment & eligibility
    enrollment: str
    enrollment_type: str = ""
    sex: str
    minimum_age: str
    maximum_age: str = ""
    # Outcomes
    primary_outcomes: str = ""
    # Sponsor & PI
    sponsor: str = ""
    sponsor_class: str = ""
    pi_name: str = ""
    pi_affiliation: str = ""
    # Dates
    start_date: str = ""
    completion_date: str = ""
    last_updated: str = ""
    # Location & contact
    location: str
    location_countries: str = ""
    location_count: int = 0
    contact: str


class SearchResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]


class ProtocolRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class ProtocolResponse(BaseModel):
    protocol: dict
    reference_trials: list[SourceDoc]


class DocxRequest(BaseModel):
    protocol: dict


class RefinementMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class RefineProtocolRequest(BaseModel):
    protocol: dict
    refinement_messages: list[RefinementMessage]
    original_summary: str = ""


class RefineProtocolResponse(BaseModel):
    protocol: dict
    assistant_message: str
    changed_fields: list[str]


class HealthResponse(BaseModel):
    status: str
    trials_loaded: int
    model: str


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class FeedbackContext(BaseModel):
    """Snapshot of the conversation around a feedback event so an admin
    reviewing the queue later can see what the user was reacting to."""
    query: str = ""
    assistant_message: str = ""
    # Server-side truncation happens in the route handler; allow extra fields
    # silently in case the frontend later attaches NCT IDs or trial slices.
    model_config = {"extra": "ignore"}


class RatingFeedback(BaseModel):
    type: Literal["rating_up", "rating_down"]
    context: FeedbackContext = Field(default_factory=FeedbackContext)
    reason: str = ""  # optional free text on a thumbs-down


class CorrectionFeedback(BaseModel):
    type: Literal["correction"]
    correction_kind: Literal["missing_alias", "inaccurate_trial", "other"]
    # For correction_kind="missing_alias":
    alias: str = ""
    canonical: str = ""
    # For all kinds: free-text notes from the user
    notes: str = ""
    context: FeedbackContext = Field(default_factory=FeedbackContext)


# Discriminated union via the `type` field.
FeedbackRequest = RatingFeedback | CorrectionFeedback


class FeedbackResponse(BaseModel):
    id: str
    status: str = "open"


class FeedbackListResponse(BaseModel):
    entries: list[dict]
    stats: dict


class PromoteAliasRequest(BaseModel):
    feedback_id: str
    alias: str
    canonical: str


class AliasesResponse(BaseModel):
    aliases: dict[str, str]


class FeedbackStatusUpdate(BaseModel):
    status: Literal["open", "resolved", "dismissed"]
