from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    cancer_type: str = "breast_cancer"


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
    cancer_type: str = "breast_cancer"


class ProtocolResponse(BaseModel):
    protocol: dict
    reference_trials: list[SourceDoc]
    # Set when the protocol was auto-saved to history; lets the frontend
    # tag subsequent refinements with the same id so versions accumulate.
    protocol_id: str = ""
    version: int = 0
    title: str = ""


class DocxRequest(BaseModel):
    protocol: dict


class RefinementMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class RefineProtocolRequest(BaseModel):
    protocol: dict
    refinement_messages: list[RefinementMessage]
    original_summary: str = ""
    # Optional: id from the initial generation. When present, the refinement
    # is appended as a new version under the same history entry.
    protocol_id: str = ""


class RefineProtocolResponse(BaseModel):
    protocol: dict
    assistant_message: str
    changed_fields: list[str]
    protocol_id: str = ""
    version: int = 0


class HealthResponse(BaseModel):
    status: str
    total_trials_loaded: int
    loaded_cancer_types: list[dict] = Field(default_factory=list)
    available_cancer_types: int = 0
    model: str
    data_source: str = "gcs"


class CancerTypeInfo(BaseModel):
    key: str
    display_name: str
    trial_count: int | None = None
    loaded: bool = False


class CancerTypesResponse(BaseModel):
    cancer_types: list[CancerTypeInfo]
    default: str = "breast_cancer"


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class FeedbackContext(BaseModel):
    """Snapshot of the conversation around a feedback event so an admin
    reviewing the queue later can see what the user was reacting to.

    `source_nct_ids` is the list of trial NCT IDs that were shown to the user
    when they rated the answer — feeds the down-vote-aware reranker."""
    query: str = ""
    assistant_message: str = ""
    source_nct_ids: list[str] = Field(default_factory=list)
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


# ---------------------------------------------------------------------------
# Protocol history
# ---------------------------------------------------------------------------

class ProtocolEntry(BaseModel):
    """Lightweight metadata row for a stored protocol — what the history
    strip in the UI renders. Full protocol JSON loads on demand."""
    id: str
    title: str
    phase: str = ""
    conditions: str = ""
    study_design: str = ""
    intervention_name: str = ""
    summary_brief: str = ""
    source_nct_ids: list[str] = Field(default_factory=list)
    version_count: int = 1
    created_at: str
    last_modified: str


class ProtocolListResponse(BaseModel):
    protocols: list[ProtocolEntry]


class ProtocolVersionEntry(BaseModel):
    """One row in a protocol's version history. Lets the planner rehydrate
    every version chip (label + change note) when a protocol is loaded from
    a fresh session."""
    version: int
    protocol: dict
    user_request: str = ""
    assistant_note: str = ""
    changed_fields: list[str] = Field(default_factory=list)
    timestamp: str = ""


class ProtocolLoadResponse(BaseModel):
    """Returned when a stored protocol + meta are loaded for editing."""
    protocol: dict
    meta: ProtocolEntry
    version: int
    # Populated when the client requests ``?include_versions=true``. Each
    # entry has its own JSON + change-log metadata so the planner can
    # rebuild the full version chip strip and support compare-any-two
    # across the protocol's lifetime.
    all_versions: list[ProtocolVersionEntry] = Field(default_factory=list)
