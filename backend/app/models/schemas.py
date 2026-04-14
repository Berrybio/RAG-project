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


class HealthResponse(BaseModel):
    status: str
    trials_loaded: int
    model: str
