from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class SourceDoc(BaseModel):
    nct_id: str
    title: str
    score: float
    phases: str
    status: str
    conditions: str
    intervention: str
    enrollment: str
    sex: str
    minimum_age: str
    location: str
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
