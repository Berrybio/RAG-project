from fastapi import APIRouter, Depends

from ..dependencies import get_pipeline
from ..models.schemas import HealthResponse
from ..core.pipeline import ClinicalTrialRAG

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(pipeline: ClinicalTrialRAG = Depends(get_pipeline)):
    return HealthResponse(
        status="ok",
        trials_loaded=len(pipeline.documents),
        model=pipeline.model,
    )
