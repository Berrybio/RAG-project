from fastapi import APIRouter, Depends

from ..config import settings
from ..dependencies import get_pipeline_manager
from ..models.schemas import (
    CancerTypeInfo,
    CancerTypesResponse,
    HealthResponse,
)
from ..core.pipeline_manager import PipelineManager

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    manager: PipelineManager = Depends(get_pipeline_manager),
):
    loaded = manager.available_cancer_types
    total_trials = sum(
        ct["trial_count"] for ct in loaded if ct.get("trial_count")
    )
    loaded_details = [
        {
            "key": ct["key"],
            "display_name": ct["display_name"],
            "trials_loaded": ct["trial_count"],
        }
        for ct in loaded
        if ct.get("loaded")
    ]
    # Use the first loaded pipeline's model as representative.
    model = "unknown"
    for ct_key in manager.loaded_cancer_types:
        try:
            pipeline = manager._pipelines[ct_key]
            model = pipeline.model
            break
        except Exception:
            pass

    return HealthResponse(
        status="ok",
        total_trials_loaded=total_trials,
        loaded_cancer_types=loaded_details,
        available_cancer_types=len(loaded),
        model=model,
        data_source="gcs",
    )


@router.get("/cancer-types", response_model=CancerTypesResponse)
async def list_cancer_types(
    manager: PipelineManager = Depends(get_pipeline_manager),
):
    """Return the list of available cancer types for the frontend dropdown."""
    items = [
        CancerTypeInfo(**ct) for ct in manager.available_cancer_types
    ]
    return CancerTypesResponse(
        cancer_types=items,
        default=settings.default_cancer_type,
    )
