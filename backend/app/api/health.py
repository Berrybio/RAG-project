import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends

from ..config import settings
from ..dependencies import get_pipeline
from ..models.schemas import HealthResponse
from ..core.pipeline import ClinicalTrialRAG

router = APIRouter()


def _csv_updated_at(csv_path: str) -> str | None:
    """Best-effort timestamp for when the trial data was last refreshed."""
    if settings.data_source == "gcs":
        try:
            from ..core.gcs_storage import get_csv_metadata
            meta = get_csv_metadata(settings.gcs_bucket, settings.gcs_csv_blob)
            return meta.get("updated")
        except Exception:
            pass
    p = Path(csv_path)
    if p.exists():
        mtime = p.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    return None


@router.get("/health", response_model=HealthResponse)
async def health_check(pipeline: ClinicalTrialRAG = Depends(get_pipeline)):
    return HealthResponse(
        status="ok",
        trials_loaded=len(pipeline.documents),
        model=pipeline.model,
        data_source=settings.data_source,
        csv_updated_at=_csv_updated_at(pipeline.csv_path),
    )
