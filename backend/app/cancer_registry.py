"""Central registry of supported cancer types.

Each entry maps a snake_case key to its display name and the condition
query string used against the ClinicalTrials.gov API v2.  The key also
determines the GCS path layout::

    gs://<bucket>/data/<cancer_type>/trials.csv
    gs://<bucket>/data/<cancer_type>/embeddings_cache/
"""

from __future__ import annotations

CANCER_TYPES: dict[str, dict[str, str]] = {
    "breast_cancer": {
        "display_name": "Breast Cancer",
        "condition_query": "Breast Cancer",
    },
    "lung_cancer": {
        "display_name": "Lung Cancer",
        "condition_query": "Lung Cancer",
    },
    "lymphoma": {
        "display_name": "Lymphoma",
        "condition_query": "Lymphoma",
    },
    "leukemia": {
        "display_name": "Leukemia",
        "condition_query": "Leukemia",
    },
    "colorectal_cancer": {
        "display_name": "Colorectal Cancer",
        "condition_query": "Colon Cancer",
    },
    "prostate_cancer": {
        "display_name": "Prostate Cancer",
        "condition_query": "Prostate Cancer",
    },
    "liver_cancer": {
        "display_name": "Liver Cancer",
        "condition_query": "Liver Cancer",
    },
    "ovarian_cancer": {
        "display_name": "Ovarian Cancer",
        "condition_query": "Ovarian Cancer",
    },
    "pancreatic_cancer": {
        "display_name": "Pancreatic Cancer",
        "condition_query": "Pancreatic Cancer",
    },
    "brain_cancer": {
        "display_name": "Brain Cancer",
        "condition_query": "Brain Cancer",
    },
    "stomach_cancer": {
        "display_name": "Stomach Cancer",
        "condition_query": "Stomach Cancer",
    },
    "melanoma": {
        "display_name": "Melanoma",
        "condition_query": "Melanoma",
    },
    "bladder_cancer": {
        "display_name": "Bladder Cancer",
        "condition_query": "Bladder Cancer",
    },
}


def gcs_csv_blob(cancer_type: str) -> str:
    """Return the GCS blob path for a cancer type's trial CSV."""
    return f"data/{cancer_type}/trials.csv"


def gcs_embeddings_prefix(cancer_type: str) -> str:
    """Return the GCS blob prefix for a cancer type's embeddings cache."""
    return f"data/{cancer_type}/embeddings_cache/"


def validate_cancer_type(cancer_type: str) -> None:
    """Raise ValueError if the cancer type is not in the registry."""
    if cancer_type not in CANCER_TYPES:
        valid = ", ".join(sorted(CANCER_TYPES))
        raise ValueError(
            f"Unknown cancer type {cancer_type!r}. Valid types: {valid}"
        )
