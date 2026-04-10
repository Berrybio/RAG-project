import argparse
import os
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic


# ---------------------------------------------------------------------------
# 1. LOAD & PREPARE DATA
# ---------------------------------------------------------------------------

def load_clinical_trials(csv_path: str) -> pd.DataFrame:
    """Load the clinical trials CSV and fill missing values."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.fillna("")
    return df


def build_documents(df: pd.DataFrame) -> list[dict]:
    """
    Convert each clinical trial row into a 'document' dict with:
      - doc_id:   the NCT ID
      - text:     a combined text blob used for retrieval / embedding
      - metadata: key fields for display and filtering
    """
    documents = []

    for _, row in df.iterrows():
        text_parts = [
            f"Title: {row.get('bTitle', '')}",
            f"Official Title: {row.get('oTitle', '')}",
            f"Summary: {row.get('summary', '')}",
            f"Description: {row.get('description', '')}",
            f"Conditions: {row.get('conditions', '')}",
            f"Keywords: {row.get('keywords', '')}",
            f"Intervention: {row.get('interventionName', '')}",
            f"Intervention Description: {row.get('interventionDescription', '')}",
            f"Inclusion Criteria: {row.get('includeCriteria', '')}",
            f"Exclusion Criteria: {row.get('excludeCriteria', '')}",
            f"Phase: {row.get('phases', '')}",
            f"Status: {row.get('oStatus', '')}",
            f"Location: {row.get('locationInfo', '')}",
        ]
        combined_text = "\n".join(text_parts)

        documents.append({
            "doc_id": row.get("nctId", ""),
            "text": combined_text,
            "metadata": {
                "nctId": row.get("nctId", ""),
                "title": row.get("bTitle", ""),
                "phases": row.get("phases", ""),
                "status": row.get("oStatus", ""),
                "conditions": row.get("conditions", ""),
                "interventionName": row.get("interventionName", ""),
                "enrollmentCont": row.get("enrollmentCont", ""),
                "sex": row.get("sex", ""),
                "minimumAge": row.get("minimumAge", ""),
                "locationInfo": str(row.get("locationInfo", ""))[:300],
                "contactInfo": str(row.get("contactInfo", ""))[:300],
            },
        })

    return documents


# row 1 in .csv as an example  #
# {
#    "nctId": "NCT04567420",
#    "title": "DNA-Guided Second Line Adjuvant Therapy...",
#    "phases": "PHASE2",
#    "status": "RECRUITING",
#    "conditions": "Breast Cancer",
#    "interventionName": "Palbociclib, Fulvestrant, Adjuvant Therapy",
#    "enrollmentCont": 100,
#    "sex": "ALL",
#    "minimumAge": "",
#    "locationInfo": "University of Arizona Cancer Center...",  # truncated to 300 chars
#    "contactInfo": "Femi Okubanjo (CONTACT) - fokubanjo..."   # truncated to 300 chars
# }

# access to it 
# doc = documents[0]
# top levels keys:
# doc["doc_id"]
# doc["text"]
# doc["metadata"]["title"]

