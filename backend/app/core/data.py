import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_clinical_trials(csv_path: str) -> pd.DataFrame:
    """Load the clinical trials CSV and fill missing values."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.fillna("")
    logger.info("Loaded %d trials from %s", len(df), csv_path)
    return df


def build_documents(df: pd.DataFrame) -> list[dict]:
    """Convert each clinical trial row into a document dict with text + metadata."""
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

    logger.info("Built %d documents", len(documents))
    return documents
