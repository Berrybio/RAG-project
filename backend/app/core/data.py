import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _str(val) -> str:
    """Safely convert a value to string, treating NaN/None as empty."""
    if pd.isna(val):
        return ""
    return str(val).strip()


def load_clinical_trials(csv_path: str) -> pd.DataFrame:
    """Load the clinical trials CSV and fill missing values."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.fillna("")
    logger.info("Loaded %d trials from %s", len(df), csv_path)
    return df


def build_documents(df: pd.DataFrame) -> list[dict]:
    """Convert each clinical trial row into a document dict with text + metadata.

    The text field is used for TF-IDF indexing and retrieval.
    The metadata dict is passed through to the LLM context and API responses.
    """
    documents = []

    for _, row in df.iterrows():
        # ----- Searchable text (used for TF-IDF indexing) -----
        text_parts = [
            # Identification
            f"Title: {_str(row.get('bTitle', ''))}",
            f"Official Title: {_str(row.get('oTitle', ''))}",
            f"Acronym: {_str(row.get('acronym', ''))}",
            # Description
            f"Summary: {_str(row.get('summary', ''))}",
            f"Description: {_str(row.get('description', ''))}",
            # Conditions & keywords (critical for search)
            f"Conditions: {_str(row.get('conditions', ''))}",
            f"Keywords: {_str(row.get('keywords', ''))}",
            f"MeSH Condition Terms: {_str(row.get('meshTermsCondition', ''))}",
            f"MeSH Intervention Terms: {_str(row.get('meshTermsIntervention', ''))}",
            # Interventions (include aliases for keyword matching)
            f"Intervention: {_str(row.get('interventionName', ''))}",
            f"Intervention Type: {_str(row.get('interventionType', ''))}",
            f"Intervention Description: {_str(row.get('interventionDescription', ''))}",
            f"Drug Aliases: {_str(row.get('interventionOtherNames', ''))}",
            # Arms
            f"Arm Groups: {_str(row.get('armGroups', ''))}",
            # Eligibility
            f"Inclusion Criteria: {_str(row.get('includeCriteria', ''))}",
            f"Exclusion Criteria: {_str(row.get('excludeCriteria', ''))}",
            # Outcomes (important for "what does this trial measure" queries)
            f"Primary Outcomes: {_str(row.get('primaryOutcomes', ''))}",
            f"Secondary Outcomes: {_str(row.get('secondaryOutcomes', ''))}",
            # Study design
            f"Phase: {_str(row.get('phases', ''))}",
            f"Status: {_str(row.get('oStatus', ''))}",
            f"Study Type: {_str(row.get('studyType', ''))}",
            f"Allocation: {_str(row.get('allocation', ''))}",
            f"Intervention Model: {_str(row.get('interventionModel', ''))}",
            f"Primary Purpose: {_str(row.get('primaryPurpose', ''))}",
            f"Masking: {_str(row.get('masking', ''))}",
            # Sponsor & PI
            f"Sponsor: {_str(row.get('sponsorName', ''))}",
            f"Principal Investigator: {_str(row.get('piName', ''))}",
            f"PI Affiliation: {_str(row.get('piAffiliation', ''))}",
            # Location
            f"Location: {_str(row.get('locationInfo', ''))}",
            f"Countries: {_str(row.get('locationCountries', ''))}",
        ]
        combined_text = "\n".join(p for p in text_parts if not p.endswith(": "))

        # ----- Metadata (passed to LLM context and API) -----
        documents.append({
            "doc_id": _str(row.get("nctId", "")),
            "text": combined_text,
            "metadata": {
                # Identification
                "nctId": _str(row.get("nctId", "")),
                "title": _str(row.get("bTitle", "")),
                "officialTitle": _str(row.get("oTitle", "")),
                "acronym": _str(row.get("acronym", "")),
                # Status & Dates
                "phases": _str(row.get("phases", "")),
                "status": _str(row.get("oStatus", "")),
                "studyType": _str(row.get("studyType", "")),
                "startDate": _str(row.get("startDate", "")),
                "primaryCompletionDate": _str(row.get("primaryCompletionDate", "")),
                "completionETA": _str(row.get("completionETA", "")),
                "lastUpdateDate": _str(row.get("lastUpdateDate", "")),
                # Conditions
                "conditions": _str(row.get("conditions", "")),
                "keywords": _str(row.get("keywords", "")),
                "meshTermsCondition": _str(row.get("meshTermsCondition", "")),
                # Design
                "allocation": _str(row.get("allocation", "")),
                "interventionModel": _str(row.get("interventionModel", "")),
                "primaryPurpose": _str(row.get("primaryPurpose", "")),
                "masking": _str(row.get("masking", "")),
                "enrollmentCont": row.get("enrollmentCont", 0),
                "enrollmentType": _str(row.get("enrollmentType", "")),
                # Interventions
                "interventionName": _str(row.get("interventionName", "")),
                "interventionType": _str(row.get("interventionType", "")),
                "interventionOtherNames": _str(row.get("interventionOtherNames", "")),
                "armGroups": _str(row.get("armGroups", ""))[:500],
                # Eligibility
                "sex": _str(row.get("sex", "")),
                "minimumAge": _str(row.get("minimumAge", "")),
                "maximumAge": _str(row.get("maximumAge", "")),
                "healthyVolunteers": _str(row.get("healthyVolunteers", "")),
                # Outcomes
                "primaryOutcomes": _str(row.get("primaryOutcomes", ""))[:500],
                "secondaryOutcomes": _str(row.get("secondaryOutcomes", ""))[:500],
                # Sponsor & PI
                "sponsorName": _str(row.get("sponsorName", "")),
                "sponsorClass": _str(row.get("sponsorClass", "")),
                "piName": _str(row.get("piName", "")),
                "piAffiliation": _str(row.get("piAffiliation", "")),
                # Contacts & Locations
                "locationInfo": _str(row.get("locationInfo", ""))[:500],
                # Full location string (untruncated) used for state/city filtering.
                # Not surfaced in API responses; see pipeline filters.
                "locationInfoFull": _str(row.get("locationInfo", "")),
                "locationCountries": _str(row.get("locationCountries", "")),
                "locationCount": row.get("locationCount", 0),
                "contactInfo": _str(row.get("contactInfo", ""))[:300],
                # Oversight
                "isFdaRegulatedDrug": _str(row.get("isFdaRegulatedDrug", "")),
            },
        })

    logger.info("Built %d documents", len(documents))
    return documents
