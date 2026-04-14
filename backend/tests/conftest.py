import os
import tempfile

import pandas as pd
import pytest

from app.core.data import build_documents
from app.core.retriever import TFIDFRetriever


SAMPLE_DATA = [
    {
        "nctId": "NCT00000001",
        "bTitle": "CDK4/6 Inhibitor Trial for HR+ Breast Cancer",
        "oTitle": "Phase II Study of Palbociclib in HR+/HER2- Breast Cancer",
        "summary": "A study of CDK4/6 inhibitor palbociclib in hormone receptor positive breast cancer",
        "description": "This trial evaluates the efficacy of palbociclib combined with letrozole",
        "conditions": "Breast Cancer",
        "keywords": "CDK4/6, palbociclib, HR+, HER2-",
        "interventionName": "Palbociclib",
        "interventionDescription": "Palbociclib 125mg daily for 21 days of 28-day cycle",
        "includeCriteria": "Age >= 18, HR+/HER2- breast cancer",
        "excludeCriteria": "Prior CDK4/6 inhibitor therapy",
        "phases": "PHASE2",
        "oStatus": "RECRUITING",
        "locationInfo": "Memorial Sloan Kettering, New York",
        "contactInfo": "Dr. Smith, smith@msk.org",
        "enrollmentCont": 100,
        "sex": "FEMALE",
        "minimumAge": "18 Years",
    },
    {
        "nctId": "NCT00000002",
        "bTitle": "Immunotherapy for Triple-Negative Breast Cancer",
        "oTitle": "Phase II Pembrolizumab in TNBC",
        "summary": "Pembrolizumab in triple-negative breast cancer patients",
        "description": "Evaluating checkpoint inhibitor immunotherapy in TNBC",
        "conditions": "Triple Negative Breast Cancer",
        "keywords": "TNBC, immunotherapy, pembrolizumab, checkpoint inhibitor",
        "interventionName": "Pembrolizumab",
        "interventionDescription": "Pembrolizumab 200mg IV every 3 weeks",
        "includeCriteria": "Age >= 18, TNBC confirmed",
        "excludeCriteria": "Autoimmune disease",
        "phases": "PHASE2",
        "oStatus": "RECRUITING",
        "locationInfo": "MD Anderson, Houston, Texas",
        "contactInfo": "Dr. Jones, jones@mdanderson.org",
        "enrollmentCont": 75,
        "sex": "ALL",
        "minimumAge": "18 Years",
    },
    {
        "nctId": "NCT00000003",
        "bTitle": "HER2-Positive Breast Cancer Targeted Therapy",
        "oTitle": "Phase II Trastuzumab Deruxtecan in HER2+ BC",
        "summary": "T-DXd in previously treated HER2-positive breast cancer",
        "description": "Antibody-drug conjugate study for HER2-positive disease",
        "conditions": "HER2 Positive Breast Cancer",
        "keywords": "HER2+, T-DXd, trastuzumab deruxtecan, ADC",
        "interventionName": "Trastuzumab Deruxtecan",
        "interventionDescription": "T-DXd 5.4mg/kg IV every 3 weeks",
        "includeCriteria": "Age >= 18, HER2+ breast cancer, prior trastuzumab",
        "excludeCriteria": "ILD history",
        "phases": "PHASE2",
        "oStatus": "RECRUITING",
        "locationInfo": "Dana-Farber Cancer Institute, Boston",
        "contactInfo": "Dr. Lee, lee@dfci.org",
        "enrollmentCont": 120,
        "sex": "FEMALE",
        "minimumAge": "18 Years",
    },
]


@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV file with sample clinical trial data."""
    df = pd.DataFrame(SAMPLE_DATA)
    csv_path = tmp_path / "test_trials.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_documents():
    """Build documents from sample data."""
    df = pd.DataFrame(SAMPLE_DATA)
    return build_documents(df)


@pytest.fixture
def sample_retriever(sample_documents):
    """Create a TFIDFRetriever from sample documents."""
    return TFIDFRetriever(sample_documents)
