import os
import tempfile

import pandas as pd
import pytest

from app.core.data import build_documents
from app.core.retriever import TFIDFRetriever


SAMPLE_DATA = [
    {
        "nctId": "NCT00000001",
        "orgID": "ORG001",
        "acronym": "CDK-TRIAL",
        "bTitle": "CDK4/6 Inhibitor Trial for HR+ Breast Cancer",
        "oTitle": "Phase II Study of Palbociclib in HR+/HER2- Breast Cancer",
        "sponsorName": "National Cancer Institute",
        "sponsorClass": "NIH",
        "oStatus": "RECRUITING",
        "startDate": "2023-01-15",
        "lastUpdateDate": "2024-01-15",
        "summary": "A study of CDK4/6 inhibitor palbociclib in hormone receptor positive breast cancer",
        "description": "This trial evaluates the efficacy of palbociclib combined with letrozole",
        "conditions": "Breast Cancer",
        "keywords": "CDK4/6, palbociclib, HR+, HER2-",
        "meshTermsCondition": "Breast Neoplasms",
        "meshTermsIntervention": "Palbociclib",
        "studyType": "INTERVENTIONAL",
        "phases": "PHASE2",
        "allocation": "RANDOMIZED",
        "interventionModel": "PARALLEL",
        "primaryPurpose": "TREATMENT",
        "masking": "DOUBLE",
        "enrollmentCont": 100,
        "armGroups": "Arm A: EXPERIMENTAL - Palbociclib + Letrozole",
        "interventionName": "Palbociclib",
        "interventionType": "DRUG",
        "interventionDescription": "Palbociclib 125mg daily for 21 days of 28-day cycle",
        "interventionOtherNames": "IBRANCE",
        "includeCriteria": "Age >= 18, HR+/HER2- breast cancer",
        "excludeCriteria": "Prior CDK4/6 inhibitor therapy",
        "sex": "FEMALE",
        "minimumAge": "18 Years",
        "primaryOutcomes": "Progression Free Survival [24 months]",
        "secondaryOutcomes": "Overall Survival [48 months]",
        "piName": "Dr. Smith",
        "piAffiliation": "Memorial Sloan Kettering Cancer Center",
        "contactInfo": "Dr. Smith, smith@msk.org",
        "locationInfo": "Memorial Sloan Kettering (RECRUITING) - New York, New York, United States",
        "locationCount": 3,
        "locationCountries": "United States",
    },
    {
        "nctId": "NCT00000002",
        "orgID": "ORG002",
        "bTitle": "Immunotherapy for Triple-Negative Breast Cancer",
        "oTitle": "Phase II Pembrolizumab in TNBC",
        "sponsorName": "Merck Sharp & Dohme",
        "sponsorClass": "INDUSTRY",
        "oStatus": "RECRUITING",
        "startDate": "2023-06-01",
        "lastUpdateDate": "2024-02-20",
        "summary": "Pembrolizumab in triple-negative breast cancer patients",
        "description": "Evaluating checkpoint inhibitor immunotherapy in TNBC",
        "conditions": "Triple Negative Breast Cancer",
        "keywords": "TNBC, immunotherapy, pembrolizumab, checkpoint inhibitor",
        "meshTermsCondition": "Triple Negative Breast Neoplasms",
        "meshTermsIntervention": "Pembrolizumab",
        "studyType": "INTERVENTIONAL",
        "phases": "PHASE2",
        "allocation": "NON_RANDOMIZED",
        "primaryPurpose": "TREATMENT",
        "masking": "NONE",
        "enrollmentCont": 75,
        "interventionName": "Pembrolizumab",
        "interventionType": "DRUG",
        "interventionDescription": "Pembrolizumab 200mg IV every 3 weeks",
        "interventionOtherNames": "Keytruda",
        "includeCriteria": "Age >= 18, TNBC confirmed",
        "excludeCriteria": "Autoimmune disease",
        "sex": "ALL",
        "minimumAge": "18 Years",
        "primaryOutcomes": "Objective Response Rate [12 months]",
        "piName": "Dr. Jones",
        "piAffiliation": "MD Anderson Cancer Center",
        "contactInfo": "Dr. Jones, jones@mdanderson.org",
        "locationInfo": "MD Anderson (RECRUITING) - Houston, Texas, United States",
        "locationCount": 5,
        "locationCountries": "United States",
    },
    {
        "nctId": "NCT00000003",
        "orgID": "ORG003",
        "acronym": "DESTINY",
        "bTitle": "HER2-Positive Breast Cancer Targeted Therapy",
        "oTitle": "Phase II Trastuzumab Deruxtecan in HER2+ BC",
        "sponsorName": "Daiichi Sankyo",
        "sponsorClass": "INDUSTRY",
        "oStatus": "RECRUITING",
        "startDate": "2023-09-15",
        "lastUpdateDate": "2024-03-10",
        "summary": "T-DXd in previously treated HER2-positive breast cancer",
        "description": "Antibody-drug conjugate study for HER2-positive disease",
        "conditions": "HER2 Positive Breast Cancer",
        "keywords": "HER2+, T-DXd, trastuzumab deruxtecan, ADC",
        "meshTermsCondition": "Breast Neoplasms",
        "meshTermsIntervention": "Trastuzumab",
        "studyType": "INTERVENTIONAL",
        "phases": "PHASE2",
        "allocation": "RANDOMIZED",
        "primaryPurpose": "TREATMENT",
        "masking": "NONE",
        "enrollmentCont": 120,
        "interventionName": "Trastuzumab Deruxtecan",
        "interventionType": "DRUG",
        "interventionDescription": "T-DXd 5.4mg/kg IV every 3 weeks",
        "interventionOtherNames": "Enhertu, T-DXd",
        "includeCriteria": "Age >= 18, HER2+ breast cancer, prior trastuzumab",
        "excludeCriteria": "ILD history",
        "sex": "FEMALE",
        "minimumAge": "18 Years",
        "maximumAge": "85 Years",
        "primaryOutcomes": "Progression Free Survival [18 months]",
        "piName": "Dr. Lee",
        "piAffiliation": "Dana-Farber Cancer Institute",
        "contactInfo": "Dr. Lee, lee@dfci.org",
        "locationInfo": "Dana-Farber Cancer Institute (RECRUITING) - Boston, Massachusetts, United States",
        "locationCount": 8,
        "locationCountries": "United States, United Kingdom",
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
