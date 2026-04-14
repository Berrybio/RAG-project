from app.core.data import load_clinical_trials, build_documents


def test_load_clinical_trials(sample_csv):
    df = load_clinical_trials(sample_csv)
    assert len(df) == 3
    assert "nctId" in df.columns
    assert df.isna().sum().sum() == 0  # no NaN after fillna


def test_build_documents(sample_csv):
    df = load_clinical_trials(sample_csv)
    docs = build_documents(df)
    assert len(docs) == 3

    doc = docs[0]
    assert "doc_id" in doc
    assert "text" in doc
    assert "metadata" in doc
    assert doc["doc_id"] == "NCT00000001"
    assert "CDK4/6" in doc["text"]

    meta = doc["metadata"]
    assert meta["nctId"] == "NCT00000001"
    assert meta["phases"] == "PHASE2"
    assert meta["status"] == "RECRUITING"

    # New fields in text (for TF-IDF indexing)
    assert "IBRANCE" in doc["text"]           # drug alias
    assert "MeSH Condition Terms" in doc["text"]  # MeSH terms
    assert "Sponsor" in doc["text"]
    assert "Principal Investigator" in doc["text"]
    assert "Primary Outcomes" in doc["text"]

    # New fields in metadata
    assert meta["sponsorName"] == "National Cancer Institute"
    assert meta["sponsorClass"] == "NIH"
    assert meta["piName"] == "Dr. Smith"
    assert meta["studyType"] == "INTERVENTIONAL"
    assert meta["allocation"] == "RANDOMIZED"
    assert meta["interventionOtherNames"] == "IBRANCE"
    assert meta["primaryOutcomes"] != ""
    assert meta["locationCountries"] == "United States"
