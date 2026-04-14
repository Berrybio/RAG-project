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
