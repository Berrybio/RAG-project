"""Tests for sponsor extraction + post-retrieval filtering."""
from app.core.pipeline import _doc_matches_sponsors, _extract_sponsors


def test_extract_full_name():
    assert _extract_sponsors("Show me AstraZeneca's ADC trials") == ["astrazeneca"]


def test_extract_case_insensitive():
    assert _extract_sponsors("any astrazeneca trials in TNBC?") == ["astrazeneca"]


def test_msd_collapses_to_merck():
    assert _extract_sponsors("MSD-sponsored studies") == ["merck"]
    assert _extract_sponsors("Merck trials") == ["merck"]


def test_daiichi_alias():
    assert _extract_sponsors("Daiichi Sankyo ADC trials") == ["daiichi sankyo"]
    assert _extract_sponsors("daiichi monotherapy") == ["daiichi sankyo"]


def test_ampersand_alias():
    # J&J → janssen. Word-boundary regex doesn't handle "&", so the matcher
    # falls back to substring containment for aliases with special chars.
    assert _extract_sponsors("J&J pipeline trials") == ["janssen"]


def test_no_match_returns_empty():
    assert _extract_sponsors("Phase II TNBC trials") == []


def test_longest_alias_wins():
    # "bristol myers squibb" canonicalizes to "bristol", and the shorter
    # alias "bms" should not double-match within the same query.
    out = _extract_sponsors("Bristol Myers Squibb trials")
    assert out == ["bristol"]


def test_doc_matches_substring():
    doc = {"metadata": {"sponsorName": "AstraZeneca AB"}}
    assert _doc_matches_sponsors(doc, ["astrazeneca"]) is True


def test_doc_does_not_match_unrelated_sponsor():
    doc = {"metadata": {"sponsorName": "Memorial Sloan Kettering Cancer Center"}}
    assert _doc_matches_sponsors(doc, ["astrazeneca"]) is False


def test_doc_matches_handles_missing_field():
    assert _doc_matches_sponsors({"metadata": {}}, ["astrazeneca"]) is False
    assert _doc_matches_sponsors({}, ["astrazeneca"]) is False


def test_pipeline_sponsor_filter_end_to_end(sample_csv, monkeypatch):
    """End-to-end: a query naming a sponsor that's present in the corpus
    must surface that sponsor's trial even though the conftest sample
    contains higher-similarity matches without sponsor context."""
    from unittest.mock import MagicMock

    from app.core.pipeline import ClinicalTrialRAG

    pipeline = ClinicalTrialRAG(
        csv_path=sample_csv,
        client=MagicMock(),
        retriever_type="tfidf",
        voyage_api_key="",
    )
    # The Daiichi Sankyo trial is the HER2+ T-DXd one (NCT00000003).
    results = pipeline.retrieve("Daiichi Sankyo trials in breast cancer", top_k=3)
    nct_ids = [r["doc_id"] for r in results]
    assert "NCT00000003" in nct_ids
    # All returned trials must be Daiichi-sponsored when the sponsor filter
    # actually matched something (it does for this fixture).
    sponsors = {(r["metadata"].get("sponsorName") or "").lower() for r in results}
    assert all("daiichi" in s for s in sponsors)
