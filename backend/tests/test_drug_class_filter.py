"""Tests for drug-class extraction, query expansion, and intersection filtering."""
from app.core.pipeline import (
    _doc_matches_drug_classes,
    _expand_query_with_drug_classes,
    _extract_drug_classes,
)


def test_extract_adc_alone():
    assert _extract_drug_classes("Show me ADC trials") == ["ADC (antibody-drug conjugate)"]


def test_extract_adc_with_punctuation():
    # "ADC +" must still match — punctuation around the alias shouldn't break
    # the boundary check.
    out = _extract_drug_classes("ADC + checkpoint inhibitor combinations in TNBC")
    assert "ADC (antibody-drug conjugate)" in out
    assert "PD-(L)1 / immune checkpoint" in out


def test_extract_full_phrase():
    assert _extract_drug_classes("antibody-drug conjugate trials") == [
        "ADC (antibody-drug conjugate)"
    ]


def test_extract_pd1_alias():
    assert _extract_drug_classes("anti-PD-1 monotherapy") == [
        "PD-(L)1 / immune checkpoint"
    ]


def test_extract_parp():
    assert _extract_drug_classes("Phase 3 PARP inhibitor for BRCA") == [
        "PARP inhibitor"
    ]


def test_extract_cdk46_with_slash():
    # CDK4/6 contains a slash; the lenient boundary must still match.
    assert _extract_drug_classes("CDK4/6 inhibitor in HR+") == ["CDK4/6 inhibitor"]


def test_no_false_match_inside_word():
    # "advanced" contains "adc" as a substring; we must NOT match it.
    assert _extract_drug_classes("advanced metastatic breast cancer") == []


def test_no_match_returns_empty():
    assert _extract_drug_classes("Phase 2 trials in TNBC") == []


def test_expansion_appends_drug_names():
    classes = ["ADC (antibody-drug conjugate)"]
    out = _expand_query_with_drug_classes("ADC trials", classes)
    assert "Datopotamab deruxtecan" in out
    assert "Sacituzumab govitecan" in out
    # Original query preserved verbatim, expansions appended.
    assert out.startswith("ADC trials ")


def test_expansion_skips_drugs_already_in_query():
    classes = ["ADC (antibody-drug conjugate)"]
    out = _expand_query_with_drug_classes(
        "trials with Sacituzumab govitecan in TNBC", classes
    )
    # Should still expand with other drugs in the class but skip Sacituzumab govitecan
    assert out.lower().count("sacituzumab govitecan") == 1


def test_expansion_passthrough_when_no_classes():
    assert _expand_query_with_drug_classes("plain query", []) == "plain query"


def test_doc_matches_intersection():
    # A trial with both Datopotamab deruxtecan (ADC) and Durvalumab (PD-L1)
    # must match an "ADC + checkpoint" requirement under require_all=True.
    doc = {"metadata": {
        "interventionName": "Datopotamab deruxtecan, Durvalumab",
        "interventionOtherNames": "Dato-DXd",
    }}
    target = ["ADC (antibody-drug conjugate)", "PD-(L)1 / immune checkpoint"]
    assert _doc_matches_drug_classes(doc, target, require_all=True) is True
    assert _doc_matches_drug_classes(doc, target, require_all=False) is True


def test_doc_misses_intersection():
    # A trial with only an ADC must NOT satisfy "ADC + checkpoint" intersection.
    doc = {"metadata": {"interventionName": "Sacituzumab govitecan", "interventionOtherNames": ""}}
    target = ["ADC (antibody-drug conjugate)", "PD-(L)1 / immune checkpoint"]
    assert _doc_matches_drug_classes(doc, target, require_all=True) is False
    # But it satisfies any-match.
    assert _doc_matches_drug_classes(doc, target, require_all=False) is True


def test_doc_unrelated_classes():
    doc = {"metadata": {"interventionName": "Letrozole", "interventionOtherNames": ""}}
    target = ["ADC (antibody-drug conjugate)"]
    assert _doc_matches_drug_classes(doc, target, require_all=True) is False
    assert _doc_matches_drug_classes(doc, target, require_all=False) is False


def test_empty_target_passes():
    doc = {"metadata": {"interventionName": "Anything"}}
    assert _doc_matches_drug_classes(doc, [], require_all=True) is True
    assert _doc_matches_drug_classes(doc, [], require_all=False) is True
