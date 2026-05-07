"""Tests for the per-user protocol history module."""
from __future__ import annotations

import json

import pytest

from app.core.protocol_history import (
    ProtocolHistoryPaths,
    auto_title,
    list_protocols,
    load_protocol,
    match_query_to_entry,
    query_mentions_previous_protocol,
    save_initial,
    save_revision,
)


# ---------------------------------------------------------------------------
# Auto-title
# ---------------------------------------------------------------------------

def test_auto_title_combines_phase_population_intervention():
    p = {
        "title": "Some long official title",
        "phase": "Phase II",
        "conditions": "Triple-negative breast cancer (TNBC)",
        "intervention_name": "Trastuzumab deruxtecan (T-DXd)",
    }
    assert auto_title(p) == "Phase II — TNBC — Trastuzumab deruxtecan"


def test_auto_title_observational_uses_rwe_tag():
    p = {
        "title": "Observational registry",
        "phase": "N/A",
        "conditions": "HER2-low breast cancer",
        "intervention_name": "",
    }
    title = auto_title(p)
    assert title.startswith("RWE")
    assert "HER2-low" in title


def test_auto_title_falls_back_to_protocol_title_when_minimal():
    p = {"title": "A perfectly fine title", "phase": "", "conditions": "", "intervention_name": ""}
    assert auto_title(p) == "A perfectly fine title"


def test_auto_title_handles_empty_protocol():
    assert auto_title({}) == "Untitled protocol"


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def _proto(phase="Phase II", conditions="TNBC", intervention="Pembrolizumab"):
    return {
        "title": f"{phase} {conditions} trial",
        "phase": phase,
        "conditions": conditions,
        "intervention_name": intervention,
        "study_design": "Randomized, double-blind",
        "primary_endpoints": ["pCR"],
    }


def test_save_initial_creates_files_and_indexes(tmp_path):
    paths = ProtocolHistoryPaths.from_data_dir(tmp_path)
    meta = save_initial(paths, "user1", _proto(), summary_brief="Some brief")
    assert meta["id"]
    assert meta["version_count"] == 1
    assert paths.meta_file("user1", meta["id"]).exists()
    assert paths.version_file("user1", meta["id"], 1).exists()
    assert paths.index_file("user1").exists()
    listed = list_protocols(paths, "user1")
    assert len(listed) == 1
    assert listed[0]["id"] == meta["id"]


def test_save_revision_appends_version_and_updates_meta(tmp_path):
    paths = ProtocolHistoryPaths.from_data_dir(tmp_path)
    meta1 = save_initial(paths, "user1", _proto(phase="Phase II"))
    meta2 = save_revision(
        paths, "user1", meta1["id"], _proto(phase="Phase III", conditions="HR+/HER2-"),
    )
    assert meta2["version_count"] == 2
    assert meta2["phase"] == "Phase III"
    assert meta2["conditions"] == "HR+/HER2-"
    # v1 + v2 files both exist; meta in index reflects v2.
    assert paths.version_file("user1", meta1["id"], 1).exists()
    assert paths.version_file("user1", meta1["id"], 2).exists()
    rows = list_protocols(paths, "user1")
    assert len(rows) == 1  # still one entry, two versions inside
    assert rows[0]["version_count"] == 2


def test_save_revision_unknown_id_raises(tmp_path):
    paths = ProtocolHistoryPaths.from_data_dir(tmp_path)
    with pytest.raises(FileNotFoundError):
        save_revision(paths, "user1", "no-such-id", _proto())


def test_load_protocol_default_returns_latest_version(tmp_path):
    paths = ProtocolHistoryPaths.from_data_dir(tmp_path)
    meta = save_initial(paths, "user1", _proto(phase="Phase II"))
    save_revision(paths, "user1", meta["id"], _proto(phase="Phase III"))
    p, m = load_protocol(paths, "user1", meta["id"])
    assert p["phase"] == "Phase III"  # latest
    assert m["version_count"] == 2


def test_load_protocol_specific_version(tmp_path):
    paths = ProtocolHistoryPaths.from_data_dir(tmp_path)
    meta = save_initial(paths, "user1", _proto(phase="Phase II"))
    save_revision(paths, "user1", meta["id"], _proto(phase="Phase III"))
    p, _ = load_protocol(paths, "user1", meta["id"], version=1)
    assert p["phase"] == "Phase II"


def test_users_are_isolated(tmp_path):
    paths = ProtocolHistoryPaths.from_data_dir(tmp_path)
    save_initial(paths, "user_a", _proto(conditions="TNBC"))
    save_initial(paths, "user_b", _proto(conditions="HR+"))
    assert len(list_protocols(paths, "user_a")) == 1
    assert len(list_protocols(paths, "user_b")) == 1
    # And they can't read each other's by id.
    a_id = list_protocols(paths, "user_a")[0]["id"]
    with pytest.raises(FileNotFoundError):
        load_protocol(paths, "user_b", a_id)


def test_user_id_sanitized_against_path_traversal(tmp_path):
    paths = ProtocolHistoryPaths.from_data_dir(tmp_path)
    # "../escape" must NOT escape the protocols dir.
    save_initial(paths, "../escape", _proto())
    # The bucket name was sanitized to "._escape" or similar; what matters is
    # that the resulting dir is still inside paths.root.
    user_dir = paths.user_dir("../escape")
    assert paths.root in user_dir.parents or user_dir.parent == paths.root


# ---------------------------------------------------------------------------
# NL match: "based on the previous protocol" → entry
# ---------------------------------------------------------------------------

def test_query_mentions_previous_protocol_positive():
    assert query_mentions_previous_protocol("based on the previous protocol, change inclusion")
    assert query_mentions_previous_protocol("Modify my previous TNBC report")
    assert query_mentions_previous_protocol("the protocol I generated earlier needs ECOG 0-1")


def test_query_mentions_previous_protocol_negative():
    assert not query_mentions_previous_protocol("show me HER2-low trials")
    assert not query_mentions_previous_protocol("design a new phase II protocol for TNBC")


def test_match_query_picks_relevant_entry(tmp_path):
    paths = ProtocolHistoryPaths.from_data_dir(tmp_path)
    save_initial(
        paths, "user1",
        _proto(phase="Phase II", conditions="Triple-negative breast cancer", intervention="Pembrolizumab"),
        summary_brief="Phase II TNBC checkpoint inhibitor",
    )
    save_initial(
        paths, "user1",
        _proto(phase="Phase III", conditions="HR+/HER2-", intervention="Palbociclib"),
        summary_brief="Phase III HR+ CDK4/6 trial",
    )
    matched = match_query_to_entry(paths, "user1", "edit my previous TNBC protocol")
    assert len(matched) == 1
    assert "TNBC" in (matched[0]["title"] + matched[0]["conditions"])


def test_match_query_falls_back_to_recent_when_no_semantic_match(tmp_path):
    paths = ProtocolHistoryPaths.from_data_dir(tmp_path)
    save_initial(paths, "user1", _proto(phase="Phase II"))
    save_initial(paths, "user1", _proto(phase="Phase III"))
    # Wholly unrelated phrase — should still return the most-recent entry
    # rather than nothing, so the user always gets a clickable load chip.
    matched = match_query_to_entry(paths, "user1", "based on the previous unrelated thing")
    assert len(matched) == 1


def test_match_query_empty_history(tmp_path):
    paths = ProtocolHistoryPaths.from_data_dir(tmp_path)
    assert match_query_to_entry(paths, "user1", "based on the previous report") == []
