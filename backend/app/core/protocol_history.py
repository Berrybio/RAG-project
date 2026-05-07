"""Per-user protocol history: persist generated protocols + their versions
for cross-session iteration.

Storage layout (rooted at ``settings.csv_path``'s data dir):

    protocols/
      <user_id>/
        index.jsonl                 # one row per protocol (entry meta)
        <protocol_id>/
          v1.json                   # full protocol JSON for version 1
          v2.json                   # ...
          meta.json                 # title, conditions, phase, summary brief

Each ``index.jsonl`` row mirrors ``meta.json`` so the list endpoint is a
single file read. ``meta.json`` is the source of truth on save/load; the
index is rebuilt from meta files if it ever gets out of sync.

Why per-user dirs and not a single SQLite database: protocols are infrequent
(handfuls per session), human-curated, and grep-friendly inspection during
dev is more useful than a query language. JSONL + plain JSON is also
trivial to back up.

Anonymous users get the literal string ``anonymous`` as their user_id so
direct API hits (curl, smoke tests) still write somewhere predictable
without breaking out the missing-header path everywhere.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Per-file locks. Saves are infrequent so a coarse global lock is fine; the
# alternative (per-user lock map) doesn't help when a single user runs two
# refinements in flight, which is the actual concurrency case here.
_history_lock = threading.Lock()

# Cap how many entries we return in list responses to keep payloads bounded.
# 100 is many sessions worth of work; the UI strip only shows the top few
# anyway.
_LIST_LIMIT_DEFAULT = 100

# When matching an NL phrase like "based on the previous TNBC protocol" against
# stored entries, anything below this TF-IDF cosine is too unrelated to surface
# (matches the threshold used by the few-shot examples store).
_NL_MIN_SIMILARITY = 0.15


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

@dataclass
class ProtocolHistoryPaths:
    root: Path

    @classmethod
    def from_data_dir(cls, data_dir: str | Path) -> "ProtocolHistoryPaths":
        d = Path(data_dir) / "protocols"
        d.mkdir(parents=True, exist_ok=True)
        return cls(root=d)

    def user_dir(self, user_id: str) -> Path:
        # Sanitize aggressively — anonymous user_id is whatever localStorage
        # gave us, but a hostile header value shouldn't be able to traverse out.
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", (user_id or "anonymous").strip())[:128]
        if not safe:
            safe = "anonymous"
        d = self.root / safe
        d.mkdir(parents=True, exist_ok=True)
        return d

    def index_file(self, user_id: str) -> Path:
        return self.user_dir(user_id) / "index.jsonl"

    def protocol_dir(self, user_id: str, protocol_id: str) -> Path:
        return self.user_dir(user_id) / protocol_id

    def meta_file(self, user_id: str, protocol_id: str) -> Path:
        return self.protocol_dir(user_id, protocol_id) / "meta.json"

    def version_file(self, user_id: str, protocol_id: str, version: int) -> Path:
        return self.protocol_dir(user_id, protocol_id) / f"v{version}.json"


# ---------------------------------------------------------------------------
# Title auto-derivation
# ---------------------------------------------------------------------------

# Population shorthand → canonical (used to pull a recognisable population
# token out of conditions text, since the LLM often writes the long form).
_POPULATION_SHORTHAND: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\btriple[\s-]?negative\b", re.I), "TNBC"),
    (re.compile(r"\bHER2[\s-]?low\b", re.I), "HER2-low"),
    (re.compile(r"\bHER2[\s-]?positive\b|\bHER2\+\b", re.I), "HER2+"),
    (re.compile(r"\bHER2[\s-]?negative\b|\bHER2\-\b", re.I), "HER2-"),
    (re.compile(r"\bHR[\s-]?positive\b|\bHR\+\b", re.I), "HR+"),
    (re.compile(r"\bER[\s-]?positive\b|\bER\+\b", re.I), "ER+"),
    (re.compile(r"\bBRCA\b", re.I), "BRCA"),
    (re.compile(r"\bmetastatic\b", re.I), "metastatic"),
    (re.compile(r"\bearly[\s-]?stage\b", re.I), "early-stage"),
    (re.compile(r"\bneoadjuvant\b", re.I), "neoadjuvant"),
    (re.compile(r"(?<!neo)\badjuvant\b", re.I), "adjuvant"),
]


def _short_population(text: str) -> str:
    """Pick a short, recognisable population token out of free text. Empty
    string when nothing matches."""
    if not text:
        return ""
    for pattern, label in _POPULATION_SHORTHAND:
        if pattern.search(text):
            return label
    return ""


def auto_title(protocol: dict) -> str:
    """Build a short title from a generated protocol.

    Priority:
    1. Phase + population shorthand + intervention if all present
    2. Phase + first 60 chars of the protocol's own title field
    3. The protocol's own title verbatim (truncated)

    Falls back to "Untitled protocol" only when nothing is parseable.
    """
    phase = (protocol.get("phase") or "").strip()
    title = (protocol.get("title") or "").strip()
    conditions = (protocol.get("conditions") or "").strip()
    intervention = (protocol.get("intervention_name") or "").strip()

    population = _short_population(conditions) or _short_population(title)

    parts: list[str] = []
    if phase and phase.upper() != "N/A":
        parts.append(phase)
    elif phase.upper() == "N/A":
        # Observational / RWE — flag explicitly so the user can tell apart from
        # a missing-phase row.
        parts.append("RWE")

    if population:
        parts.append(population)

    if intervention:
        # Cut at the first slash / parenthesis so "Trastuzumab deruxtecan
        # (T-DXd, Enhertu)" becomes just "Trastuzumab deruxtecan".
        clean = re.split(r"[/(]", intervention, maxsplit=1)[0].strip()
        if clean:
            parts.append(clean[:50])

    if parts:
        return " — ".join(parts)
    if title:
        return title[:80]
    return "Untitled protocol"


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_meta(
    *,
    protocol_id: str,
    title: str,
    protocol: dict,
    summary_brief: str,
    source_nct_ids: list[str],
    version_count: int,
    created_at: str,
    last_modified: str,
) -> dict:
    return {
        "id": protocol_id,
        "title": title,
        "phase": (protocol.get("phase") or "").strip(),
        "conditions": (protocol.get("conditions") or "").strip(),
        "study_design": (protocol.get("study_design") or "").strip()[:200],
        "intervention_name": (protocol.get("intervention_name") or "").strip()[:200],
        "summary_brief": (summary_brief or "")[:600],
        "source_nct_ids": list(source_nct_ids or [])[:25],
        "version_count": version_count,
        "created_at": created_at,
        "last_modified": last_modified,
    }


def _append_to_index(paths: ProtocolHistoryPaths, user_id: str, meta: dict) -> None:
    line = json.dumps(meta, ensure_ascii=False)
    with paths.index_file(user_id).open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _rewrite_index(paths: ProtocolHistoryPaths, user_id: str, all_meta: list[dict]) -> None:
    tmp = paths.index_file(user_id).with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for m in all_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    tmp.replace(paths.index_file(user_id))


def save_initial(
    paths: ProtocolHistoryPaths,
    user_id: str,
    protocol: dict,
    summary_brief: str = "",
    source_nct_ids: list[str] | None = None,
) -> dict:
    """Persist a freshly-generated protocol as version 1. Returns the entry meta."""
    protocol_id = str(uuid.uuid4())
    title = auto_title(protocol)
    now = _now_iso()
    meta = _build_meta(
        protocol_id=protocol_id,
        title=title,
        protocol=protocol,
        summary_brief=summary_brief,
        source_nct_ids=source_nct_ids or [],
        version_count=1,
        created_at=now,
        last_modified=now,
    )
    with _history_lock:
        paths.protocol_dir(user_id, protocol_id).mkdir(parents=True, exist_ok=True)
        paths.version_file(user_id, protocol_id, 1).write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        paths.meta_file(user_id, protocol_id).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _append_to_index(paths, user_id, meta)
    logger.info(
        "Saved protocol v1 for user=%s id=%s title=%r",
        user_id, protocol_id, title,
    )
    return meta


def save_revision(
    paths: ProtocolHistoryPaths,
    user_id: str,
    protocol_id: str,
    protocol: dict,
) -> dict:
    """Persist a refined protocol as the next version. Returns the updated meta."""
    if not protocol_id:
        raise ValueError("protocol_id is required to save a revision")
    with _history_lock:
        meta_path = paths.meta_file(user_id, protocol_id)
        if not meta_path.exists():
            raise FileNotFoundError(f"No protocol meta found for {user_id}/{protocol_id}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        next_version = int(meta.get("version_count", 1)) + 1
        # Refresh derived fields from the new protocol — title / conditions /
        # phase can shift across refinements (e.g. user changes phase).
        new_title = auto_title(protocol)
        # Preserve the original title if user hasn't changed any of the title-
        # determining fields; otherwise update so the index stays meaningful.
        if new_title and new_title != "Untitled protocol":
            meta["title"] = new_title
        meta["phase"] = (protocol.get("phase") or "").strip()
        meta["conditions"] = (protocol.get("conditions") or "").strip()
        meta["study_design"] = (protocol.get("study_design") or "").strip()[:200]
        meta["intervention_name"] = (protocol.get("intervention_name") or "").strip()[:200]
        meta["version_count"] = next_version
        meta["last_modified"] = _now_iso()

        paths.version_file(user_id, protocol_id, next_version).write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Rewrite index so the row's last_modified / version_count / title stay current.
        all_rows = _read_index(paths, user_id)
        replaced = False
        for i, row in enumerate(all_rows):
            if row.get("id") == protocol_id:
                all_rows[i] = meta
                replaced = True
                break
        if not replaced:
            all_rows.append(meta)
        _rewrite_index(paths, user_id, all_rows)
    logger.info(
        "Saved protocol v%d for user=%s id=%s",
        next_version, user_id, protocol_id,
    )
    return meta


def _read_index(paths: ProtocolHistoryPaths, user_id: str) -> list[dict]:
    f = paths.index_file(user_id)
    if not f.exists():
        return []
    rows: list[dict] = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed protocol index row: %r", line[:120])
    return rows


def list_protocols(
    paths: ProtocolHistoryPaths,
    user_id: str,
    limit: int = _LIST_LIMIT_DEFAULT,
) -> list[dict]:
    """Return entry metas, newest-first."""
    rows = _read_index(paths, user_id)
    rows.sort(key=lambda r: r.get("last_modified") or r.get("created_at") or "", reverse=True)
    return rows[:limit]


def load_protocol(
    paths: ProtocolHistoryPaths,
    user_id: str,
    protocol_id: str,
    version: int | None = None,
) -> tuple[dict, dict]:
    """Return ``(protocol_json, meta)`` for a stored protocol.

    ``version=None`` returns the latest. Raises ``FileNotFoundError`` if the
    protocol or version doesn't exist.
    """
    meta_path = paths.meta_file(user_id, protocol_id)
    if not meta_path.exists():
        raise FileNotFoundError(f"No protocol meta found for {user_id}/{protocol_id}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    target_version = version if version is not None else int(meta.get("version_count", 1))
    vfile = paths.version_file(user_id, protocol_id, target_version)
    if not vfile.exists():
        raise FileNotFoundError(
            f"No version {target_version} for {user_id}/{protocol_id}"
        )
    protocol = json.loads(vfile.read_text(encoding="utf-8"))
    return protocol, meta


# ---------------------------------------------------------------------------
# NL match: "based on the previous TNBC protocol" → entry
# ---------------------------------------------------------------------------

# Phrases that indicate the user is asking us to load a previously-generated
# protocol. Matched case-insensitively as substrings — false positives are
# fine because we still verify there's a protocol to load before acting.
_PREV_PROTOCOL_PHRASES = [
    "previous report",
    "previous protocol",
    "previous draft",
    "last report",
    "last protocol",
    "last draft",
    "prior protocol",
    "earlier protocol",
    "the protocol i generated",
    "the report i generated",
    "based on the previous",
    "based on my previous",
    "edit my last",
    "modify my previous",
    "modify the previous",
    "update my previous",
    "from earlier",
    "the one we did",
]


def query_mentions_previous_protocol(query: str) -> bool:
    """Cheap heuristic: does the user's message hint at loading a prior protocol?"""
    if not query:
        return False
    lower = query.lower()
    return any(phrase in lower for phrase in _PREV_PROTOCOL_PHRASES)


def match_query_to_entry(
    paths: ProtocolHistoryPaths,
    user_id: str,
    query: str,
    top_n: int = 1,
) -> list[dict]:
    """Pick the best stored entries for a user's NL hint.

    Strategy:
    1. If the user has stored exactly one protocol, return that.
    2. Otherwise rank by TF-IDF cosine over (title + conditions +
       intervention_name + study_design + summary_brief).
    3. If no entry clears _NL_MIN_SIMILARITY, fall back to the most recent.

    Returns up to ``top_n`` entry metas, best first.
    """
    rows = list_protocols(paths, user_id, limit=_LIST_LIMIT_DEFAULT)
    if not rows:
        return []
    if len(rows) == 1:
        return rows[:1]

    docs = [
        " ".join([
            r.get("title", ""),
            r.get("conditions", ""),
            r.get("intervention_name", ""),
            r.get("study_design", ""),
            r.get("summary_brief", ""),
        ])
        for r in rows
    ]
    try:
        vec = TfidfVectorizer(max_features=2048).fit(docs + [query or ""])
        matrix = vec.transform(docs)
        qv = vec.transform([query or ""])
    except ValueError:
        return rows[:top_n]
    sims = cosine_similarity(qv, matrix)[0]
    order = sims.argsort()[::-1]
    picked: list[dict] = []
    for rank, idx in enumerate(order[:top_n]):
        score = float(sims[int(idx)])
        # Always accept the #1 match if it has any positive overlap — sparse
        # 2-3 entry corpora produce small absolute cosines but the top match
        # is still the right one. The threshold only filters secondary matches
        # to avoid surfacing irrelevant additional chips.
        if rank == 0 and score > 0:
            picked.append(rows[int(idx)])
            continue
        if score >= _NL_MIN_SIMILARITY:
            picked.append(rows[int(idx)])
    if not picked:
        # No semantic match at all — fall back to most recent so the user
        # always gets *something* clickable.
        picked = rows[:1]
    return picked
