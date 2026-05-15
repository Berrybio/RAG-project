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

# Cap on how many entries we surface to the chat UI when the user invokes
# the "previous protocol" NL hook. We rank by relevance but always return
# everything (up to this cap) so the user can pick — filtering by similarity
# threshold here would silently drop legitimate options when the new query
# doesn't lexically overlap with the old protocol's title.
_NL_LIST_CAP = 10


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
    change_log: list[dict] | None = None,
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
        # One row per saved version describing what changed. Lets the planner
        # rebuild the full version-chip history (with labels + notes) when the
        # user reloads the protocol from a fresh session.
        "change_log": list(change_log or []),
    }


def _change_log_entry(
    *,
    version: int,
    user_request: str,
    assistant_note: str,
    changed_fields: list[str],
    timestamp: str,
) -> dict:
    return {
        "version": version,
        "user_request": (user_request or "")[:600],
        "assistant_note": (assistant_note or "")[:1500],
        "changed_fields": list(changed_fields or []),
        "timestamp": timestamp,
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
        change_log=[_change_log_entry(
            version=1,
            user_request=summary_brief or "Initial generation from the planning brief",
            assistant_note="Initial protocol generated from the conversation summary.",
            changed_fields=[],
            timestamp=now,
        )],
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
    user_request: str = "",
    assistant_note: str = "",
    changed_fields: list[str] | None = None,
) -> dict:
    """Persist a refined protocol as the next version. Returns the updated meta.

    The new ``user_request`` / ``assistant_note`` / ``changed_fields`` are
    appended to the meta's ``change_log`` so the planner can rehydrate the
    full version-chip history (with labels + tooltips) when the user
    reloads the protocol from a fresh session.
    """
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
        # Append change-log entry. Backward-compat: meta files written by an
        # earlier version of this module don't have change_log yet — start one.
        log = meta.get("change_log")
        if not isinstance(log, list):
            log = []
        log.append(_change_log_entry(
            version=next_version,
            user_request=user_request,
            assistant_note=assistant_note,
            changed_fields=list(changed_fields or []),
            timestamp=meta["last_modified"],
        ))
        meta["change_log"] = log

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


def load_all_versions(
    paths: ProtocolHistoryPaths,
    user_id: str,
    protocol_id: str,
) -> list[dict]:
    """Return every saved version of a protocol with its change-log metadata.

    Each item is ``{version, protocol, user_request, assistant_note,
    changed_fields, timestamp}``. Pre-change-log protocols (older entries
    without the ``change_log`` field) fall back to empty notes so the planner
    can still render version chips with at least version + timestamp.
    """
    meta_path = paths.meta_file(user_id, protocol_id)
    if not meta_path.exists():
        raise FileNotFoundError(f"No protocol meta found for {user_id}/{protocol_id}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    total = int(meta.get("version_count", 1))
    raw_log = meta.get("change_log") or []
    log_by_version = {int(e.get("version", 0)): e for e in raw_log if isinstance(e, dict)}
    fallback_ts = meta.get("created_at") or meta.get("last_modified") or _now_iso()

    versions: list[dict] = []
    for v in range(1, total + 1):
        vfile = paths.version_file(user_id, protocol_id, v)
        if not vfile.exists():
            # Skip missing version files rather than failing — a manual cleanup
            # shouldn't tank the load endpoint for the rest of the history.
            logger.warning("Missing version file v%d for %s/%s", v, user_id, protocol_id)
            continue
        protocol = json.loads(vfile.read_text(encoding="utf-8"))
        entry = log_by_version.get(v, {})
        versions.append({
            "version": v,
            "protocol": protocol,
            "user_request": entry.get("user_request", ""),
            "assistant_note": entry.get("assistant_note", ""),
            "changed_fields": entry.get("changed_fields", []),
            "timestamp": entry.get("timestamp", fallback_ts),
        })
    return versions


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
    top_n: int = _NL_LIST_CAP,
) -> list[dict]:
    """Return the user's stored protocols, ranked by relevance to ``query``.

    Behaviour: when the user invokes the "previous protocol" NL hook, we
    surface ALL their stored protocols (up to ``top_n``) so they can pick.
    No similarity threshold filtering — that would silently drop legitimate
    options whenever the new query doesn't lexically overlap with the old
    protocol's title (e.g. "continue working on the previous protocol" has
    no overlap with any specific population term).

    Within the cap, entries are ordered by TF-IDF cosine over
    ``(title + conditions + intervention + study_design + summary_brief)``,
    breaking ties by recency (newest first). When the query has no useful
    tokens, ranking collapses to pure recency.
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
    # Stable sort: TF-IDF score desc, then preserve original (recency-desc)
    # order for ties — Python's sort is stable, so we reverse-enumerate to
    # encode "later index = older" as a tiebreaker.
    indexed = sorted(
        enumerate(sims),
        key=lambda pair: (-float(pair[1]), pair[0]),
    )
    return [rows[i] for i, _ in indexed[:top_n]]
