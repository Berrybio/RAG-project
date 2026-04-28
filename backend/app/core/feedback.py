"""User feedback storage + drug-alias dictionary.

Two on-disk artifacts:

- `user_feedback.jsonl` — append-only log of every feedback event
  (thumbs up/down + correction suggestions). One JSON object per line so
  the file stays grep-friendly and crash-safe.
- `drug_aliases.json` — curated alias dictionary, the source of truth for
  query-time expansion. Populated by an admin promoting items out of the
  feedback log; never written to by the public feedback endpoint.

Both files live under the configured data directory (default
`backend/data/`) and are created lazily.
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

logger = logging.getLogger(__name__)

# Per-file locks. Append-mode writes on POSIX are atomic for small payloads,
# but we still serialize to avoid interleaving when many requests arrive at
# once and to keep reads consistent with concurrent promotes.
_feedback_lock = threading.Lock()
_aliases_lock = threading.Lock()


@dataclass
class FeedbackPaths:
    feedback_log: Path
    aliases_file: Path

    @classmethod
    def from_data_dir(cls, data_dir: str | Path) -> "FeedbackPaths":
        d = Path(data_dir)
        d.mkdir(parents=True, exist_ok=True)
        return cls(
            feedback_log=d / "user_feedback.jsonl",
            aliases_file=d / "drug_aliases.json",
        )


# ---------------------------------------------------------------------------
# Feedback log (jsonl, append-only)
# ---------------------------------------------------------------------------

def append_feedback(paths: FeedbackPaths, entry: dict) -> dict:
    """Append a feedback entry to the jsonl log; returns the stored entry.

    Each entry is augmented with an `id` (uuid4) and `created_at` (UTC ISO),
    and `status` defaults to "open" so the admin queue can distinguish
    pending items from those already actioned.
    """
    enriched = {
        "id": entry.get("id") or str(uuid.uuid4()),
        "created_at": entry.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "status": entry.get("status", "open"),
        **{k: v for k, v in entry.items() if k not in ("id", "created_at", "status")},
    }
    line = json.dumps(enriched, ensure_ascii=False)
    with _feedback_lock:
        with paths.feedback_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    logger.info("Feedback recorded: type=%s id=%s", enriched.get("type"), enriched["id"])
    return enriched


def read_feedback(paths: FeedbackPaths) -> list[dict]:
    """Read all feedback entries. Returns newest-first."""
    if not paths.feedback_log.exists():
        return []
    entries: list[dict] = []
    with _feedback_lock:
        with paths.feedback_log.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed feedback line: %r", line[:120])
    entries.reverse()
    return entries


def update_feedback_status(paths: FeedbackPaths, feedback_id: str, status: str) -> dict | None:
    """Update the `status` field on a feedback entry (rewrites the log).

    The log is small (human-scale) so a full rewrite is fine. Returns the
    updated entry, or None if the id wasn't found.
    """
    if not paths.feedback_log.exists():
        return None
    updated: dict | None = None
    with _feedback_lock:
        rows = []
        with paths.feedback_log.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("id") == feedback_id:
                    row["status"] = status
                    updated = row
                rows.append(row)
        if updated is not None:
            tmp = paths.feedback_log.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            tmp.replace(paths.feedback_log)
    return updated


def feedback_stats(paths: FeedbackPaths) -> dict:
    """Aggregate counts: ratings up/down and correction queue depth."""
    rows = read_feedback(paths)
    up = sum(1 for r in rows if r.get("type") == "rating_up")
    down = sum(1 for r in rows if r.get("type") == "rating_down")
    open_corrections = sum(
        1 for r in rows if r.get("type") == "correction" and r.get("status") == "open"
    )
    return {
        "rating_up": up,
        "rating_down": down,
        "open_corrections": open_corrections,
        "total": len(rows),
    }


# ---------------------------------------------------------------------------
# Drug alias dictionary
# ---------------------------------------------------------------------------

def load_aliases(paths: FeedbackPaths) -> dict[str, str]:
    """Return {alias_lower_stripped: canonical_name}. Empty if file missing."""
    if not paths.aliases_file.exists():
        return {}
    with _aliases_lock:
        try:
            data = json.loads(paths.aliases_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("drug_aliases.json is malformed; treating as empty")
            return {}
    if not isinstance(data, dict):
        return {}
    # Normalize keys to lowercase/stripped on read so the rest of the code
    # doesn't have to care about how they were entered.
    return {str(k).strip().lower(): str(v).strip() for k, v in data.items() if k and v}


def add_alias(paths: FeedbackPaths, alias: str, canonical: str) -> dict[str, str]:
    """Add or update a single alias, persist, and return the new dictionary."""
    alias_key = alias.strip().lower()
    canonical_value = canonical.strip()
    if not alias_key or not canonical_value:
        raise ValueError("alias and canonical must both be non-empty")
    with _aliases_lock:
        current: dict[str, str] = {}
        if paths.aliases_file.exists():
            try:
                raw = json.loads(paths.aliases_file.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    current = {str(k): str(v) for k, v in raw.items()}
            except json.JSONDecodeError:
                logger.warning("drug_aliases.json malformed; recreating")
        current[alias_key] = canonical_value
        paths.aliases_file.write_text(
            json.dumps(current, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    logger.info("Alias added: %r -> %r", alias_key, canonical_value)
    return {k.strip().lower(): v.strip() for k, v in current.items()}


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

def expand_query(query: str, aliases: dict[str, str]) -> str:
    """Append canonical names for any aliases mentioned in the query.

    We append rather than replace so the original wording is preserved for
    the LLM context, while the embedder/TF-IDF retriever sees both forms.
    Whole-word match, case-insensitive. If `aliases` is empty, returns the
    original query unchanged.
    """
    if not aliases or not query:
        return query
    matched: list[str] = []
    seen: set[str] = set()
    for alias, canonical in aliases.items():
        # Whole-word match. We escape the alias and require word boundaries
        # on both sides so "Dato-DXd" matches but "Dato-DXdX" does not.
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, query, flags=re.IGNORECASE):
            key = canonical.lower()
            if key not in seen and canonical.lower() not in query.lower():
                matched.append(canonical)
                seen.add(key)
    if not matched:
        return query
    return query + " " + " ".join(matched)


def aliases_prompt_block(aliases: dict[str, str]) -> str:
    """Render a short 'Known drug aliases' block for inclusion in a system
    prompt. Returns empty string when there are no aliases.

    The block is grouped by canonical name so the model sees one entry per
    drug regardless of how many aliases it has.
    """
    if not aliases:
        return ""
    by_canonical: dict[str, list[str]] = {}
    for alias, canonical in aliases.items():
        by_canonical.setdefault(canonical, []).append(alias)
    lines = ["Known drug aliases (use the canonical name in answers):"]
    for canonical in sorted(by_canonical):
        alts = ", ".join(sorted(set(by_canonical[canonical])))
        lines.append(f"- {canonical} (also: {alts})")
    return "\n".join(lines)
