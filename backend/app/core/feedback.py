"""User feedback storage + drug-alias dictionary.

Supports two backends, chosen at startup based on config:

1. **Supabase** (preferred for Cloud Run): feedback rows live in a
   ``public.feedback`` table; aliases in ``public.drug_aliases``. Durable
   across container restarts and deploys.
2. **Local files** (fallback for local dev / Docker Compose): append-only
   ``user_feedback.jsonl`` + ``drug_aliases.json`` on disk.

The caller-facing API is identical regardless of backend — every public
function takes a ``FeedbackPaths`` instance whose optional ``supabase``
field controls dispatch.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_feedback_lock = threading.Lock()
_aliases_lock = threading.Lock()


@dataclass
class FeedbackPaths:
    feedback_log: Path
    aliases_file: Path
    supabase: Any = field(default=None, repr=False)

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str | Path,
        supabase_client: Any = None,
    ) -> "FeedbackPaths":
        d = Path(data_dir)
        d.mkdir(parents=True, exist_ok=True)
        return cls(
            feedback_log=d / "user_feedback.jsonl",
            aliases_file=d / "drug_aliases.json",
            supabase=supabase_client,
        )


# ---------------------------------------------------------------------------
# Feedback log
# ---------------------------------------------------------------------------

def append_feedback(paths: FeedbackPaths, entry: dict) -> dict:
    enriched = {
        "id": entry.get("id") or str(uuid.uuid4()),
        "created_at": entry.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "status": entry.get("status", "open"),
        **{k: v for k, v in entry.items() if k not in ("id", "created_at", "status")},
    }

    if paths.supabase:
        row = {
            "id": enriched["id"],
            "type": enriched.get("type", ""),
            "status": enriched.get("status", "open"),
            "reason": enriched.get("reason", ""),
            "correction_kind": enriched.get("correction_kind") or None,
            "alias": enriched.get("alias", ""),
            "canonical": enriched.get("canonical", ""),
            "notes": enriched.get("notes", ""),
            "context": enriched.get("context", {}),
            "created_at": enriched["created_at"],
        }
        try:
            paths.supabase.table("feedback").insert(row).execute()
        except Exception:
            logger.exception("Supabase insert failed; falling back to file")
            _append_feedback_file(paths, enriched)
    else:
        _append_feedback_file(paths, enriched)

    logger.info("Feedback recorded: type=%s id=%s", enriched.get("type"), enriched["id"])
    return enriched


def _append_feedback_file(paths: FeedbackPaths, enriched: dict) -> None:
    line = json.dumps(enriched, ensure_ascii=False)
    with _feedback_lock:
        with paths.feedback_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_feedback(paths: FeedbackPaths) -> list[dict]:
    if paths.supabase:
        try:
            resp = (
                paths.supabase.table("feedback")
                .select("*")
                .order("created_at", desc=True)
                .execute()
            )
            return [_supabase_row_to_dict(r) for r in (resp.data or [])]
        except Exception:
            logger.exception("Supabase read failed; falling back to file")

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


def _supabase_row_to_dict(row: dict) -> dict:
    """Normalize a Supabase row back to the dict shape the rest of the code expects."""
    out = dict(row)
    if "created_at" in out and out["created_at"]:
        ts = out["created_at"]
        if not isinstance(ts, str):
            ts = str(ts)
        out["created_at"] = ts
    if out.get("correction_kind") is None:
        out.pop("correction_kind", None)
    return out


def update_feedback_status(paths: FeedbackPaths, feedback_id: str, status: str) -> dict | None:
    if paths.supabase:
        try:
            resp = (
                paths.supabase.table("feedback")
                .update({"status": status})
                .eq("id", feedback_id)
                .execute()
            )
            if resp.data:
                return _supabase_row_to_dict(resp.data[0])
            return None
        except Exception:
            logger.exception("Supabase update failed; falling back to file")

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
    if paths.supabase:
        try:
            resp = paths.supabase.table("drug_aliases").select("alias, canonical").execute()
            return {
                r["alias"].strip().lower(): r["canonical"].strip()
                for r in (resp.data or [])
                if r.get("alias") and r.get("canonical")
            }
        except Exception:
            logger.exception("Supabase alias read failed; falling back to file")

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
    return {str(k).strip().lower(): str(v).strip() for k, v in data.items() if k and v}


def add_alias(paths: FeedbackPaths, alias: str, canonical: str) -> dict[str, str]:
    alias_key = alias.strip().lower()
    canonical_value = canonical.strip()
    if not alias_key or not canonical_value:
        raise ValueError("alias and canonical must both be non-empty")

    if paths.supabase:
        try:
            paths.supabase.table("drug_aliases").upsert(
                {"alias": alias_key, "canonical": canonical_value},
            ).execute()
            logger.info("Alias added (Supabase): %r -> %r", alias_key, canonical_value)
            return load_aliases(paths)
        except Exception:
            logger.exception("Supabase alias upsert failed; falling back to file")

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
    if not aliases or not query:
        return query
    matched: list[str] = []
    seen: set[str] = set()
    for alias, canonical in aliases.items():
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
