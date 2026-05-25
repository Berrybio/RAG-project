"""Structured event logging for product analytics.

Events are always written to stdout (Cloud Logging on GCP picks these up
automatically). When a Supabase client is configured via ``configure()``,
events are also persisted to the ``analytics_events`` table and
``user_sessions`` is upserted so session-level metrics stay current.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("analytics")
logger.setLevel(logging.INFO)

_MAX_TEXT_CHARS = 2000
_TEXT_FIELDS = {"query_text", "correction_text", "exception_message"}

_supabase = None
_supabase_lock = threading.Lock()


def configure(supabase_client: Any = None) -> None:
    global _supabase
    _supabase = supabase_client
    if _supabase:
        logger.info("Analytics: Supabase persistence enabled")


def _truncate(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_TEXT_CHARS:
        return value[:_MAX_TEXT_CHARS] + "…"
    return value


def log_event(event_type: str, **fields: Any) -> None:
    """Emit one structured analytics event.

    Always synchronous and best-effort: a failure here must never break a
    user-facing request, so we swallow any exception inside the logger.
    """
    try:
        now = datetime.now(timezone.utc).isoformat()
        payload: dict[str, Any] = {"event_type": event_type, "ts": now}
        for key, value in fields.items():
            if value is None:
                continue
            if key in _TEXT_FIELDS:
                value = _truncate(value)
            payload[key] = value
        logger.info(json.dumps(payload, default=str))

        if _supabase:
            _persist_to_supabase(event_type, now, payload, fields)
    except Exception:
        logger.warning("analytics: failed to emit event_type=%s", event_type)


def _persist_to_supabase(
    event_type: str,
    ts: str,
    payload: dict[str, Any],
    fields: dict[str, Any],
) -> None:
    try:
        user_id = fields.get("user_id")
        session_id = fields.get("session_id")
        properties = {
            k: v for k, v in payload.items()
            if k not in ("event_type", "ts", "user_id", "session_id")
        }

        _supabase.table("analytics_events").insert({
            "event_type": event_type,
            "user_id": user_id,
            "session_id": session_id,
            "properties": properties,
            "created_at": ts,
        }).execute()

        if session_id:
            _upsert_session(session_id, user_id, event_type, ts)
    except Exception:
        logger.warning("analytics: Supabase persist failed for %s", event_type)


def _upsert_session(
    session_id: str,
    user_id: str | None,
    event_type: str,
    ts: str,
) -> None:
    try:
        resp = (
            _supabase.table("user_sessions")
            .select("event_count, features_used")
            .eq("session_id", session_id)
            .execute()
        )
        if resp.data:
            row = resp.data[0]
            count = (row.get("event_count") or 0) + 1
            features = list(set(row.get("features_used") or []) | {event_type})
            _supabase.table("user_sessions").update({
                "last_active_at": ts,
                "event_count": count,
                "features_used": features,
                "user_id": user_id or row.get("user_id"),
            }).eq("session_id", session_id).execute()
        else:
            _supabase.table("user_sessions").insert({
                "session_id": session_id,
                "user_id": user_id,
                "started_at": ts,
                "last_active_at": ts,
                "event_count": 1,
                "features_used": [event_type],
            }).execute()
    except Exception:
        logger.warning("analytics: session upsert failed for %s", session_id)
