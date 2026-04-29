"""Structured event logging for product analytics.

Every interesting user action gets one JSON-formatted log line with a
consistent shape. Cloud Logging on GCP automatically detects JSON in
log output and lands typed columns in BigQuery via a log sink, so this
module deliberately stays as a thin wrapper over `logging`:

    log_event("query_received", user_id=uid, session_id=sid, query=q,
              num_sources=len(sources), latency_ms=elapsed)

→ a single line like:
    {"event_type":"query_received","user_id":"...","ts":"2026-04-29T...","...":...}

Fields with `None` values are dropped so JSON columns in BigQuery stay
sparse rather than full of nulls. `query_text` and `correction_text`
are truncated to keep log volume bounded under abusive input.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("analytics")
logger.setLevel(logging.INFO)

# Cap free-text fields. 2 KB is enough to inspect a query post-hoc but
# small enough that 100k events stay well under Cloud Logging's free tier.
_MAX_TEXT_CHARS = 2000

_TEXT_FIELDS = {"query_text", "correction_text", "exception_message"}


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
        payload: dict[str, Any] = {
            "event_type": event_type,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        for key, value in fields.items():
            if value is None:
                continue
            if key in _TEXT_FIELDS:
                value = _truncate(value)
            payload[key] = value
        # logger.info() with a single JSON string is what GCP Cloud Logging
        # picks up automatically as `jsonPayload`. Locally it just prints
        # to stdout, which docker-compose forwards to the user's terminal.
        logger.info(json.dumps(payload, default=str))
    except Exception:  # pragma: no cover - analytics must never raise
        logger.warning("analytics: failed to emit event_type=%s", event_type)
