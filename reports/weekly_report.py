"""Weekly product analytics report → email.

Designed to run as a Cloud Run Job on a weekly Cloud Scheduler cron.
Pulls the last 7 days from BigQuery, computes the same metrics we
discussed for the weekly review, renders an HTML email, and sends via
SendGrid.

Required env vars (all read at startup; missing ones fail fast):
    GCP_PROJECT       e.g. "rag-project-494802"
    BQ_DATASET        e.g. "berrybio_analytics"
    REPORT_TO         destination email
    REPORT_FROM       verified sender email in SendGrid
    SENDGRID_API_KEY  from Secret Manager

The script is idempotent: it CREATE OR REPLACE's the `events` view on
every run, so a fresh dataset (or schema drift in the underlying log
table) self-heals without manual DDL.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from html import escape

from google.cloud import bigquery
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("weekly_report")


# ---------- env ----------

def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        log.error("Missing required env var: %s", name)
        sys.exit(2)
    return value


PROJECT = _require_env("GCP_PROJECT")
DATASET = os.environ.get("BQ_DATASET", "berrybio_analytics")
TO_EMAIL = _require_env("REPORT_TO")
FROM_EMAIL = _require_env("REPORT_FROM")
SG_API_KEY = _require_env("SENDGRID_API_KEY")


# ---------- BigQuery ----------

bq = bigquery.Client(project=PROJECT)


def ensure_view() -> None:
    """Create or replace the typed `events` view over the raw sink table.

    The Cloud Logging → BigQuery sink lands one row per log line in tables
    named `run_googleapis_com_stdout_YYYYMMDD` (date-partitioned). The view
    flattens the JSON in `text_payload` into typed columns so the report
    queries stay readable. CREATE OR REPLACE so this is safe to re-run.
    """
    sql = f"""
    CREATE OR REPLACE VIEW `{PROJECT}.{DATASET}.events` AS
    SELECT
      timestamp,
      JSON_VALUE(text_payload, '$.event_type')    AS event_type,
      JSON_VALUE(text_payload, '$.user_id')       AS user_id,
      JSON_VALUE(text_payload, '$.session_id')    AS session_id,
      JSON_VALUE(text_payload, '$.endpoint')      AS endpoint,
      JSON_VALUE(text_payload, '$.query_text')    AS query_text,
      CAST(JSON_VALUE(text_payload, '$.num_sources') AS INT64) AS num_sources,
      CAST(JSON_VALUE(text_payload, '$.latency_ms')  AS INT64) AS latency_ms,
      CAST(JSON_VALUE(text_payload, '$.turn_number') AS INT64) AS turn_number,
      JSON_VALUE(text_payload, '$.kind')          AS kind,
      JSON_VALUE(text_payload, '$.format')        AS format,
      JSON_VALUE(text_payload, '$.phase')         AS phase,
      JSON_VALUE(text_payload, '$.study_type')    AS study_type,
      JSON_VALUE(text_payload, '$.alias')         AS promoted_alias,
      JSON_VALUE(text_payload, '$.canonical')     AS promoted_canonical,
      JSON_VALUE(text_payload, '$.correction_text') AS correction_text,
      text_payload AS raw
    FROM `{PROJECT}.{DATASET}.run_googleapis_com_stdout_*`
    WHERE JSON_VALUE(text_payload, '$.event_type') IS NOT NULL
    """
    bq.query(sql).result()


def q(sql: str) -> list[dict]:
    rows = bq.query(sql).result()
    return [dict(r) for r in rows]


def _wow_pct(this: int | None, prev: int | None) -> str:
    """Format a week-over-week percentage delta: +12.5% / −3.2% / —."""
    if not prev:
        return "—" if not this else "(new)"
    delta = (this or 0) - prev
    pct = (delta / prev) * 100
    sign = "+" if delta >= 0 else "−"  # using a real minus sign
    return f"{sign}{abs(pct):.1f}%"


# ---------- queries ----------

def fetch_metrics() -> dict:
    base = f"`{PROJECT}.{DATASET}.events`"

    # Headline numbers + week-over-week comparison.
    headline = q(f"""
        SELECT
          (SELECT COUNT(DISTINCT user_id) FROM {base}
            WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL  7 DAY)) AS wau_this,
          (SELECT COUNT(DISTINCT user_id) FROM {base}
            WHERE timestamp BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
                                AND TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL  7 DAY)) AS wau_prev,
          (SELECT COUNT(DISTINCT session_id) FROM {base}
            WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL  7 DAY)) AS sessions_this,
          (SELECT COUNT(DISTINCT session_id) FROM {base}
            WHERE timestamp BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
                                AND TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL  7 DAY)) AS sessions_prev,
          (SELECT COUNT(*) FROM {base}
            WHERE event_type='query_received'
              AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)) AS queries_this,
          (SELECT COUNT(*) FROM {base}
            WHERE event_type='query_received'
              AND timestamp BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
                                AND TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL  7 DAY)) AS queries_prev
    """)[0]

    # New vs returning — uses lifetime first_seen.
    new_returning = q(f"""
        WITH first_seen AS (
          SELECT user_id, MIN(DATE(timestamp)) AS first_day
          FROM {base} GROUP BY user_id
        )
        SELECT
          COUNTIF(first_day >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)) AS new_users,
          COUNTIF(first_day <  DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
                  AND user_id IN (
                    SELECT DISTINCT user_id FROM {base}
                    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
                  )) AS returning_users
        FROM first_seen
    """)[0]

    # Funnel: how far did sessions get this week?
    funnel = q(f"""
        SELECT
          COUNT(DISTINCT IF(event_type='query_received',     session_id, NULL)) AS searched,
          COUNT(DISTINCT IF(event_type='chat_message',       session_id, NULL)) AS reached_chat,
          COUNT(DISTINCT IF(event_type='protocol_generated', session_id, NULL)) AS generated_protocol,
          COUNT(DISTINCT IF(event_type='protocol_refined',   session_id, NULL)) AS refined_protocol,
          COUNT(DISTINCT IF(event_type='protocol_downloaded',session_id, NULL)) AS downloaded
        FROM {base}
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    """)[0]

    # Quality signals from feedback events.
    quality = q(f"""
        SELECT
          COUNTIF(kind='rating_up')   AS thumbs_up,
          COUNTIF(kind='rating_down') AS thumbs_down,
          COUNTIF(kind='correction')  AS corrections,
          SAFE_DIVIDE(COUNTIF(kind='rating_down'),
                      COUNTIF(kind IN ('rating_up','rating_down'))) AS down_rate
        FROM {base}
        WHERE event_type='feedback_submitted'
          AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    """)[0]

    # Top thumbs-down queries — what to investigate.
    bad_queries = q(f"""
        SELECT query_text, COUNT(*) AS thumbs_down_count
        FROM {base}
        WHERE event_type='feedback_submitted' AND kind='rating_down'
          AND query_text IS NOT NULL
          AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        GROUP BY query_text
        ORDER BY thumbs_down_count DESC, query_text
        LIMIT 10
    """)

    # Zero-result queries — retrieval gaps.
    zero_results = q(f"""
        SELECT query_text, COUNT(*) AS occurrences
        FROM {base}
        WHERE event_type='query_received' AND num_sources = 0
          AND query_text IS NOT NULL
          AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        GROUP BY query_text
        ORDER BY occurrences DESC
        LIMIT 10
    """)

    # Latency.
    latency = q(f"""
        SELECT endpoint,
               APPROX_QUANTILES(latency_ms, 100)[OFFSET(50)] AS p50_ms,
               APPROX_QUANTILES(latency_ms, 100)[OFFSET(95)] AS p95_ms,
               COUNT(*) AS calls
        FROM {base}
        WHERE event_type='query_received' AND latency_ms IS NOT NULL
          AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        GROUP BY endpoint
        ORDER BY calls DESC
    """)

    return {
        "headline": headline,
        "new_returning": new_returning,
        "funnel": funnel,
        "quality": quality,
        "bad_queries": bad_queries,
        "zero_results": zero_results,
        "latency": latency,
    }


# ---------- HTML rendering ----------

CSS = """
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
         color: #222; line-height: 1.5; max-width: 720px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  h2 { font-size: 15px; margin-top: 28px; margin-bottom: 8px;
       text-transform: uppercase; letter-spacing: 0.05em; color: #555; border-bottom: 1px solid #eee; padding-bottom: 4px; }
  .subtitle { color: #888; font-size: 13px; margin-bottom: 24px; }
  .kpis { display: table; width: 100%; border-collapse: collapse; margin: 12px 0; }
  .kpi { display: table-cell; padding: 14px 12px; border: 1px solid #eee; text-align: center; vertical-align: top; }
  .kpi-num { font-size: 26px; font-weight: 600; color: #111; }
  .kpi-lbl { font-size: 12px; color: #777; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.04em; }
  .kpi-delta { font-size: 12px; margin-top: 6px; }
  .kpi-up   { color: #1a7f3c; }
  .kpi-down { color: #b3261e; }
  .kpi-flat { color: #888; }
  table.data { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }
  table.data th, table.data td { border: 1px solid #eee; padding: 6px 10px; text-align: left; }
  table.data th { background: #fafafa; font-weight: 600; color: #555; }
  table.data td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .empty { color: #999; font-style: italic; font-size: 13px; }
  .footer { color: #999; font-size: 12px; margin-top: 36px; border-top: 1px solid #eee; padding-top: 12px; }
"""


def _delta_class(this: int | None, prev: int | None) -> str:
    if not prev or this is None:
        return "kpi-flat"
    return "kpi-up" if this >= prev else "kpi-down"


def _kpi(label: str, value: int | None, prev: int | None) -> str:
    val = "—" if value is None else f"{value:,}"
    return (
        f'<div class="kpi">'
        f'<div class="kpi-num">{val}</div>'
        f'<div class="kpi-lbl">{escape(label)}</div>'
        f'<div class="kpi-delta {_delta_class(value, prev)}">{_wow_pct(value, prev)}</div>'
        f'</div>'
    )


def _table(rows: list[dict], cols: list[tuple[str, str, bool]]) -> str:
    """cols: list of (key, header, is_numeric)."""
    if not rows:
        return '<p class="empty">No data this week.</p>'
    header = "".join(f"<th>{escape(h)}</th>" for _, h, _ in cols)
    body_rows = []
    for r in rows:
        cells = []
        for key, _, num in cols:
            v = r.get(key)
            if v is None:
                v = ""
            elif isinstance(v, float):
                v = f"{v:.1%}" if 0 < v < 1 else f"{v:,.0f}"
            elif isinstance(v, int):
                v = f"{v:,}"
            else:
                v = escape(str(v))
            cls = ' class="num"' if num else ""
            cells.append(f"<td{cls}>{v}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="data"><thead><tr>{header}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def render_html(m: dict) -> str:
    h = m["headline"]
    nr = m["new_returning"]
    f = m["funnel"]
    q_ = m["quality"]

    kpis = (
        '<div class="kpis">'
        + _kpi("Active users (WAU)", h.get("wau_this"), h.get("wau_prev"))
        + _kpi("Sessions",            h.get("sessions_this"), h.get("sessions_prev"))
        + _kpi("Queries",             h.get("queries_this"), h.get("queries_prev"))
        + "</div>"
    )

    new_v_returning = (
        '<div class="kpis">'
        + _kpi("New users",       nr.get("new_users"), None)
        + _kpi("Returning users", nr.get("returning_users"), None)
        + "</div>"
    )

    funnel_rows = [
        {"step": "1. Searched",          "sessions": f.get("searched")},
        {"step": "2. Reached chat",      "sessions": f.get("reached_chat")},
        {"step": "3. Generated protocol","sessions": f.get("generated_protocol")},
        {"step": "4. Refined protocol",  "sessions": f.get("refined_protocol")},
        {"step": "5. Downloaded",        "sessions": f.get("downloaded")},
    ]

    quality_rows = []
    if any(q_.get(k) for k in ("thumbs_up", "thumbs_down", "corrections")):
        quality_rows = [{
            "metric": "Thumbs up",   "count": q_.get("thumbs_up", 0),
        }, {
            "metric": "Thumbs down", "count": q_.get("thumbs_down", 0),
        }, {
            "metric": "Corrections", "count": q_.get("corrections", 0),
        }, {
            "metric": "Thumbs-down rate",
            "count": (q_.get("down_rate") or 0),
        }]

    today = datetime.now(timezone.utc).strftime("%B %d, %Y")
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
  <h1>Berrybio RAG · Weekly Report</h1>
  <div class="subtitle">Week ending {today} (last 7 days vs. previous 7 days)</div>

  <h2>Headline</h2>
  {kpis}

  <h2>User growth</h2>
  {new_v_returning}

  <h2>Engagement funnel (sessions)</h2>
  {_table(funnel_rows, [("step", "Step", False), ("sessions", "Sessions", True)])}

  <h2>Feedback quality</h2>
  {_table(quality_rows, [("metric", "Metric", False), ("count", "Count / rate", True)])}

  <h2>Top thumbs-down queries (investigate)</h2>
  {_table(m["bad_queries"], [("query_text", "Query", False), ("thumbs_down_count", "Down votes", True)])}

  <h2>Zero-result queries (retrieval gaps)</h2>
  {_table(m["zero_results"], [("query_text", "Query", False), ("occurrences", "Hits", True)])}

  <h2>Latency by endpoint</h2>
  {_table(m["latency"], [("endpoint", "Endpoint", False), ("p50_ms", "p50 (ms)", True),
                         ("p95_ms", "p95 (ms)", True), ("calls", "Calls", True)])}

  <div class="footer">
    Generated automatically · BerryBio RAG analytics ·
    Data source: <code>{escape(PROJECT)}.{escape(DATASET)}.events</code>
  </div>
</body></html>
"""


# ---------- send ----------

def send_email(html: str) -> None:
    today = datetime.now(timezone.utc).strftime("%b %d")
    msg = Mail(
        from_email=FROM_EMAIL,
        to_emails=TO_EMAIL,
        subject=f"Berrybio RAG · Weekly Report · {today}",
        html_content=html,
    )
    sg = SendGridAPIClient(SG_API_KEY)
    resp = sg.send(msg)
    log.info("SendGrid response: status=%s", resp.status_code)
    if resp.status_code >= 300:
        # 4xx/5xx — surface the body so the Cloud Run Job log shows what failed.
        log.error("SendGrid error body: %s", resp.body)
        sys.exit(1)


def main() -> None:
    log.info("Project=%s Dataset=%s To=%s From=%s", PROJECT, DATASET, TO_EMAIL, FROM_EMAIL)
    ensure_view()
    metrics = fetch_metrics()
    html = render_html(metrics)
    send_email(html)
    log.info("Weekly report sent successfully.")


if __name__ == "__main__":
    main()
