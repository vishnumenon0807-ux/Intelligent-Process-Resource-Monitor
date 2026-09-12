"""
IPRM :: REST API.

Read-mostly layer over the existing schema. No ORM -- these are queries you
already know how to write, and an ORM would only obscure them.

Run:
    pip install fastapi uvicorn
    uvicorn api.app:app --reload --port 8000

Interactive docs at http://localhost:8000/docs
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator, Literal

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

DSN = os.environ.get(
    "IPRM_DSN", "host=localhost port=5432 dbname=iprm user=postgres"
)

# Change in one place if you rename the table back to `recommendations`.
ADVISORY_TABLE = "advisories"
ADVISORY_VIEW = "v_advisory_summary"

app = FastAPI(
    title="Intelligent Process Resource Monitor",
    description="Anomaly detection, root cause analysis, and advisory API.",
    version="0.1.0",
)

# Vite dev server. Add your deployed origin here if you ever host the frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Database
#
# One connection per request. At dashboard polling rates this is cheap, and it
# avoids the psycopg_pool dependency. Swap in a pool if you ever need it.
# ---------------------------------------------------------------------------
@contextmanager
def _connect() -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(DSN, row_factory=dict_row, autocommit=True)
    try:
        yield conn
    finally:
        conn.close()


def db() -> Iterator[psycopg.Connection]:
    with _connect() as conn:
        yield conn


def fetch_all(conn, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def fetch_one(conn, sql: str, params: tuple = ()) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class Advisory(BaseModel):
    id: int
    rule_id: str
    category: str
    confidence: float
    severity: float | None = None
    title: str
    diagnosis: str
    cause: str
    suggestions: list[str] = []
    target_pid: int | None = None
    target_name: str | None = None
    metrics: dict[str, Any] = {}
    outcome: str
    metric_delta_pct: float | None = None
    user_feedback: str | None = None
    created_at: datetime


class SummaryRow(BaseModel):
    rule_id: str
    category: str
    issued: int
    avg_confidence: float | None = None
    resolved: int
    persisted: int
    marked_helpful: int
    last_seen: datetime | None = None


class MetricPoint(BaseModel):
    ts: datetime
    cpu_percent: float | None = None
    ram_percent: float | None = None
    disk_percent: float | None = None
    gpu_percent: float | None = None
    swap_percent: float | None = None
    process_count: float | None = None


class LatestMetrics(BaseModel):
    ts: datetime
    cpu_percent: float | None = None
    cpu_temp_c: float | None = None
    ram_percent: float | None = None
    ram_used_mb: float | None = None
    ram_available_mb: float | None = None
    swap_percent: float | None = None
    disk_percent: float | None = None
    gpu_percent: float | None = None
    gpu_temp_c: float | None = None
    gpu_memory_percent: float | None = None
    net_sent_rate_bps: float | None = None
    net_recv_rate_bps: float | None = None
    process_count: int | None = None
    thread_count_total: int | None = None
    battery_percent: float | None = None
    on_battery: bool | None = None


class FeedbackIn(BaseModel):
    feedback: Literal["helpful", "not_helpful", "dismissed"]


# ---------------------------------------------------------------------------
# Window handling
#
# 19k+ snapshots means a 7-day window would return thousands of points and
# choke the chart. Bucket server-side so the payload stays roughly 200-400
# points regardless of window.
# ---------------------------------------------------------------------------
WINDOWS: dict[str, tuple[str, int]] = {
    # name: (postgres interval, bucket width in seconds)
    "15m": ("15 minutes", 10),
    "1h":  ("1 hour", 30),
    "6h":  ("6 hours", 120),
    "24h": ("24 hours", 300),
    "7d":  ("7 days", 1800),
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", tags=["meta"])
def health(conn=Depends(db)) -> dict[str, Any]:
    """Liveness probe that actually checks the database, not just the process."""
    row = fetch_one(conn, "SELECT count(*) AS n FROM system_snapshots")
    return {"status": "ok", "snapshots": row["n"] if row else 0}


@app.get("/api/advisories", response_model=list[Advisory], tags=["advisories"])
def list_advisories(
    conn=Depends(db),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    category: str | None = Query(None, pattern="^(cpu|memory|disk|startup)$"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    hours: int | None = Query(None, ge=1, le=720,
                              description="Only advisories from the last N hours"),
) -> list[dict[str, Any]]:
    """The dashboard feed. Newest first."""
    clauses = ["confidence >= %s"]
    params: list[Any] = [min_confidence]

    if category:
        clauses.append("category = %s")
        params.append(category)
    if hours:
        clauses.append("created_at > now() - make_interval(hours => %s)")
        params.append(hours)

    params.extend([limit, offset])
    return fetch_all(
        conn,
        f"""
        SELECT id, rule_id, category, confidence, severity,
               title, diagnosis, cause, suggestions,
               target_pid, target_name, metrics,
               outcome, metric_delta_pct, user_feedback, created_at
          FROM {ADVISORY_TABLE}
         WHERE {' AND '.join(clauses)}
         ORDER BY created_at DESC
         LIMIT %s OFFSET %s
        """,
        tuple(params),
    )


@app.get("/api/advisories/{advisory_id}", response_model=Advisory, tags=["advisories"])
def get_advisory(advisory_id: int, conn=Depends(db)) -> dict[str, Any]:
    row = fetch_one(
        conn,
        f"""
        SELECT id, rule_id, category, confidence, severity,
               title, diagnosis, cause, suggestions,
               target_pid, target_name, metrics,
               outcome, metric_delta_pct, user_feedback, created_at
          FROM {ADVISORY_TABLE} WHERE id = %s
        """,
        (advisory_id,),
    )
    if row is None:
        raise HTTPException(404, f"advisory {advisory_id} not found")
    return row


@app.post("/api/advisories/{advisory_id}/feedback", tags=["advisories"])
def set_feedback(
    advisory_id: int, body: FeedbackIn, conn=Depends(db)
) -> dict[str, Any]:
    """Thumbs up/down from the dashboard. Feeds the summary view."""
    row = fetch_one(
        conn,
        f"UPDATE {ADVISORY_TABLE} SET user_feedback = %s WHERE id = %s "
        f"RETURNING id, user_feedback",
        (body.feedback, advisory_id),
    )
    if row is None:
        raise HTTPException(404, f"advisory {advisory_id} not found")
    return row


@app.get("/api/summary", response_model=list[SummaryRow], tags=["advisories"])
def summary(conn=Depends(db)) -> list[dict[str, Any]]:
    """Per-rule counts. Which rules fire most, and how often conditions clear."""
    return fetch_all(
        conn,
        f"""
        SELECT rule_id, category, issued, avg_confidence,
               resolved, persisted, marked_helpful, last_seen
          FROM {ADVISORY_VIEW}
        """,
    )


@app.get("/api/metrics", response_model=list[MetricPoint], tags=["metrics"])
def metrics(
    conn=Depends(db),
    window: str = Query("1h", description=f"One of: {', '.join(WINDOWS)}"),
) -> list[dict[str, Any]]:
    """
    Bucketed host timeseries for the charts. Bucket width scales with the
    window so the payload stays chart-sized.
    """
    if window not in WINDOWS:
        raise HTTPException(422, f"window must be one of {list(WINDOWS)}")
    interval, bucket = WINDOWS[window]

    return fetch_all(
        conn,
        """
        SELECT to_timestamp(floor(extract(epoch FROM timestamp) / %s) * %s) AS ts,
               avg(cpu_percent)    AS cpu_percent,
               avg(ram_percent)    AS ram_percent,
               avg(disk_percent)   AS disk_percent,
               avg(gpu_percent)    AS gpu_percent,
               avg(swap_percent)   AS swap_percent,
               avg(process_count)  AS process_count
          FROM system_snapshots
         WHERE timestamp > now() - %s::interval
         GROUP BY 1
         ORDER BY 1
        """,
        (bucket, bucket, interval),
    )


@app.get("/api/metrics/latest", response_model=LatestMetrics, tags=["metrics"])
def latest_metrics(conn=Depends(db)) -> dict[str, Any]:
    """Current values for the gauges at the top of the dashboard."""
    row = fetch_one(
        conn,
        """
        SELECT timestamp AS ts, cpu_percent, cpu_temp_c,
               ram_percent, ram_used_mb, ram_available_mb, swap_percent,
               disk_percent, gpu_percent, gpu_temp_c, gpu_memory_percent,
               net_sent_rate_bps, net_recv_rate_bps,
               process_count, thread_count_total,
               battery_percent, on_battery
          FROM system_snapshots
         ORDER BY timestamp DESC
         LIMIT 1
        """,
    )
    if row is None:
        raise HTTPException(404, "no snapshots recorded yet")
    return row