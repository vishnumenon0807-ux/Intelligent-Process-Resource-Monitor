"""
IPRM :: REST API

    python -m uvicorn api.app:app --port 8000

Nine endpoints over the tables the rest of the pipeline writes. This
layer only reads and serves: detection, attribution and forecasting
have already happened, and their results are sitting in PostgreSQL.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator, Literal

import psycopg
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from psycopg.rows import dict_row
from pydantic import BaseModel

# Must run before DSN is read: that assignment happens at import time,
# so loading .env afterwards would be too late and the connection would
# go out with no password.
load_dotenv()

DSN = os.environ.get(
    "IPRM_DSN", "host=localhost port=5432 dbname=iprm user=postgres"
)

ADVISORY_TABLE = "advisories"
ADVISORY_VIEW = "v_advisory_summary"

app = FastAPI(
    title="Intelligent Process Resource Monitor",
    description="Anomaly detection, root cause analysis, and advisory API.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================================================================
# Database: one connection per request
# ===================================================================

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


# ===================================================================
# Response models
# ===================================================================

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


class ForecastPoint(BaseModel):
    ts: datetime
    actual: float | None = None
    predicted: float | None = None


class Forecast(BaseModel):
    # `now` comes from the database rather than the browser. The chart
    # draws a boundary between measured and predicted, and a browser
    # clock a few minutes off would put that line in the wrong place.
    now: datetime
    metric: str
    horizon_s: int | None = None
    points: list[ForecastPoint]


class AccuracyRow(BaseModel):
    metric: str
    rmse: float | None = None
    mae: float | None = None
    scored: int
    horizon_s: int | None = None


class ForecastAccuracy(BaseModel):
    version: str | None = None
    trained_at: datetime | None = None
    hours: int
    metrics: list[AccuracyRow]


# ===================================================================
# Bucketing: (postgres interval, bucket width in seconds)
# ===================================================================

WINDOWS: dict[str, tuple[str, int]] = {
    "15m": ("15 minutes", 10),
    "1h": ("1 hour", 30),
    "6h": ("6 hours", 120),
    "24h": ("24 hours", 300),
    "7d": ("7 days", 1800),
}

# A metric name becomes a column name in the forecast query below, and
# a column name cannot be passed as a bound parameter. This whitelist
# is what makes that interpolation safe: anything not a key here is
# rejected before the SQL is built.
FORECAST_METRICS: dict[str, str] = {
    "cpu_percent": "cpu_percent",
    "ram_percent": "ram_percent",
    "gpu_percent": "gpu_percent",
}

ADVISORY_COLUMNS = """
    id, rule_id, category, confidence, severity,
    title, diagnosis, cause, suggestions,
    target_pid, target_name, metrics,
    outcome, metric_delta_pct, user_feedback, created_at
"""


# ===================================================================
# Meta
# ===================================================================

@app.get("/api/health", tags=["meta"])
def health(conn=Depends(db)) -> dict[str, Any]:
    """Liveness probe that checks the database, not just the process."""
    row = fetch_one(conn, "SELECT count(*) AS n FROM system_snapshots")
    return {"status": "ok", "snapshots": row["n"] if row else 0}


# ===================================================================
# Advisories
# ===================================================================

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
        SELECT {ADVISORY_COLUMNS}
        FROM {ADVISORY_TABLE}
        WHERE {' AND '.join(clauses)}
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
        """,
        tuple(params),
    )


@app.get("/api/advisories/{advisory_id}", response_model=Advisory,
         tags=["advisories"])
def get_advisory(advisory_id: int, conn=Depends(db)) -> dict[str, Any]:
    row = fetch_one(
        conn,
        f"SELECT {ADVISORY_COLUMNS} FROM {ADVISORY_TABLE} WHERE id = %s",
        (advisory_id,),
    )
    if row is None:
        raise HTTPException(404, f"advisory {advisory_id} not found")
    return row


@app.post("/api/advisories/{advisory_id}/feedback", tags=["advisories"])
def set_feedback(advisory_id: int, body: FeedbackIn,
                 conn=Depends(db)) -> dict[str, Any]:
    """Thumbs up or down from the dashboard. Feeds the summary view."""
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
    """Per-rule counts. Which rules fire most, and how often they clear."""
    return fetch_all(
        conn,
        f"""
        SELECT rule_id, category, issued, avg_confidence,
               resolved, persisted, marked_helpful, last_seen
        FROM {ADVISORY_VIEW}
        """,
    )


# ===================================================================
# Metrics
# ===================================================================

@app.get("/api/metrics", response_model=list[MetricPoint], tags=["metrics"])
def metrics(
    conn=Depends(db),
    window: str = Query("1h", description=f"One of: {', '.join(WINDOWS)}"),
) -> list[dict[str, Any]]:
    """Bucketed host timeseries for the charts."""
    if window not in WINDOWS:
        raise HTTPException(422, f"window must be one of {list(WINDOWS)}")
    interval, bucket = WINDOWS[window]
    return fetch_all(
        conn,
        """
        SELECT to_timestamp(floor(extract(epoch FROM timestamp) / %s) * %s) AS ts,
               avg(cpu_percent) AS cpu_percent,
               avg(ram_percent) AS ram_percent,
               avg(disk_percent) AS disk_percent,
               avg(gpu_percent) AS gpu_percent,
               avg(swap_percent) AS swap_percent,
               avg(process_count) AS process_count
        FROM system_snapshots
        WHERE timestamp > now() - %s::interval
        GROUP BY 1
        ORDER BY 1
        """,
        (bucket, bucket, interval),
    )


@app.get("/api/metrics/latest", response_model=LatestMetrics, tags=["metrics"])
def latest_metrics(conn=Depends(db)) -> dict[str, Any]:
    """Current values for the dashboard gauges."""
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


# ===================================================================
# Forecasting
# ===================================================================

@app.get("/api/forecast", response_model=Forecast, tags=["forecast"])
def forecast(
    conn=Depends(db),
    window: str = Query("1h", description=f"One of: {', '.join(WINDOWS)}"),
    metric: str = Query("cpu_percent",
                        description=f"One of: {', '.join(FORECAST_METRICS)}"),
) -> dict[str, Any]:
    """
    What actually happened, and what the LSTM said would happen, on one
    shared time grid.

    Both series are bucketed to the same width, so the front end can
    plot them directly without aligning anything. Buckets past the
    present have a `predicted` value and no `actual`: that is the part
    of the line extending into the future. Buckets already in the past
    have both, which is what makes the chart evidence rather than
    decoration.
    """
    if window not in WINDOWS:
        raise HTTPException(422, f"window must be one of {list(WINDOWS)}")
    if metric not in FORECAST_METRICS:
        raise HTTPException(
            422, f"metric must be one of {list(FORECAST_METRICS)}")

    interval, bucket = WINDOWS[window]
    column = FORECAST_METRICS[metric]   # whitelisted above

    points = fetch_all(
        conn,
        f"""
        WITH actual AS (
            SELECT to_timestamp(
                       floor(extract(epoch FROM timestamp) / %s) * %s) AS ts,
                   avg({column}) AS actual
            FROM system_snapshots
            WHERE timestamp > now() - %s::interval
            GROUP BY 1
        ),
        pred AS (
            SELECT to_timestamp(
                       floor(extract(epoch FROM target_ts) / %s) * %s) AS ts,
                   avg(predicted) AS predicted
            FROM predictions
            WHERE metric = %s
              AND target_ts > now() - %s::interval
            GROUP BY 1
        )
        SELECT coalesce(a.ts, p.ts) AS ts, a.actual, p.predicted
        FROM actual a
        FULL OUTER JOIN pred p ON a.ts = p.ts
        ORDER BY 1
        """,
        (bucket, bucket, interval, bucket, bucket, metric, interval),
    )

    meta = fetch_one(
        conn,
        """
        SELECT now() AS now,
               (SELECT max(horizon_s) FROM predictions
                WHERE metric = %s
                  AND made_at > now() - interval '1 hour') AS horizon_s
        """,
        (metric,),
    )

    return {
        "now": meta["now"],
        "metric": metric,
        "horizon_s": meta["horizon_s"],
        "points": points,
    }


@app.get("/api/forecast/accuracy", response_model=ForecastAccuracy,
         tags=["forecast"])
def forecast_accuracy(
    conn=Depends(db),
    hours: int = Query(24, ge=1, le=720),
) -> dict[str, Any]:
    """
    Error measured on forecasts whose moment has since arrived.

    A stronger number than the training RMSE, because every row in it
    was predicted before the value existed. It is also the one to quote,
    since the training figure pools three targets and one of them (GPU)
    is currently a flat zero that drags the average down.
    """
    rows = fetch_all(
        conn,
        """
        SELECT metric,
               sqrt(avg(power(predicted - actual, 2))) AS rmse,
               avg(abs(predicted - actual)) AS mae,
               count(*) AS scored,
               max(horizon_s) AS horizon_s
        FROM v_prediction_accuracy
        WHERE actual IS NOT NULL
          AND made_at > now() - make_interval(hours => %s)
        GROUP BY metric
        ORDER BY metric
        """,
        (hours,),
    )

    model = fetch_one(
        conn,
        """
        SELECT model_version, trained_at
        FROM model_registry
        WHERE model_type = 'lstm_forecaster' AND is_active
        ORDER BY trained_at DESC
        LIMIT 1
        """,
    )

    return {
        "version": model["model_version"] if model else None,
        "trained_at": model["trained_at"] if model else None,
        "hours": hours,
        "metrics": rows,
    }