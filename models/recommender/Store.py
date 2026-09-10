"""
IPRM :: recommendation persistence.

Three functions. Swap psycopg for whatever collector.py already uses.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .recommender import Recommendation, SystemState

UTC = timezone.utc

DEDUPE_WINDOW_SECONDS = 600      # do not re-advise the same rule+process for 10 min
RECHECK_AFTER_SECONDS = 120      # how long before we look again


def already_advised(conn, rule_id: str, pid: int | None) -> bool:
    """Without this you get the same card 200 times in an afternoon."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT 1 FROM advisories
             WHERE rule_id = %s
               AND target_pid IS NOT DISTINCT FROM %s
               AND created_at > now() - make_interval(secs => %s)
             LIMIT 1
            """,
            (rule_id, pid, DEDUPE_WINDOW_SECONDS),
        )
        return cur.fetchone() is not None


def save(conn, rec: Recommendation) -> int | None:
    if already_advised(conn, rec.rule_id, rec.state.pid):
        return None

    s: SystemState = rec.state
    recheck = datetime.now(UTC) + timedelta(seconds=RECHECK_AFTER_SECONDS)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO advisories (
                anomaly_id, rule_id, category, confidence, severity,
                title, diagnosis, cause, suggestions,
                target_pid, target_name, target_create_time, metrics,
                recheck_due_at
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s
            ) RETURNING id
            """,
            (
                s.anomaly_id, rec.rule_id, rec.category, rec.confidence, s.severity,
                rec.title, rec.diagnosis, rec.cause, Jsonb(list(rec.suggestions)),
                s.pid, s.name or None, s.create_time, Jsonb(s.as_metrics()),
                recheck,
            ),
        )
        return cur.fetchone()["id"]


def sweep_rechecks(conn, sample_fn) -> int:
    """
    Look again at conditions we flagged a couple of minutes ago.

    `sample_fn(pid, create_time) -> dict | None` should return current metrics,
    or None if the process is gone. Reuse whatever collector.py already has.

    This records whether the condition persisted. It does NOT establish that
    any suggestion caused the change -- label the dashboard column accordingly.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, target_pid, target_create_time, metrics, rule_id
              FROM advisories
             WHERE outcome = 'pending' AND recheck_due_at <= now()
             ORDER BY recheck_due_at
             LIMIT 20
            """
        )
        rows = cur.fetchall()

    for row in rows:
        current = sample_fn(row["target_pid"], row["target_create_time"])
        before = (row["metrics"] or {}).get("cpu_percent")

        if current is None:
            outcome, delta = "process_gone", None
        elif before is None or before < 5:
            outcome, delta = "unknown", None
        else:
            after = current.get("cpu_percent", 0.0)
            delta = (before - after) / before * 100.0
            outcome = "resolved" if delta >= 30 else "persisted"

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE advisories
                   SET outcome = %s, metric_delta_pct = %s, rechecked_at = now()
                 WHERE id = %s
                """,
                (outcome, delta, row["id"]),
            )

    return len(rows)