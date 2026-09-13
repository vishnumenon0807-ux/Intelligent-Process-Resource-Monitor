"""
IPRM :: bridge from the root cause engine to the recommender.

The engine already knows more than scan.py ever did -- it has the Isolation
Forest score, the analyzer's cause text, and process trees aggregated across
children. This maps that onto SystemState and files an advisory.

Call it from Root_cause_engine.py right after insert_root_cause:

    from recommender.bridge import advise_from_anomaly
    advise_from_anomaly(conn, anomaly, tree, result)

Once this is running, scan.py is redundant -- retire it.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from .recommender import RECOMMENDER, SystemState
from .store import save

log = logging.getLogger("iprm.bridge")


def _first(d: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Tolerate key-name drift between the engine and this module."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def normalise_severity(score: float | None) -> float | None:
    """
    sklearn's decision_function is NEGATIVE for anomalies, and more negative
    means more anomalous. The recommender expects 0-1 where higher is more
    severe, so flip the sign and clamp. Passing the raw -0.638 through would
    read as 0.0 -- i.e. the most anomalous events would get the LOWEST
    confidence, which is exactly backwards.
    """
    if score is None:
        return None
    return max(0.0, min(1.0, -float(score)))


def build_state(
    anomaly: Mapping[str, Any],
    tree: Mapping[str, Any],
    result: Mapping[str, Any] | None = None,
    *,
    memory_growth_mb_per_min: float = 0.0,
) -> SystemState:
    result = result or {}
    pids = _first(tree, "pids", "process_tree_pids", default=[]) or []

    return SystemState(
        pid=pids[0] if pids else _first(tree, "pid"),
        name=_first(tree, "name", default="") or "",
        create_time=_first(tree, "create_time"),
        cpu_percent=float(_first(tree, "cpu_percent", "cpu", default=0.0)),
        memory_mb=float(_first(tree, "ram_mb", "memory_mb", "ram", default=0.0)),
        memory_growth_mb_per_min=memory_growth_mb_per_min,
        # The engine aggregates the whole tree, so this is the real child
        # count -- more accurate than scan.py's parent-PID tally.
        child_count=max(0, len(pids) - 1),
        thread_count=int(_first(tree, "thread_count", default=0) or 0),
        uptime_minutes=float(_first(tree, "uptime_minutes", default=0.0) or 0.0),
        host_cpu_percent=float(_first(anomaly, "sys_cpu", default=0.0) or 0.0),
        host_mem_percent=float(_first(anomaly, "sys_ram", default=0.0) or 0.0),
        process_count=int(_first(anomaly, "process_count", default=0) or 0),
        anomaly_id=_first(anomaly, "id", "anomaly_id"),
        cause_label=_first(result, "analyzer_used", "cause", "category"),
        cause_text=_first(result, "explanation", "reason"),
        severity=normalise_severity(_first(anomaly, "score", "anomaly_score")),
    )


def advise_from_anomaly(
    conn,
    anomaly: Mapping[str, Any],
    tree: Mapping[str, Any],
    result: Mapping[str, Any] | None = None,
    *,
    memory_growth_mb_per_min: float = 0.0,
) -> int | None:
    """
    Returns the new advisory id, or None if no rule matched or it was deduped.
    Never raises -- an advisory failure must not stop the engine from writing
    its root cause.
    """
    try:
        state = build_state(
            anomaly, tree, result,
            memory_growth_mb_per_min=memory_growth_mb_per_min,
        )
        rec = RECOMMENDER.advise(state)
        if rec is None:
            return None
        advisory_id = save(conn, rec)
        if advisory_id:
            log.info("advisory %s [%s] %s", advisory_id, rec.rule_id, rec.title)
        return advisory_id
    except Exception:
        log.exception("advisory generation failed for anomaly %s",
                      anomaly.get("id"))
        return None