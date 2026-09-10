"""
IPRM :: standalone advisory scanner.

Samples live process state with psutil, runs it through the recommender, and
writes advisories to Postgres. Independent of collector.py and the RCA engine
-- the point is to prove the chain works and to tune rule thresholds against
your real machine before wiring anything together.

Run as a fourth terminal:
    python -m recommender.scan

Stop with Ctrl+C.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict

import psutil
import psycopg

from .recommender import RECOMMENDER, SystemState
from .store import save, sweep_rechecks

log = logging.getLogger("iprm.scan")

DSN = os.environ.get("IPRM_DSN", "postgresql://postgres@localhost:5432/iprm")
INTERVAL = 30                # seconds between passes
CPU_COUNT = psutil.cpu_count(logical=True) or 1
CPU_FLOOR = 10.0             # ignore processes quieter than this
MEM_FLOOR_MB = 500.0         # ...unless they are holding this much

# Windows pseudo-processes. "System Idle Process" accounts for UNUSED cycles,
# so a high reading there means the machine is quiet -- the opposite of a
# problem. PID 0 is Idle, PID 4 is the System process; neither is advisable.
IGNORE_PIDS = {0, 4}
IGNORE_NAMES = {
    "system idle process", "system", "registry",
    "memory compression", "secure system", "idle",
}

# (pid, create_time) -> (last_memory_mb, last_seen_monotonic)
_mem_history: dict[tuple[int, float], tuple[float, float]] = {}


def _growth_mb_per_min(key: tuple[int, float], memory_mb: float) -> float:
    """Needs two passes. Returns 0.0 on the first sighting of a process."""
    now = time.monotonic()
    prev = _mem_history.get(key)
    _mem_history[key] = (memory_mb, now)
    if prev is None:
        return 0.0
    prev_mb, prev_t = prev
    minutes = (now - prev_t) / 60.0
    return (memory_mb - prev_mb) / minutes if minutes > 0.01 else 0.0


def _collect() -> list[SystemState]:
    """One pass over all processes. Returns the ones worth advising on."""
    attrs = ["pid", "name", "ppid", "create_time", "memory_info", "num_threads"]
    procs: list[psutil.Process] = []
    children: dict[int, int] = defaultdict(int)

    for p in psutil.process_iter(attrs):
        try:
            p.cpu_percent(None)          # prime; real value arrives next pass
            children[p.info["ppid"]] += 1
            procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    host_cpu = psutil.cpu_percent(interval=1.0)
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    process_count = len(procs)

    out: list[SystemState] = []
    for p in procs:
        try:
            info = p.info
            cpu = p.cpu_percent(None) / CPU_COUNT
            mem_mb = info["memory_info"].rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError, TypeError):
            continue

        if info["pid"] in IGNORE_PIDS or (info["name"] or "").lower() in IGNORE_NAMES:
            continue

        key = (info["pid"], info["create_time"])
        growth = _growth_mb_per_min(key, mem_mb)

        if cpu < CPU_FLOOR and mem_mb < MEM_FLOOR_MB:
            continue

        out.append(SystemState(
            pid=info["pid"],
            name=info["name"] or "",
            create_time=info["create_time"],
            cpu_percent=cpu,
            memory_mb=mem_mb,
            memory_growth_mb_per_min=growth,
            child_count=children.get(info["pid"], 0),
            thread_count=info["num_threads"] or 0,
            uptime_minutes=(time.time() - info["create_time"]) / 60.0,
            host_cpu_percent=host_cpu,
            host_mem_percent=vm.percent,
            host_swap_percent=sw.percent,
            process_count=process_count,
        ))

    # Host-wide findings need a stateless row with no process attached.
    out.append(SystemState(
        host_cpu_percent=host_cpu,
        host_mem_percent=vm.percent,
        host_swap_percent=sw.percent,
        process_count=process_count,
    ))
    return out


def _sample_fn(pid: int | None, create_time: float | None) -> dict | None:
    """Used by sweep_rechecks to see whether the condition persisted."""
    if pid is None:
        return {"cpu_percent": psutil.cpu_percent(interval=0.5)}
    try:
        p = psutil.Process(pid)
        if create_time and abs(p.create_time() - create_time) > 1.0:
            return None                  # PID was reused; different process
        p.cpu_percent(None)
        time.sleep(0.5)
        return {"cpu_percent": p.cpu_percent(None) / CPU_COUNT}
    except psutil.Error:
        return None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s :: %(message)s",
        datefmt="%H:%M:%S",
    )
    conn = psycopg.connect(DSN, autocommit=True)
    log.info("scanner up -- first pass has no CPU data, that is expected")

    try:
        while True:
            written = 0
            for state in _collect():
                rec = RECOMMENDER.advise(state)
                if rec is None:
                    continue
                if save(conn, rec) is None:
                    continue             # deduped
                written += 1
                log.info("[%s] %.0f%% :: %s", rec.rule_id,
                         rec.confidence * 100, rec.title)
                for s in rec.suggestions[:2]:
                    log.info("        -> %s", s)

            rechecked = sweep_rechecks(conn, _sample_fn)
            if written or rechecked:
                log.info("pass complete: %d new, %d rechecked", written, rechecked)
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        log.info("stopped")
    finally:
        conn.close()


if __name__ == "__main__":
    main()