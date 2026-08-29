"""
db.py
-----
Database layer for the Intelligent Process Resource Monitor.

Install:
    pip install "psycopg[binary,pool]"

Set credentials via environment variables (don't hardcode them):
    PGHOST=localhost  PGPORT=5432  PGDATABASE=iprm
    PGUSER=postgres   PGPASSWORD=yourpassword

Design notes:
  * One connection pool, opened once at startup. Opening a fresh
    connection every sample would dominate your loop time.
  * insert_system_snapshot() RETURNING id gives you the FK that
    every other table in that sample hangs off.
  * Process rows go in via COPY, not INSERT. At ~250 processes per
    sample, individual INSERTs will not keep up with a 1 Hz loop.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import json
from datetime import datetime, timezone
from psycopg_pool import ConnectionPool
from psycopg.types.json import Jsonb


# ----------------------------------------------------------------
# Connection pool
# ----------------------------------------------------------------

CONNINFO = (
    f"host={os.getenv('PGHOST', 'localhost')} "
    f"port={os.getenv('PGPORT', '5432')} "
    f"dbname={os.getenv('PGDATABASE', 'iprm')} "
    f"user={os.getenv('PGUSER', 'postgres')} "
    f"password={os.getenv('PGPASSWORD', '')}"
)

pool = ConnectionPool(CONNINFO, min_size=1, max_size=5, open=False)


def init_db():
    pool.open()
    pool.wait()


def close_db():
    pool.close()


def now():
    """Always store timezone-aware UTC — the columns are TIMESTAMPTZ."""
    return datetime.now(timezone.utc)


# ----------------------------------------------------------------
# 1. monitoring_sessions
# ----------------------------------------------------------------

def insert_session(info: dict) -> int:
    """
    `info` is the dict from hardware_info.collect_session_info().
    Returns the new session id, which you hold for the whole run.
    """
    sql = """
        INSERT INTO monitoring_sessions (
            started_at, hostname, os_name, os_version,
            cpu_model, cpu_cores, cpu_threads,
            total_ram_mb, gpu_model, total_gpu_memory_mb, agent_version
        ) VALUES (
            %(started_at)s, %(hostname)s, %(os_name)s, %(os_version)s,
            %(cpu_model)s, %(cpu_cores)s, %(cpu_threads)s,
            %(total_ram_mb)s, %(gpu_model)s, %(total_gpu_memory_mb)s, %(agent_version)s
        ) RETURNING id;
    """
    params = {"started_at": now(), **info}
    # Tolerate missing keys rather than raising on a partial probe
    for key in ("hostname", "os_name", "os_version", "cpu_model", "cpu_cores",
                "cpu_threads", "total_ram_mb", "gpu_model",
                "total_gpu_memory_mb", "agent_version"):
        params.setdefault(key, None)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()[0]


def close_session(session_id: int):
    with pool.connection() as conn:
        conn.execute(
            "UPDATE monitoring_sessions SET ended_at = %s WHERE id = %s;",
            (now(), session_id),
        )


# ----------------------------------------------------------------
# 2. system_snapshots
# ----------------------------------------------------------------

SNAPSHOT_COLUMNS = [
    "session_id", "timestamp",
    "cpu_percent", "cpu_freq_mhz", "cpu_temp_c", "load_avg_1m",
    "ram_percent", "ram_used_mb", "ram_available_mb",
    "swap_percent", "swap_used_mb",
    "disk_percent", "disk_read_bytes", "disk_write_bytes",
    "disk_read_rate_bps", "disk_write_rate_bps",
    "gpu_percent", "gpu_memory_used_mb", "gpu_memory_percent",
    "gpu_temp_c", "gpu_power_w",
    "net_sent_bytes", "net_recv_bytes",
    "net_sent_rate_bps", "net_recv_rate_bps",
    "process_count", "thread_count_total", "boot_time",
    "battery_percent", "on_battery",
]


def insert_system_snapshot(data: dict) -> int:
    """
    Insert one system-wide sample. Missing keys become NULL, so the
    collector can skip probes that aren't available on this machine
    (no GPU, no battery, no temperature sensors).

    Returns snapshot_id — pass this to every other insert for the
    same sampling tick.
    """
    cols = ", ".join(SNAPSHOT_COLUMNS)
    placeholders = ", ".join(f"%({c})s" for c in SNAPSHOT_COLUMNS)
    sql = f"INSERT INTO system_snapshots ({cols}) VALUES ({placeholders}) RETURNING id;"

    params = {c: data.get(c) for c in SNAPSHOT_COLUMNS}
    params["timestamp"] = params["timestamp"] or now()

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()[0]


# ----------------------------------------------------------------
# 3. process_snapshots  (bulk via COPY)
# ----------------------------------------------------------------

PROCESS_COLUMNS = [
    "snapshot_id", "timestamp", "pid", "ppid", "name", "exe_path",
    "cmdline", "username", "status", "cpu_percent", "ram_percent",
    "ram_rss_mb", "ram_vms_mb", "thread_count", "open_files_count",
    "io_read_bytes", "io_write_bytes", "gpu_percent", "nice_value",
    "create_time", "is_dominant",
]


def insert_process_snapshots(snapshot_id: int, processes: list[dict]):
    """
    Bulk-load all process rows for one tick.

    COPY is roughly an order of magnitude faster than looped INSERTs
    here, and it matters: 250 processes x 1 sample/sec is 250 rows
    every second.
    """
    if not processes:
        return

    ts = now()
    cols = ", ".join(PROCESS_COLUMNS)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            with cur.copy(f"COPY process_snapshots ({cols}) FROM STDIN") as copy:
                for p in processes:
                    row = {**p, "snapshot_id": snapshot_id}
                    row.setdefault("timestamp", ts)
                    row.setdefault("is_dominant", False)
                    copy.write_row([row.get(c) for c in PROCESS_COLUMNS])


# ----------------------------------------------------------------
# 4. Analyzer tables
# ----------------------------------------------------------------

def insert_chrome_tabs(snapshot_id: int, tabs: list[dict]):
    if not tabs:
        return
    sql = """
        INSERT INTO chrome_tabs (
            snapshot_id, timestamp, target_id, tab_title, tab_url, domain,
            target_type, is_audible, is_active, is_media_playing,
            renderer_pid, estimated_cpu, estimated_ram_mb
        ) VALUES (
            %(snapshot_id)s, %(timestamp)s, %(target_id)s, %(tab_title)s,
            %(tab_url)s, %(domain)s, %(target_type)s, %(is_audible)s,
            %(is_active)s, %(is_media_playing)s, %(renderer_pid)s,
            %(estimated_cpu)s, %(estimated_ram_mb)s
        );
    """
    ts = now()
    rows = []
    for t in tabs:
        row = {k: None for k in (
            "target_id", "tab_title", "tab_url", "domain", "target_type",
            "is_audible", "is_active", "is_media_playing", "renderer_pid",
            "estimated_cpu", "estimated_ram_mb")}
        row.update(t)
        row["snapshot_id"] = snapshot_id
        row["timestamp"] = ts
        rows.append(row)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)      # executemany is fine at ~20-40 tabs


# ----------------------------------------------------------------
# 5. ML output
# ----------------------------------------------------------------

def insert_anomaly(snapshot_id: int, model_id: int, score: float,
                   is_anomaly: bool, severity: str,
                   triggered_metrics: dict, feature_vector: dict) -> int:
    """
    JSONB columns need the Jsonb() wrapper — passing a raw dict
    raises, and passing json.dumps() stores it as a plain string.
    """
    sql = """
        INSERT INTO anomalies (
            snapshot_id, model_id, timestamp, anomaly_score, is_anomaly,
            severity, triggered_metrics, feature_vector
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                snapshot_id, model_id, now(), score, is_anomaly, severity,
                Jsonb(triggered_metrics), Jsonb(feature_vector),
            ))
            return cur.fetchone()[0]


def insert_root_cause(anomaly_id: int, cause: dict) -> int:
    sql = """
        INSERT INTO root_causes (
            anomaly_id, timestamp, responsible_process, responsible_pid,
            process_tree_pids, analyzer_used, analyzer_had_deep_data,
            explanation, evidence, confidence, explains_percent,
            contributing_factors
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                anomaly_id, now(),
                cause.get("responsible_process"),
                cause.get("responsible_pid"),
                cause.get("process_tree_pids"),        # Python list -> INTEGER[]
                cause.get("analyzer_used"),
                cause.get("analyzer_had_deep_data"),
                cause.get("explanation"),
                Jsonb(cause.get("evidence", {})),
                cause.get("confidence"),
                cause.get("explains_percent"),
                Jsonb(cause.get("contributing_factors", {})),
            ))
            return cur.fetchone()[0]


def insert_prediction(model_id: int, target_metric: str, horizon_sec: int,
                      target_timestamp, predicted_value: float,
                      lower=None, upper=None) -> int:
    sql = """
        INSERT INTO predictions (
            model_id, predicted_at, target_metric, horizon_seconds,
            target_timestamp, predicted_value, confidence_lower, confidence_upper
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (model_id, now(), target_metric, horizon_sec,
                              target_timestamp, predicted_value, lower, upper))
            return cur.fetchone()[0]


def backfill_prediction_actuals():
    """
    Run periodically. Matches each due prediction against the nearest
    real snapshot and fills in actual_value + error columns.

    The prediction error this produces is both your accuracy metric
    and a candidate input feature for the Isolation Forest.
    """
    sql = """
        UPDATE predictions p
        SET actual_value = s.actual,
            prediction_error = s.actual - p.predicted_value,
            abs_percentage_error = CASE
                WHEN s.actual = 0 THEN NULL
                ELSE abs((s.actual - p.predicted_value) / s.actual) * 100
            END
        FROM (
            SELECT DISTINCT ON (pr.id)
                   pr.id AS pred_id,
                   CASE pr.target_metric
                       WHEN 'cpu' THEN ss.cpu_percent
                       WHEN 'gpu' THEN ss.gpu_percent
                       WHEN 'ram' THEN ss.ram_percent
                   END AS actual
            FROM predictions pr
            JOIN system_snapshots ss
              ON ss.timestamp BETWEEN pr.target_timestamp - interval '5 seconds'
                                  AND pr.target_timestamp + interval '5 seconds'
            WHERE pr.actual_value IS NULL
              AND pr.target_timestamp < now()
            ORDER BY pr.id,
                     abs(extract(epoch FROM ss.timestamp - pr.target_timestamp))
        ) s
        WHERE p.id = s.pred_id AND s.actual IS NOT NULL;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.rowcount


# ----------------------------------------------------------------
# 6. Remediation
# ----------------------------------------------------------------

def insert_remediation(action: dict) -> int:
    sql = """
        INSERT INTO remediation_actions (
            anomaly_id, root_cause_id, recommendation_id, timestamp,
            action_type, trigger_source, target_pid, target_process_name,
            action_parameters, pre_action_cpu, pre_action_ram_mb, success
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                action.get("anomaly_id"), action.get("root_cause_id"),
                action.get("recommendation_id"), now(),
                action["action_type"], action.get("trigger_source"),
                action.get("target_pid"), action.get("target_process_name"),
                Jsonb(action.get("action_parameters", {})),
                action.get("pre_action_cpu"), action.get("pre_action_ram_mb"),
                action.get("success"),
            ))
            return cur.fetchone()[0]


def update_remediation_effect(action_id: int, post_cpu: float,
                              post_ram: float, delay_sec: int):
    """
    Called a few seconds after the action. These pre/post numbers are
    what let you quantify that the intervention actually worked.
    """
    sql = """
        UPDATE remediation_actions
        SET post_action_cpu = %s,
            post_action_ram_mb = %s,
            measurement_delay_sec = %s,
            improvement_percent = CASE
                WHEN pre_action_cpu > 0
                THEN (pre_action_cpu - %s) / pre_action_cpu * 100
                ELSE NULL
            END
        WHERE id = %s;
    """
    with pool.connection() as conn:
        conn.execute(sql, (post_cpu, post_ram, delay_sec, post_cpu, action_id))


# ----------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------

if __name__ == "__main__":
    import psutil
    from hardware_info import collect_session_info

    init_db()
    try:
        session_id = insert_session(collect_session_info())
        print("session id:", session_id)

        psutil.cpu_percent()                      # prime the counter
        vm = psutil.virtual_memory()

        snap_id = insert_system_snapshot({
            "session_id": session_id,
            "timestamp": now(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "ram_percent": vm.percent,
            "ram_used_mb": vm.used / (1024 ** 2),
            "ram_available_mb": vm.available / (1024 ** 2),
            "process_count": len(psutil.pids()),
        })
        print("snapshot id:", snap_id)

        procs = []
        for p in psutil.process_iter(["pid", "ppid", "name", "cpu_percent",
                                      "memory_percent", "memory_info",
                                      "num_threads", "status"]):
            try:
                i = p.info
                procs.append({
                    "pid": i["pid"],
                    "ppid": i["ppid"],
                    "name": i["name"],
                    "cpu_percent": i["cpu_percent"],
                    "ram_percent": i["memory_percent"],
                    "ram_rss_mb": i["memory_info"].rss / (1024 ** 2)
                                  if i["memory_info"] else None,
                    "thread_count": i["num_threads"],
                    "status": i["status"],
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        insert_process_snapshots(snap_id, procs)
        print("inserted", len(procs), "process rows")

        close_session(session_id)
    finally:
        close_db()
