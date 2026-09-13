"""
root_cause_engine.py
--------------------
Takes anomalies and answers "which process caused this, and what
was it doing".

Run:
    python root_cause_engine.py --backfill     # explain all unexplained anomalies
    python root_cause_engine.py                # watch for new anomalies
    python root_cause_engine.py --anomaly 42   # explain one, verbosely

Pipeline for a single anomaly:

    anomaly_id
        ↓
    load the snapshot's process rows
        ↓
    aggregate into process TREES via ppid
        (Chrome's 28 processes become one entry)
        ↓
    pick the dominant tree by CPU
        ↓
    dispatch to the analyzer registered for that name
        (chrome -> chrome_analyzer, everything else -> generic)
        ↓
    write root_causes + recommendations

The engine itself knows nothing about Chrome, VS Code, or any other
application. It only knows how to find the culprit and who to ask.
Adding an analyzer means registering a name, not editing this file.
"""

import time
import signal
import argparse
from collections import defaultdict
from recommender.bridge import advise_from_anomaly
import psycopg
from psycopg.types.json import Jsonb

from trainer import connect          # reuses the same .env connection
import generic_analyzer

try:
    import chrome_analyzer
    HAVE_CHROME = True
except ImportError:
    HAVE_CHROME = False


# Process names that route to a dedicated analyzer.
# Everything else falls through to generic_analyzer.
ANALYZER_REGISTRY = {
    "chrome.exe": "chrome",
    "chrome": "chrome",
    "msedge.exe": "chrome",          # Chromium family, same protocol
    "brave.exe": "chrome",
}

# A tree must account for at least this share of system CPU before
# the engine will name it as the cause. Below this, no single
# process explains the anomaly.
DOMINANCE_THRESHOLD = 25.0
# Pseudo-processes that are never a root cause.
# "System Idle Process" measures UNUSED cpu — high values mean the
# machine is free, not busy.
EXCLUDED_PROCESSES = {
    "system idle process",
    "idle",
    "system",
    "registry",
    "memory compression",
    "secure system",
}
# ----------------------------------------------------------------
# Load
# ----------------------------------------------------------------

def load_anomaly(conn, anomaly_id):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.id, a.snapshot_id, a.timestamp, a.anomaly_score,
                   a.severity, a.triggered_metrics,
                   s.cpu_percent, s.ram_percent, s.gpu_percent, s.ram_used_mb
            FROM anomalies a
            JOIN system_snapshots s ON s.id = a.snapshot_id
            WHERE a.id = %s;
            """,
            (anomaly_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    keys = ["id", "snapshot_id", "timestamp", "score", "severity",
            "triggered", "sys_cpu", "sys_ram", "sys_gpu", "ram_used_mb"]
    return dict(zip(keys, row))


def load_processes(conn, snapshot_id):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT pid, ppid, name, exe_path, cpu_percent, ram_rss_mb,
                   thread_count, create_time, status
            FROM process_snapshots
            WHERE snapshot_id = %s;
            """,
            (snapshot_id,),
        )
        rows = cur.fetchall()
        names = [d.name for d in cur.description]
    return [dict(zip(names, r)) for r in rows]


def memory_trend(conn, process_name, snapshot_id, lookback=15):
    """
    How much memory this process name has gained over recent
    snapshots. Distinguishes a steady heavy process from one that
    is climbing.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.snapshot_id, SUM(p.ram_rss_mb) AS total
            FROM process_snapshots p
            WHERE p.name = %s
              AND p.snapshot_id <= %s
              AND p.snapshot_id > %s - %s
            GROUP BY p.snapshot_id
            ORDER BY p.snapshot_id;
            """,
            (process_name, snapshot_id, snapshot_id, lookback),
        )
        rows = cur.fetchall()

    if len(rows) < 3:
        return None
    first = float(rows[0][1] or 0)
    last = float(rows[-1][1] or 0)
    return {"growth_mb": last - first, "samples": len(rows)}


# ----------------------------------------------------------------
# Tree aggregation
# ----------------------------------------------------------------

def build_trees(processes: list[dict]) -> list[dict]:
    """
    Groups processes by executable name and rolls their usage
    together.

    Grouping by name rather than walking the ppid graph is
    deliberate: Chrome's children are often reparented, and on
    Windows a parent can exit while children continue, orphaning
    the subtree. Name grouping is more robust and gives the same
    answer for the case that matters — many processes, one
    application.

    ppid is still used to find the tree root, which is the pid the
    remediation layer would act on first.
    """
    groups = defaultdict(list)
    for p in processes:
        if not p.get("name"):
            continue
        if p["name"].lower() in EXCLUDED_PROCESSES:
            continue
        groups[p["name"]].append(p)

    trees = []
    for name, members in groups.items():
        pids = {m["pid"] for m in members}
        # Root = a member whose parent is outside this group
        roots = [m for m in members if m.get("ppid") not in pids]
        root = roots[0] if roots else members[0]

        trees.append({
            "name": name,
            "root_pid": root.get("pid"),
            "pids": sorted(m["pid"] for m in members if m.get("pid")),
            "process_count": len(members),
            "cpu_percent": sum(float(m.get("cpu_percent") or 0) for m in members),
            "ram_mb": sum(float(m.get("ram_rss_mb") or 0) for m in members),
            "thread_count": sum(int(m.get("thread_count") or 0) for m in members),
            "create_time": root.get("create_time"),
            "exe_path": root.get("exe_path"),
        })

    trees.sort(key=lambda t: t["cpu_percent"], reverse=True)
    return trees


# ----------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------

def run_analyzer(conn, tree, anomaly, snapshot_id):
    """Route the dominant tree to its analyzer and normalise the result."""
    key = ANALYZER_REGISTRY.get((tree["name"] or "").lower())

    if key == "chrome" and HAVE_CHROME:
        # The Chrome analyzer inspects the LIVE browser. For a
        # backfilled anomaly from hours ago that live state is not
        # the state that caused it, so we only trust it when
        # scoring something recent.
        analysis = chrome_analyzer.analyze(anomaly["sys_cpu"], anomaly["sys_ram"])
        if analysis:
            result = chrome_analyzer.explain(analysis)
            result.setdefault("is_expected", False)
            # Prefer the stored snapshot's numbers over live ones
            result["process_tree_pids"] = tree["pids"]
            return result
        # Chrome not running now — fall through to generic

    trend = memory_trend(conn, tree["name"], snapshot_id)
    result = generic_analyzer.analyze(tree, anomaly["sys_cpu"], trend)
    result["process_tree_pids"] = tree["pids"]
    return result


# ----------------------------------------------------------------
# Write
# ----------------------------------------------------------------

def insert_root_cause(conn, anomaly, tree, result, contributing):
    advise_from_anomaly(conn, anomaly, dominant, result)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO root_causes (
                anomaly_id, timestamp, responsible_process, responsible_pid,
                process_tree_pids, analyzer_used, analyzer_had_deep_data,
                explanation, evidence, confidence, explains_percent,
                contributing_factors
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id;
            """,
            (
                anomaly["id"], anomaly["timestamp"],
                result.get("responsible_process") or tree["name"],
                tree["root_pid"],
                result.get("process_tree_pids") or tree["pids"],
                result.get("analyzer_used"),
                result.get("analyzer_had_deep_data", False),
                result.get("explanation"),
                Jsonb(result.get("evidence", {})),
                result.get("confidence"),
                result.get("explains_percent"),
                Jsonb(contributing),
            ),
        )
        return cur.fetchone()[0]


def insert_recommendations(conn, root_cause_id, timestamp, recs):
    if not recs:
        return
    with conn.cursor() as cur:
        for i, text in enumerate(recs, 1):
            cur.execute(
                """
                INSERT INTO recommendations (
                    root_cause_id, timestamp, recommendation_text,
                    priority, is_automatable
                ) VALUES (%s,%s,%s,%s,%s);
                """,
                (root_cause_id, timestamp, text, i, False),
            )


def mark_suppressed(conn, anomaly_id, reason):
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE anomalies SET was_suppressed = TRUE, "
            "suppression_reason = %s WHERE id = %s;",
            (reason, anomaly_id),
        )


# ----------------------------------------------------------------
# Explain one anomaly
# ----------------------------------------------------------------

def explain_anomaly(conn, anomaly_id, verbose=False):
    anomaly = load_anomaly(conn, anomaly_id)
    if not anomaly:
        print(f"anomaly {anomaly_id} not found")
        return None

    processes = load_processes(conn, anomaly["snapshot_id"])
    if not processes:
        if verbose:
            print(f"anomaly {anomaly_id}: no process rows for this snapshot")
        return None

    trees = build_trees(processes)
    if not trees:
        if verbose:
            print(f"anomaly {anomaly_id}: no attributable processes")
        return None

    sys_cpu = float(anomaly["sys_cpu"] or 0)
    sys_ram = float(anomaly["sys_ram"] or 0)

    # Blame by whichever resource is actually under pressure.
    # A RAM-driven anomaly should not be attributed by CPU share.
    if sys_ram > 70 and sys_cpu < 50:
        trees.sort(key=lambda t: t["ram_mb"], reverse=True)
        dominant = trees[0]
        # Measure against RAM actually in use, not the sum of process
        # RSS — summing RSS double-counts shared memory and inflates
        # the denominator so nothing ever looks dominant.
        used_mb = float(anomaly.get("ram_used_mb") or 0)
        if used_mb <= 0:
            used_mb = sum(t["ram_mb"] for t in trees) or 1
        share = dominant["ram_mb"] / used_mb * 100
        basis = "memory"
    else:
        dominant = trees[0]
        share = (dominant["cpu_percent"] / sys_cpu * 100) if sys_cpu > 0 else 0.0
        basis = "cpu"

    if verbose:
        print(f"\nanomaly {anomaly_id} @ {anomaly['timestamp']:%Y-%m-%d %H:%M:%S}")
        print(f"  system: cpu {sys_cpu:.1f}%  ram {sys_ram:.1f}%  "
              f"score {anomaly['score']:.3f}  (blaming by {basis})")
        print("  top process trees:")
        for t in trees[:5]:
            print(f"    {t['name'][:32]:<32} cpu {t['cpu_percent']:6.2f}%  "
                  f"ram {t['ram_mb']:8.1f} MB  x{t['process_count']}")

    # Nothing dominant enough to blame
    if share < DOMINANCE_THRESHOLD:
        if verbose:
            print(f"  no dominant process ({share:.1f}% < {DOMINANCE_THRESHOLD}%)")
        unit = "memory" if basis == "memory" else "CPU"
        result = {
            "responsible_process": None,
            "analyzer_used": "none",
            "analyzer_had_deep_data": False,
            "explanation": (
                f"No single process accounts for this anomaly. The heaviest, "
                f"{dominant['name']}, explains only {share:.0f}% of {unit} usage. "
                f"The cause is distributed across many processes."
            ),
            "evidence": {"top_process": dominant["name"],
                         "top_share_percent": round(share, 1),
                         "attribution_basis": basis,
                         "data_source": "process_inspection"},
            "confidence": 0.30,
            "explains_percent": round(share, 1),
            "recommendations": ["Review the process list — load is spread "
                                "across several applications."],
            "is_expected": False,
        }
    else:
        result = run_analyzer(conn, dominant, anomaly, anomaly["snapshot_id"])
        if result.get("explains_percent") is None:
            result["explains_percent"] = round(share, 1)
        result.setdefault("evidence", {})["attribution_basis"] = basis

    # Rank contributors by the same resource we blamed on
    key = "ram_mb" if basis == "memory" else "cpu_percent"
    contributing = {
        t["name"]: round(t[key], 2)
        for t in trees[1:4] if t[key] > (50 if basis == "memory" else 1.0)
    }

    root_cause_id = insert_root_cause(conn, anomaly, dominant, result, contributing)
    insert_recommendations(conn, root_cause_id, anomaly["timestamp"],
                           result.get("recommendations", []))

    if result.get("is_expected"):
        mark_suppressed(conn, anomaly["id"],
                        result.get("evidence", {}).get("task_category", "expected load"))

    conn.commit()

    if verbose:
        print(f"\n  ROOT CAUSE  ({result['analyzer_used']}, "
              f"confidence {result.get('confidence')})")
        print(f"  {result['explanation']}")
        if result.get("recommendations"):
            print("  recommendations:")
            for r in result["recommendations"]:
                print(f"    - {r}")

    return root_cause_id, result


# ----------------------------------------------------------------
# Modes
# ----------------------------------------------------------------

def backfill(conn, verbose=False):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT a.id FROM anomalies a
            LEFT JOIN root_causes rc ON rc.anomaly_id = a.id
            WHERE a.is_anomaly AND rc.id IS NULL
            ORDER BY a.id;
        """)
        ids = [r[0] for r in cur.fetchall()]

    if not ids:
        print("no unexplained anomalies")
        return

    print(f"explaining {len(ids)} anomalies...\n")
    for aid in ids:
        out = explain_anomaly(conn, aid, verbose=verbose)
        if out and not verbose:
            _, result = out
            proc = result.get("responsible_process") or "distributed"
            print(f"  anomaly {aid:<5} -> {proc[:28]:<28} "
                  f"{result['analyzer_used']:<16} "
                  f"conf {result.get('confidence', 0):.2f}")
    print(f"\ndone — {len(ids)} explained")


running = True


def _stop(signum, frame):
    global running
    running = False
    print("\nstopping...")


def watch(conn, interval):
    signal.signal(signal.SIGINT, _stop)
    print("watching for new anomalies — Ctrl+C to stop\n")

    while running:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT a.id FROM anomalies a
                LEFT JOIN root_causes rc ON rc.anomaly_id = a.id
                WHERE a.is_anomaly AND rc.id IS NULL
                ORDER BY a.id;
            """)
            ids = [r[0] for r in cur.fetchall()]

        for aid in ids:
            explain_anomaly(conn, aid, verbose=True)

        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--anomaly", type=int, help="explain a single anomaly id")
    ap.add_argument("--interval", type=float, default=4.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    conn = connect()
    print(f"connected to {conn.info.dbname}")
    print(f"chrome analyzer: {'available' if HAVE_CHROME else 'not found'}")

    try:
        if args.anomaly:
            explain_anomaly(conn, args.anomaly, verbose=True)
        elif args.backfill:
            backfill(conn, verbose=args.verbose)
        else:
            watch(conn, args.interval)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
