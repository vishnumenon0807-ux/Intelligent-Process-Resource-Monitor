"""
score_anomalies.py
------------------
Loads the active Isolation Forest from model_registry and scores
snapshots, writing detections to the anomalies table.

Run:
    python score_anomalies.py --backfill      # score all past snapshots
    python score_anomalies.py                 # watch for new ones, live
    python score_anomalies.py --interval 5

Run this alongside collector.py. The collector writes snapshots;
this reads them, scores them, and records anomalies.

Why it rebuilds features from a window rather than a single row:
delta and rolling-mean features need the preceding samples. Scoring
one row in isolation would produce zeros for every dynamic feature
and the model would see a vector unlike anything it trained on.
"""

import os
import time
import signal
import argparse

import numpy as np
import pandas as pd
import joblib
import psycopg
from psycopg.types.json import Jsonb
from dotenv import load_dotenv

from trainer import (
    BASE_COLUMNS, build_features, connect, MODEL_TYPE,
)
load_dotenv()

# Enough history to fill the rolling windows
WINDOW = 20


def load_active_model(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, model_version, artifact_path, feature_list
            FROM model_registry
            WHERE model_type = %s AND is_active
            LIMIT 1;
            """,
            (MODEL_TYPE,),
        )
        row = cur.fetchone()

    if row is None:
        raise SystemExit(
            "No active isolation_forest model. Run train_isolation_forest.py first."
        )

    model_id, version, path, feature_list = row
    if not os.path.exists(path):
        raise SystemExit(f"Model artifact missing: {path}")

    bundle = joblib.load(path)
    print(f"loaded model {model_id} ({version}) — {len(bundle['features'])} features")
    return model_id, bundle


def fetch_window(conn, up_to_id=None, limit=WINDOW):
    """
    Returns the `limit` snapshots ending at up_to_id (inclusive),
    oldest first — the tail is the row we actually score.
    """
    cols = ", ".join(BASE_COLUMNS)
    where = "WHERE id <= %s" if up_to_id else ""
    params = (up_to_id, limit) if up_to_id else (limit,)

    sql = f"""
        SELECT * FROM (
            SELECT id, timestamp, {cols}
            FROM system_snapshots
            {where}
            ORDER BY id DESC
            LIMIT %s
        ) t ORDER BY id;
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        names = [d.name for d in cur.description]
    return pd.DataFrame(rows, columns=names)


def severity_of(score: float) -> str:
    """
    score_samples returns negative values; more negative is more
    abnormal. These cutoffs are a starting point — adjust once you
    see the score distribution your own machine produces.
    """
    if score < -0.65:
        return "critical"
    if score < -0.58:
        return "high"
    if score < -0.52:
        return "medium"
    return "low"


def triggered_metrics(row) -> dict:
    """The raw values worth showing a human, for the anomalies row."""
    out = {}
    for col in ("cpu_percent", "ram_percent", "gpu_percent",
                "disk_read_rate_bps", "disk_write_rate_bps", "process_count"):
        val = row.get(col)
        if val is not None and not pd.isna(val):
            out[col] = round(float(val), 2)
    return out


def score_snapshot(conn, model_id, bundle, snapshot_id) -> dict | None:
    df = fetch_window(conn, up_to_id=snapshot_id)
    if df.empty:
        return None

    features, names = build_features(df)

    # Guard against a model trained on a different feature set
    if names != bundle["features"]:
        raise SystemExit(
            "Feature mismatch between model and current code.\n"
            "Retrain: python train_isolation_forest.py"
        )

    X = bundle["scaler"].transform(features.values)
    x_last = X[-1].reshape(1, -1)

    score = float(bundle["model"].score_samples(x_last)[0])
    is_anom = bool(bundle["model"].predict(x_last)[0] == -1)

    row = df.iloc[-1]
    return {
        "snapshot_id": int(row["id"]),
        "timestamp": row["timestamp"],
        "score": score,
        "is_anomaly": is_anom,
        "severity": severity_of(score) if is_anom else "low",
        "triggered": triggered_metrics(row),
        "vector": {n: round(float(v), 4)
                   for n, v in zip(names, features.iloc[-1].values)},
    }


def insert_anomaly(conn, model_id, result):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO anomalies (
                snapshot_id, model_id, timestamp, anomaly_score,
                is_anomaly, severity, triggered_metrics, feature_vector
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id;
            """,
            (
                result["snapshot_id"], model_id, result["timestamp"],
                result["score"], result["is_anomaly"], result["severity"],
                Jsonb(result["triggered"]), Jsonb(result["vector"]),
            ),
        )
        anomaly_id = cur.fetchone()[0]
    conn.commit()
    return anomaly_id


def already_scored(conn, snapshot_id) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM anomalies WHERE snapshot_id = %s LIMIT 1;",
                    (snapshot_id,))
        return cur.fetchone() is not None


# ----------------------------------------------------------------
# Modes
# ----------------------------------------------------------------

def backfill(conn, model_id, bundle, only_anomalies=True):
    """Score every snapshot already in the database."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT s.id FROM system_snapshots s
            LEFT JOIN anomalies a ON a.snapshot_id = s.id
            WHERE a.id IS NULL
            ORDER BY s.id;
        """)
        ids = [r[0] for r in cur.fetchall()]

    print(f"backfilling {len(ids)} unscored snapshots...")
    found = 0
    for n, sid in enumerate(ids, 1):
        result = score_snapshot(conn, model_id, bundle, sid)
        if result is None:
            continue
        # Storing only detections keeps the table small. Pass
        # only_anomalies=False to record every score instead.
        if result["is_anomaly"] or not only_anomalies:
            insert_anomaly(conn, model_id, result)
            if result["is_anomaly"]:
                found += 1
                print(f"  [{result['timestamp']:%H:%M:%S}] "
                      f"{result['severity']:<8} score {result['score']:.3f}  "
                      f"cpu {result['triggered'].get('cpu_percent', 0):5.1f}%  "
                      f"ram {result['triggered'].get('ram_percent', 0):5.1f}%")
        if n % 500 == 0:
            print(f"  ...{n}/{len(ids)}")

    print(f"\ndone — {found} anomalies in {len(ids)} snapshots "
          f"({found/max(len(ids),1)*100:.2f}%)")


running = True


def _stop(signum, frame):
    global running
    running = False
    print("\nstopping...")


def watch(conn, model_id, bundle, interval):
    """Poll for new snapshots and score them as they arrive."""
    signal.signal(signal.SIGINT, _stop)

    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(max(id), 0) FROM system_snapshots;")
        last_id = cur.fetchone()[0]

    print(f"watching from snapshot {last_id} — Ctrl+C to stop\n")

    while running:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM system_snapshots WHERE id > %s ORDER BY id;",
                (last_id,),
            )
            new_ids = [r[0] for r in cur.fetchall()]

        for sid in new_ids:
            if already_scored(conn, sid):
                last_id = sid
                continue
            result = score_snapshot(conn, model_id, bundle, sid)
            last_id = sid
            if result and result["is_anomaly"]:
                aid = insert_anomaly(conn, model_id, result)
                print(f"[{result['timestamp']:%H:%M:%S}] ANOMALY {aid}  "
                      f"{result['severity']:<8} score {result['score']:.3f}  "
                      f"cpu {result['triggered'].get('cpu_percent', 0):5.1f}%  "
                      f"ram {result['triggered'].get('ram_percent', 0):5.1f}%")

        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true",
                    help="score existing snapshots instead of watching")
    ap.add_argument("--interval", type=float, default=3.0,
                    help="seconds between polls in watch mode")
    ap.add_argument("--store-all", action="store_true",
                    help="store every score, not just detections")
    args = ap.parse_args()

    conn = connect()
    print(f"connected to {conn.info.dbname}")
    model_id, bundle = load_active_model(conn)

    try:
        if args.backfill:
            backfill(conn, model_id, bundle, only_anomalies=not args.store_all)
        else:
            watch(conn, model_id, bundle, args.interval)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
