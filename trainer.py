"""
train_isolation_forest.py
-------------------------
Trains the anomaly detector on collected system_snapshots and
registers it in model_registry.

Install:
    python -m pip install scikit-learn pandas numpy joblib

Run:
    python train_isolation_forest.py
    python train_isolation_forest.py --contamination 0.02
    python train_isolation_forest.py --hours 24      # last 24h only

Notes on the two decisions that matter most:

  FEATURES. Raw metrics alone are not enough. A machine sitting at
  85% CPU for an hour is not anomalous; a machine that jumped from
  10% to 85% in one sample is. So alongside the raw values we
  compute deltas (change since the previous sample) and rolling
  means. Rate of change carries as much signal as absolute level.

  CONTAMINATION. This is the expected proportion of anomalies in
  the training data — effectively a sensitivity dial. 0.01 means
  "assume 1% of what I collected was abnormal". Too high and the
  model flags ordinary work; too low and it never fires. Start at
  0.01 and tune against your own data.
"""

import os
import json
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
import psycopg
from psycopg.types.json import Jsonb
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

load_dotenv()

MODEL_DIR = "models"
MODEL_TYPE = "isolation_forest"

# Raw columns pulled from system_snapshots
BASE_COLUMNS = [
    "cpu_percent",
    "ram_percent",
    "swap_percent",
    "disk_read_rate_bps",
    "disk_write_rate_bps",
    "gpu_percent",
    "gpu_memory_percent",
    "net_sent_rate_bps",
    "net_recv_rate_bps",
    "process_count",
    "thread_count_total",
]

# Deltas and rolling means are computed for these
DYNAMIC_COLUMNS = ["cpu_percent", "ram_percent", "gpu_percent", "process_count"]


def connect():
    return psycopg.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
    )


# ----------------------------------------------------------------
# Load
# ----------------------------------------------------------------

def load_snapshots(conn, hours=None) -> pd.DataFrame:
    cols = ", ".join(BASE_COLUMNS)
    sql = f"SELECT id, timestamp, {cols} FROM system_snapshots"
    if hours:
        sql += f" WHERE timestamp > now() - interval '{int(hours)} hours'"
    sql += " ORDER BY timestamp"

    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
        names = [d.name for d in cur.description]

    df = pd.DataFrame(rows, columns=names)
    return df


# ----------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns the feature matrix and the ordered feature name list.
    The name list is stored in model_registry so inference builds
    the vector in exactly the same order — a mismatch here produces
    silently wrong predictions rather than an error.
    """
    f = pd.DataFrame(index=df.index)

    # Raw levels. Columns that are entirely NULL (no GPU, no swap)
    # are filled with 0 rather than dropped, so the feature vector
    # keeps a stable shape across machines.
    for col in BASE_COLUMNS:
        f[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Rate of change since the previous sample.
    for col in DYNAMIC_COLUMNS:
        f[f"{col}_delta"] = f[col].diff().fillna(0.0)

    # Short rolling mean — separates a spike from a sustained level.
    for col in DYNAMIC_COLUMNS:
        f[f"{col}_roll5"] = f[col].rolling(window=5, min_periods=1).mean()

    # Distance from the recent baseline. Large positive values mean
    # "currently well above what this machine has been doing".
    for col in DYNAMIC_COLUMNS:
        f[f"{col}_dev"] = f[col] - f[f"{col}_roll5"]

    # Combined pressure — captures states where nothing individually
    # looks extreme but everything is elevated at once.
    f["resource_pressure"] = (
        f["cpu_percent"] * 0.4 + f["ram_percent"] * 0.4 + f["gpu_percent"] * 0.2
    )

    f = f.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return f, list(f.columns)


# ----------------------------------------------------------------
# Train
# ----------------------------------------------------------------

def train(features: pd.DataFrame, contamination: float, seed: int = 42):
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X)
    return model, scaler, X


# ----------------------------------------------------------------
# Register
# ----------------------------------------------------------------

def register(conn, version, feature_names, scaler, params,
             n_rows, t_start, t_end, anomaly_rate, artifact_path):
    """
    Deactivates any previous active model, then inserts this one as
    active. The partial unique index on model_registry enforces one
    active model per type, so this must happen in one transaction.
    """
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE model_registry SET is_active = FALSE "
            "WHERE model_type = %s AND is_active;",
            (MODEL_TYPE,),
        )
        cur.execute(
            """
            INSERT INTO model_registry (
                model_type, model_version, trained_at, training_rows,
                training_start, training_end, feature_list,
                hyperparameters, scaler_params, anomaly_rate,
                artifact_path, is_active
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE)
            RETURNING id;
            """,
            (
                MODEL_TYPE, version, datetime.now(timezone.utc), n_rows,
                t_start, t_end, Jsonb(feature_names),
                Jsonb(params), Jsonb(scaler_params), anomaly_rate,
                artifact_path,
            ),
        )
        model_id = cur.fetchone()[0]
    conn.commit()
    return model_id


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contamination", type=float, default=0.01,
                    help="expected anomaly proportion (sensitivity)")
    ap.add_argument("--hours", type=float, default=None,
                    help="train on the last N hours only")
    ap.add_argument("--min-rows", type=int, default=500,
                    help="refuse to train below this many samples")
    args = ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    conn = connect()
    print(f"connected to {conn.info.dbname}")

    df = load_snapshots(conn, args.hours)
    print(f"loaded {len(df)} snapshots")

    if len(df) < args.min_rows:
        print(f"\nNot enough data. Have {len(df)}, want at least {args.min_rows}.")
        print("Run the collector for longer — and use the machine normally")
        print("while it collects, so the model learns a realistic baseline.")
        conn.close()
        return

    features, feature_names = build_features(df)
    print(f"built {len(feature_names)} features")

    model, scaler, X = train(features, args.contamination)

    # How the model scores its own training data — a sanity check,
    # not a real evaluation. Unsupervised means no ground truth.
    preds = model.predict(X)                    # -1 anomaly, 1 normal
    scores = model.score_samples(X)
    rate = float((preds == -1).mean())

    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    artifact = os.path.join(MODEL_DIR, f"iforest_{version}.joblib")
    joblib.dump({"model": model, "scaler": scaler,
                 "features": feature_names}, artifact)

    params = {
        "n_estimators": 200,
        "contamination": args.contamination,
        "max_samples": "auto",
        "random_state": 42,
    }

    model_id = register(
        conn, version, feature_names, scaler, params,
        len(df), df["timestamp"].min(), df["timestamp"].max(),
        rate, artifact,
    )
    conn.close()

    print(f"\nmodel {model_id} ({version}) trained and set active")
    print(f"artifact: {artifact}")
    print(f"flagged {rate*100:.2f}% of training data as anomalous")
    print(f"score range: {scores.min():.3f} to {scores.max():.3f}")

    # Show what it considers the most abnormal moments — the fastest
    # way to tell whether the model learned something sensible.
    worst = np.argsort(scores)[:5]
    print("\nmost anomalous samples in training data:")
    for i in worst:
        row = df.iloc[i]
        print(f"  {row['timestamp']:%Y-%m-%d %H:%M:%S}  "
              f"cpu {row['cpu_percent'] or 0:5.1f}%  "
              f"ram {row['ram_percent'] or 0:5.1f}%  "
              f"score {scores[i]:.3f}")


if __name__ == "__main__":
    main()
