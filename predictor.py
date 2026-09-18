"""
predictor.py
------------
Loads the active LSTM from model_registry, runs it against recent
snapshots on a timer, and writes forecasts to the predictions table.

Run:
    python predictor.py                  # loop, predict every 30s
    python predictor.py --once           # single prediction, for testing
    python predictor.py --interval 60

The model artifact carries its own window, horizon and scaler, so this
script never hardcodes them -- whatever lstm_trainer.py saved is what
gets applied here. A mismatch between training and inference scaling
produces confident nonsense rather than an error, so the values are
always read from the checkpoint, never reconstructed.
"""

import time
import argparse
from datetime import timedelta

import numpy as np
import torch

from trainer import connect
from lstm_trainer import Forecaster, INPUT_COLUMNS, TARGET_COLUMNS, MODEL_TYPE


# ================================================================
# Load the active model
# ================================================================

def load_active_model(conn):
    """Read the active artifact path from the registry and rebuild it."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, model_version, artifact_path
            FROM model_registry
            WHERE model_type = %s AND is_active
            ORDER BY trained_at DESC
            LIMIT 1;
            """,
            (MODEL_TYPE,),
        )
        row = cur.fetchone()

    if row is None:
        raise RuntimeError(
            f"No active {MODEL_TYPE} in model_registry. "
            f"Run lstm_trainer.py first."
        )

    model_id, version, artifact_path = row

    # Our own file, written by lstm_trainer.py -- not untrusted input.
    ckpt = torch.load(artifact_path, map_location="cpu", weights_only=False)

    arch = ckpt["arch"]
    model = Forecaster(
        n_features=len(ckpt["input_columns"]),
        n_targets=len(ckpt["target_columns"]),
        hidden=arch["hidden"],
        layers=arch["layers"],
        dropout=arch["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    meta = {
        "model_id": model_id,
        "version": version,
        "input_columns": ckpt["input_columns"],
        "target_columns": ckpt["target_columns"],
        "window": ckpt["window"],
        "horizon": ckpt["horizon"],
        "scaler_min": np.array(ckpt["scaler_min"], dtype=np.float32),
        "scaler_range": np.array(ckpt["scaler_range"], dtype=np.float32),
    }
    return model, meta


# ================================================================
# Data
# ================================================================

def fetch_window(conn, columns, window):
    """
    The most recent `window` snapshots, oldest first.

    The query pulls newest-first with a LIMIT so it can use the
    timestamp index, then reverses -- ordering ascending and taking
    the tail would scan the whole table.
    """
    cols = ", ".join(columns)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT timestamp, {cols}
            FROM system_snapshots
            ORDER BY timestamp DESC
            LIMIT %s;
            """,
            (window,),
        )
        rows = cur.fetchall()

    if len(rows) < window:
        return None, None

    rows = rows[::-1]
    timestamps = [r[0] for r in rows]
    values = np.array(
        [[float(v) if v is not None else 0.0 for v in r[1:]] for r in rows],
        dtype=np.float32,
    )
    return timestamps, values


def sample_interval_seconds(timestamps, default=2.0):
    """
    Median gap between snapshots, used to convert a horizon measured
    in samples into wall-clock seconds. Median rather than mean so a
    single collector stall doesn't skew it.
    """
    if len(timestamps) < 2:
        return default
    gaps = [
        (timestamps[i + 1] - timestamps[i]).total_seconds()
        for i in range(len(timestamps) - 1)
    ]
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return default
    return float(np.median(gaps))


# ================================================================
# Predict
# ================================================================

def predict_once(conn, model, meta, verbose=True):
    timestamps, values = fetch_window(
        conn, meta["input_columns"], meta["window"])

    if values is None:
        if verbose:
            print(f"Not enough snapshots yet "
                  f"(need {meta['window']}). Waiting.")
        return None

    # Identical transform to training: (x - min) / range.
    scaled = (values - meta["scaler_min"]) / meta["scaler_range"]

    with torch.no_grad():
        x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
        out = model(x).squeeze(0).numpy()

    # Back to percentage points using the target columns' own scaler
    # entries, which sit at their positions within INPUT_COLUMNS.
    target_idx = [meta["input_columns"].index(c)
                  for c in meta["target_columns"]]
    tmin = meta["scaler_min"][target_idx]
    trng = meta["scaler_range"][target_idx]
    predicted = out * trng + tmin

    # Clamp to the physically meaningful range. The model works in
    # scaled space and can extrapolate slightly outside 0-100.
    predicted = np.clip(predicted, 0.0, 100.0)

    interval = sample_interval_seconds(timestamps)
    horizon_s = meta["horizon"] * interval
    last_ts = timestamps[-1]
    target_ts = last_ts + timedelta(seconds=horizon_s)

    rows = [
        (meta["model_id"], target_ts, int(round(horizon_s)),
         name, float(value))
        for name, value in zip(meta["target_columns"], predicted)
    ]

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO predictions
                (model_id, target_ts, horizon_s, metric, predicted)
            VALUES (%s, %s, %s, %s, %s);
            """,
            rows,
        )
    conn.commit()

    if verbose:
        parts = " ".join(
            f"{n.replace('_percent', '')}={v:.1f}%"
            for n, v in zip(meta["target_columns"], predicted)
        )
        print(f"[{last_ts:%H:%M:%S}] +{horizon_s:.0f}s -> {parts}")

    return dict(zip(meta["target_columns"], predicted))


# ================================================================
# Main
# ================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=30.0,
                    help="seconds between predictions")
    ap.add_argument("--once", action="store_true",
                    help="predict once and exit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    conn = connect()
    model, meta = load_active_model(conn)

    print(f"model {meta['version']} (id {meta['model_id']})")
    print(f"window {meta['window']} samples, "
          f"horizon {meta['horizon']} samples")
    print(f"predicting: {', '.join(meta['target_columns'])}")

    if args.once:
        predict_once(conn, model, meta, verbose=not args.quiet)
        conn.close()
        return

    print(f"running every {args.interval:.0f}s -- ctrl-c to stop\n")
    try:
        while True:
            try:
                predict_once(conn, model, meta, verbose=not args.quiet)
            except Exception as exc:
                # A transient DB error should not kill a long-running
                # loop; reconnect and carry on.
                print(f"prediction failed: {exc}")
                try:
                    conn.close()
                except Exception:
                    pass
                time.sleep(5)
                conn = connect()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()