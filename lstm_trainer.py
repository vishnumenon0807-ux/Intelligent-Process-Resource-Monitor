"""
lstm_trainer.py
---------------
Trains an LSTM to forecast CPU, RAM and GPU utilisation from recent
history, and registers it in model_registry.

Install:
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

Run:
    python lstm_trainer.py
    python lstm_trainer.py --window 30 --horizon 30
    python lstm_trainer.py --epochs 100

Two decisions that matter on a small dataset:

  CHRONOLOGICAL SPLIT. Training windows overlap by all but one
  sample, so a random train/test split leaks future data into
  training through those shared samples and produces a validation
  score that means nothing. The split here is by time: earliest 80%
  trains, latest 20% validates.

  SMALL NETWORK. With a few thousand snapshots the effective sample
  count is far lower than it looks, because consecutive windows are
  near-duplicates. Two layers of 64 units with dropout and early
  stopping is deliberately modest — a larger model memorises
  instead of learning.
"""

import os
import json
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import psycopg
from psycopg.types.json import Jsonb
from dotenv import load_dotenv

from trainer import connect          # same .env connection

load_dotenv()

MODEL_DIR = "models"
MODEL_TYPE = "lstm_forecaster"

# What the model reads as input
INPUT_COLUMNS = [
    "cpu_percent", "ram_percent", "gpu_percent",
    "disk_read_rate_bps", "disk_write_rate_bps", "process_count",
]

# What it predicts
TARGET_COLUMNS = ["cpu_percent", "ram_percent", "gpu_percent"]


# ================================================================
# Model
# ================================================================

class Forecaster(nn.Module):
    def __init__(self, n_features, n_targets, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, n_targets),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        # Only the final timestep's hidden state feeds the head —
        # it carries the summary of the whole window.
        return self.head(out[:, -1, :])


# ================================================================
# Data
# ================================================================

def load_data(conn, hours=None) -> pd.DataFrame:
    cols = ", ".join(INPUT_COLUMNS)
    sql = f"SELECT timestamp, {cols} FROM system_snapshots"
    if hours:
        sql += f" WHERE timestamp > now() - interval '{int(hours)} hours'"
    sql += " ORDER BY timestamp"

    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
        names = [d.name for d in cur.description]

    df = pd.DataFrame(rows, columns=names)
    for c in INPUT_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def make_sequences(values: np.ndarray, target_idx: list[int],
                   window: int, horizon: int):
    """
    X[i] = values[i : i+window]                  (the history)
    y[i] = values[i+window+horizon-1, targets]   (the future point)
    """
    X, y = [], []
    limit = len(values) - window - horizon + 1
    for i in range(limit):
        X.append(values[i:i + window])
        y.append(values[i + window + horizon - 1, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ================================================================
# Train
# ================================================================

def train_model(Xtr, ytr, Xva, yva, n_features, n_targets,
                epochs, lr, patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = Forecaster(n_features, n_targets).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr).to(device)
    ytr_t = torch.tensor(ytr).to(device)
    Xva_t = torch.tensor(Xva).to(device)
    yva_t = torch.tensor(yva).to(device)

    batch = 64
    n = len(Xtr_t)
    best_loss = float("inf")
    best_state = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        # Shuffling batches is fine — the sequences themselves keep
        # their internal time order, and the split was chronological.
        perm = torch.randperm(n, device=device)
        total = 0.0

        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            pred = model(Xtr_t[idx])
            loss = loss_fn(pred, ytr_t[idx])
            loss.backward()
            # LSTMs are prone to exploding gradients; clipping keeps
            # training stable without needing a tiny learning rate.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva_t), yva_t).item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}  train {total/n:.5f}  val {val_loss:.5f}")

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_loss, device


def evaluate(model, X, y, scaler_min, scaler_range, target_idx, device):
    """Metrics in original units (percent), not scaled space."""
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X).to(device)).cpu().numpy()

    tmin = scaler_min[target_idx]
    trng = scaler_range[target_idx]
    pred_real = pred * trng + tmin
    true_real = y * trng + tmin

    rmse = float(np.sqrt(np.mean((pred_real - true_real) ** 2)))
    mae = float(np.mean(np.abs(pred_real - true_real)))

    per_target = {}
    for i, name in enumerate(TARGET_COLUMNS):
        per_target[name] = {
            "rmse": round(float(np.sqrt(np.mean(
                (pred_real[:, i] - true_real[:, i]) ** 2))), 3),
            "mae": round(float(np.mean(
                np.abs(pred_real[:, i] - true_real[:, i]))), 3),
        }
    return rmse, mae, per_target


# ================================================================
# Register
# ================================================================

def register(conn, version, params, scaler_params, n_rows,
             t_start, t_end, rmse, mae, notes, artifact):
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE model_registry SET is_active = FALSE "
            "WHERE model_type = %s AND is_active;", (MODEL_TYPE,))
        cur.execute(
            """
            INSERT INTO model_registry (
                model_type, model_version, trained_at, training_rows,
                training_start, training_end, feature_list,
                hyperparameters, scaler_params, rmse, mae, notes,
                artifact_path, is_active
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE)
            RETURNING id;
            """,
            (MODEL_TYPE, version, datetime.now(timezone.utc), n_rows,
             t_start, t_end, Jsonb(INPUT_COLUMNS), Jsonb(params),
             Jsonb(scaler_params), rmse, mae, notes, artifact),
        )
        model_id = cur.fetchone()[0]
    conn.commit()
    return model_id


# ================================================================
# Main
# ================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=30,
                    help="input sequence length in samples")
    ap.add_argument("--horizon", type=int, default=30,
                    help="how many samples ahead to predict")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--min-rows", type=int, default=800)
    args = ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    conn = connect()
    print(f"connected to {conn.info.dbname}")

    df = load_data(conn, args.hours)
    print(f"loaded {len(df)} snapshots")

    need = args.window + args.horizon + args.min_rows
    if len(df) < need:
        print(f"\nNot enough data: have {len(df)}, need about {need} "
              f"for window={args.window} horizon={args.horizon}.")
        print("Keep the collector running and try again.")
        conn.close()
        return

    values = df[INPUT_COLUMNS].values.astype(np.float32)

    # Min-max scaling to 0-1. Stored in model_registry so inference
    # applies the identical transform — a mismatch here produces
    # confident nonsense rather than an error.
    vmin = values.min(axis=0)
    vmax = values.max(axis=0)
    vrange = np.where(vmax - vmin < 1e-6, 1.0, vmax - vmin)
    scaled = (values - vmin) / vrange

    target_idx = [INPUT_COLUMNS.index(c) for c in TARGET_COLUMNS]
    X, y = make_sequences(scaled, target_idx, args.window, args.horizon)
    print(f"built {len(X)} sequences  "
          f"(window {args.window}, horizon {args.horizon})")

    # Chronological split — see the module docstring.
    split = int(len(X) * 0.8)
    Xtr, Xva = X[:split], X[split:]
    ytr, yva = y[:split], y[split:]
    print(f"train {len(Xtr)}  validate {len(Xva)}")

    model, val_loss, device = train_model(
        Xtr, ytr, Xva, yva, len(INPUT_COLUMNS), len(TARGET_COLUMNS),
        args.epochs, args.lr, args.patience,
    )

    rmse, mae, per_target = evaluate(
        model, Xva, yva, vmin, vrange, target_idx, device)

    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    artifact = os.path.join(MODEL_DIR, f"lstm_{version}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "input_columns": INPUT_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "window": args.window,
        "horizon": args.horizon,
        "scaler_min": vmin.tolist(),
        "scaler_range": vrange.tolist(),
        "arch": {"hidden": 64, "layers": 2, "dropout": 0.2},
    }, artifact)

    params = {
        "window": args.window, "horizon": args.horizon,
        "hidden": 64, "layers": 2, "dropout": 0.2,
        "lr": args.lr, "batch_size": 64, "optimizer": "adam",
        "targets": TARGET_COLUMNS,
    }
    scaler_params = {"min": vmin.tolist(), "range": vrange.tolist()}
    notes = json.dumps(per_target)

    model_id = register(
        conn, version, params, scaler_params, len(df),
        df["timestamp"].min(), df["timestamp"].max(),
        rmse, mae, notes, artifact,
    )
    conn.close()

    print(f"\nmodel {model_id} ({version}) trained and set active")
    print(f"artifact: {artifact}")
    print(f"\nvalidation (original units, percentage points):")
    print(f"  overall   RMSE {rmse:6.2f}   MAE {mae:6.2f}")
    for name, m in per_target.items():
        print(f"  {name:<14} RMSE {m['rmse']:6.2f}   MAE {m['mae']:6.2f}")

    # A naive baseline: predict that nothing changes. If the LSTM
    # cannot beat this, it has not learned anything useful.
    naive = Xva[:, -1, target_idx]
    naive_real = naive * vrange[target_idx] + vmin[target_idx]
    true_real = yva * vrange[target_idx] + vmin[target_idx]
    naive_rmse = float(np.sqrt(np.mean((naive_real - true_real) ** 2)))
    print(f"\n  naive baseline (assume no change)  RMSE {naive_rmse:6.2f}")
    if rmse < naive_rmse:
        print(f"  LSTM beats baseline by {naive_rmse - rmse:.2f} points")
    else:
        print(f"  LSTM does NOT beat the baseline — needs more data "
              f"or tuning")


if __name__ == "__main__":
    main()
