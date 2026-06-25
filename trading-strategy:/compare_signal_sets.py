"""
Signal Set Comparison — Autoencoder Experiment
===============================================
Trains three separate autoencoder instances with identical architecture
and hyperparameters, differing only in the feature subset used as input:

  Set A  — Macroeconomic cycle (10 features)
  Set B  — Market risk and sentiment (10 features)
  Full   — All 30 features (main paper experiment)

For each model, reports:
  - Reconstruction loss (train / val / test)
  - Regime-conditioned trading strategy: annualised Sharpe and IC
    by split, derived from a position rule fixed on training data only

Results correspond to Table II in the paper (Section V-B: Choice of Signals).

Inputs (relative to repo root)
-------------------------------
  DATA LAYER/windows.npz
  DATA LAYER/schema.json

Outputs (saved to trading-strategy/outputs/)
--------------------------------------------
  signal_set_comparison.csv   — full results table

Usage
-----
  cd <repo-root>
  python trading-strategy/compare_signal_sets.py

Requirements
------------
  torch, numpy, pandas, scikit-learn, scipy
  The ae_regimes package must be installed:
    pip install -e market-regime-autoencoder --no-build-isolation
"""

from __future__ import annotations

import copy
import json
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# ── Make config.py importable from repo root ─────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config as cfg   # LATENT_DIM, N_CLUSTERS, RANDOM_SEED

sys.path.insert(0, str(ROOT / "market-regime-autoencoder" / "src"))
from ae_regimes.model import Autoencoder

DATA_DIR = ROOT / "DATA LAYER"
OUT_DIR  = Path(__file__).resolve().parent / "outputs"

ANNUALIZATION_FACTOR = np.sqrt(52)

# ── Signal set definitions ────────────────────────────────────────────
SIGNAL_SET_A = ["GDP", "CPI", "M2", "UN", "CF", "IR", "RR", "STP", "Y02", "Y10"]
SIGNAL_SET_B = ["_MKT", "MOV", "DY", "YSS", "_LCP", "_TY", "_OIL", "_AU", "_DXY", "NYF"]


# ── Data helpers ─────────────────────────────────────────────────────

def load_feature_indices() -> tuple[list[str], int]:
    schema = json.load(open(DATA_DIR / "schema.json"))
    feat   = schema["feature_order"]
    mkt    = feat.index("_MKT")
    return feat, mkt


def get_windows(windows_npz, feat_idx: list[int]) -> dict[str, np.ndarray]:
    """Return flattened (n, window_size * n_feat) arrays for each split."""
    out = {}
    for split in ("train", "val", "test"):
        w = windows_npz[split][:, :, feat_idx]
        out[split] = w.reshape(w.shape[0], -1).astype(np.float32)
    return out


def get_forward_returns(windows_npz, mkt_idx: int) -> dict[str, np.ndarray]:
    """One-week-forward _MKT return for every window in every split."""
    fwd = {}
    for split in ("train", "val", "test"):
        mkt    = windows_npz[split][:, :, mkt_idx]
        n      = mkt.shape[0]
        f      = np.full(n, np.nan)
        f[:-1] = mkt[1:, -1]
        fwd[split] = f
    return fwd


# ── Training ─────────────────────────────────────────────────────────

def train_autoencoder(
    train_w: np.ndarray,
    val_w:   np.ndarray,
    input_dim: int,
    epochs:   int = 200,
    patience: int = 40,
    lr:       float = 1e-3,
    wd:       float = 5e-4,
    batch:    int = 16,
    seed:     int = 42,
) -> Autoencoder:
    torch.manual_seed(seed)
    model  = Autoencoder(input_dim=input_dim)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit   = nn.MSELoss()
    tr_t   = torch.tensor(train_w, dtype=torch.float32)
    va_t   = torch.tensor(val_w,   dtype=torch.float32)
    loader = DataLoader(TensorDataset(tr_t), batch_size=batch, shuffle=True)

    best_val, stale = float("inf"), 0
    best_state = copy.deepcopy(model.state_dict())

    for ep in range(1, epochs + 1):
        model.train()
        for (b,) in loader:
            opt.zero_grad()
            crit(model(b), b).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(va_t), va_t).item()
        if vl < best_val:
            best_val, stale = vl, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    print(f"    early stop ep {ep}, best val {best_val:.4f}")
    return model


# ── Evaluation helpers ────────────────────────────────────────────────

def reconstruction_loss(model: Autoencoder, windows: dict) -> dict:
    model.eval()
    crit = nn.MSELoss()
    out  = {}
    with torch.no_grad():
        for split, w in windows.items():
            t = torch.tensor(w, dtype=torch.float32)
            out[split] = round(crit(model(t), t).item(), 4)
    return out


def cluster_latents(model: Autoencoder, windows: dict) -> dict:
    """Fit KMeans on train latents, assign regimes to all splits."""
    model.eval()
    latents = {}
    with torch.no_grad():
        for split, w in windows.items():
            latents[split] = model.encoder(
                torch.tensor(w, dtype=torch.float32)
            ).numpy()

    scaler  = StandardScaler()
    tr_sc   = scaler.fit_transform(latents["train"])
    km      = KMeans(n_clusters=cfg.N_CLUSTERS, n_init="auto",
                     random_state=cfg.RANDOM_SEED)
    km.fit(tr_sc)
    return {s: km.predict(scaler.transform(lat))
            for s, lat in latents.items()}


def run_backtest(labels: dict, fwd: dict) -> pd.DataFrame:
    """Long/short rule from train; evaluate on all splits."""
    train_df = (
        pd.DataFrame({"regime": labels["train"], "fwd": fwd["train"]})
        .dropna()
    )
    ranked  = (train_df.groupby("regime")["fwd"].mean()
               .sort_values(ascending=False))
    pos_map = {int(ranked.index[0]): 1, int(ranked.index[-1]): -1}
    for r in ranked.index[1:-1]:
        pos_map[int(r)] = 0

    rows = []
    for split in ("train", "val", "test"):
        df         = pd.DataFrame({"regime": labels[split], "fwd": fwd[split]}).dropna()
        df["pos"]  = df["regime"].map(pos_map)
        pnl        = df["pos"] * df["fwd"]
        sharpe     = (pnl.mean() / pnl.std() * ANNUALIZATION_FACTOR
                      if pnl.std() > 0 else np.nan)
        ic, _      = spearmanr(df["pos"], df["fwd"])
        rows.append({"split": split,
                     "sharpe": round(sharpe, 3),
                     "ic":     round(ic, 3)})
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    windows_npz = np.load(DATA_DIR / "windows.npz")
    feat, mkt_idx = load_feature_indices()
    fwd = get_forward_returns(windows_npz, mkt_idx)

    experiments = [
        ("Set A (macro, 10 feat.)",         [feat.index(f) for f in SIGNAL_SET_A]),
        ("Set B (risk/sentiment, 10 feat.)", [feat.index(f) for f in SIGNAL_SET_B]),
        ("Full (all 30 feat.)",              list(range(len(feat)))),
    ]

    summary_rows = []

    for name, idx in experiments:
        print(f"\n{'='*55}\n{name}\n{'='*55}")
        wins   = get_windows(windows_npz, idx)
        model  = train_autoencoder(wins["train"], wins["val"],
                                   input_dim=wins["train"].shape[1],
                                   seed=cfg.RANDOM_SEED)
        rl     = reconstruction_loss(model, wins)
        labels = cluster_latents(model, wins)
        bt     = run_backtest(labels, fwd)

        print(f"  Recon loss — train:{rl['train']}  val:{rl['val']}  test:{rl['test']}")
        print(bt.to_string(index=False))

        def row_for(split):
            r = bt[bt["split"] == split].iloc[0]
            return r["sharpe"], r["ic"]

        tr_s, tr_ic = row_for("train")
        va_s, va_ic = row_for("val")
        te_s, te_ic = row_for("test")

        summary_rows.append({
            "signal_set":    name,
            "recon_train":   rl["train"],
            "recon_val":     rl["val"],
            "recon_test":    rl["test"],
            "train_sharpe":  tr_s, "train_ic": tr_ic,
            "val_sharpe":    va_s, "val_ic":   va_ic,
            "test_sharpe":   te_s, "test_ic":  te_ic,
        })

    summary = pd.DataFrame(summary_rows)
    out_path = OUT_DIR / "signal_set_comparison.csv"
    summary.to_csv(out_path, index=False)

    print("\n\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(summary.to_string(index=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
