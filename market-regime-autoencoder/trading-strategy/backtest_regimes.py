"""
Regime-Conditioned Trading Strategy Backtest
=============================================
Tests whether the autoencoder-derived market regimes carry economic value
by backtesting a simple long/flat/short rule conditioned on regime label.

Methodology
-----------
1. For every window, the realized one-week-forward _MKT return is the new
   observation introduced by window i+1 (the last time-step of window i+1).
2. A position rule is derived from TRAIN ONLY: regimes are ranked by their
   average forward return in training; the best regime is assigned long (+1),
   the worst short (-1), and the remaining two flat (0).
3. That rule is frozen and applied unchanged to validation and test.

Inputs (relative to repo root)
-------------------------------
  DATA LAYER/windows.npz
  DATA LAYER/schema.json
  market-regime-autoencoder/outputs/handoff/clustered_regimes.csv

Outputs (saved to trading-strategy/outputs/)
--------------------------------------------
  regime_forward_returns.csv
  position_rule.csv
  strategy_results.csv

Usage
-----
  cd <repo-root>
  python trading-strategy/backtest_regimes.py

Note on units
-------------
Sharpe and IC are computed on StandardScaler-transformed log-returns.
To obtain Sharpe in raw percentage-return units, inverse-transform _MKT
with DATA LAYER/scaler.pkl first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "DATA LAYER"
AE_DIR   = ROOT / "market-regime-autoencoder" / "outputs" / "handoff"
OUT_DIR  = Path(__file__).resolve().parent / "outputs"

ANNUALIZATION_FACTOR = np.sqrt(52)


def load_forward_returns(data_dir: Path, ae_dir: Path) -> pd.DataFrame:
    """Attach realised one-week-forward _MKT return to each regime window."""
    windows       = np.load(data_dir / "windows.npz")
    feature_names = json.load(open(data_dir / "schema.json"))["feature_order"]
    mkt_idx       = feature_names.index("_MKT")
    ae_df         = pd.read_csv(ae_dir / "clustered_regimes.csv", parse_dates=["date"])

    parts = []
    for split in ("train", "val", "test"):
        w   = windows[split][:, :, mkt_idx]
        n   = w.shape[0]
        fwd = np.full(n, np.nan)
        fwd[:-1] = w[1:, -1]
        sub = ae_df[ae_df["split"] == split].reset_index(drop=True).copy()
        if len(sub) != n:
            raise ValueError(f"Row mismatch for '{split}': {len(sub)} vs {n}")
        sub["fwd_return_1w"] = fwd
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def derive_position_rule(full_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Rank regimes by train mean forward return; long best, short worst."""
    train  = full_df[full_df["split"] == "train"].dropna(subset=["fwd_return_1w"])
    ranked = train.groupby("regime")["fwd_return_1w"].mean().sort_values(ascending=False)
    if len(ranked) < 2:
        raise ValueError("Need at least 2 regimes.")
    pos_map = {int(ranked.index[0]): 1, int(ranked.index[-1]): -1}
    for r in ranked.index[1:-1]:
        pos_map[int(r)] = 0
    rule_df = ranked.reset_index()
    rule_df.columns = ["regime", "mean_forward_return_train"]
    rule_df["position"] = rule_df["regime"].map(pos_map)
    return pos_map, rule_df


def backtest_split(df: pd.DataFrame, pos_map: dict, split: str) -> dict:
    sub    = df[df["split"] == split].dropna(subset=["fwd_return_1w"]).copy()
    sub["position"] = sub["regime"].map(pos_map)
    pnl    = sub["position"] * sub["fwd_return_1w"]
    sharpe = pnl.mean() / pnl.std() * ANNUALIZATION_FACTOR if pnl.std() > 0 else np.nan
    bh     = (sub["fwd_return_1w"].mean() / sub["fwd_return_1w"].std()
              * ANNUALIZATION_FACTOR if sub["fwd_return_1w"].std() > 0 else np.nan)
    ic, p  = spearmanr(sub["position"], sub["fwd_return_1w"])
    active = sub[sub["position"] != 0]
    hit    = ((np.sign(active["position"]) == np.sign(active["fwd_return_1w"])).mean()
              if len(active) > 0 else np.nan)
    return {
        "split":                    split,
        "n_windows":                len(sub),
        "n_active_positions":       len(active),
        "strategy_sharpe":          round(sharpe, 4),
        "buy_and_hold_sharpe":      round(bh, 4),
        "information_coefficient":  round(ic, 4),
        "ic_pvalue":                round(p, 4),
        "hit_rate":                 round(hit, 4),
        "cumulative_strategy_pnl":  round(pnl.sum(), 4),
        "cumulative_bh_return":     round(sub["fwd_return_1w"].sum(), 4),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",   default=str(DATA_DIR))
    p.add_argument("--ae-dir",     default=str(AE_DIR))
    p.add_argument("--output-dir", default=str(OUT_DIR))
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    full_df = load_forward_returns(Path(args.data_dir), Path(args.ae_dir))

    print("Deriving position rule from train only …")
    pos_map, rule_df = derive_position_rule(full_df)
    print(rule_df.to_string(index=False))

    print("\nBacktesting …")
    results_df = pd.DataFrame([
        backtest_split(full_df, pos_map, s) for s in ("train", "val", "test")
    ])
    print(results_df[["split","strategy_sharpe","buy_and_hold_sharpe",
                       "information_coefficient"]].to_string(index=False))

    full_df.to_csv(out_dir / "regime_forward_returns.csv", index=False)
    rule_df.to_csv(out_dir / "position_rule.csv",          index=False)
    results_df.to_csv(out_dir / "strategy_results.csv",    index=False)
    print(f"\nOutputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
