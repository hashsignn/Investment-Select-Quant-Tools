import numpy as np, json, pandas as pd
from scipy.stats import spearmanr

windows = np.load("DATA LAYER/windows.npz")
feat = json.load(open("DATA LAYER/schema.json"))["feature_order"]
mkt_idx = feat.index("_MKT")

ae_df = pd.read_csv("market-regime-autoencoder/outputs/handoff/clustered_regimes.csv", parse_dates=["date"])

results = []
for split in ["train", "val", "test"]:
    w = windows[split][:, :, mkt_idx]
    n = w.shape[0]
    fwd_1w = np.full(n, np.nan)
    fwd_1w[:-1] = w[1:, -1]
    sub = ae_df[ae_df["split"] == split].reset_index(drop=True)
    sub["fwd_return_1w"] = fwd_1w
    results.append(sub)
full = pd.concat(results, ignore_index=True)

# derive position rule from train only, then apply out of sample
train = full[full["split"] == "train"].dropna(subset=["fwd_return_1w"])
ranked = train.groupby("regime")["fwd_return_1w"].mean().sort_values(ascending=False)
position_map = {ranked.index[0]: 1, ranked.index[-1]: -1}
for r in ranked.index[1:-1]:
    position_map[r] = 0

for split in ["train", "val", "test"]:
    sub = full[full["split"] == split].dropna(subset=["fwd_return_1w"]).copy()
    sub["position"] = sub["regime"].map(position_map)
    pnl = sub["position"] * sub["fwd_return_1w"]
    sharpe = pnl.mean() / pnl.std() * np.sqrt(52)
    ic, _ = spearmanr(sub["position"], sub["fwd_return_1w"])
    print(split, "Sharpe:", round(sharpe, 3), "IC:", round(ic, 3))
