"""
Preprocessing pipeline

Decisions
  - 8 underscore price/index columns -> log returns (they are non-stationary levels)
  - 22 macro/valuation columns kept as-is (already stationary / YoY-processed)
  - Split 70/15/15 by TIME (forward chaining, never shuffled)
  - StandardScaler fit on TRAIN ONLY, then applied to val/test (anti-leakage)
  - Sliding windows of 26 weeks (~6 months), stride 1, overlapping
    (window length chosen on economic grounds; regimes proved robust to it)

Output: handoff/ folder. load windows.npz and trains directly.
Reproducible: fixed seed, fixed float32 dtype.
"""

import hashlib
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# 0. Config
# ----------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

SRC = "market_data.xlsx"
SHEET = "US"
OUT = Path("handoff")
OUT.mkdir(exist_ok=True)

WINDOW_SIZE = 26          # weeks (~6 months)
STRIDE = 1                # overlapping windows
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15   # test = remaining 0.15

# 8 raw price/index levels -> converted to log returns
RETURN_COLS = ["_AU", "_DXY", "_LCP", "_TY", "_OIL", "_MKT", "_VA", "_GR"]


# ----------------------------------------------------------------------
# 1. Ingest + clean column names
# ----------------------------------------------------------------------
df = pd.read_excel(SRC, sheet_name=SHEET)
df.columns = [c.strip() for c in df.columns]   # fixes 'MOV ' trailing space
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

raw_sha = hashlib.sha256(
    pd.util.hash_pandas_object(df, index=True).values.tobytes()
).hexdigest()

feature_cols = [c for c in df.columns if c != "Date"]

# sanity: clean weekly data, no gaps expected
assert df["Date"].is_monotonic_increasing
assert not df["Date"].duplicated().any(), "duplicate dates"


# ----------------------------------------------------------------------
# 2. Stationarity transform: log returns on the 8 price/index levels
#    log return r_t = ln(P_t) - ln(P_{t-1}); first row becomes NaN -> dropped
# ----------------------------------------------------------------------
for c in RETURN_COLS:
    df[c] = np.log(df[c]).diff()

df = df.dropna().reset_index(drop=True)   # drops the single first row (return NaN)

dates = df["Date"].values
X = df[feature_cols].astype("float64").values
n = len(X)


# ----------------------------------------------------------------------
# 3. Split FIRST (forward chaining) — before fitting any transform
# ----------------------------------------------------------------------
i_tr = int(n * TRAIN_FRAC)
i_va = int(n * (TRAIN_FRAC + VAL_FRAC))

idx = {
    "train": slice(0, i_tr),
    "val":   slice(i_tr, i_va),
    "test":  slice(i_va, n),
}
splits_raw = {k: X[v] for k, v in idx.items()}
split_dates = {k: dates[v] for k, v in idx.items()}


# ----------------------------------------------------------------------
# 4. Scale: fit on TRAIN ONLY, apply to all (anti-leakage)
# ----------------------------------------------------------------------
scaler = StandardScaler()
scaler.fit(splits_raw["train"])

splits = {k: scaler.transform(v).astype("float32") for k, v in splits_raw.items()}

joblib.dump(scaler, OUT / "scaler.pkl")


# ----------------------------------------------------------------------
# 5. Window: sliding, per split separately (no window straddles a boundary)
#    Autoencoder target == the input window itself.
# ----------------------------------------------------------------------
def make_windows(arr, w, stride):
    n_rows = len(arr)
    if n_rows < w:
        return np.empty((0, w, arr.shape[1]), dtype="float32")
    starts = range(0, n_rows - w + 1, stride)
    return np.stack([arr[s:s + w] for s in starts]).astype("float32")

windows = {k: make_windows(v, WINDOW_SIZE, STRIDE) for k, v in splits.items()}


# ----------------------------------------------------------------------
# 6. Serialize
# ----------------------------------------------------------------------
np.savez_compressed(OUT / "splits.npz", **splits)
np.savez_compressed(OUT / "windows.npz", **windows)

# schema / provenance
schema = {
    "source_file": SRC,
    "sheet": SHEET,
    "raw_sha256": raw_sha,
    "n_rows_after_returns": int(n),
    "feature_order": feature_cols,
    "n_features": len(feature_cols),
    "return_cols_log": RETURN_COLS,
    "macro_cols_asis": [c for c in feature_cols if c not in RETURN_COLS],
}
(OUT / "schema.json").write_text(json.dumps(schema, indent=2))

metadata = {
    "frequency": "weekly (7-day)",
    "date_start": str(pd.Timestamp(dates[0]).date()),
    "date_end": str(pd.Timestamp(dates[-1]).date()),
    "seed": SEED,
    "dtype": "float32",
    "split": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": round(1 - TRAIN_FRAC - VAL_FRAC, 2)},
    "split_boundaries_dates": {
        "train": [str(pd.Timestamp(split_dates['train'][0]).date()),
                  str(pd.Timestamp(split_dates['train'][-1]).date())],
        "val":   [str(pd.Timestamp(split_dates['val'][0]).date()),
                  str(pd.Timestamp(split_dates['val'][-1]).date())],
        "test":  [str(pd.Timestamp(split_dates['test'][0]).date()),
                  str(pd.Timestamp(split_dates['test'][-1]).date())],
    },
    "window": {"size": WINDOW_SIZE, "stride": STRIDE, "type": "sliding/overlapping",
               "autoencoder_target": "input window itself"},
    "shapes": {
        "splits": {k: list(v.shape) for k, v in splits.items()},
        "windows": {k: list(v.shape) for k, v in windows.items()},
    },
    "scaler": "StandardScaler, fit on TRAIN only",
}
(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

print("PIPELINE COMPLETE")
print(json.dumps(metadata["shapes"], indent=2))
print("\nSplit date ranges:")
for k, v in metadata["split_boundaries_dates"].items():
    print(f"  {k:>5}: {v[0]} -> {v[1]}")
