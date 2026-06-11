# Investment-Select-Quant-Tools

# Pre-Processing Handoff
Model-ready, leak-free data. **You do not need the raw Excel or any re-cleaning.**
Load `windows.npz` and train.

## Quick start
Unzip the package, then run this:
```python
import numpy as np
w = np.load("windows.npz")["train"]   # (1365, 26, 30)
batch = w[:32]                         # autoencoder: input == target
assert not np.isnan(batch).any()
print(batch.shape)                     # (32, 26, 30)
```
Sanity Check
```python
import numpy as np, json
W = np.load("windows.npz")
Xtr, Xva, Xte = W["train"], W["val"], W["test"]
feat = json.load(open("schema.json"))["feature_order"]   # 30 names
print(Xtr.shape, "no NaNs:", not np.isnan(Xtr).any())     # (1365, 26, 30) True
```

That is the whole onboarding. windows.npz is the model-ready data; everything else in the folder supports it or documents it. The rest of this doc explains each file, and what we found in the data analysis.

Input is a 26×30 window. The bottleneck (latent dimension) is the key knob. **See the analysis in §4 for why 8–16 is a sensible starting range**

## What's in the box

| File | Contents |
|---|---|
| `windows.npz` | **Main input.** Sliding 26-week windows. Arrays `train` / `val` / `test`, shape `(n_windows, 26, 30)`, `float32`. AE reconstruction target = the input window itself. |
| `splits.npz`  | Row-level scaled features (no windowing), same 3 splits, shape `(n_rows, 30)`. Use if you want per-week vectors instead of sequences. |
| `scaler.pkl`  | `StandardScaler` fit on **train only**. `joblib.load` it to inverse-transform or to scale any new rows. |
| `schema.json` | `feature_order` (the 30 column names, in array order), which 8 cols are log-returns vs 22 macro-as-is, raw-data SHA. |
| `metadata.json` | Split dates, window params, shapes, seed. |
| `pipeline.py` | The script that produced all of this (reproducible). |
| `test_handoff.py` | Contract test (13 checks, all pass). |

## Shapes

```
splits :  train (1390, 30)   val (298, 30)   test (298, 30)
windows:  train (1365,26,30) val (273,26,30) test (273,26,30)
```

## What was done (so you can describe it in the paper)

1. **8 price/index columns → log returns** (`_AU _DXY _LCP _TY _OIL _MKT _VA _GR`); they were non-stationary levels.
2. **The 22 macro/valuation signals** were kept as-is (already stationary / YoY) - interest rates, yield curve, PE, CAPE, dividend yield, unemployment, GDP, CPI, money supply, consumer confidence, recession probability, etc. These describe the economic environment. 
3. **Split 70/15/15 by time** (forward chaining — train is oldest, test is newest, never shuffled).
   - train 1988-04-17 → 2014-11-30
   - val   2014-12-07 → 2020-08-16
   - test  2020-08-23 → 2026-05-03
4. **StandardScaler fit on TRAIN only**, applied to val/test (no leakage). Verified: train ≈ mean 0 / std 1, val/test drift away.
5. **Sliding windows, 26 weeks (~6mo), stride 1**, built per-split so no window crosses a split boundary.

## Analysis
1. Prices were non-stationary, turned to log-returns.** All 8 price/index columns were strongly non-stationary as raw levels (ADF p ≈ 0.18–1.0, they just trend). After the log-return transform, all 8 became strongly stationary (ADF p < 0.0001).

| Column | ADF p (raw level) | ADF p (log return) |
|---|---|---|
| _MKT (market | 1.00 | <0.0001 |
| _AU (gold) | 1.00 | <0.0001 |
| _GR (growth) | 1.00 | <0.0001 |
| _OIL (oil) | 0.30 | <0.0001 |

(p < 0.05 = stationary. All 8 behaved like the rows above.)

2. **Finding 2: Heavy Redundancy:** Many features move together: 11 feature pairs have |correlation| > 0.8 (e.g. 2Y vs 10Y yield = 0.93, short rate vs 2Y = 0.97, PE vs CAPE = 0.89, value vs growth = 0.83). The 30 features carry far fewer than 30 independent dimensions. Practical consequence: a latent dimension of ~8–16 should reconstruct well, you don't need a wide bottleneck. As a reference point, a simple 10-component PCA on the train windows already reconstructs ~67% of the variance.

## Notes
- tested 13/26/39/52/65/78-week windows and measured how stable the resulting regime labels were. Stability was essentially flat across all sizes (regime flip rate ~0.5% regardless), because the slow macro signals dominate. So rather than overfit the window to the data, we chose 26 weeks (~6 months): long enough to smooth single-week noise, short enough to catch a regime change within a quarter or two, and a natural half-year economic horizon. If you want to try alternatives, please tell me.
- One thing we deliberately did NOT do: extra smoothing of the returns. That would throw away real signal and is a modelling choice, not a preprocessing one. If you want a smoothed variant, ask and we can produce one but it shouldn't be the default.
- **Feature order is fixed** use `schema.json["feature_order"]`; don't reorder.
- **Noise control is your side now**: bottleneck size + regularization (dropout / weight decay / early stopping) are what stop the AE from fitting noise. Returns + windowing already removed the trend/jitter from the inputs.
- A 10-component PCA on the train windows reconstructs ~67% of variance, so the data carries compressible structure. good sign for both the AE and PCA regime work.
- To re-generate from scratch: `python pipeline.py` (needs `market_data.xlsx`), then `python test_handoff.py`.

## Autoencoder Regime Outputs

Eleni's autoencoder and regime-clustering pipeline is in:

```text
market-regime-autoencoder/
```

Main downstream file for PCA comparison, visualizations, and economic regime interpretation:

```text
market-regime-autoencoder/outputs/handoff/clustered_regimes.csv
```

It contains:

```text
split,window_index,date,z1,z2,z3,regime
```

The model uses a 3-dimensional PyTorch autoencoder latent space (`z1`, `z2`, `z3`) and KMeans with 4 regimes. KMeans is fitted only on the training latent vectors, then applied to validation and test.

To reproduce the autoencoder outputs:

```bash
cd market-regime-autoencoder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-build-isolation
python -m ae_regimes.train_handoff --handoff-dir "../DATA LAYER" --epochs 200 --clusters 4 --output-dir outputs/handoff
```
