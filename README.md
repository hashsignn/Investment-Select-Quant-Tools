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

The model uses a 3-dimensional PyTorch autoencoder latent space (`z1`, `z2`, `z3`) and KMeans with 4 regimes. The autoencoder checkpoint is selected using validation loss with early stopping. KMeans is fitted only on the training latent vectors, then applied to validation and test.

To reproduce the autoencoder outputs:

```bash
cd market-regime-autoencoder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-build-isolation
python -m ae_regimes.train_handoff --handoff-dir "../DATA LAYER" --epochs 200 --early-stopping-patience 40 --clusters 4 --output-dir outputs/handoff
```
## PCA Regime Outputs

PCA implementation, comparison with the autoencoder, visualizations, and economic regime interpretation are in:

```text
pca-regime-analysis/
```

Main output file, merged with the autoencoder for direct comparison:

```text
pca-regime-analysis/outputs/pca_ae_comparison.csv
```

It contains:

```text
split,date,pc1,pc2,pc3,pca_regime,z1,z2,z3,ae_regime
```

If only the PCA latent representation is needed, use:

```text
pca-regime-analysis/outputs/pca_latent_space.csv
pca-regime-analysis/outputs/pca_clustered_regimes.csv
```

The model uses a 3-component PCA fit on the training windows only (`pc1`, `pc2`, `pc3`), matching the autoencoder's latent dimensionality for a fair comparison, and KMeans with 4 regimes fitted on the training PCA projections, then applied to validation and test.

Agreement between PCA and autoencoder regimes is measured with Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI), both computed on the merged comparison set.

Figures (saved to `pca-regime-analysis/outputs/figures/`):

```text
01_scree_plot.png
02_pca_2d_regimes.png
03_ae_2d_regimes.png
04_pca_vs_ae_3d.png
05_regime_timeline.png
06_regime_feature_profiles.png
07_regime_agreement_heatmap.png
08_reconstruction_loss_vs_pca_variance.png
```

To reproduce the PCA outputs:

```bash
cd pca-regime-analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pca_analysis.py
```

This expects `../DATA LAYER/windows.npz` and `../market-regime-autoencoder/outputs/handoff/clustered_regimes.csv` to already exist, so run the data pipeline and the autoencoder handoff first.

## PCA vs. Autoencoder Agreement

3 principal components explain 49.7% of variance (10 components: 67.2%, 20: 72.7%, 50: 82.4%), so a 3-D linear PCA discards more than half the variance the 30 features carry. The autoencoder's non-linear bottleneck is compressing into the same 3 dimensions but can capture structure PCA cannot.

```text
Adjusted Rand Index (ARI):        0.448   (1 = perfect, 0 = random)
Normalized Mutual Info (NMI):     0.548   (1 = perfect, 0 = none)
```

This is moderate agreement: the two methods share some regime structure but disagree on boundary placement, consistent with the AE's non-linear capacity picking up patterns linear PCA misses.

## Regime Interpretation (train set, PCA + KMeans)

Labels are derived from the dominant standardised features per regime (`build_interpretation()` in `pca_analysis.py`) and should be validated against domain knowledge before being treated as final.

| Regime | Windows (train) | Dominant features | Suggested label |
|---|---|---|---|
| 0 | 378 (27.7%) | CPI +1.24, dividend yield +1.18, 10Y yield +1.12 | inflationary pressure, rising long rates |
| 1 | 368 (27.0%) | 2Y yield −1.17, short rate −1.15, 10Y yield −1.10 | low volatility, falling long rates |
| 2 | 460 (33.7%) | CAPE +1.08, employment +0.96, real rate +0.91 | mixed / transitional |
| 3 | 159 (11.6%) | unemployment +1.99, yield-spread stress +1.97, GDP −1.62 | high volatility, elevated unemployment, falling long rates |

Regime 3 is the smallest and most distinct (highest unemployment, lowest GDP, highest volatility), and is the most useful candidate for a "crisis" label, pending cross-check against known recession dates. Regime 2 is the largest and most diffuse, consistent with a "normal market" baseline rather than a sharply defined state.

## Regime-Conditioned Trading Strategy

To test whether the regimes carry economic value rather than just descriptive structure, we built a minimal regime-conditioned long/short rule and backtested it strictly out of sample.

**How it's built:**
1. For every window, compute the realized one-week-forward return of `_MKT` (the market factor). Since windows slide one week at a time, window `i`'s forward return is just the new week added in window `i+1` — `windows[i+1][-1, mkt_idx]`.
2. Using **train only**, compute the average forward return per autoencoder regime (Table below), rank the regimes, and assign a fixed position: long the best regime, short the worst, flat the middle two.
3. Freeze that position map and apply it unchanged to validation and test. No out-of-sample information touches the rule.

```text
Regime   Mean forward return (train)   Position
1        +0.074                        Long (+1)
2        +0.034                        Flat (0)
0        +0.001                        Flat (0)
3        −0.057                        Short (−1)
```

**Backtest results (Sharpe annualized, weekly returns, √52 scaling; IC = Spearman rank correlation between position and realized forward return):**

```text
Split               Strategy Sharpe   Buy-and-hold Sharpe   IC
Train (in-sample)   0.29              0.00                  0.05
Validation (OOS)    0.15              -0.02                 0.10
Test (OOS)          -0.24             0.07                  -0.04
```

**Verdict:** the strategy is profitable and economically coherent in sample (the regime with elevated unemployment and volatility precedes the lowest forward returns, the low-volatility regime precedes the highest), but it does **not** generalize: on test it loses to a flat buy-and-hold benchmark, and IC drops to indistinguishable from zero. This tracks the same generalization gap seen elsewhere in this README — the validation-period regime collapse and the elevated test reconstruction loss both point to the same three-dimensional bottleneck struggling outside the training period. We would not recommend this as a stand-alone investment signal in its current form.

**Caveat on units:** Sharpe and IC above are computed on the *standardized* `_MKT` values (the StandardScaler-transformed log-return, not the raw percentage return), since reproducing this analysis from scratch without `scaler.pkl` doesn't give back the original scale. IC and the sign/ranking-based comparisons are unaffected by this, but if you want Sharpe in actual percentage-return terms, inverse-transform `_MKT` with `scaler.pkl` first.

To reproduce:

```python
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
```
