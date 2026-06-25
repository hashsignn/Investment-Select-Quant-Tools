# Investment-Select-Quant-Tools

Unsupervised market regime detection using autoencoders and PCA, applied to weekly US financial and macroeconomic data (1988–2026). The repo is organized into four self-contained folders, each owned by a separate pipeline stage.

```
DATA LAYER/                   ← pre-processing (Harjot)
market-regime-autoencoder/    ← autoencoder + KMeans (Eleni)
pca-regime-analysis/          ← PCA baseline + comparison (Adene)
trading-strategy/             ← backtest + signal-set experiment
```

Run order: **DATA LAYER → market-regime-autoencoder → pca-regime-analysis → trading-strategy**

---

## 1. DATA LAYER

Model-ready, leak-free data. **You do not need the raw Excel or any re-cleaning.** Load `windows.npz` and train.

### Quick start

```python
import numpy as np, json

W    = np.load("DATA LAYER/windows.npz")
feat = json.load(open("DATA LAYER/schema.json"))["feature_order"]  # 30 names

Xtr, Xva, Xte = W["train"], W["val"], W["test"]
print(Xtr.shape, "no NaNs:", not np.isnan(Xtr).any())  # (1365, 26, 30) True
```

### Files

| File | Contents |
|---|---|
| `windows.npz` | **Main input.** Sliding 26-week windows. Arrays `train` / `val` / `test`, shape `(n_windows, 26, 30)`, `float32`. AE reconstruction target = the input itself. |
| `splits.npz` | Row-level scaled features (no windowing), same 3 splits, shape `(n_rows, 30)`. Use for per-week vectors. |
| `scaler.pkl` | `StandardScaler` fit on **train only**. `joblib.load` it to inverse-transform or to scale new rows. |
| `schema.json` | `feature_order` (30 column names in array order), which 8 cols are log-returns vs 22 macro-as-is. |
| `metadata.json` | Split dates, window params, shapes, seed. |
| `pipeline.py` | Reproduces all files above from `market_data.xlsx`. |
| `python_test_handoff.py` | 13-check contract test (all pass). |

### Shapes

```
splits :  train (1390, 30)    val (298, 30)    test (298, 30)
windows:  train (1365, 26, 30) val (273, 26, 30) test (273, 26, 30)
```

### What was done

1. **Log-returns** — 8 price/index columns (`_AU _DXY _LCP _TY _OIL _MKT _VA _GR`) were non-stationary as raw levels (ADF p ≈ 0.18–1.0). After the log-return transform all 8 are strongly stationary (ADF p < 0.0001).
2. **22 macro/valuation signals** kept as-is (already stationary): interest rates, yield curve, PE, CAPE, dividend yield, unemployment, GDP, CPI, money supply, etc.
3. **70 / 15 / 15 time split** — forward chaining, never shuffled.
   - train: 1988-04-17 → 2014-11-30
   - val:   2014-12-07 → 2020-08-16
   - test:  2020-08-23 → 2026-05-03
4. **StandardScaler fit on train only**, applied to val/test. No leakage.
5. **Sliding windows, 26 weeks, stride 1**, built per split so no window crosses a boundary.

### Data analysis findings

**Stationarity** — ADF test (representative rows):

| Column | ADF p (raw) | ADF p (log-return) |
|---|---|---|
| `_MKT` (market) | 1.00 | < 0.0001 |
| `_AU` (gold) | 1.00 | < 0.0001 |
| `_OIL` (oil) | 0.30 | < 0.0001 |

**Redundancy** — 11 feature pairs have |correlation| > 0.8 (e.g. 2Y vs 10Y yield = 0.93, short rate vs 2Y = 0.97, PE vs CAPE = 0.89). The 30 features carry far fewer than 30 independent dimensions. A 10-component PCA on the train windows already captures ~67% of variance — a good sign for both AE and PCA regime work. This is why a latent dimension of 16 is a sensible choice: wide enough to avoid mode collapse, compact enough to force genuine compression.

**Window size** — tested 13/26/39/52/65/78-week windows; regime-label stability was flat across all sizes (flip rate ~0.5%), because slow macro signals dominate. 26 weeks (~6 months) was chosen as a natural half-year economic horizon.

---

## 2. Autoencoder Regime Outputs

Pipeline in `market-regime-autoencoder/`. Main downstream file:

```
market-regime-autoencoder/outputs/handoff/clustered_regimes.csv
```

Schema: `split, window_index, date, z1, z2, …, z16, regime`

The model uses a 16-dimensional PyTorch autoencoder (architecture: `Input(780) → Dense(256) → ReLU → Dense(64) → ReLU → Latent(16)`, symmetric decoder) with KMeans (k=4). Scaler and KMeans are fitted on train only, then applied to val/test.

**Reconstruction loss (MSE):**

| Split | Loss |
|---|---|
| Train | 0.299 |
| Validation | 0.611 |
| Test | 1.205 |

The elevated test loss reflects market conditions post-2020 (inflation spike, rate-hike cycle) that differ structurally from the training period — a regime shift, not a model failure.

### Reproduce

```bash
cd market-regime-autoencoder
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-build-isolation
python -m ae_regimes.train_handoff \
    --handoff-dir "../DATA LAYER" \
    --epochs 200 --early-stopping-patience 40 \
    --clusters 4 --output-dir outputs/handoff
```

---

## 3. PCA Regime Outputs

Pipeline in `pca-regime-analysis/`. Main output files:

```
pca-regime-analysis/outputs/pca_ae_comparison.csv   ← merged PCA + AE regimes
pca-regime-analysis/outputs/pca_latent_space.csv    ← PCA projections only
pca-regime-analysis/outputs/pca_clustered_regimes.csv
```

Merged schema: `split, date, pc1, …, pc16, pca_regime, z1, …, z16, ae_regime`

The model uses 16-component PCA (matching AE dimensionality for a fair comparison) fitted on training windows only, followed by KMeans (k=4) on train projections.

**PCA explained variance** — 16 components retain 70.8% of training variance (10 components: 67.2%; 20: 72.7%; 50: 82.4%). The AE compresses into the same 16 dimensions nonlinearly.

**Agreement between methods:**

```
Adjusted Rand Index (ARI):       0.515   (1 = perfect, 0 = random)
Normalized Mutual Information:   0.577   (1 = perfect, 0 = none)
```

Moderate agreement: both methods share regime structure but differ on boundary placement, consistent with the AE capturing nonlinear patterns PCA misses.

**Regime interpretation (train set, PCA + KMeans):**

| Regime | Size | Dominant features | Interpretation |
|---|---|---|---|
| 0 | 363 (26.6%) | 2Y yield −1.17, short rate −1.15, 10Y yield −1.11 | low volatility, falling rates |
| 1 | 464 (34.0%) | CAPE +1.08, employment +0.97, real rate +0.91 | mixed / transitional |
| 2 | 373 (27.3%) | CPI +1.25, dividend yield +1.19, 10Y yield +1.13 | inflationary pressure, rising rates |
| 3 | 165 (12.1%) | yield-spread stress +1.94, unemployment +1.93, GDP −1.58 | crisis: high unemployment, falling GDP |

Regime 3 is the most distinct and the strongest candidate for a "crisis" label. Regime 1 is the largest and most diffuse, consistent with a normal-market baseline.

**Figures** (saved to `pca-regime-analysis/outputs/figures/`):

```
01_scree_plot.png               05_regime_timeline.png
02_pca_2d_regimes.png           06_regime_feature_profiles.png
03_ae_2d_regimes.png            07_regime_agreement_heatmap.png
04_pca_vs_ae_3d.png             08_reconstruction_loss_vs_pca_variance.png
```

### Reproduce

Requires `../DATA LAYER/windows.npz` and `../market-regime-autoencoder/outputs/handoff/clustered_regimes.csv`.

```bash
cd pca-regime-analysis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python pca_analysis.py
```

---

## 4. Trading Strategy

Pipeline in `trading-strategy/`. Tests whether the AE regimes carry economic value via a regime-conditioned long/short rule evaluated strictly out of sample.

### Files

| File | Description |
|---|---|
| `backtest_regimes.py` | Derives position rule from AE regime labels (train only) and backtests on val/test |
| `compare_signal_sets.py` | Trains three AE models (Set A, Set B, all 30 features) and compares Sharpe and IC |
| `requirements.txt` | Dependencies |

### Setup

```bash
cd <repo-root>
pip install -r trading-strategy/requirements.txt
pip install -e market-regime-autoencoder --no-build-isolation
```

### Run

```bash
# Main backtest (needs market-regime-autoencoder outputs)
python trading-strategy/backtest_regimes.py

# Signal set comparison (trains models from scratch, ~2 min)
python trading-strategy/compare_signal_sets.py
```

### Methodology

1. **Forward return** — window `i`'s one-week-forward `_MKT` return is `windows[i+1][-1, mkt_idx]` (the new week added by the next window).
2. **Position rule** — regimes ranked by train-set mean forward return; best regime = long (+1), worst = short (−1), middle two = flat (0). Rule is frozen before touching val/test.
3. **Metrics** — Sharpe annualised (×√52) on StandardScaler-transformed log-returns. To obtain raw-percentage Sharpe, inverse-transform `_MKT` with `DATA LAYER/scaler.pkl`.

### Results

**Main backtest (all 30 features, AE regime labels):**

| Split | Strategy Sharpe | Buy-and-hold Sharpe | IC |
|---|---|---|---|
| Train (in-sample) | 0.23 | 0.00 | 0.03 |
| Validation (OOS) | 0.13 | −0.02 | −0.02 |
| Test (OOS) | −0.28 | 0.07 | −0.04 |

**Signal set comparison:**

| Signal set | Recon train | Recon test | Train Sharpe | Test Sharpe | Test IC |
|---|---|---|---|---|---|
| Set A — macro (10 feat.) | 0.029 | 0.973 | 0.30 | −0.05 | −0.00 |
| Set B — risk/sentiment (10 feat.) | 0.619 | 0.653 | 0.44 | −0.33 | −0.05 |
| Full — all 30 features | 0.327 | 1.208 | 0.36 | −0.23 | −0.03 |

**Verdict** — the strategy is economically coherent in sample (crisis regime precedes the lowest forward returns, low-volatility regime precedes the highest) but fails to generalize on the test set regardless of signal composition. None of the three configurations achieves positive out-of-sample Sharpe. The generalization gap is a structural property of the post-2020 period, not an artifact of feature choice. We do not recommend this regime signal as a stand-alone investment signal in its current form.
