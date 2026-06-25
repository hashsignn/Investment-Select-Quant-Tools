# Market Regime Autoencoder

This project covers Person 2's part of the assignment:

- train an autoencoder on preprocessed financial time-series features,
- extract a 16-dimensional latent representation,
- cluster the latent vectors into market regimes,
- export results for visualization and interpretation.

The pipeline is intentionally independent from the final OLAT dataset. Until Person 1 provides the cleaned dataset, use the synthetic sample data generator.

## For PCA / Visualization Person

Use this file for downstream analysis:

```text
market-regime-autoencoder/outputs/handoff/clustered_regimes.csv
```

It contains:

```text
split,window_index,date,z1,z2,...,z16,regime
```

Column meanings:

- `split`: original time split from the data handoff (`train`, `val`, `test`),
- `window_index`: index of the 26-week window inside its split,
- `date`: end date of the 26-week window,
- `z1` through `z16`: 16-dimensional autoencoder latent representation,
- `regime`: KMeans regime label assigned from the latent space.

If only the latent representation is needed, use:

```text
market-regime-autoencoder/outputs/handoff/latent_space.csv
```

Diagnostics:

```text
market-regime-autoencoder/outputs/handoff/training_loss.csv
market-regime-autoencoder/outputs/handoff/split_reconstruction_loss.csv
```

Current reconstruction losses:

```text
train = 0.299371
val   = 0.610532
test  = 1.204732
```

Current regime counts:

```text
regime 0 = 379
regime 1 = 666
regime 2 = 345
regime 3 = 521
```

The test reconstruction loss is higher because the test period is the most recent sample, including 2020-2026 market conditions. This should be discussed as a possible regime shift rather than treated as classification accuracy.

## Model Used

The model is a fully connected autoencoder implemented in PyTorch:

- input: one flattened 26-week window with 30 features, so `26 x 30 = 780` inputs,
- encoder: `Linear(780, 256) -> ReLU -> Linear(256, 64) -> ReLU -> Linear(64, 16)`,
- latent space: 16 dimensions, `z1` through `z16`,
- decoder: `Linear(16, 64) -> ReLU -> Linear(64, 256) -> ReLU -> Linear(256, 780)`,
- objective: mean squared reconstruction error,
- optimizer: Adam,
- maximum epochs for the handoff run: 200,
- model selection: best validation-loss epoch with early stopping,
- regularization: Adam weight decay of `5e-4`,
- clustering: KMeans with 4 regimes.

KMeans is fitted only on the training latent vectors and then used to assign regimes to validation and test. This avoids using validation/test data to define the clusters.

We use a 16-dimensional latent space (widened from an earlier 3-dimensional architecture) to prevent representation mode collapse. The root data README mentions 8-16 dimensions as a sensible reconstruction-oriented range, and 16 dimensions gives the network sufficient capacity to accurately differentiate market regimes out-of-sample. For interpretability and visualization, we subsequently apply t-SNE to project the 16-D representations down to 2-D and 3-D.

## Expected Data Contract

Person 1 should provide a CSV in `data/processed/` with this shape:

```text
date,feature_1,feature_2,feature_3,...
2000-01-31,0.012,0.034,-0.008,...
2000-02-29,-0.021,0.045,0.003,...
```

Requirements:

- one row per period,
- consistent frequency, preferably monthly or quarterly,
- `date` column parseable as a date,
- all feature columns numeric,
- missing values handled before training or clearly documented,
- features should be comparable in frequency and scale.

Examples of useful features:

- month-to-month index returns,
- rolling volatility computed on the same frequency,
- factor returns,
- macro variables transformed to monthly or quarterly changes.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

## Quick Run With Synthetic Data

```bash
python scripts/make_sample_data.py
python -m ae_regimes.train --input data/processed/sample_monthly_features.csv --epochs 200 --window-size 4
```

Outputs are written to `outputs/`:

- `latent_space.csv`: date plus 3 latent dimensions,
- `clustered_regimes.csv`: latent dimensions plus cluster labels,
- `training_loss.csv`: epoch loss history.

## Run With Final Data

After Person 1 adds the final processed CSV:

```bash
python -m ae_regimes.train --input data/processed/final_features.csv --epochs 300 --clusters 4
```

## Run With Person 1 Handoff

If Person 1 provides the data layer files `windows.npz`, `splits.npz`, `schema.json`, `metadata.json`, and `scaler.pkl`, train directly from the handoff folder:

```bash
python -m ae_regimes.train_handoff \
  --handoff-dir "../DATA LAYER" \
  --epochs 200 \
  --early-stopping-patience 40 \
  --clusters 4 \
  --output-dir outputs/handoff
```

This uses the already preprocessed 26-week windows:

- train: used for fitting the autoencoder,
- validation/test: used only for reconstruction diagnostics and latent/regime extraction,
- output: `latent_space.csv`, `clustered_regimes.csv`, `training_loss.csv`, `split_reconstruction_loss.csv`.

Expected terminal output shape summary:

```text
Trained on handoff train windows: (1365, 780)
Validation windows: (273, 780)
Test windows: (273, 780)
```

## Convert The Course Excel File

The lecture example uses `market_data.xlsx`, sheet `US`, and columns `_TY`, `ED`, `_MKT`. To convert it to monthly returns:

```bash
python scripts/prepare_course_market_data.py \
  --input /Users/elenetsaouse/Downloads/ae_vector_quantized/market_data.xlsx \
  --frequency M \
  --transform pct_change \
  --output data/processed/course_market_features.csv
```

Then train:

```bash
python -m ae_regimes.train --input data/processed/course_market_features.csv --epochs 200 --window-size 4 --clusters 4
```

Use `--frequency Q` for quarterly data if the final group dataset is quarterly.

## Project Boundary

This folder focuses on the autoencoder and clustering. PCA comparison and final regime visualization can use the exported `latent_space.csv` and `clustered_regimes.csv`.
