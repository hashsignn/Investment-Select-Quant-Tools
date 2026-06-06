# Market Regime Autoencoder

This project covers Person 2's part of the assignment:

- train an autoencoder on preprocessed financial time-series features,
- extract a 3-dimensional latent representation,
- cluster the latent vectors into market regimes,
- export results for visualization and interpretation.

The pipeline is intentionally independent from the final OLAT dataset. Until Person 1 provides the cleaned dataset, use the synthetic sample data generator.

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
pip install -e .
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
  --clusters 4 \
  --output-dir outputs/handoff
```

This uses the already preprocessed 26-week windows:

- train: used for fitting the autoencoder,
- validation/test: used only for reconstruction diagnostics and latent/regime extraction,
- output: `latent_space.csv`, `clustered_regimes.csv`, `training_loss.csv`, `split_reconstruction_loss.csv`.

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
