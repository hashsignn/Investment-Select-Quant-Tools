from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ae_regimes.clustering import cluster_latent_space
from ae_regimes.config import TrainingConfig
from ae_regimes.data import load_feature_table, make_sliding_windows, prepare_features
from ae_regimes.model import Autoencoder


def train_autoencoder(features: np.ndarray, config: TrainingConfig) -> tuple[Autoencoder, pd.DataFrame]:
    torch.manual_seed(config.random_state)

    tensor = torch.tensor(features, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=config.batch_size,
        shuffle=True,
    )

    model = Autoencoder(
        input_dim=features.shape[1],
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []

        for (batch,) in loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        losses.append({"epoch": epoch, "loss": float(np.mean(epoch_losses))})

    return model, pd.DataFrame(losses)


def extract_latent_space(model: Autoencoder, features: np.ndarray, dates: pd.Series) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(features, dtype=torch.float32)
        latent = model.encode(tensor).numpy()

    latent_df = pd.DataFrame(latent, columns=[f"z{i + 1}" for i in range(latent.shape[1])])
    latent_df.insert(0, "date", dates.dt.strftime("%Y-%m-%d"))
    return latent_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train autoencoder and cluster market regimes.")
    parser.add_argument("--input", required=True, help="Path to processed feature CSV.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated outputs.")
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--hidden-dim", type=int, default=TrainingConfig.hidden_dim)
    parser.add_argument("--latent-dim", type=int, default=TrainingConfig.latent_dim)
    parser.add_argument("--window-size", type=int, default=TrainingConfig.window_size)
    parser.add_argument("--train-fraction", type=float, default=TrainingConfig.train_fraction)
    parser.add_argument("--clusters", type=int, default=TrainingConfig.clusters)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--random-state", type=int, default=TrainingConfig.random_state)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        window_size=args.window_size,
        train_fraction=args.train_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        clusters=args.clusters,
        random_state=args.random_state,
    )

    df = load_feature_table(args.input)
    prepared = prepare_features(df, train_fraction=config.train_fraction)
    train_windows, _ = make_sliding_windows(
        prepared.train_scaled_features,
        prepared.dates.iloc[: len(prepared.train_scaled_features)],
        config.window_size,
    )
    all_windows, window_dates = make_sliding_windows(
        prepared.scaled_features,
        prepared.dates,
        config.window_size,
    )

    model, loss_history = train_autoencoder(train_windows, config)
    latent_df = extract_latent_space(model, all_windows, window_dates)
    clustered_df = cluster_latent_space(latent_df, config.clusters, config.random_state)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_history.to_csv(output_dir / "training_loss.csv", index=False)
    latent_df.to_csv(output_dir / "latent_space.csv", index=False)
    clustered_df.to_csv(output_dir / "clustered_regimes.csv", index=False)

    print(f"Trained autoencoder with {train_windows.shape[0]} training windows.")
    print(f"Window size: {config.window_size} periods.")
    print(f"Input features: {', '.join(prepared.feature_names)}")
    print(f"Final training loss: {loss_history['loss'].iloc[-1]:.6f}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
