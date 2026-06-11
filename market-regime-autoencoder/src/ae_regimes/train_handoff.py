from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ae_regimes.clustering import cluster_from_train_split
from ae_regimes.config import TrainingConfig
from ae_regimes.model import Autoencoder
from ae_regimes.train import extract_latent_space


def load_handoff_dates(handoff_dir: Path) -> dict[str, pd.Series] | None:
    schema_path = handoff_dir / "schema.json"
    metadata_path = handoff_dir / "metadata.json"
    source_path = handoff_dir / "market_data.xlsx"
    if not schema_path.exists() or not metadata_path.exists() or not source_path.exists():
        return None

    schema = json.loads(schema_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    window_size = int(metadata["window"]["size"])

    df = pd.read_excel(source_path, sheet_name=schema["sheet"], engine="openpyxl")
    df.columns = [col.strip() for col in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    for col in schema["return_cols_log"]:
        df[col] = np.log(df[col]).diff()
    df = df.dropna().reset_index(drop=True)

    dates = df["Date"].reset_index(drop=True)
    n = len(dates)
    train_frac = float(metadata["split"]["train"])
    val_frac = float(metadata["split"]["val"])
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))

    split_dates = {
        "train": dates.iloc[:i_train].reset_index(drop=True),
        "val": dates.iloc[i_train:i_val].reset_index(drop=True),
        "test": dates.iloc[i_val:].reset_index(drop=True),
    }
    return {
        split: values.iloc[window_size - 1 :].reset_index(drop=True)
        for split, values in split_dates.items()
    }


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    if windows.ndim != 3:
        raise ValueError("Expected windows with shape (n_windows, window_size, n_features).")
    return windows.reshape(windows.shape[0], -1).astype("float32")


def reconstruction_loss(model: torch.nn.Module, features: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(features, dtype=torch.float32)
        reconstructed = model(tensor)
        return float(torch.mean((reconstructed - tensor) ** 2).item())


def train_autoencoder_with_validation(
    train_features: np.ndarray,
    val_features: np.ndarray,
    config: TrainingConfig,
    weight_decay: float,
    early_stopping_patience: int,
) -> tuple[Autoencoder, pd.DataFrame]:
    torch.manual_seed(config.random_state)

    train_tensor = torch.tensor(train_features, dtype=torch.float32)
    val_tensor = torch.tensor(val_features, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=config.batch_size,
        shuffle=True,
    )

    model = Autoencoder(
        input_dim=train_features.shape[1],
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        batch_losses = []

        for (batch,) in loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_reconstructed = model(val_tensor)
            val_loss = float(criterion(val_reconstructed, val_tensor).item())

        train_loss = float(np.mean(batch_losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "is_best": val_loss < best_val_loss,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if early_stopping_patience > 0 and stale_epochs >= early_stopping_patience:
            break

    model.load_state_dict(best_state)
    history_df = pd.DataFrame(history)
    history_df["best_epoch"] = best_epoch
    return model, history_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train autoencoder directly from Person 1 handoff windows.npz."
    )
    parser.add_argument("--handoff-dir", default="../DATA LAYER", help="Path to handoff folder.")
    parser.add_argument("--output-dir", default="outputs/handoff", help="Directory for outputs.")
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=TrainingConfig.latent_dim)
    parser.add_argument("--clusters", type=int, default=TrainingConfig.clusters)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument("--random-state", type=int, default=TrainingConfig.random_state)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handoff_dir = Path(args.handoff_dir)
    windows_path = handoff_dir / "windows.npz"
    if not windows_path.exists():
        raise FileNotFoundError(f"Could not find {windows_path}")

    windows = dict(np.load(windows_path))
    required = {"train", "val", "test"}
    missing = required.difference(windows)
    if missing:
        raise ValueError(f"windows.npz is missing splits: {sorted(missing)}")

    features = {split: flatten_windows(windows[split]) for split in ["train", "val", "test"]}
    config = TrainingConfig(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        clusters=args.clusters,
        random_state=args.random_state,
    )

    model, loss_history = train_autoencoder_with_validation(
        features["train"],
        features["val"],
        config,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
    )
    split_dates = load_handoff_dates(handoff_dir)

    latent_parts = []
    for split in ["train", "val", "test"]:
        if split_dates and len(split_dates[split]) == len(features[split]):
            dates = split_dates[split]
        else:
            dates = pd.Series([f"{split}_{i}" for i in range(len(features[split]))])

        latent = extract_latent_space(model, features[split], dates)
        latent.insert(0, "split", split)
        latent.insert(1, "window_index", range(len(latent)))
        latent_parts.append(latent)

    latent_df = pd.concat(latent_parts, ignore_index=True)
    clustered_df = cluster_from_train_split(latent_df, config.clusters, config.random_state)

    losses = {
        split: reconstruction_loss(model, values)
        for split, values in features.items()
    }
    eval_df = pd.DataFrame(
        [{"split": split, "reconstruction_loss": loss} for split, loss in losses.items()]
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_history.to_csv(output_dir / "training_loss.csv", index=False)
    eval_df.to_csv(output_dir / "split_reconstruction_loss.csv", index=False)
    latent_df.to_csv(output_dir / "latent_space.csv", index=False)
    clustered_df.to_csv(output_dir / "clustered_regimes.csv", index=False)

    print(f"Trained on handoff train windows: {features['train'].shape}")
    print(f"Validation windows: {features['val'].shape}")
    print(f"Test windows: {features['test'].shape}")
    best_epoch = int(loss_history["best_epoch"].iloc[-1])
    best_val_loss = float(loss_history.loc[loss_history["epoch"] == best_epoch, "val_loss"].iloc[0])
    print(f"Best validation epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("Reconstruction loss by split:")
    for split, loss in losses.items():
        print(f"  {split}: {loss:.6f}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
