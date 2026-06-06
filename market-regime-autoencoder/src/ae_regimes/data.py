from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class PreparedData:
    dates: pd.Series
    feature_names: list[str]
    scaled_features: np.ndarray
    train_scaled_features: np.ndarray
    scaler: StandardScaler


def load_feature_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Input CSV must contain a 'date' column.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = [col for col in df.columns if col != "date"]
    if not feature_cols:
        raise ValueError("Input CSV must contain at least one numeric feature column.")

    non_numeric = [
        col for col in feature_cols if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise ValueError(f"Feature columns must be numeric: {non_numeric}")

    if df[feature_cols].isna().any().any():
        raise ValueError(
            "Input data contains missing feature values. Handle missing values in preprocessing."
        )

    return df


def prepare_features(df: pd.DataFrame, train_fraction: float = 0.8) -> PreparedData:
    feature_names = [col for col in df.columns if col != "date"]
    split_idx = int(len(df) * train_fraction)
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("train_fraction must leave observations for both train and test periods.")

    scaler = StandardScaler()
    train_features = df.loc[: split_idx - 1, feature_names]
    all_features = df[feature_names]

    train_scaled = scaler.fit_transform(train_features)
    scaled = scaler.transform(all_features)

    return PreparedData(
        dates=df["date"],
        feature_names=feature_names,
        scaled_features=scaled,
        train_scaled_features=train_scaled,
        scaler=scaler,
    )


def make_sliding_windows(
    features: np.ndarray,
    dates: pd.Series,
    window_size: int,
) -> tuple[np.ndarray, pd.Series]:
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    if len(features) < window_size:
        raise ValueError("Not enough observations for the requested window_size.")

    windows = []
    for start in range(0, len(features) - window_size + 1):
        end = start + window_size
        windows.append(features[start:end].reshape(-1))

    window_dates = dates.iloc[window_size - 1 :].reset_index(drop=True)
    return np.asarray(windows, dtype=np.float32), window_dates
