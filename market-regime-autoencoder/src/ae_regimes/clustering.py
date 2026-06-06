from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans


def latent_columns(latent_df: pd.DataFrame) -> list[str]:
    cols = [col for col in latent_df.columns if col.startswith("z")]
    if not cols:
        raise ValueError("Latent dataframe must include columns named z1, z2, ...")
    return cols


def cluster_latent_space(
    latent_df: pd.DataFrame,
    clusters: int,
    random_state: int,
) -> pd.DataFrame:
    latent_cols = latent_columns(latent_df)

    model = KMeans(n_clusters=clusters, n_init="auto", random_state=random_state)
    result = latent_df.copy()
    result["regime"] = model.fit_predict(result[latent_cols])
    return result


def cluster_from_train_split(
    latent_df: pd.DataFrame,
    clusters: int,
    random_state: int,
    split_col: str = "split",
    train_value: str = "train",
) -> pd.DataFrame:
    if split_col not in latent_df.columns:
        raise ValueError(f"Latent dataframe must include a '{split_col}' column.")

    train_mask = latent_df[split_col] == train_value
    if not train_mask.any():
        raise ValueError(f"No rows found for train split value: {train_value}")

    latent_cols = latent_columns(latent_df)
    model = KMeans(n_clusters=clusters, n_init="auto", random_state=random_state)
    model.fit(latent_df.loc[train_mask, latent_cols])

    result = latent_df.copy()
    result["regime"] = model.predict(result[latent_cols])
    return result
