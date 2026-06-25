"""
Latent-dimension ablation for the market-regime autoencoder.

Retrains the autoencoder at several bottleneck widths under identical settings
and reports, for each width:
  - train and test reconstruction MSE (best-validation checkpoint),
  - the number of the four K-Means regimes actually populated per split.

This reproduces Table~\ref{tab:ablation} in the paper. It shows that (1) the
16-dimensional bottleneck minimizes out-of-sample reconstruction loss, and
(2) out-of-sample regime coverage does NOT improve with capacity, i.e. the
validation/test regime collapse is a property of the data (regime shift), not of
the bottleneck width.

Run from the repo root (needs the Person 1 handoff windows):
    python market-regime-autoencoder/scripts/latent_dim_ablation.py
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- settings (mirror ae_regimes.train_handoff defaults) ---
WINDOWS = Path("DATA LAYER/windows.npz")
LATENT_DIMS = [2, 3, 8, 16, 32]
N_CLUSTERS = 4
SEED = 42
EPOCHS = 200
PATIENCE = 40
BATCH = 16
LR = 1e-3
WEIGHT_DECAY = 5e-4


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


def flatten(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(arr.shape[0], -1).astype("float32")


def train_one(latent_dim: int, splits: dict[str, np.ndarray]):
    torch.manual_seed(SEED)
    x_tr = torch.tensor(splits["train"])
    x_va = torch.tensor(splits["val"])
    x_te = torch.tensor(splits["test"])

    model = Autoencoder(x_tr.shape[1], latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.MSELoss()
    loader = DataLoader(TensorDataset(x_tr), batch_size=BATCH, shuffle=True)

    best, best_state, stale = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train()
        for (batch,) in loader:
            opt.zero_grad()
            loss = crit(model(batch), batch)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(x_va), x_va).item()
        if vl < best:
            best, best_state, stale = vl, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        mse_tr = crit(model(x_tr), x_tr).item()
        mse_te = crit(model(x_te), x_te).item()
        z = {k: model.encode(torch.tensor(v)).numpy() for k, v in splits.items()}

    scaler = StandardScaler().fit(z["train"])
    km = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=SEED).fit(
        scaler.transform(z["train"])
    )
    coverage = {
        k: len(np.unique(km.predict(scaler.transform(v)))) for k, v in z.items()
    }
    return mse_tr, mse_te, coverage


def main() -> None:
    raw = np.load(WINDOWS)
    splits = {s: flatten(raw[s]) for s in ("train", "val", "test")}

    print(f"{'latent':>6} {'train_MSE':>10} {'test_MSE':>9}  regimes (train/val/test)")
    for d in LATENT_DIMS:
        mse_tr, mse_te, cov = train_one(d, splits)
        print(
            f"{d:>6} {mse_tr:>10.3f} {mse_te:>9.3f}"
            f"  {cov['train']}/{cov['val']}/{cov['test']}"
        )


if __name__ == "__main__":
    main()
