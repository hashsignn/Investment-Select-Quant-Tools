from __future__ import annotations

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
import config

import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, config.LATENT_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.LATENT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

