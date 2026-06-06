from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    latent_dim: int = 3
    hidden_dim: int = 16
    window_size: int = 4
    train_fraction: float = 0.8
    epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 1e-3
    clusters: int = 4
    random_state: int = 42
