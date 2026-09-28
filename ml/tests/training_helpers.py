"""Shared fixtures for training tests: a tiny learnable synthetic dataset and model config."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from cardiomamba.models import CardioMambaConfig
from cardiomamba.training import TrainConfig

TINY_MODEL = CardioMambaConfig(input_length=64, d_model=16, n_layers=1, d_state=4, dt_rank=2,
                               dropout=0.1)


class SyntheticECG(Dataset):
    """[64, 12] signals; class c is positive when lead c's mean is > 0 (learnable)."""

    def __init__(self, n: int, seed: int = 0, nan_at: int | None = None) -> None:
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 64, 12, generator=g)
        self.x[:, :, :5] += torch.randn(n, 1, 5, generator=g)       # per-record lead offsets
        self.y = (self.x[:, :, :5].mean(dim=1) > 0).float()
        if nan_at is not None:
            self.x[nan_at, 0, 0] = float("nan")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> dict:
        return {"signal": self.x[i], "target": self.y[i], "ecg_id": i + 1}


def tiny_config(**overrides) -> TrainConfig:
    base = TrainConfig(epochs=3, batch_size=8, num_workers=0, device="cpu",
                       mixed_precision="fp32", activation_checkpointing=False,
                       log_every_steps=0, learning_rate=3e-3, min_learning_rate=1e-5)
    return base.with_overrides(**overrides)
