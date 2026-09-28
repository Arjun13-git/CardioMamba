"""Training configuration (dataclass, loadable from / savable to YAML).

The model itself is configured by `cardiomamba.models.CardioMambaConfig` (locked defaults);
this config only covers the optimisation/run protocol. `activation_checkpointing` is passed
through to the model's `checkpoint_blocks` switch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

import yaml

Precision = Literal["auto", "bf16", "fp32"]

# Changing any of these between an interrupted run and its resumption would silently mix two
# different experiments, so `resume` refuses to proceed if they differ.
RESUME_LOCKED_FIELDS = (
    "seed", "epochs", "batch_size", "gradient_accumulation_steps", "learning_rate",
    "weight_decay", "adam_betas", "warmup_ratio", "min_learning_rate", "gradient_clip_norm",
    "early_stopping_patience", "mixed_precision", "activation_checkpointing",
    "dataset_dir", "normalization_stats", "max_train_records", "max_val_records",
)


@dataclass(frozen=True)
class TrainConfig:
    # reproducibility
    seed: int = 42
    # schedule / duration
    epochs: int = 50
    early_stopping_patience: int = 10
    # batches (effective batch = batch_size * gradient_accumulation_steps)
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    # optimiser: AdamW; no weight decay on biases, norm weights, A_log, D
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    adam_betas: tuple[float, float] = (0.9, 0.999)
    # LR: linear warmup over warmup_ratio of optimiser steps, then cosine to min_learning_rate
    warmup_ratio: float = 0.05
    min_learning_rate: float = 1e-5
    gradient_clip_norm: float = 1.0
    # hardware
    device: str = "auto"                     # "auto" | "cuda" | "cpu"
    mixed_precision: Precision = "auto"      # auto: bf16 if CUDA supports it, else fp32
    activation_checkpointing: bool = True
    num_workers: int = 4
    # data (paths relative to the repository root unless absolute)
    dataset_dir: str = "ml/data/raw/ptb-xl-1.0.3"
    normalization_stats: str = "ml/configs/normalization_stats_prep-v1.json"
    # optional deterministic subsets for smoke runs only (None = full split)
    max_train_records: int | None = None
    max_val_records: int | None = None
    # outputs
    output_dir: str = "outputs/cardiomamba"
    run_name: str | None = None              # default: timestamp
    log_every_steps: int = 50
    # threshold calibration grid (validation only)
    threshold_grid_start: float = 0.01
    threshold_grid_stop: float = 0.99
    threshold_grid_step: float = 0.01

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1 or self.epochs < 1:
            raise ValueError("batch_size, gradient_accumulation_steps and epochs must be >= 1")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.mixed_precision not in ("auto", "bf16", "fp32"):
            raise ValueError(f"Unknown mixed_precision {self.mixed_precision!r}")
        object.__setattr__(self, "adam_betas", tuple(self.adam_betas))

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["adam_betas"] = list(self.adam_betas)
        return d

    def save_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown training config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})

    def with_overrides(self, **overrides: Any) -> TrainConfig:
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})


def resume_mismatches(current: TrainConfig, saved: dict[str, Any]) -> dict[str, tuple]:
    """Fields in RESUME_LOCKED_FIELDS whose value differs from the checkpoint's config."""
    now = current.to_dict()
    return {k: (saved.get(k), now[k]) for k in RESUME_LOCKED_FIELDS if saved.get(k) != now[k]}
