"""Step-based learning-rate schedule: linear warmup, then cosine decay to a floor.

Steps are *optimiser* steps: with gradient accumulation, one step = `accumulation` micro-
batches. The schedule is a LambdaLR, so its position is restored exactly from its state dict
on resume.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def steps_per_epoch(num_microbatches: int, accumulation: int) -> int:
    """Optimiser steps per epoch; a final partial accumulation group still takes a step."""
    return math.ceil(num_microbatches / accumulation)


def warmup_steps_for(total_steps: int, warmup_ratio: float) -> int:
    return max(1, round(warmup_ratio * total_steps)) if warmup_ratio > 0 else 0


def lr_factor(step: int, total_steps: int, warmup_steps: int, min_ratio: float) -> float:
    """Multiplier of the base LR before optimiser step `step + 1` (step counts from 0).

    step < warmup:  (step + 1) / warmup            (reaches 1.0 at the last warmup step)
    afterwards:     min_ratio + (1 - min_ratio) * 0.5 * (1 + cos(pi * progress)),
                    progress = (step - warmup) / (total - warmup) in [0, 1]
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(optimizer: Optimizer, total_steps: int, warmup_ratio: float,
                    base_lr: float, min_lr: float) -> tuple[LambdaLR, int]:
    warmup = warmup_steps_for(total_steps, warmup_ratio)
    min_ratio = min_lr / base_lr
    scheduler = LambdaLR(optimizer, lambda s: lr_factor(s, total_steps, warmup, min_ratio))
    return scheduler, warmup
