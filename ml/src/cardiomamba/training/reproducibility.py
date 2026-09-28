"""Seeding, DataLoader worker seeding and RNG-state capture for exact resume.

Determinism notes: cuDNN autotuning is disabled and deterministic cuDNN kernels requested.
`torch.use_deterministic_algorithms(True)` is NOT enabled (some CUDA kernels used by autograd
have no deterministic variant and it would require CUBLAS_WORKSPACE_CONFIG), so GPU runs are
reproducible up to floating-point reduction-order effects; CPU runs are verified
bit-identical in the tests.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)             # also seeds all CUDA devices
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Derive Python/NumPy seeds in each worker from the torch seed the DataLoader assigned."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng_state() -> dict:
    """RNG states as tensors/primitives only (so checkpoints load with weights_only=True)."""
    np_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {"name": np_state[0], "keys": torch.from_numpy(np_state[1].copy()),
                  "pos": int(np_state[2]), "has_gauss": int(np_state[3]),
                  "cached_gaussian": float(np_state[4])},
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(_to_tuple(state["python"]))
    n = state["numpy"]
    np.random.set_state((n["name"], n["keys"].numpy(), n["pos"], n["has_gauss"],
                         n["cached_gaussian"]))
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _to_tuple(value):
    return tuple(_to_tuple(v) for v in value) if isinstance(value, list | tuple) else value
