"""Checkpoint save/load (atomic writes; payload restricted to tensors and primitives so that
checkpoints load with `torch.load(weights_only=True)`)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)          # never leaves a half-written best.pt / last.pt


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=True)


def git_state() -> dict[str, Any]:
    """Commit hash and dirty flag of the working tree (None if git is unavailable)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
