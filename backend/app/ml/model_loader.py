"""Load the frozen checkpoint, its normalization statistics and validation thresholds once.

Reuses the training package's own loaders (`load_checkpoint` with weights_only=True,
`load_model_from_checkpoint`) and the production `PTBXLPreprocessor` statistics; nothing is
re-implemented or recomputed here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from cardiomamba.data.labels import CLASS_NAMES
from cardiomamba.data.preprocessing import PREPROCESSING_VERSION, NormalizationStats
from cardiomamba.models import CardioMamba, count_parameters
from cardiomamba.training.checkpoint import load_checkpoint
from cardiomamba.training.engine import load_model_from_checkpoint


class ModelLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelRuntime:
    model: CardioMamba
    stats: NormalizationStats
    thresholds: dict[str, float]
    device: torch.device
    info: dict


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_runtime(checkpoint_path: Path, thresholds_path: Path, test_metrics_path: Path,
                 repo_root: Path, device: str) -> ModelRuntime:
    for p in (checkpoint_path, thresholds_path):
        if not p.is_file():
            raise ModelLoadError(f"Required artifact not found: {p}")
    ckpt = load_checkpoint(checkpoint_path)
    if ckpt["class_names"] != list(CLASS_NAMES):
        raise ModelLoadError("Checkpoint class order does not match the project class order")
    if ckpt["preprocessing_version"] != PREPROCESSING_VERSION:
        raise ModelLoadError(f"Checkpoint expects {ckpt['preprocessing_version']}, this code "
                             f"implements {PREPROCESSING_VERSION}")
    stats = NormalizationStats.load(repo_root / ckpt["normalization_stats"])  # frozen, train-only
    thr_file = json.loads(thresholds_path.read_text())
    if thr_file.get("checkpoint_epoch") != ckpt["epoch"]:
        raise ModelLoadError("thresholds.json was not calibrated for this checkpoint")
    thresholds = {c: float(thr_file["per_class"][c]["threshold"]) for c in CLASS_NAMES}

    dev = _resolve_device(device)
    model = load_model_from_checkpoint(ckpt, dev)            # eval mode, no checkpointing
    test = None
    if test_metrics_path.is_file():
        t = json.loads(test_metrics_path.read_text())
        m = t["metrics_frozen_thresholds"]
        test = {"split": t["split"], "n_records": t["n_records"],
                "macro_auroc": m["macro_auroc"], "macro_auprc": m["macro_auprc"],
                "macro_f1": m["macro_f1"], "micro_f1": m["micro_f1"]}
    info = {
        "name": "CardioMamba",
        "version": "0.1.0",
        "architecture": "Bidirectional Mamba-1 / S6 (pure PyTorch), Conv1d patch stem, "
                        "4 blocks, RMSNorm, mean pooling, linear head",
        "task": "ptbxl-superdiagnostic-5 (multi-label)",
        "dataset": "PTB-XL 1.0.3",
        "class_order": list(CLASS_NAMES),
        "parameters": count_parameters(model),
        "model_config": ckpt["model_config"],
        "checkpoint_epoch": ckpt["epoch"],
        "best_val_macro_auroc": ckpt["best_val_macro_auroc"],
        "preprocessing_version": ckpt["preprocessing_version"],
        "thresholds": thresholds,
        "thresholds_source": "validation fold 9 (per-class F1 maximisation), frozen",
        "input": {"leads": 12, "duration_s": 10, "sampling_rate_hz": 100, "shape": [1000, 12]},
        "device": str(dev),
        "precision": "fp32",
        "test_metrics": test,
    }
    return ModelRuntime(model, stats, thresholds, dev, info)
