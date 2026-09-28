"""Training loop, validation, early stopping, checkpointing and threshold calibration.

Protocol (see System-Design/09-training-evaluation):
  * loss: BCEWithLogitsLoss on raw logits (sigmoid only for metrics);
  * AdamW with decoupled weight decay (not on biases, norm weights, A_log, D);
  * step-based linear warmup + cosine LR, stepped once per optimiser step;
  * gradient accumulation, clipping (max-norm) before every optimiser step; a non-finite
    gradient norm aborts the run (it is never skipped silently);
  * model selection and early stopping on validation macro AUROC only;
  * per-class thresholds calibrated on validation predictions of the best checkpoint.
The test split is not an input to anything in this module.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import platform
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cardiomamba.data.labels import CLASS_NAMES, LABEL_MAPPING_VERSION
from cardiomamba.data.preprocessing import PREPROCESSING_VERSION
from cardiomamba.models import CardioMamba, CardioMambaConfig, count_parameters
from cardiomamba.training.checkpoint import git_state, load_checkpoint, save_checkpoint
from cardiomamba.training.config import TrainConfig, resume_mismatches
from cardiomamba.training.data import make_loader
from cardiomamba.training.metrics import compute_metrics
from cardiomamba.training.reproducibility import (
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)
from cardiomamba.training.schedule import build_scheduler, steps_per_epoch
from cardiomamba.training.thresholds import calibrate_thresholds, threshold_grid

SELECTION_METRIC = "macro_auroc"
CSV_FIELDS = (
    ["epoch", "global_step", "learning_rate", "train_loss", "val_loss", "val_macro_auroc",
     "val_macro_auprc", "val_macro_f1", "val_micro_f1"]
    + [f"val_auroc_{c}" for c in CLASS_NAMES] + [f"val_auprc_{c}" for c in CLASS_NAMES]
    + ["epoch_time_s", "peak_gpu_mem_mib", "is_best"]
)


class NonFiniteError(RuntimeError):
    """Raised when the loss or gradients become NaN/Inf."""


class ConfigMismatchError(ValueError):
    """Raised when a resume would mix two different experiment configurations."""


@dataclass
class RunState:
    epoch: int = 0                          # last completed epoch (0 = none)
    global_step: int = 0                    # optimiser steps taken
    best_metric: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement: int = 0
    finished: bool = False                  # max epochs reached or early-stopped
    history: list[dict] = field(default_factory=list)


@dataclass
class TrainingResult:
    run_dir: Path
    state: RunState
    thresholds: dict | None


# --------------------------------------------------------------------------- helpers

def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(requested)


def resolve_precision(requested: str, device: torch.device) -> str:
    """'bf16' only on CUDA devices that support it; otherwise 'fp32' (no FP16 path)."""
    bf16_ok = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if requested == "bf16" and not bf16_ok:
        raise RuntimeError("bf16 requested but not supported on this device")
    return "bf16" if requested in ("auto", "bf16") and bf16_ok else "fp32"


def autocast_ctx(device: torch.device, precision: str) -> torch.autocast:
    """bf16 autocast for matmuls/convs; the model's selective scan always stays FP32."""
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=precision == "bf16")


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # biases / norm weights (1-D) and parameters tagged by the model (A_log, D)
        (no_decay if p.ndim < 2 or getattr(p, "_no_weight_decay", False) else decay).append(p)
    groups = [{"params": decay, "weight_decay": config.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=config.learning_rate, betas=config.adam_betas)


def loss_fn() -> nn.Module:
    return nn.BCEWithLogitsLoss()           # expects logits; never apply sigmoid before it


def _setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"cardiomamba.train.{run_dir.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(run_dir / "train.log"), logging.StreamHandler()):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def _fmt(v: float | None, digits: int = 4) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


# --------------------------------------------------------------------------- evaluation

@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device,
            precision: str) -> dict[str, Any]:
    """Validation pass: mean BCE loss, sigmoid probabilities, targets, ecg_ids (numpy)."""
    model.eval()
    criterion = loss_fn()
    loss_sum = torch.zeros((), device=device)
    n = 0
    probs, targets, ids = [], [], []
    for batch in loader:
        x = batch["signal"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        with autocast_ctx(device, precision):
            logits = model(x)
        loss_sum += criterion(logits.float(), y) * x.shape[0]
        n += x.shape[0]
        probs.append(torch.sigmoid(logits.float()))
        targets.append(y)
        ids.append(batch["ecg_id"])
    return {
        "loss": (loss_sum / max(n, 1)).item(),
        "probs": torch.cat(probs).cpu().numpy(),
        "targets": torch.cat(targets).cpu().numpy(),
        "ecg_ids": torch.cat(ids).numpy(),
    }


# --------------------------------------------------------------------------- training

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler, config: TrainConfig,
                    device: torch.device, precision: str, state: RunState,
                    logger: logging.Logger) -> float:
    """One pass over the training loader; returns the sample-weighted mean training loss."""
    model.train()
    criterion = loss_fn()
    accum = config.gradient_accumulation_steps
    n_micro = len(loader)
    spe = steps_per_epoch(n_micro, accum)
    last_group = n_micro - accum * (spe - 1)             # size of the final (partial) group
    params = [p for p in model.parameters() if p.requires_grad]
    loss_sum = torch.zeros((), device=device)
    n_seen = 0
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        x = batch["signal"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        with autocast_ctx(device, precision):
            logits = model(x)
        loss = criterion(logits.float(), y)
        group_len = accum if i // accum < spe - 1 else last_group
        (loss / group_len).backward()                     # mean over the accumulation group
        loss_sum += loss.detach() * x.shape[0]
        n_seen += x.shape[0]
        if (i + 1) % accum == 0 or i + 1 == n_micro:
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    params, config.gradient_clip_norm, error_if_nonfinite=True)
            except RuntimeError as exc:
                raise NonFiniteError(
                    f"Non-finite gradient norm at epoch {state.epoch + 1}, optimiser step "
                    f"{state.global_step + 1} (last micro-batch loss {loss.item()})") from exc
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state.global_step += 1
            if config.log_every_steps and state.global_step % config.log_every_steps == 0:
                logger.info("step %d | loss %.4f | grad_norm %.3f | lr %.3e", state.global_step,
                            (loss_sum / n_seen).item(), grad_norm.item(),
                            optimizer.param_groups[0]["lr"])
    mean_loss = (loss_sum / max(n_seen, 1)).item()
    if not math.isfinite(mean_loss):
        raise NonFiniteError(f"Non-finite training loss in epoch {state.epoch + 1}")
    return mean_loss


def _checkpoint_payload(model, optimizer, scheduler, state: RunState, config: TrainConfig,
                        model_config: CardioMambaConfig, precision: str,
                        extra: dict) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": state.epoch,
        "global_step": state.global_step,
        "best_val_macro_auroc": state.best_metric,
        "best_epoch": state.best_epoch,
        "epochs_without_improvement": state.epochs_without_improvement,
        "finished": state.finished,
        "history": state.history,
        "train_config": config.to_dict(),
        "model_config": model_config.to_dict(),
        "random_seed": config.seed,
        "precision": precision,
        "amp_scaler_state_dict": None,          # bf16 needs no GradScaler
        "rng_state": capture_rng_state(),
        "class_names": list(CLASS_NAMES),
        "preprocessing_version": PREPROCESSING_VERSION,
        "label_mapping_version": LABEL_MAPPING_VERSION,
        **extra,
    }


def _truncate_logs(run_dir: Path, last_epoch: int) -> None:
    """On resume, drop log rows written after the checkpoint being resumed from."""
    csv_path, jsonl_path = run_dir / "metrics.csv", run_dir / "metrics.jsonl"
    if csv_path.exists():
        with csv_path.open() as f:
            rows = [r for r in csv.DictReader(f) if int(r["epoch"]) <= last_epoch]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
    if jsonl_path.exists():
        lines = [ln for ln in jsonl_path.read_text().splitlines()
                 if ln and json.loads(ln)["epoch"] <= last_epoch]
        jsonl_path.write_text("".join(ln + "\n" for ln in lines))


def _append_logs(run_dir: Path, row: dict, full: dict) -> None:
    csv_path = run_dir / "metrics.csv"
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    with (run_dir / "metrics.jsonl").open("a") as f:
        f.write(json.dumps(full) + "\n")


def _environment(device: torch.device, precision: str, model: nn.Module) -> dict:
    env = {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "device": str(device), "precision": precision,
        "parameters": count_parameters(model), "git": git_state(),
    }
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
    return env


def run_training(config: TrainConfig, train_dataset: Dataset, val_dataset: Dataset,
                 run_dir: str | Path | None = None, resume_from: str | Path | None = None,
                 model_config: CardioMambaConfig | None = None,
                 max_epochs_this_run: int | None = None) -> TrainingResult:
    """Train CardioMamba with validation-based model selection, then calibrate thresholds.

    `model_config` defaults to the locked CardioMamba configuration (tests pass tiny ones).
    `max_epochs_this_run` stops after that many epochs *in this invocation* (a resumable
    interruption, used for time-boxed runs and resume tests).
    """
    ckpt = load_checkpoint(resume_from) if resume_from else None
    if ckpt is not None:
        mismatches = resume_mismatches(config, ckpt["train_config"])
        if mismatches:
            raise ConfigMismatchError(
                "Requested config differs from the checkpoint's: "
                + ", ".join(f"{k}: checkpoint={a!r} requested={b!r}"
                            for k, (a, b) in mismatches.items()))
        model_config = CardioMambaConfig(**ckpt["model_config"])
        run_dir = Path(resume_from).parent
    else:
        model_config = model_config or CardioMambaConfig()
        run_dir = Path(run_dir or Path(config.output_dir)
                       / (config.run_name or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")))
    model_config = replace(model_config, checkpoint_blocks=config.activation_checkpointing)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(run_dir)

    seed_everything(config.seed)
    device = resolve_device(config.device)
    precision = resolve_precision(config.mixed_precision, device)
    model = CardioMamba(model_config).to(device)
    optimizer = build_optimizer(model, config)
    pin = device.type == "cuda"
    shuffle_gen = torch.Generator()                      # reseeded every epoch (below)
    train_loader = make_loader(train_dataset, config.batch_size, True, config.num_workers, pin,
                               shuffle_gen, seed=config.seed)
    val_loader = make_loader(val_dataset, config.batch_size, False, config.num_workers, pin,
                             seed=config.seed)
    spe = steps_per_epoch(len(train_loader), config.gradient_accumulation_steps)
    total_steps = spe * config.epochs
    scheduler, warmup = build_scheduler(optimizer, total_steps, config.warmup_ratio,
                                        config.learning_rate, config.min_learning_rate)
    state = RunState()

    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        state = RunState(epoch=ckpt["epoch"], global_step=ckpt["global_step"],
                         best_metric=ckpt["best_val_macro_auroc"], best_epoch=ckpt["best_epoch"],
                         epochs_without_improvement=ckpt["epochs_without_improvement"],
                         finished=ckpt["finished"], history=list(ckpt["history"]))
        restore_rng_state(ckpt["rng_state"])
        _truncate_logs(run_dir, state.epoch)
        logger.info("Resuming from epoch %d | global_step = %d | best_val_macro_auroc = %s "
                    "(epoch %s)", state.epoch, state.global_step, _fmt(state.best_metric),
                    state.best_epoch)
    else:
        config.save_yaml(run_dir / "config.yaml")
        (run_dir / "model_config.json").write_text(json.dumps(model_config.to_dict(), indent=2))
        (run_dir / "environment.json").write_text(
            json.dumps(_environment(device, precision, model), indent=2))

    logger.info("Run dir: %s", run_dir)
    logger.info("Device %s | precision %s (selective scan always FP32) | activation "
                "checkpointing %s | parameters %s", device, precision,
                config.activation_checkpointing, f"{count_parameters(model):,}")
    logger.info("Train records %d | val records %d | batch %d x accumulation %d = effective %d "
                "| %d optimiser steps/epoch | total %d | warmup %d", len(train_dataset),
                len(val_dataset), config.batch_size, config.gradient_accumulation_steps,
                config.effective_batch_size, spe, total_steps, warmup)

    extra = {"normalization_stats": config.normalization_stats}
    epochs_run = 0
    while not state.finished:
        if max_epochs_this_run is not None and epochs_run >= max_epochs_this_run:
            logger.info("Stopping after %d epoch(s) in this invocation (resumable).", epochs_run)
            return TrainingResult(run_dir, state, None)
        epoch = state.epoch + 1
        shuffle_gen.manual_seed(config.seed + epoch)     # epoch order independent of resume
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, config, device,
                                     precision, state, logger)
        val = predict(model, val_loader, device, precision)
        metrics = compute_metrics(val["targets"], val["probs"], 0.5)
        epoch_time = time.perf_counter() - t0
        peak_mem = (torch.cuda.max_memory_allocated(device) / 2**20
                    if device.type == "cuda" else None)

        score = metrics[SELECTION_METRIC]
        if score is None:
            raise RuntimeError("Validation macro AUROC is undefined (degenerate targets)")
        improved = state.best_metric is None or score > state.best_metric
        state.epoch = epoch
        if improved:
            state.best_metric, state.best_epoch = score, epoch
            state.epochs_without_improvement = 0
        else:
            state.epochs_without_improvement += 1
        if epoch >= config.epochs or state.epochs_without_improvement >= \
                config.early_stopping_patience:
            state.finished = True
        summary = {"epoch": epoch, "global_step": state.global_step, "learning_rate": lr,
                   "train_loss": train_loss, "val_loss": val["loss"],
                   "val_macro_auroc": score, "val_macro_auprc": metrics["macro_auprc"],
                   "val_macro_f1": metrics["macro_f1"], "val_micro_f1": metrics["micro_f1"],
                   "epoch_time_s": round(epoch_time, 2), "peak_gpu_mem_mib": peak_mem,
                   "is_best": improved}
        state.history.append(summary)

        payload = _checkpoint_payload(model, optimizer, scheduler, state, config, model_config,
                                      precision, extra)
        save_checkpoint(run_dir / "last.pt", payload)
        if improved:
            save_checkpoint(run_dir / "best.pt", payload)
        row = {**summary,
               **{f"val_auroc_{c}": metrics["per_class"][c]["auroc"] for c in CLASS_NAMES},
               **{f"val_auprc_{c}": metrics["per_class"][c]["auprc"] for c in CLASS_NAMES}}
        _append_logs(run_dir, row, {**summary, "val_metrics_at_0.5": metrics})
        logger.info(
            "epoch %d/%d | train_loss %.4f | val_loss %.4f | val macro AUROC %s AUPRC %s | "
            "F1@0.5 macro %s micro %s | lr %.3e | %.1fs | peak %s MiB%s", epoch, config.epochs,
            train_loss, val["loss"], _fmt(score), _fmt(metrics["macro_auprc"]),
            _fmt(metrics["macro_f1"]), _fmt(metrics["micro_f1"]), lr, epoch_time,
            "n/a" if peak_mem is None else f"{peak_mem:.0f}", "  *best*" if improved else "")
        logger.info("  per-class AUROC %s", {c: _fmt(metrics["per_class"][c]["auroc"], 3)
                                             for c in CLASS_NAMES})
        epochs_run += 1

    reason = ("early stopping" if state.epochs_without_improvement >= config.early_stopping_patience
              and state.epoch < config.epochs else "max epochs")
    logger.info("Training finished (%s). Best val macro AUROC %s at epoch %s.", reason,
                _fmt(state.best_metric), state.best_epoch)
    thresholds = finalize_thresholds(run_dir / "best.pt", val_loader, device, precision,
                                     config, logger)
    return TrainingResult(run_dir, state, thresholds)


# --------------------------------------------------------------------------- calibration

def load_model_from_checkpoint(ckpt: dict, device: torch.device) -> CardioMamba:
    cfg = replace(CardioMambaConfig(**ckpt["model_config"]), checkpoint_blocks=False)
    model = CardioMamba(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


def evaluate_validation(checkpoint_path: Path, val_loader: DataLoader, device: torch.device,
                        precision: str, config: TrainConfig, calibrate: bool) -> dict:
    """Validation metrics of a checkpoint at 0.5 and (optionally) at calibrated thresholds."""
    ckpt = load_checkpoint(checkpoint_path)
    model = load_model_from_checkpoint(ckpt, device)
    val = predict(model, val_loader, device, precision)
    out = {
        "split": "validation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt["epoch"],
        "best_epoch": ckpt["best_epoch"],
        "n_records": len(val["targets"]),
        "val_loss": val["loss"],
        "metrics_at_0.5": compute_metrics(val["targets"], val["probs"], 0.5),
    }
    if calibrate:
        grid = threshold_grid(config.threshold_grid_start, config.threshold_grid_stop,
                              config.threshold_grid_step)
        per_class = calibrate_thresholds(val["targets"], val["probs"], grid)
        thr = [per_class[c]["threshold"] for c in CLASS_NAMES]
        out["thresholds"] = {
            "objective": "per-class F1 maximisation on the validation split (fold 9); "
                         "ties -> threshold closest to 0.5",
            "grid": {"start": config.threshold_grid_start, "stop": config.threshold_grid_stop,
                     "step": config.threshold_grid_step},
            "class_order": list(CLASS_NAMES),
            "per_class": per_class,
        }
        out["metrics_at_calibrated"] = compute_metrics(val["targets"], val["probs"], thr)
    return out


def finalize_thresholds(best_path: Path, val_loader: DataLoader, device: torch.device,
                        precision: str, config: TrainConfig, logger: logging.Logger) -> dict:
    """Calibrate thresholds on the best checkpoint's validation predictions and freeze them."""
    result = evaluate_validation(best_path, val_loader, device, precision, config,
                                 calibrate=True)
    payload = {**result["thresholds"], "checkpoint": "best.pt",
               "checkpoint_epoch": result["checkpoint_epoch"],
               "val_macro_auroc": result["metrics_at_0.5"]["macro_auroc"],
               "val_macro_f1_at_0.5": result["metrics_at_0.5"]["macro_f1"],
               "val_macro_f1_at_calibrated": result["metrics_at_calibrated"]["macro_f1"],
               "val_micro_f1_at_calibrated": result["metrics_at_calibrated"]["micro_f1"]}
    run_dir = best_path.parent
    (run_dir / "thresholds.json").write_text(json.dumps(payload, indent=2))
    (run_dir / "validation_best.json").write_text(json.dumps(result, indent=2))
    logger.info("Calibrated thresholds (validation only): %s",
                {c: v["threshold"] for c, v in payload["per_class"].items()})
    logger.info("Validation macro F1: %s at 0.5 -> %s at calibrated thresholds",
                _fmt(payload["val_macro_f1_at_0.5"]), _fmt(payload["val_macro_f1_at_calibrated"]))
    return payload
