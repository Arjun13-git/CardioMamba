"""Phase 4 pre-training sanity checks on real PTB-XL data with the real training settings.

Checks: batch loading, forward, backward, finite gradients, optimiser step, LR schedule,
checkpoint save/reload, validation metrics, threshold calibration, the training-data builder
never requesting the test split, and a tiny overfit on 32 training records (optimisation
sanity only; not evidence of model performance). No test-split data is loaded.

Usage:  uv run python scripts/sanity_check_training.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from cardiomamba.models import CardioMamba, CardioMambaConfig
from cardiomamba.training import TrainConfig, calibrate_thresholds, compute_metrics
from cardiomamba.training import data as training_data
from cardiomamba.training.checkpoint import load_checkpoint, save_checkpoint
from cardiomamba.training.data import build_train_val_datasets, make_loader
from cardiomamba.training.engine import (
    autocast_ctx,
    build_optimizer,
    loss_fn,
    predict,
    resolve_device,
    resolve_precision,
)
from cardiomamba.training.schedule import build_scheduler
from cardiomamba.training.thresholds import threshold_grid

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")


def main() -> int:
    # 12. the training builders must never ask for the test split
    requested: list[str] = []
    original_split = training_data.PTBXLMetadata.split

    def guarded_split(self, name, labelled_only=True):
        requested.append(name)
        if name == "test":
            raise AssertionError("test split requested")
        return original_split(self, name, labelled_only)

    training_data.PTBXLMetadata.split = guarded_split
    config = TrainConfig(max_train_records=256, max_val_records=256, num_workers=2)
    train_ds, val_ds = build_train_val_datasets(config)
    training_data.PTBXLMetadata.split = original_split
    check("training builders never request the test split", "test" not in requested,
          f"requested {requested}")

    device = resolve_device(config.device)
    precision = resolve_precision(config.mixed_precision, device)
    print(f"device {device}, precision {precision}, activation checkpointing "
          f"{config.activation_checkpointing}, batch {config.batch_size}")
    torch.manual_seed(config.seed)
    model = CardioMamba(replace(CardioMambaConfig(), checkpoint_blocks=True)).to(device)
    opt = build_optimizer(model, config)
    sched, warmup = build_scheduler(opt, 100, config.warmup_ratio, config.learning_rate,
                                    config.min_learning_rate)
    loader = make_loader(train_ds, config.batch_size, True, 2, device.type == "cuda",
                         torch.Generator().manual_seed(config.seed), seed=config.seed)

    # 1. batch
    batch = next(iter(loader))
    x = batch["signal"].to(device, non_blocking=True)
    y = batch["target"].to(device, non_blocking=True)
    check("training batch loads", x.shape == (32, 1000, 12) and y.shape == (32, 5)
          and x.dtype == torch.float32 and bool(torch.isfinite(x).all()),
          f"signal {list(x.shape)} target {list(y.shape)}")
    # 2-4. forward / backward / finite gradients
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with autocast_ctx(device, precision):
        logits = model(x)
    loss = loss_fn()(logits.float(), y)
    check("forward pass", logits.shape == (32, 5) and bool(torch.isfinite(loss)),
          f"logits {list(logits.shape)} {logits.dtype}, loss {loss.item():.4f}")
    loss.backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())
    check("backward pass, all gradients present and finite", grads_ok)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    # 5-6. optimiser step, scheduler
    before = [p.detach().clone() for p in model.parameters()]
    lr0 = opt.param_groups[0]["lr"]
    opt.step()
    sched.step()
    changed = sum(not torch.equal(a, b) for a, b in zip(before, model.parameters(),
                                                          strict=True))
    check("optimiser step updates parameters", changed == len(before),
          f"{changed}/{len(before)} tensors changed; pre-clip grad norm {norm.item():.3f}")
    lr1 = opt.param_groups[0]["lr"]
    check("scheduler advances LR (warmup)", lr1 > lr0,
          f"{lr0:.2e} -> {lr1:.2e} (warmup {warmup} of 100 steps)")
    if device.type == "cuda":
        print(f"      peak GPU memory for one step: "
              f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    # 7-8. checkpoint save / reload
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        save_checkpoint(path, {"model_state_dict": model.state_dict(),
                               "optimizer_state_dict": opt.state_dict(),
                               "scheduler_state_dict": sched.state_dict()})
        ck = load_checkpoint(path)
        m2 = CardioMamba().to(device)
        m2.load_state_dict(ck["model_state_dict"])
        o2 = build_optimizer(m2, config)
        o2.load_state_dict(ck["optimizer_state_dict"])
        same = all(torch.equal(a, b.to(a.device)) for a, b in
                   zip(model.state_dict().values(), ck["model_state_dict"].values(),
                       strict=True))
        check("checkpoint save + weights_only reload restores state",
              same and ck["scheduler_state_dict"]["last_epoch"] == 1
              and len(o2.state_dict()["state"]) == len(opt.state_dict()["state"]))
    # 10-11. validation metrics + calibration (validation subset)
    val = predict(model, make_loader(val_ds, 32, False, 2, device.type == "cuda"), device,
                  precision)
    m = compute_metrics(val["targets"], val["probs"], 0.5)
    check("validation metrics", m["macro_auroc"] is not None and np.isfinite(val["loss"]),
          f"{m['n_samples']} records, loss {val['loss']:.4f}, macro AUROC {m['macro_auroc']:.3f}"
          " (untrained model; value meaningless)")
    thr = calibrate_thresholds(val["targets"], val["probs"], threshold_grid())
    check("threshold calibration", len(thr) == 5 and all(0 < v["threshold"] < 1
                                                         for v in thr.values()),
          str({c: v["threshold"] for c, v in thr.items()}))

    # tiny overfit on 32 real training records
    torch.manual_seed(0)
    small = torch.utils.data.Subset(train_ds, range(32))
    xs = torch.stack([small[i]["signal"] for i in range(32)]).to(device)
    ys = torch.stack([small[i]["target"] for i in range(32)]).to(device)
    model = CardioMamba(replace(CardioMambaConfig(), checkpoint_blocks=True)).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    losses, t0 = [], time.perf_counter()
    for _ in range(150):
        with autocast_ctx(device, precision):
            loss = loss_fn()(model(xs).float(), ys)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    check("tiny overfit (32 real records, 150 steps)", losses[-1] < 0.1 * losses[0],
          f"loss {losses[0]:.4f} -> {losses[-1]:.4f} in {time.perf_counter() - t0:.0f}s")

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
