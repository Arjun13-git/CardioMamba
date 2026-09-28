"""CardioMamba training / validation-only CLI.

Train (run from the repository root):
    uv run python -m cardiomamba.train --config ml/configs/train_default.yaml
Resume an interrupted run (same config):
    uv run python -m cardiomamba.train --config ml/configs/train_default.yaml \\
        --resume outputs/cardiomamba/<run>/last.pt
Validation-only evaluation of a checkpoint (never touches the test split):
    uv run python -m cardiomamba.train --checkpoint outputs/cardiomamba/<run>/best.pt \\
        --eval-validation [--calibrate]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cardiomamba.training.checkpoint import load_checkpoint
from cardiomamba.training.config import TrainConfig
from cardiomamba.training.data import (
    build_train_val_datasets,
    build_validation_dataset,
    make_loader,
)
from cardiomamba.training.engine import (
    evaluate_validation,
    resolve_device,
    resolve_precision,
    run_training,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train or validate CardioMamba on PTB-XL")
    p.add_argument("--config", type=Path, help="YAML training config (defaults if omitted)")
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--grad-accumulation", type=int, dest="gradient_accumulation_steps")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"])
    p.add_argument("--precision", choices=["auto", "bf16", "fp32"], dest="mixed_precision")
    p.add_argument("--num-workers", type=int)
    p.add_argument("--run-name")
    p.add_argument("--max-train-records", type=int, help="smoke runs only")
    p.add_argument("--max-val-records", type=int, help="smoke runs only")
    p.add_argument("--max-epochs-this-run", type=int,
                   help="stop (resumably) after this many epochs in this invocation")
    p.add_argument("--resume", type=Path, help="checkpoint (last.pt) to resume from")
    p.add_argument("--checkpoint", type=Path, help="checkpoint for --eval-validation")
    p.add_argument("--eval-validation", action="store_true",
                   help="evaluate --checkpoint on the validation split only")
    p.add_argument("--calibrate", action="store_true",
                   help="with --eval-validation: also calibrate per-class thresholds")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> TrainConfig:
    base = TrainConfig.from_yaml(args.config) if args.config else TrainConfig()
    return base.with_overrides(
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, device=args.device,
        mixed_precision=args.mixed_precision, num_workers=args.num_workers,
        run_name=args.run_name, max_train_records=args.max_train_records,
        max_val_records=args.max_val_records)


def eval_validation(args: argparse.Namespace) -> int:
    if args.checkpoint is None:
        print("--eval-validation requires --checkpoint", file=sys.stderr)
        return 2
    saved = TrainConfig.from_dict(load_checkpoint(args.checkpoint)["train_config"])
    config = saved.with_overrides(device=args.device, num_workers=args.num_workers,
                                  batch_size=args.batch_size,
                                  max_val_records=args.max_val_records)
    device = resolve_device(config.device)
    precision = resolve_precision(config.mixed_precision, device)
    loader = make_loader(build_validation_dataset(config), config.batch_size, False,
                         config.num_workers, device.type == "cuda")
    result = evaluate_validation(args.checkpoint, loader, device, precision, config,
                                 calibrate=args.calibrate)
    out_path = args.checkpoint.with_name(f"validation_{args.checkpoint.stem}.json")
    out_path.write_text(json.dumps(result, indent=2))
    m = result["metrics_at_0.5"]
    print(f"validation ({result['n_records']} records) loss {result['val_loss']:.4f} | "
          f"macro AUROC {m['macro_auroc']:.4f} | macro AUPRC {m['macro_auprc']:.4f} | "
          f"F1@0.5 macro {m['macro_f1']:.4f} micro {m['micro_f1']:.4f}")
    if args.calibrate:
        c = result["metrics_at_calibrated"]
        print("calibrated thresholds:",
              {k: v["threshold"] for k, v in result["thresholds"]["per_class"].items()},
              f"-> macro F1 {c['macro_f1']:.4f} micro F1 {c['micro_f1']:.4f}")
    print(f"wrote {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.eval_validation:
        return eval_validation(args)
    config = build_config(args)
    train_ds, val_ds = build_train_val_datasets(config)
    result = run_training(config, train_ds, val_ds, resume_from=args.resume,
                          max_epochs_this_run=args.max_epochs_this_run)
    print(f"run directory: {result.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
