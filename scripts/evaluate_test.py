"""ONE-TIME final held-out test evaluation of a frozen CardioMamba run (PTB-XL fold 10).

Evaluation only: no training, no threshold calibration, no model or hyper-parameter
selection. The checkpoint and the validation-calibrated thresholds are read, never written.

  1. integrity checks (split, labels, patients, checkpoint, preprocessing, normalization,
     thresholds, code state) -- abort before any inference if one fails;
  2. inference on the complete fold-10 test split (sigmoid of logits);
  3. threshold-free metrics (AUROC / AUPRC) and metrics at the frozen validation thresholds
     (0.5 results reported additionally);
  4. artifacts: test_metrics.json, test_predictions.csv, test_evaluation_report.md.

Refuses to run again once test_metrics.json exists (the test set is evaluated once).

Usage:
    uv run python scripts/evaluate_test.py --check-only     # integrity checks, no inference
    uv run python scripts/evaluate_test.py                  # the official evaluation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from cardiomamba.data import (
    CLASS_NAMES,
    LABEL_MAPPING_VERSION,
    PREPROCESSING_VERSION,
    SPLIT_FOLDS,
    NormalizationStats,
    PTBXLMetadata,
    PTBXLMultilabelDataset,
    PTBXLPreprocessor,
)
from cardiomamba.models import CardioMambaConfig
from cardiomamba.training.checkpoint import load_checkpoint
from cardiomamba.training.config import TrainConfig
from cardiomamba.training.data import make_loader
from cardiomamba.training.engine import (
    load_model_from_checkpoint,
    predict,
    resolve_device,
    resolve_precision,
)
from cardiomamba.training.metrics import compute_metrics, confusion_counts

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "outputs" / "cardiomamba" / "full-30ep-seed42"
EXPECTED = {
    "checkpoint_epoch": 5,
    "test_records": 2158,
    "thresholds": {"NORM": 0.38, "MI": 0.26, "STTC": 0.41, "CD": 0.35, "HYP": 0.27},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


class Checks:
    def __init__(self) -> None:
        self.items: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.items.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{' -- ' + detail if detail else ''}")

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.items)


def integrity_checks(run_dir: Path) -> tuple[Checks, dict]:
    c = Checks()
    ckpt_path = (run_dir / "best.pt").resolve()
    ckpt = load_checkpoint(ckpt_path)
    config = TrainConfig.from_dict(ckpt["train_config"])
    env = json.loads((run_dir / "environment.json").read_text())

    # 6-7 checkpoint identity
    c.add("checkpoint is the frozen run's best.pt",
          ckpt_path == (RUN_DIR / "best.pt").resolve() and ckpt_path.is_file(), str(ckpt_path))
    c.add("checkpoint epoch is the validation-selected best epoch",
          ckpt["epoch"] == ckpt["best_epoch"] == EXPECTED["checkpoint_epoch"],
          f"epoch {ckpt['epoch']}, best_epoch {ckpt['best_epoch']}, "
          f"best_val_macro_auroc {ckpt['best_val_macro_auroc']:.6f}")

    # 8 preprocessing / model / code state identical to training
    run_model_cfg = json.loads((run_dir / "model_config.json").read_text())
    c.add("preprocessing / label-mapping versions match training",
          ckpt["preprocessing_version"] == PREPROCESSING_VERSION
          and ckpt["label_mapping_version"] == LABEL_MAPPING_VERSION
          and ckpt["class_names"] == list(CLASS_NAMES),
          f"{ckpt['preprocessing_version']}, {ckpt['label_mapping_version']}, "
          f"class order {ckpt['class_names']}")
    c.add("model config matches the run's model_config.json",
          ckpt["model_config"] == run_model_cfg
          and CardioMambaConfig(**ckpt["model_config"]).d_model == 128)
    head = git("rev-parse", "HEAD")
    tracked_dirty = git("status", "--porcelain", "--untracked-files=no")
    c.add("tracked source/config files identical to the training commit",
          head == env["git"]["commit"] and not tracked_dirty and env["git"]["dirty"] is False,
          f"HEAD {head[:7]} == training {env['git']['commit'][:7]}; tracked changes: "
          f"{'none' if not tracked_dirty else tracked_dirty}")

    # 9 frozen train-derived normalization statistics
    stats_path = REPO_ROOT / ckpt["normalization_stats"]
    stats = NormalizationStats.load(stats_path)
    c.add("frozen train-fold normalization statistics",
          stats.metadata["source_folds"] == list(SPLIT_FOLDS["train"])
          and stats.metadata["n_records"] == 17084
          and git("log", "-1", "--format=%H", "--", str(stats_path)) != ""
          and not git("status", "--porcelain", "--", str(stats_path)),
          f"{ckpt['normalization_stats']} (sha256 {sha256(stats_path)[:12]}), folds "
          f"{stats.metadata['source_folds']}, {stats.metadata['n_records']} records")

    # 10-11 thresholds: validation fold 9 only, frozen values, same checkpoint
    thr = json.loads((run_dir / "thresholds.json").read_text())
    val = json.loads((run_dir / "validation_best.json").read_text())
    loaded = {k: v["threshold"] for k, v in thr["per_class"].items()}
    c.add("thresholds are the frozen validation (fold 9) thresholds",
          loaded == EXPECTED["thresholds"] and thr["class_order"] == list(CLASS_NAMES)
          and "fold 9" in thr["objective"] and thr["checkpoint_epoch"] == ckpt["epoch"]
          and val["split"] == "validation"
          and {k: v["threshold"] for k, v in val["thresholds"]["per_class"].items()} == loaded,
          str(loaded))

    # 1-5 test split
    meta = PTBXLMetadata.load(REPO_ROOT / config.dataset_dir)
    test = meta.split("test", labelled_only=True)
    train, valid = meta.split("train", False), meta.split("val", False)
    c.add("exactly fold 10", set(test["strat_fold"]) == set(SPLIT_FOLDS["test"]) == {10})
    c.add("no fold 1-9 record in the test split",
          not (set(test.index) & (set(train.index) | set(valid.index))))
    c.add("labelled test record count", len(test) == EXPECTED["test_records"],
          f"{len(test)} labelled records (fold 10 total "
          f"{len(meta.split('test', labelled_only=False))}, 40 without a superclass excluded)")
    y = meta.targets(test)
    c.add("no missing test labels",
          y.shape == (len(test), 5) and np.isfinite(y).all() and (y.sum(axis=1) > 0).all()
          and set(np.unique(y)) <= {0.0, 1.0})
    pats = test["patient_id"]
    other = set(train["patient_id"]) | set(valid["patient_id"])
    c.add("patient IDs present, consistent and disjoint from folds 1-9",
          pats.notna().all() and not (set(pats) & other),
          f"{pats.nunique()} unique patients among the labelled test records "
          f"({meta.split('test', labelled_only=False)['patient_id'].nunique()} over all fold-10 "
          "records)")

    context = {"ckpt": ckpt, "ckpt_path": ckpt_path, "config": config, "meta": meta,
               "test": test, "stats": stats, "stats_path": stats_path, "thresholds": loaded,
               "thresholds_file": thr, "env": env}
    return c, context


def summarize(y: np.ndarray, probs: np.ndarray, thresholds: list[float]) -> dict:
    m = compute_metrics(y, probs, thresholds)
    counts = confusion_counts(y, probs >= np.asarray(thresholds))
    tp, fp, fn = (int(counts[k].sum()) for k in ("tp", "fp", "fn"))
    precisions = [m["per_class"][c]["precision"] for c in CLASS_NAMES]
    recalls = [m["per_class"][c]["recall"] for c in CLASS_NAMES]
    m["macro_precision"] = (float(np.mean(precisions))
                            if all(p is not None for p in precisions) else None)
    m["macro_recall"] = float(np.mean(recalls)) if all(r is not None for r in recalls) else None
    m["micro_precision"] = tp / (tp + fp) if tp + fp else None
    m["micro_recall"] = tp / (tp + fn) if tp + fn else None
    m["micro_counts"] = {"tp": tp, "fp": fp, "fn": fn}
    return m


def write_report(path: Path, result: dict) -> None:
    r, t, h = result["metrics_frozen_thresholds"], result["thresholds"], result["metrics_at_0.5"]
    v = result["validation_reference"]
    lines = [
        "# CardioMamba — Final Held-Out Test Evaluation (PTB-XL fold 10)", "",
        (f"Evaluated: {result['evaluated_at']} — one-time evaluation of a frozen model. "
        "No training, threshold calibration, model selection or tuning was performed on the "
        "test split."), "",
        ("> Educational/research project; these are model classification metrics, not a "
        "clinical diagnostic validation."), "",
        "## Evaluation configuration", "",
        (f"- Run: `{result['run_name']}`; checkpoint `{result['checkpoint']}` "
        f"(epoch {result['checkpoint_epoch']}, selected by validation macro AUROC "
        f"{result['checkpoint_best_val_macro_auroc']:.4f}; sha256 "
        f"`{result['checkpoint_sha256'][:16]}…`)"),
        (f"- Test split: PTB-XL 1.0.3 fold 10, {result['n_records']} labelled records, "
        f"{result['n_patients']} patients"),
        (f"- Preprocessing: {result['preprocessing']['version']} — 500 Hz WFDB → resample_poly "
        "500→100 Hz (Kaiser 5.0, padtype line) → [1000, 12] → per-lead normalization with "
        "frozen training-fold statistics"),
        (f"- Device {result['environment']['device']} ({result['environment'].get('gpu', '')}), "
        f"precision {result['environment']['precision']} (selective scan FP32)"),
        f"- Probabilities: sigmoid(logits); class order {result['class_order']}", "",
        "## Integrity checks", "",
        *[f"- {'PASS' if ok else 'FAIL'} — {name}" + (f" ({d})" if d else "")
          for name, ok, d in result["integrity_checks"]], "",
        "## Primary results (threshold-free)", "",
        "| Class | AUROC | AUPRC |", "|---|---:|---:|",
        *[f"| {c} | {r['per_class'][c]['auroc']:.4f} | {r['per_class'][c]['auprc']:.4f} |"
          for c in CLASS_NAMES],
        f"| **Macro** | **{r['macro_auroc']:.4f}** | **{r['macro_auprc']:.4f}** |", "",
        "## Thresholded results (frozen validation thresholds — official)", "",
        "| Class | Threshold | Precision | Recall | F1 | Support | Predicted positives |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[f"| {c} | {t[c]:.2f} | {r['per_class'][c]['precision']:.4f} | "
          f"{r['per_class'][c]['recall']:.4f} | {r['per_class'][c]['f1']:.4f} | "
          f"{r['per_class'][c]['support']} | {r['per_class'][c]['predicted_positive']} |"
          for c in CLASS_NAMES],
        (f"| **Macro** | — | {r['macro_precision']:.4f} | {r['macro_recall']:.4f} | "
        f"{r['macro_f1']:.4f} | — | — |"),
        (f"| **Micro** | — | {r['micro_precision']:.4f} | {r['micro_recall']:.4f} | "
        f"{r['micro_f1']:.4f} | {sum(r['per_class'][c]['support'] for c in CLASS_NAMES)} | "
        f"{sum(r['per_class'][c]['predicted_positive'] for c in CLASS_NAMES)} |"), "",
        (f"Supplementary (global 0.5 threshold, not official): macro F1 {h['macro_f1']:.4f}, "
        f"micro F1 {h['micro_f1']:.4f}."), "",
        "## Validation reference (fold 9, same checkpoint; for context only)", "",
        "| Metric | Validation | Test |", "|---|---:|---:|",
        f"| Macro AUROC | {v['macro_auroc']:.4f} | {r['macro_auroc']:.4f} |",
        f"| Macro AUPRC | {v['macro_auprc']:.4f} | {r['macro_auprc']:.4f} |",
        f"| Macro F1 (frozen thresholds) | {v['macro_f1_calibrated']:.4f} | {r['macro_f1']:.4f} |",
        f"| Micro F1 (frozen thresholds) | {v['micro_f1_calibrated']:.4f} | {r['micro_f1']:.4f} |",
        "", ("Validation F1 at the calibrated thresholds is optimistic (thresholds were tuned "
        "on that split); the test values are the unbiased estimate."), "",
        "## Reproducibility", "",
        "```json", json.dumps(result["reproducibility"], indent=2), "```", "",
        ("Command: `uv run python scripts/evaluate_test.py` (refuses to overwrite an existing "
        "`test_metrics.json`)."), "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--check-only", action="store_true",
                        help="run integrity checks only; no test inference")
    args = parser.parse_args()
    run_dir = args.run_dir
    out_metrics = run_dir / "test_metrics.json"
    if out_metrics.exists() and not args.check_only:
        print(f"{out_metrics} already exists: the test set has been evaluated. Refusing to "
              "re-run.", file=sys.stderr)
        return 2

    checks, ctx = integrity_checks(run_dir)
    if not checks.ok:
        print("\nIntegrity checks FAILED -- no test evaluation performed.", file=sys.stderr)
        return 1
    print(f"\nAll {len(checks.items)} integrity checks passed.")
    if args.check_only:
        return 0

    # ---- the one-time test inference --------------------------------------------------
    ckpt, config, test, meta = ctx["ckpt"], ctx["config"], ctx["test"], ctx["meta"]
    device = resolve_device("auto")
    precision = resolve_precision(config.mixed_precision, device)
    model = load_model_from_checkpoint(ckpt, device)
    dataset = PTBXLMultilabelDataset(test, PTBXLPreprocessor(meta.dataset_dir, ctx["stats"]))
    loader = make_loader(dataset, config.batch_size, False, config.num_workers,
                         device.type == "cuda", seed=config.seed)
    out = predict(model, loader, device, precision)
    ids = out["ecg_ids"]
    if not (np.array_equal(ids, test.index.to_numpy())
            and np.array_equal(out["targets"], meta.targets(test))):
        print("Prediction order / targets do not match the test split -- aborting.",
              file=sys.stderr)
        return 1

    thr = [ctx["thresholds"][c] for c in CLASS_NAMES]
    frozen = summarize(out["targets"], out["probs"], thr)
    at_half = summarize(out["targets"], out["probs"], [0.5] * 5)
    val = json.loads((run_dir / "validation_best.json").read_text())
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "cuda_runtime": torch.version.cuda, "device": str(device), "precision": precision}
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
    result = {
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_name": run_dir.name,
        "split": "test (PTB-XL 1.0.3 fold 10)",
        "checkpoint": str(ctx["ckpt_path"].relative_to(REPO_ROOT)),
        "checkpoint_sha256": sha256(ctx["ckpt_path"]),
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_best_val_macro_auroc": ckpt["best_val_macro_auroc"],
        "n_records": len(test),
        "n_patients": int(test["patient_id"].nunique()),
        "class_order": list(CLASS_NAMES),
        "thresholds": ctx["thresholds"],
        "thresholds_source": "thresholds.json (validation fold 9, epoch-5 checkpoint)",
        "test_loss_bce": out["loss"],
        "metrics_frozen_thresholds": frozen,
        "metrics_at_0.5": at_half,
        "validation_reference": {
            "macro_auroc": val["metrics_at_0.5"]["macro_auroc"],
            "macro_auprc": val["metrics_at_0.5"]["macro_auprc"],
            "macro_f1_calibrated": val["metrics_at_calibrated"]["macro_f1"],
            "micro_f1_calibrated": val["metrics_at_calibrated"]["micro_f1"],
        },
        "preprocessing": {"version": PREPROCESSING_VERSION,
                          "normalization_stats": ckpt["normalization_stats"],
                          "normalization_stats_sha256": sha256(ctx["stats_path"])},
        "environment": env,
        "integrity_checks": checks.items,
        "reproducibility": {
            "git_commit": git("rev-parse", "HEAD"), "training_git_commit":
            ctx["env"]["git"]["commit"], "train_config": ckpt["train_config"],
            "model_config": ckpt["model_config"], "seed": ckpt["random_seed"],
            "label_mapping_version": ckpt["label_mapping_version"],
            "dataset": "PTB-XL 1.0.3 (PhysioNet, CC BY 4.0)", **env,
        },
    }
    out_metrics.write_text(json.dumps(result, indent=2))

    pid = test["patient_id"]
    with (run_dir / "test_predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ecg_id", "patient_id", *[f"true_{c}" for c in CLASS_NAMES],
                    *[f"prob_{c}" for c in CLASS_NAMES], *[f"pred_{c}" for c in CLASS_NAMES]])
        preds = (out["probs"] >= np.asarray(thr)).astype(int)
        for i, ecg_id in enumerate(ids):
            w.writerow([int(ecg_id), int(pid.loc[ecg_id]),
                        *out["targets"][i].astype(int).tolist(),
                        *[f"{p:.6f}" for p in out["probs"][i]], *preds[i].tolist()])
    write_report(run_dir / "test_evaluation_report.md", result)
    print(json.dumps({"macro_auroc": frozen["macro_auroc"], "macro_auprc": frozen["macro_auprc"],
                      "macro_f1": frozen["macro_f1"], "micro_f1": frozen["micro_f1"]}, indent=2))
    print(f"wrote {out_metrics}, test_predictions.csv, test_evaluation_report.md in {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
