"""Export frontend data and demo ECGs from the FROZEN run (read-only; nothing is recomputed).

Writes
  frontend/data/results.json    official fold-10 test metrics (from test_metrics.json)
  frontend/data/training.json   per-epoch history (from metrics.csv, config.yaml, last.pt)
  frontend/data/model.json      model config, parameter count, frozen thresholds
  frontend/public/demo/*.csv    demo ECGs + manifest.json

Demo selection rule (deterministic, independent of any prediction): for each superclass,
the lowest-ecg_id fold-10 record whose PTB-XL superclass set is exactly {class} and whose
signal-quality columns (baseline_drift, static_noise, burst_noise, electrodes_problems) are
all empty. CSVs hold the original 500 Hz signal in mV (3 decimals = PTB-XL's 1 uV
resolution), so the demo exercises the full preprocessing path. PTB-XL: CC BY 4.0.

Usage:  uv run python scripts/export_demo_assets.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cardiomamba.data import CLASS_NAMES, LEAD_NAMES, PTBXLMetadata, load_wfdb_record
from cardiomamba.models import CardioMamba, CardioMambaConfig, count_parameters
from cardiomamba.training.checkpoint import load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN = REPO_ROOT / "outputs" / "cardiomamba" / "full-30ep-seed42"
DATASET = REPO_ROOT / "ml" / "data" / "raw" / "ptb-xl-1.0.3"
FRONTEND = REPO_ROOT / "frontend"
QUALITY_COLS = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]


def results() -> dict:
    t = json.loads((RUN / "test_metrics.json").read_text())
    m = t["metrics_frozen_thresholds"]
    keys = ("auroc", "auprc", "precision", "recall", "f1", "support", "predicted_positive",
            "threshold")
    return {
        "source": "outputs/cardiomamba/full-30ep-seed42/test_metrics.json",
        "split": t["split"], "n_records": t["n_records"], "n_patients": t["n_patients"],
        "checkpoint_epoch": t["checkpoint_epoch"], "evaluated_at": t["evaluated_at"],
        "overall": {k: m[k] for k in ("macro_auroc", "macro_auprc", "macro_f1", "micro_f1",
                                      "macro_precision", "macro_recall", "micro_precision",
                                      "micro_recall")},
        "per_class": {c: {k: m["per_class"][c][k] for k in keys} for c in CLASS_NAMES},
        "validation_reference": t["validation_reference"],
    }


def training() -> dict:
    cfg = yaml.safe_load((RUN / "config.yaml").read_text())
    last = load_checkpoint(RUN / "last.pt")
    with (RUN / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    epochs = [{"epoch": int(r["epoch"]), "train_loss": float(r["train_loss"]),
               "val_loss": float(r["val_loss"]), "val_macro_auroc": float(r["val_macro_auroc"]),
               "val_macro_auprc": float(r["val_macro_auprc"]),
               "learning_rate": float(r["learning_rate"]), "is_best": r["is_best"] == "True"}
              for r in rows]
    return {
        "source": "outputs/cardiomamba/full-30ep-seed42/metrics.csv",
        "max_epochs": cfg["epochs"], "patience": cfg["early_stopping_patience"],
        "epochs_completed": last["epoch"], "best_epoch": last["best_epoch"],
        "best_val_macro_auroc": last["best_val_macro_auroc"],
        "early_stopped": last["finished"] and last["epoch"] < cfg["epochs"],
        "epoch_time_s_mean": float(np.mean([float(r["epoch_time_s"]) for r in rows])),
        "config": {k: cfg[k] for k in ("batch_size", "learning_rate", "weight_decay",
                                       "warmup_ratio", "min_learning_rate", "seed")},
        "epochs": epochs,
    }


def model_info() -> dict:
    ckpt = load_checkpoint(RUN / "best.pt")
    thr = json.loads((RUN / "thresholds.json").read_text())
    return {
        "source": "outputs/cardiomamba/full-30ep-seed42/best.pt",
        "config": ckpt["model_config"],
        "parameters": count_parameters(CardioMamba(CardioMambaConfig(**ckpt["model_config"]))),
        "checkpoint_epoch": ckpt["epoch"],
        "class_order": ckpt["class_names"],
        "preprocessing_version": ckpt["preprocessing_version"],
        "thresholds": {c: thr["per_class"][c]["threshold"] for c in CLASS_NAMES},
        "thresholds_source": thr["objective"],
    }


def demo_samples() -> list[dict]:
    meta = PTBXLMetadata.load(DATASET)
    raw = pd.read_csv(DATASET / "ptbxl_database.csv", index_col="ecg_id")
    test = meta.split("test")
    clean = raw.loc[test.index, QUALITY_COLS].isna().all(axis=1)
    out_dir = FRONTEND / "public" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for c in CLASS_NAMES:
        only_c = (test[c] == 1) & (test["n_labels"] == 1) & clean
        ecg_id = int(test.index[only_c][0])
        signal = load_wfdb_record(DATASET / test.loc[ecg_id, "filename_hr"])   # [5000, 12] mV
        name = f"ptbxl-{ecg_id:05d}-{c.lower()}.csv"
        with (out_dir / name).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(LEAD_NAMES)
            w.writerows([[f"{v:.3f}" for v in row] for row in signal])
        samples.append({
            "id": f"ptbxl-{ecg_id}", "file": f"/demo/{name}", "ecg_id": ecg_id,
            "fold": int(test.loc[ecg_id, "strat_fold"]),
            "reference_labels": [k for k in CLASS_NAMES if test.loc[ecg_id, k] == 1],
            "sampling_rate_hz": 500, "samples": int(signal.shape[0]),
        })
    manifest = {
        "provenance": "PTB-XL 1.0.3 (PhysioNet, Wagner et al., CC BY 4.0), fold 10 (held-out "
                      "test fold); not newly collected ECGs",
        "selection_rule": "per class: lowest ecg_id in fold 10 with exactly that single "
                          "superclass and no signal-quality flags (independent of model "
                          "predictions)",
        "samples": samples,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return samples


def main() -> int:
    data_dir = FRONTEND / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("results", results()), ("training", training()),
                          ("model", model_info())):
        (data_dir / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote frontend/data/{name}.json")
    for s in demo_samples():
        print(f"demo sample {s['id']} ({', '.join(s['reference_labels'])}) -> {s['file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
