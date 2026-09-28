# CardioMamba

**Mamba-Based Deep Learning for Multi-Lead ECG Classification**

CardioMamba is an educational/research-oriented deep learning project that applies a Mamba/selective state-space architecture to multi-lead ECG time-series classification using the PTB-XL dataset.

> **Medical disclaimer:** CardioMamba is a student/research project and is not a clinical diagnostic device. Model outputs must not be interpreted as medical diagnoses.

## Stack
- Python 3.12 + PyTorch (CUDA wheels via `uv`)
- Pure-PyTorch bidirectional Mamba-1 / S6 model (no `mamba-ssm` / custom CUDA kernels)
- PTB-XL 1.0.3 ECG dataset
- FastAPI inference backend and React + Vite frontend (later phases)

## Repository Layout
```text
CardioMamba/
├── ml/
│   ├── src/cardiomamba/    # installable package: data/, models/, training/, train.py (CLI)
│   ├── configs/            # label mapping, normalization stats, train_default.yaml
│   └── tests/              # pytest suite
├── scripts/                # download, inspection, validation, benchmark, sanity checks
├── backend/                # FastAPI inference API (later phase)
├── frontend/               # React frontend (later phase)
├── docs/                   # public documentation
└── outputs/                # training runs (gitignored)
```

## Current Status
Implemented and tested: PTB-XL acquisition/inspection, preprocessing (`prep-v1`), the CardioMamba model (939,397 parameters), and the training/validation pipeline. The full training run, final test evaluation, backend and frontend are pending.

## Setup
```bash
uv sync                                   # Python 3.12 env with pinned dependencies (uv.lock)
uv run pytest -q                          # tests (PTB-XL-dependent tests skip if data is absent)
```

### PTB-XL preparation (prerequisite for training)
```bash
uv run python scripts/download_ptbxl.py   # PTB-XL 1.0.3 -> ml/data/raw/ptb-xl-1.0.3 (SHA-256 verified)
uv run python scripts/validate_preprocessing.py   # optional: preprocessing evidence report
```
The normalization statistics (`ml/configs/normalization_stats_prep-v1.json`, computed from training folds 1–8 only) and label mapping are committed; `scripts/prepare_ptbxl.py` regenerates them deterministically.

## Task and data protocol
- Target: 5 PTB-XL diagnostic superclasses (NORM, MI, STTC, CD, HYP), multi-label.
- Input: original 500 Hz WFDB records resampled to 100 Hz (`[1000, 12]`), per-lead normalization with frozen training-fold statistics.
- Split (patient-disjoint `strat_fold`): folds 1–8 train (17,084 labelled records), fold 9 validation (2,146), fold 10 test (2,158).

## Training
Run from the repository root:
```bash
uv run python -m cardiomamba.train --config ml/configs/train_default.yaml
```
Default configuration (`ml/configs/train_default.yaml`):

| Setting | Default |
|---|---|
| Loss | `BCEWithLogitsLoss` on logits |
| Optimizer | AdamW, lr 1e-3, weight decay 0.05 (not on biases, norm weights, `A_log`, `D`) |
| LR schedule | linear warmup over 5 % of optimiser steps, cosine decay to 1e-5 (step-based) |
| Batch | 32 × gradient accumulation 1 (use `--batch-size 16 --grad-accumulation 2` if memory is tight) |
| Epochs | max 50, early stopping on validation macro AUROC with patience 10 |
| Gradient clipping | max-norm 1.0; a non-finite gradient norm aborts the run |
| Precision | bf16 autocast on CUDA GPUs that support it, otherwise FP32; the selective scan always runs in FP32 |
| Activation checkpointing | on (per Mamba block, training only) |
| Seed | 42 (Python, NumPy, PyTorch CPU/CUDA, DataLoader shuffling) |

Useful overrides: `--seed`, `--epochs`, `--batch-size`, `--grad-accumulation`, `--device`, `--precision`, `--num-workers`, `--run-name`.

**Outputs** go to `outputs/cardiomamba/<run_name>/` (gitignored): `config.yaml`, `model_config.json`, `environment.json`, `train.log`, `metrics.csv` / `metrics.jsonl` (per-epoch losses, macro and per-class AUROC/AUPRC, F1, LR, time, GPU memory), `last.pt`, `best.pt`, `thresholds.json`, `validation_best.json`.

**Resume** an interrupted run with the same configuration (conflicting settings are rejected):
```bash
uv run python -m cardiomamba.train --config ml/configs/train_default.yaml \
    --resume outputs/cardiomamba/<run_name>/last.pt
```
Model, optimizer, scheduler position, epoch, global step, best metric, early-stopping counter and RNG states are restored; a resumed run reproduces the uninterrupted run.

**Validation-only evaluation** of a checkpoint (never uses the test split):
```bash
uv run python -m cardiomamba.train --checkpoint outputs/cardiomamba/<run_name>/best.pt \
    --eval-validation --calibrate
```

### Validation/test isolation
The training code only builds the training (folds 1–8) and validation (fold 9) splits. Checkpoint selection (`best.pt` = highest validation macro AUROC), early stopping and threshold calibration use validation data only. The test split (fold 10) is reserved for a single final evaluation after the configuration, checkpoint and thresholds are frozen.

### Threshold calibration
After training, the best checkpoint's validation predictions are used to choose one decision threshold per class: the value on the grid 0.01, 0.02, …, 0.99 that maximises that class's F1 (ties go to the threshold closest to 0.5). The thresholds, with validation F1/precision/recall/support, are written to `thresholds.json` and are frozen for the final test evaluation. Per-epoch monitoring metrics use a fixed 0.5 threshold.

## Development
See `CLAUDE.md` for agent instructions. Detailed design documents live in a local `System-Design/` folder that is intentionally not committed.
