# CardioMamba

**Mamba-Based Deep Learning for Multi-Lead ECG Classification**

A pure-PyTorch **bidirectional Mamba-1 (S6) selective state-space model** for **five-class
multi-label classification** of 10-second, 12-lead ECGs from **PTB-XL 1.0.3**, with a
reproducible training/validation pipeline, a one-time held-out test evaluation, an
inference-only FastAPI service and a Next.js research demo.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13%20(cu130)-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-inference%20API-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=nextdotjs&logoColor=white)
![Data](https://img.shields.io/badge/Data-PTB--XL%201.0.3%20(CC%20BY%204.0)-4c1)

> [!WARNING]
> **Research prototype — not a medical diagnostic system.** CardioMamba is an
> educational/research project. Its outputs are model predictions on a public research
> dataset; they are not diagnoses and must not be used for clinical decisions.

---

## Contents

- [Overview](#overview)
- [Key results (held-out test, fold 10)](#key-results-held-out-test-fold-10)
- [Architecture](#architecture)
- [Pure-PyTorch selective scan](#pure-pytorch-selective-scan)
- [Dataset, split and labels](#dataset-split-and-labels)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Metrics and threshold calibration](#metrics-and-threshold-calibration)
- [Training run results](#training-run-results)
- [Held-out test evaluation and integrity gate](#held-out-test-evaluation-and-integrity-gate)
- [Reproducibility](#reproducibility)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Running the ML pipeline](#running-the-ml-pipeline)
- [Running the demo](#running-the-demo)
- [Testing](#testing)
- [Design rationale](#design-rationale)
- [Limitations](#limitations)
- [Project status](#project-status)
- [Dataset attribution and citation](#dataset-attribution-and-citation)

---

## Overview

| | |
|---|---|
| **Task** | Multi-label classification into 5 PTB-XL diagnostic superclasses: `NORM`, `MI`, `STTC`, `CD`, `HYP`. A record can have several positive labels at once, so the model outputs five **independent** probabilities (sigmoid per class), not a softmax over mutually exclusive classes. |
| **Data** | PTB-XL 1.0.3, original 500 Hz WFDB recordings (`records500/`), 12 leads, 10 s. |
| **Input** | Resampled to 100 Hz → tensor `[1000, 12]`, normalized per lead with statistics from the training folds only. |
| **Model** | Conv1D patch stem → 4 bidirectional Mamba-1/S6 blocks → RMSNorm → mean pooling → linear head. 939,397 trainable parameters. Selective scan implemented directly in PyTorch (no `mamba-ssm`). |
| **Protocol** | Train on folds 1–8, select the checkpoint and calibrate per-class thresholds on fold 9, evaluate once on the untouched fold 10. |
| **Demo** | FastAPI service running the frozen checkpoint + Next.js dashboard (upload or bundled demo ECG → 12-lead view → probabilities with frozen thresholds). |

Class names follow PTB-XL's superclass terminology: `NORM` normal ECG, `MI` myocardial
infarction, `STTC` ST/T change, `CD` conduction disturbance, `HYP` hypertrophy.

---

## Key results (held-out test, fold 10)

Official **held-out test results on PTB-XL fold 10** (2,158 labelled records, 1,877 patients),
evaluated **once** with the frozen epoch-5 checkpoint and frozen validation thresholds.
Source: `outputs/cardiomamba/full-30ep-seed42/test_metrics.json` (local artifact, gitignored);
a copy of the metrics is committed in [`frontend/data/results.json`](frontend/data/results.json).

| Metric | Test (fold 10) |
|---|---:|
| **Macro AUROC** | **0.9223** |
| **Macro AUPRC** | **0.8121** |
| **Macro F1** (frozen thresholds) | **0.7417** |
| **Micro F1** (frozen thresholds) | **0.7765** |
| Macro precision / recall | 0.7222 / 0.7673 |
| Micro precision / recall | 0.7451 / 0.8105 |

Per class (AUROC/AUPRC from probabilities; precision/recall/F1 at the frozen thresholds):

| Class | AUROC | AUPRC | Precision | Recall | F1 | Threshold | Support |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORM | 0.9434 | 0.9257 | 0.8034 | 0.9211 | 0.8582 | 0.38 | 963 |
| MI | 0.9252 | 0.8343 | 0.6845 | 0.8127 | 0.7431 | 0.26 | 550 |
| STTC | 0.9302 | 0.8094 | 0.7138 | 0.8042 | 0.7563 | 0.41 | 521 |
| CD | 0.9187 | 0.8485 | 0.8257 | 0.7258 | 0.7725 | 0.35 | 496 |
| HYP | 0.8938 | 0.6427 | 0.5837 | 0.5725 | 0.5780 | 0.27 | 262 |
| **Macro** | **0.9223** | **0.8121** | 0.7222 | 0.7673 | **0.7417** | — | — |

**How these numbers were obtained**

- The best checkpoint (epoch 5) was selected by **validation (fold 9) macro AUROC**.
- The per-class thresholds were calibrated on **fold 9** and frozen before the test run.
- Fold 10 was not used for training, checkpoint selection, threshold calibration or
  hyper-parameter choices; it was evaluated once, after all of those were frozen.
- These are results of a single training run (seed 42); no confidence intervals or seed
  variance were estimated.

For context, the same checkpoint on validation (fold 9): macro AUROC 0.9264, macro AUPRC
0.8098, macro F1 0.7472 at the calibrated thresholds (optimistic, since the thresholds were
tuned there).

---

## Architecture

```text
Input ECG                              [B, 1000, 12]   100 Hz, 10 s, normalized
   │  transpose to channels-first
Conv1D patch stem  12 → 128, k=4, s=4  [B, 250, 128]   one token = 40 ms of all 12 leads
   │
4 × Bidirectional Mamba-1 / S6 block   [B, 250, 128]
   │     x + Dropout( Mixer_fwd(RMSNorm(x)) + flip(Mixer_rev(flip(RMSNorm(x)))) )
RMSNorm                                [B, 250, 128]
   │
Mean pooling over tokens               [B, 128]
   │
Dropout 0.1 → Linear 128 → 5           [B, 5]          logits (returned by forward)
   │
sigmoid + frozen per-class thresholds  (probabilities / labels at inference only)
```

| Setting | Value |
|---|---|
| `d_model` | 128 |
| `n_layers` | 4 |
| `d_state` | 16 |
| `d_conv` | 4 |
| `expand` | 2 (inner width 256) |
| `dt_rank` | 8 |
| `bidirectional` | true |
| `dropout` | 0.1 |
| Patch stem | Conv1D, kernel 4, stride 4, no padding (1000 → 250 tokens) |
| Positional encoding | none — token order is modelled by the recurrence |
| Parameters | **939,397** trainable (stem 6,272; per block 233,088; final norm 128; head 645) |

Implementation notes (`ml/src/cardiomamba/models/`):

- **Blocks** are pre-norm residual blocks. Each has a forward S6 mixer and a separate
  reverse S6 mixer (independent weights); the reverse mixer runs on the time-flipped
  sequence and its output is flipped back before the two are **summed**, so the width stays
  at 128.
- **RMSNorm** is computed in FP32; the residual stream is kept in FP32.
- `CardioMamba.forward` returns **logits**; `predict_proba` applies the sigmoid. Training
  uses `BCEWithLogitsLoss` on the logits.
- Configuration lives in the frozen dataclass `CardioMambaConfig` (defaults = values above).
  `scan_mode` (`parallel`/`reference`) and `checkpoint_blocks` (activation checkpointing) are
  implementation switches that do not change the model.

---

## Pure-PyTorch selective scan

CardioMamba does **not** use `mamba-ssm`, `causal-conv1d` or Triton Mamba kernels. The S6
mixer and its selective scan are implemented in plain PyTorch
([`selective_scan.py`](ml/src/cardiomamba/models/selective_scan.py),
[`mamba.py`](ml/src/cardiomamba/models/mamba.py)).

**Mixer** (per direction, per token): `in_proj` (128 → 2×256, split into `x` and gate `z`) →
causal depthwise Conv1D (kernel 4) → SiLU → `x_proj` → input-dependent `Δ` (via `dt_proj` +
softplus), `B`, `C` → selective scan → `y ⊙ SiLU(z)` → `out_proj` (256 → 128).

**Recurrence** (state size N = 16 per channel, `A = −exp(A_log)` with S4D-real init):

```text
Ā_t = exp(Δ_t · A)                 zero-order hold for A
B̄_t x_t = (Δ_t ⊙ x_t) · B_t         Euler step for B
h_t = Ā_t ⊙ h_{t−1} + B̄_t x_t       (h_{−1} = 0; state updated before readout)
y_t = h_t · C_t + D ⊙ x_t
```

Two implementations of the same recurrence:

| Mode | Implementation | Used for |
|---|---|---|
| `reference` | explicit Python loop over time with autograd | correctness oracle in tests |
| `parallel` (default) | **chunked associative scan**, time-major: sequential steps inside chunks of ≤ 16 positions (10 for L = 250), vectorised over all chunks; chunk summaries combined with a log-depth Hillis–Steele scan; one broadcast pass adds the carried state. No Python loop over the sequence length. | training and inference |

- The parallel path has a **custom `autograd.Function` backward**: the same scan run in
  reverse time (∂b = g, ∂Ā = g · h_{t−1}); only Ā and h are saved.
- The scan always runs in **FP32** with autocast disabled, even when the surrounding model
  uses bf16 autocast; Δ = softplus(·) is also computed in FP32.
- **Validation** (`ml/tests/test_selective_scan.py`): parallel vs reference forward outputs
  and gradients for all six inputs across several shapes (L up to 250 in tests, up to 1000 in
  `scripts/validate_model.py`), bf16 inputs, CPU and CUDA, compared with a float64 oracle;
  the tests assert a normalized error `max|Δ| / max|ref| < 1e-5` (observed errors are around
  1e-7). The custom backward passes **float64 `torch.autograd.gradcheck`**.

The pure-PyTorch scan is intended for transparency and control, not to match the speed of
fused CUDA kernels.

---

## Dataset, split and labels

**PTB-XL 1.0.3** (PhysioNet): 21,799 records from 18,869 patients; 12-lead, 10-second ECGs;
original sampling 500 Hz (`records500/`, used) plus a 100 Hz copy (`records100/`, **not**
used — CardioMamba generates 100 Hz itself so uploads follow the identical path).

**Split** — PTB-XL's `strat_fold` column, which is patient-disjoint:

| Split | Folds | Labelled records used |
|---|---|---:|
| Train | 1–8 | 17,084 |
| Validation | 9 | 2,146 |
| Test (held out) | 10 | 2,158 |

Records without any diagnostic superclass are excluded (411 in total: 334 train, 37
validation, 40 test). Fold 10 contains 2,198 records; 2,158 carry at least one superclass.
The tests assert zero patient overlap between the three splits.

**Label mapping** ([`labels.py`](ml/src/cardiomamba/data/labels.py),
[`metadata.py`](ml/src/cardiomamba/data/metadata.py);
artifact [`ml/configs/labels_superdiagnostic_v1.json`](ml/configs/labels_superdiagnostic_v1.json)):

1. Parse `scp_codes` from `ptbxl_database.csv`.
2. Keep SCP statements whose row in `scp_statements.csv` has `diagnostic == 1` (44 codes).
3. Map each kept code through `diagnostic_class` to one of the 5 superclasses; the target is
   the multi-hot union in the fixed order `[NORM, MI, STTC, CD, HYP]`.
4. **Likelihood values are ignored** — a diagnostic code present in `scp_codes` contributes
   its superclass regardless of the likelihood recorded for it. Form/rhythm-only statements do
   not contribute.
5. A code missing from `scp_statements.csv` raises an error instead of being dropped.

---

## Preprocessing

Preprocessing version **`prep-v1`** ([`preprocessing.py`](ml/src/cardiomamba/data/preprocessing.py)),
shared by training, evaluation and the API:

```text
500 Hz WFDB record (physical units, mV), located via `filename_hr`
  ↓ validate: 12 leads, names I…V6 matched case-insensitively (PTB-XL stores AVR/AVL/AVF),
  ↓           500 Hz, 5000 samples, units mV, finite values → otherwise ECGValidationError
  ↓ no filtering
  ↓ scipy.signal.resample_poly(up=1, down=5, window=("kaiser", 5.0), padtype="line")
[1000, 12] float32, 100 Hz
  ↓ per-lead normalization with frozen training-fold statistics
model input
```

- **Resampling**: polyphase FIR with an anti-aliasing low-pass (Kaiser, β = 5) before
  decimation by 5. `padtype="line"` (not SciPy's default zero padding) avoids artificial
  boundary distortion for records with a baseline offset.
- **Normalization**: one mean and one standard deviation per lead (population std),
  computed from the **17,084 labelled training records (folds 1–8) only**, streaming in
  `ecg_id` order (bit-identical for any worker count); std floored at 0.001 mV. Stored in
  [`ml/configs/normalization_stats_prep-v1.json`](ml/configs/normalization_stats_prep-v1.json)
  and reused unchanged for validation, test and inference. **No per-record z-scoring.**
- Preprocessing is deterministic (repeated runs on a record are identical; tested).
- Records are loaded lazily per item (no preprocessed cache).

---

## Training

Configuration: [`ml/configs/train_default.yaml`](ml/configs/train_default.yaml)
(`cardiomamba.training.TrainConfig`); engine: [`engine.py`](ml/src/cardiomamba/training/engine.py).

| Setting | Value |
|---|---|
| Loss | `BCEWithLogitsLoss` on logits |
| Optimizer | AdamW, lr 1e-3, betas (0.9, 0.999), weight decay 0.05 — no decay on biases, norm weights, `A_log`, `D` |
| Schedule | step-based: linear warmup over 5 % of optimizer steps, cosine decay to 1e-5 |
| Batch | 32 (gradient accumulation supported, e.g. 16 × 2) |
| Epochs | max 50 by default; early stopping on **validation macro AUROC**, patience 10 |
| Gradient clipping | max-norm 1.0; a non-finite gradient norm aborts the run |
| Precision | bf16 autocast on CUDA GPUs that support it, otherwise FP32; the selective scan is always FP32 |
| Activation checkpointing | on (per block, training only) |
| Seed | 42 |

**The completed run** `full-30ep-seed42` used this configuration with `--epochs 30`
(on an NVIDIA RTX 3050 Laptop GPU, bf16, ~8.1 min per epoch). Validation macro AUROC peaked
at **epoch 5**; early stopping ended training after **epoch 15** (10 epochs without
improvement).

---

## Metrics and threshold calibration

- **Primary (threshold-free)**: per-class and macro **AUROC** and **AUPRC**. AUPRC is
  informative for the less frequent classes (e.g. HYP) because it depends on prevalence.
- **Secondary (thresholded)**: per-class precision, recall, F1; macro and micro F1,
  precision and recall.
- Undefined values (a class with no positives, no predicted positives) are reported as
  `null`, never silently as 0 ([`metrics.py`](ml/src/cardiomamba/training/metrics.py)).
- Per-epoch monitoring uses a fixed 0.5 threshold; model selection uses validation macro
  AUROC only (`best.pt` is replaced only on a strict improvement).

**Threshold calibration** ([`thresholds.py`](ml/src/cardiomamba/training/thresholds.py)):
after training, the best checkpoint's **validation (fold 9)** probabilities are used to choose,
independently for each class, the threshold on the grid 0.01, 0.02, …, 0.99 that maximizes
that class's F1 (ties → the threshold closest to 0.5). The thresholds are written to
`thresholds.json` and frozen:

| NORM | MI | STTC | CD | HYP |
|---:|---:|---:|---:|---:|
| 0.38 | 0.26 | 0.41 | 0.35 | 0.27 |

Test data is never an input to calibration.

---

## Training run results

Recorded in `outputs/cardiomamba/full-30ep-seed42/metrics.csv` (per-epoch history also
committed in [`frontend/data/training.json`](frontend/data/training.json)):

| Validation (fold 9), best epoch 5 | Value |
|---|---:|
| Macro AUROC | 0.9264 |
| Macro AUPRC | 0.8098 |
| Macro F1 at 0.5 | 0.7169 |
| Micro F1 at 0.5 | 0.7665 |
| Macro F1 at calibrated thresholds | 0.7472 |
| Micro F1 at calibrated thresholds | 0.7788 |

After epoch 5 the training loss kept decreasing (0.243 → 0.064 at epoch 15) while the
validation loss rose (0.276 → 0.492) and validation macro AUROC stayed below its epoch-5
value — validation degradation consistent with overfitting, which early stopping bounded.

---

## Held-out test evaluation and integrity gate

[`scripts/evaluate_test.py`](scripts/evaluate_test.py) performs the one-time fold-10
evaluation. Before any inference it runs **12 integrity checks** and aborts if one fails:

1. the checkpoint is the frozen run's `best.pt`;
2. its epoch equals the recorded best (validation-selected) epoch, 5;
3. preprocessing version, label-mapping version and class order match the training code;
4. the checkpoint's model config matches the run's `model_config.json`;
5. the model-determining code and configs (`ml/src`, `ml/configs`, `pyproject.toml`,
   `uv.lock`) are unchanged since the training commit, with no uncommitted tracked changes;
6. the normalization statistics are the tracked, unmodified training-fold (1–8) file;
7. the thresholds equal the frozen fold-9 values and match `validation_best.json` for the same
   checkpoint;
8. the evaluated split is exactly fold 10;
9. no fold 1–9 record is in it;
10. it has the expected 2,158 labelled records;
11. every record has a valid, non-empty label vector;
12. patient IDs are present and disjoint from folds 1–9.

It then predicts (sigmoid of logits), checks that prediction order and targets match the
split, computes the metrics at the frozen thresholds (and at 0.5 as a supplement), and writes
`test_metrics.json`, `test_predictions.csv` and `test_evaluation_report.md`. It contains no
training or calibration code and **refuses to run again** once `test_metrics.json` exists.

```bash
uv run python scripts/evaluate_test.py --check-only   # integrity checks only, no inference
```

These checks protect the specific properties listed above; they are not a general guarantee
against every form of leakage.

---

## Reproducibility

- **Seeding**: Python, NumPy, PyTorch CPU/CUDA; cuDNN deterministic, no benchmarking.
  Shuffling uses a dedicated sampler generator reseeded with `seed + epoch`, separate from the
  DataLoader's worker-seed generator; workers use the `forkserver` start method.
  `torch.use_deterministic_algorithms` is not enabled.
- **Checkpoints** (`last.pt` every epoch, `best.pt` on improvement; atomic writes; loadable
  with `torch.load(weights_only=True)`) contain model, optimizer and scheduler state, epoch,
  global step, best metric/epoch, early-stopping counter, history, training and model config,
  seed, precision, RNG states, class order, preprocessing/label versions and the
  normalization-stats path.
- **Resume** restores all of the above; it refuses to continue if a protected setting
  (e.g. batch size, learning rate, epochs) differs from the checkpoint. The tests verify that
  an interrupted-and-resumed run is bit-identical to an uninterrupted one on CPU.
- **Run directory** also stores `config.yaml`, `model_config.json`, `environment.json`
  (Python/PyTorch/CUDA versions, GPU, git commit), `train.log`, `metrics.csv`/`.jsonl`,
  `thresholds.json`, `validation_best.json`.

Verify the repository state:

```bash
uv run pytest                                        # 146 tests
uv run ruff check .
uv run python scripts/evaluate_test.py --check-only  # "All 12 integrity checks passed."
```

The PTB-XL-dependent tests skip automatically when the dataset is not downloaded; the
integrity check requires the local run directory `outputs/cardiomamba/full-30ep-seed42/`.

---

## Repository structure

```text
CardioMamba/
├── ml/
│   ├── src/cardiomamba/          # installable package
│   │   ├── data/                 # labels, metadata/split, prep-v1 preprocessing, Dataset
│   │   ├── models/               # config, RMSNorm, selective scan, Mamba mixer/block, CardioMamba
│   │   ├── training/             # config, schedule, metrics, thresholds, checkpoints, engine
│   │   └── train.py              # training / resume / validation-only CLI
│   ├── configs/                  # label mapping, normalization stats, train_default.yaml
│   └── tests/                    # ML test suite
├── scripts/                      # download, inspection, validation, benchmark, test evaluation, demo export
├── backend/                      # inference-only FastAPI service (separate uv project)
├── frontend/                     # Next.js research demo
├── docs/                         # public documentation placeholder
├── pyproject.toml, uv.lock       # root Python environment (Python 3.12, CUDA 13.0 PyTorch)
└── README.md
```

Not tracked (gitignored): `ml/data/` (PTB-XL download), `outputs/` (checkpoints, logs,
evaluation artifacts), virtual environments and `node_modules/`.

---

## Installation

**Requirements**: Python 3.12, [uv](https://docs.astral.sh/uv/), git. An NVIDIA GPU with a
driver supporting CUDA 13.0 is needed for GPU training (PyTorch is installed from the
`cu130` wheel index; inference and the demo backend run on CPU by default). Node.js + npm for
the frontend (developed with Node 26 / npm 12).

```bash
git clone https://github.com/Arjun13-git/CardioMamba.git
cd CardioMamba
uv sync                 # creates .venv with the pinned environment from uv.lock
```

**PTB-XL** (≈ 3.2 GB) is downloaded to `ml/data/raw/ptb-xl-1.0.3/` (gitignored — do not
commit it):

```bash
uv run python scripts/download_ptbxl.py
```

The script fetches PhysioNet's official open-data copy
(`s3://physionet-open/ptb-xl/1.0.3/`, over HTTPS) and verifies every file against the
`SHA256SUMS.txt` published on physionet.org; it writes a manifest next to the dataset.

---

## Running the ML pipeline

All commands run from the repository root.

| Step | Command |
|---|---|
| Download + verify PTB-XL | `uv run python scripts/download_ptbxl.py [--workers 32]` |
| Inspect dataset (metadata, folds, labels, WFDB headers) | `uv run python scripts/inspect_ptbxl.py [--sample-per-fold 20]` |
| Regenerate label mapping + normalization stats | `uv run python scripts/prepare_ptbxl.py [--workers 8]` |
| Preprocessing evidence report | `uv run python scripts/validate_preprocessing.py [--per-split 100]` |
| Model / scan validation report | `uv run python scripts/validate_model.py` |
| GPU memory/time benchmark | `uv run python scripts/benchmark_model.py [--batch-sizes 32 16 8]` |
| Pre-training sanity checks (real data) | `uv run python scripts/sanity_check_training.py` |
| Train | `uv run python -m cardiomamba.train --config ml/configs/train_default.yaml` |
| Resume | `uv run python -m cardiomamba.train --config ml/configs/train_default.yaml --resume outputs/cardiomamba/<run>/last.pt` |
| Validation-only evaluation | `uv run python -m cardiomamba.train --checkpoint outputs/cardiomamba/<run>/best.pt --eval-validation --calibrate` |
| Held-out test integrity check | `uv run python scripts/evaluate_test.py --check-only` |
| Export frontend data + demo ECGs | `uv run python scripts/export_demo_assets.py` |

Training CLI overrides: `--seed`, `--epochs`, `--batch-size`, `--grad-accumulation`,
`--device`, `--precision`, `--num-workers`, `--run-name`, `--max-epochs-this-run`
(and `--max-train-records` / `--max-val-records` for smoke runs). The frozen run was started
with `--epochs 30 --seed 42 --batch-size 32 --grad-accumulation 1 --run-name full-30ep-seed42`.

> The official test evaluation (`scripts/evaluate_test.py` without `--check-only`) has
> already been performed for `full-30ep-seed42` and refuses to re-run.

---

## Running the demo

```text
Next.js frontend (port 3000)  ──/api/v1/* proxy──▶  FastAPI (port 8000)
                                                      │ decode + validate CSV
                                                      │ prep-v1 preprocessing (shared package)
                                                      │ frozen best.pt, eval mode, no_grad
                                                      │ sigmoid + frozen thresholds
                                                      ▼
                                                   JSON prediction response
```

Requires the frozen run directory `outputs/cardiomamba/full-30ep-seed42/` (`best.pt`,
`thresholds.json`) on the machine.

```bash
# terminal 1 — inference API (separate uv project; same pinned torch/numpy/scipy as the root)
cd backend
uv sync
uv run uvicorn app.main:app --port 8000

# terminal 2 — frontend
cd frontend
npm install
npm run dev            # http://localhost:3000
```

The backend is a separate uv project so the root `pyproject.toml`/`uv.lock` (protected by the
integrity check) stay unchanged. Settings use the `CARDIOMAMBA_` prefix (e.g.
`CARDIOMAMBA_DEVICE=cuda`, `CARDIOMAMBA_CHECKPOINT_PATH`); the frontend's proxy target is
`CARDIOMAMBA_API_URL` (default `http://127.0.0.1:8000`).

### API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | process liveness |
| GET | `/api/v1/ready` | 200 when the model is loaded, 503 otherwise |
| GET | `/api/v1/model` | model/config/threshold metadata and frozen test metrics |
| POST | `/api/v1/predictions` | classify one uploaded ECG (multipart field `file`) |

**Supported upload** — CSV only:

- header row with the 12 lead names `I, II, III, aVR, aVL, aVF, V1–V6` (case-insensitive;
  a complete but permuted set is reordered);
- values in mV;
- exactly **5000 rows** (10 s at 500 Hz) or **1000 rows** (10 s at 100 Hz) — the sampling rate
  is inferred from the row count;
- at most **2 MB**.

WFDB `.hea`/`.dat` uploads are **not** exposed by the demo API.

The response contains the five probabilities, the frozen threshold and a positive/negative
flag per class, input metadata, the forward-pass time, and the resampled (un-normalized)
100 Hz waveform `[12][1000]` for display.

**Input validation and errors** — structured JSON errors
`{"error": {"code", "message", "request_id"}}`: `INVALID_ECG_INPUT` (400; malformed CSV,
wrong leads/rows, non-numeric or NaN/Inf values, non-UTF-8), `PAYLOAD_TOO_LARGE` (413),
`UNSUPPORTED_FORMAT` (415), `MODEL_NOT_READY` (503), `INTERNAL_ERROR` (500, generic message —
no stack trace is sent to the client). Uploads are processed in memory and not stored; raw
signals are not logged. The model is loaded once at startup.

The API runs on **CPU in FP32** by default; the official evaluation ran on GPU with bf16
autocast, so live probabilities can differ slightly from `test_predictions.csv` (the backend
test checks agreement within 0.02 for the demo records).

### Demo samples

Five bundled records from **PTB-XL fold 10 (the held-out test fold)**, chosen by a fixed rule
independent of model predictions: for each class, the lowest-`ecg_id` fold-10 record whose
superclass set is exactly that class and whose signal-quality columns are empty
([`frontend/public/demo/manifest.json`](frontend/public/demo/manifest.json)). They are the
original 500 Hz signals exported as CSV (PTB-XL, CC BY 4.0) — **not newly collected ECGs**.

| Record | PTB-XL reference label | Frozen model output (official test predictions) |
|---|---|---|
| #40 | NORM | NORM |
| #430 | MI | MI |
| #160 | STTC | STTC, plus NORM |
| #65 | CD | CD, plus NORM |
| #1219 | HYP | NORM — **HYP not detected** (p = 0.073 vs threshold 0.27) |

The set intentionally shows real behaviour, including errors, rather than only successes.

### Frontend pages

| Route | Content |
|---|---|
| `/` | project summary, key facts, architecture pipeline with real tensor shapes |
| `/analyze` | backend status, demo samples or CSV upload, interactive 12-lead viewer (zoom/pan/hover), probabilities with frozen thresholds, and a trace of what the backend did |
| `/architecture` | tensor flow, bidirectional block, S6 mixer steps, model specification |
| `/results` | frozen fold-10 metrics, per-class table, validation AUROC and loss curves, methodology |

All metrics shown by the frontend come from `frontend/data/*.json`, exported read-only from
the frozen run by `scripts/export_demo_assets.py`.

---

## Testing

| Suite | Command | Count | Covers |
|---|---|---:|---|
| ML | `uv run pytest` | 146 | labels, metadata/split, preprocessing & normalization, Dataset, selective scan equivalence + gradcheck, Mamba block (causality, directions), full model (shapes, gradients, bf16, checkpointing, overfit sanity, real PTB-XL batch), training components & engine (schedule, metrics, thresholds, checkpoint/resume, early stopping, test-split isolation) |
| API | `cd backend && uv run pytest` | 17 | health/ready/model, invalid inputs, size limit, missing model, 100 Hz and case-insensitive input, demo records vs official predictions, determinism |
| Frontend | `cd frontend && npm run lint && npm run build` | — | ESLint, TypeScript, production build |
| Integrity | `uv run python scripts/evaluate_test.py --check-only` | 12 checks | see [above](#held-out-test-evaluation-and-integrity-gate) |

Counts are from the current repository state; no coverage percentage is measured.

---

## Design rationale

These are the design motivations, not benchmark conclusions — no comparison with other
architectures was run.

- **ECG as a sequence**: a 10-second 12-lead ECG is a long multichannel time series;
  a state-space recurrence models it with cost linear in sequence length.
- **Selectivity**: Mamba-1's input-dependent Δ, B and C let the recurrence decide what to keep
  or forget along the signal.
- **Bidirectionality**: classification is offline on a complete record, so each block also
  reads the sequence right-to-left and every token gets context from both directions.
- **Patch stem**: 1000 samples → 250 tokens of 40 ms keeps sequence length and memory
  manageable on a 4 GB laptop GPU.
- **Pure PyTorch**: the selective scan is implemented directly in PyTorch instead of relying
  on external Mamba-specific CUDA kernel packages (`mamba-ssm`, `causal-conv1d`, Triton Mamba
  kernels). This keeps the implementation readable and testable with explicit numerical checks.
  PyTorch itself is still used with CUDA for GPU training; the same code also runs on CPU,
  which the demo API uses by default.

---

## Limitations

- Research/educational prototype — **not a medical diagnostic system**, not clinically
  validated and not intended for deployment in care settings.
- Trained and evaluated on a single dataset (PTB-XL); no external or prospective validation.
- One training run (seed 42); no confidence intervals or seed variance.
- HYP is the weakest and rarest class on the test fold (F1 0.578, AUPRC 0.643, 262 of
  2,158 records).
- Per-class thresholds are tuned on fold 9 and may not transfer to other populations or
  recording setups.
- The demo accepts CSV only; it has no WFDB upload and no clinical data handling features.
- The pure-PyTorch scan prioritizes transparency over speed; it is slower than fused kernels.
- The training run used a GPU with bf16 autocast; the demo API defaults to CPU/FP32, so live
  probabilities can differ slightly from the recorded evaluation.

---

## Project status

- [x] PTB-XL 1.0.3 acquisition with SHA-256 verification, and dataset inspection
- [x] `prep-v1` preprocessing (500 → 100 Hz, training-fold normalization)
- [x] 5-class superdiagnostic multi-label mapping and patient-disjoint split
- [x] Pure-PyTorch bidirectional Mamba-1/S6 model with reference and parallel scans
- [x] Numerical scan validation (equivalence tests, float64 gradcheck)
- [x] Training pipeline (checkpointing, resume, early stopping, logging)
- [x] Validation-only threshold calibration
- [x] Full training run `full-30ep-seed42` (best epoch 5, early-stopped after epoch 15)
- [x] One-time held-out fold-10 evaluation with integrity checks
- [x] Inference-only FastAPI service
- [x] Next.js research demo

---

## Dataset attribution and citation

This project uses **PTB-XL 1.0.3**, published on **PhysioNet** under the **Creative Commons
Attribution 4.0 International (CC BY 4.0)** license: <https://physionet.org/content/ptb-xl/1.0.3/>
(DOI [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45)). The demo CSVs in
`frontend/public/demo/` are unmodified PTB-XL signal values from five records, redistributed
under that license.

> **Citation placeholder** — the repository records only the dataset name and version, the
> PhysioNet URL, the DOI above and the attribution "Wagner et al."; it does not contain the full
> bibliographic entry. Add the official PTB-XL citation(s) exactly as listed on the PhysioNet
> dataset page before publishing.

No license file for the CardioMamba code is present yet.
