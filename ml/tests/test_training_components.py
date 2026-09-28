"""Unit tests: training config, LR schedule, metrics, threshold calibration, seeding."""

import math
import random
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from cardiomamba.train import build_config, parse_args
from cardiomamba.training import TrainConfig, calibrate_thresholds, compute_metrics
from cardiomamba.training.config import resume_mismatches
from cardiomamba.training.reproducibility import (
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)
from cardiomamba.training.schedule import build_scheduler, lr_factor, steps_per_epoch
from cardiomamba.training.thresholds import threshold_grid

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- config

def test_config_defaults_match_protocol():
    c = TrainConfig()
    assert (c.seed, c.epochs, c.early_stopping_patience) == (42, 50, 10)
    assert (c.batch_size, c.gradient_accumulation_steps, c.effective_batch_size) == (32, 1, 32)
    assert (c.learning_rate, c.weight_decay, c.warmup_ratio) == (1e-3, 0.05, 0.05)
    assert (c.gradient_clip_norm, c.min_learning_rate) == (1.0, 1e-5)
    assert c.mixed_precision == "auto" and c.activation_checkpointing is True
    assert c.max_train_records is None and c.max_val_records is None


def test_default_yaml_matches_dataclass(tmp_path):
    assert TrainConfig.from_yaml(REPO_ROOT / "ml/configs/train_default.yaml") == TrainConfig()
    c = TrainConfig(batch_size=16, gradient_accumulation_steps=2)
    c.save_yaml(tmp_path / "c.yaml")
    assert TrainConfig.from_yaml(tmp_path / "c.yaml") == c


def test_unknown_config_key_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        TrainConfig.from_dict({"lr": 0.1})


def test_resume_mismatch_detection():
    saved = TrainConfig().to_dict()
    assert resume_mismatches(TrainConfig(num_workers=0, run_name="x"), saved) == {}
    assert set(resume_mismatches(TrainConfig(learning_rate=5e-4), saved)) == {"learning_rate"}


def test_cli_overrides():
    args = parse_args(["--seed", "7", "--batch-size", "16", "--grad-accumulation", "2",
                       "--epochs", "3", "--device", "cpu"])
    c = build_config(args)
    assert (c.seed, c.batch_size, c.gradient_accumulation_steps, c.epochs, c.device) == (
        7, 16, 2, 3, "cpu")
    assert c.learning_rate == 1e-3                       # untouched defaults stay


# --------------------------------------------------------------------------- schedule

def test_steps_per_epoch_counts_optimizer_steps():
    assert steps_per_epoch(534, 1) == 534
    assert steps_per_epoch(1068, 2) == 534               # B=16 x 2 == B=32
    assert steps_per_epoch(5, 2) == 3                    # partial last group still steps


def test_warmup_then_cosine_shape():
    total, warm, mr = 1000, 50, 1e-2
    f = [lr_factor(s, total, warm, mr) for s in range(total)]
    assert f[0] == pytest.approx(1 / 50) and f[49] == pytest.approx(1.0)
    assert all(a < b for a, b in pairwise(f[:50]))       # linear warmup
    assert all(a >= b for a, b in pairwise(f[50:]))       # monotone decay
    assert lr_factor(525, total, warm, mr) == pytest.approx(mr + (1 - mr) * 0.5)  # midpoint
    assert lr_factor(total, total, warm, mr) == pytest.approx(mr)


def test_scheduler_drives_optimizer_and_resumes():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-3)
    sched, warmup = build_scheduler(opt, total_steps=200, warmup_ratio=0.05, base_lr=1e-3,
                                    min_lr=1e-5)
    assert warmup == 10
    lrs = []
    for _ in range(60):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(1e-4) and lrs[9] == pytest.approx(1e-3)
    state_opt, state_sched = opt.state_dict(), sched.state_dict()
    cont = [opt.param_groups[0]["lr"]]
    # resume into fresh objects: LR continues exactly where it stopped
    p2 = torch.nn.Parameter(torch.zeros(1))
    opt2 = torch.optim.AdamW([p2], lr=1e-3)
    sched2, _ = build_scheduler(opt2, 200, 0.05, 1e-3, 1e-5)
    opt2.load_state_dict(state_opt)
    sched2.load_state_dict(state_sched)
    assert opt2.param_groups[0]["lr"] == pytest.approx(cont[0])
    assert sched2.last_epoch == 60


# --------------------------------------------------------------------------- metrics

def _random_problem(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 5)) < 0.3).astype(np.float32)
    p = np.clip(0.6 * y + 0.4 * rng.random((n, 5)), 0, 1)
    return y, p


def test_auroc_auprc_match_sklearn():
    y, p = _random_problem()
    m = compute_metrics(y, p)
    for i, c in enumerate(["NORM", "MI", "STTC", "CD", "HYP"]):
        assert m["per_class"][c]["auroc"] == pytest.approx(roc_auc_score(y[:, i], p[:, i]))
        assert m["per_class"][c]["auprc"] == pytest.approx(
            average_precision_score(y[:, i], p[:, i]))
    assert m["macro_auroc"] == pytest.approx(np.mean([roc_auc_score(y[:, i], p[:, i])
                                                      for i in range(5)]))


def test_f1_precision_recall_hand_example():
    y = np.array([[1, 0, 0, 0, 1], [1, 1, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 1, 0]], float)
    p = np.array([[.9, .2, .1, .1, .8], [.4, .7, .1, .1, .1], [.6, .6, .9, .2, .1],
                  [.1, .1, .1, .7, .1]])
    m = compute_metrics(y, p, 0.5)
    norm = m["per_class"]["NORM"]            # TP=1 (row0), FN=1 (row1), FP=1 (row2)
    assert (norm["precision"], norm["recall"], norm["f1"]) == (0.5, 0.5, 0.5)
    assert m["per_class"]["STTC"]["f1"] == 0.0   # TP 0, FP 1 (row2), FN 1 (row3)
    tp, fp, fn = 1 + 2 + 0 + 1 + 1, 1 + 0 + 1 + 0 + 0, 1 + 0 + 1 + 1 + 0
    assert m["micro_f1"] == pytest.approx(2 * tp / (2 * tp + fp + fn))


def test_undefined_metrics_are_none_not_zero():
    y, p = _random_problem()
    y[:, 4] = 0                              # HYP has no positives in this split
    p[:, 1] = 0.0                            # MI never predicted positive
    m = compute_metrics(y, p, 0.5)
    assert m["per_class"]["HYP"]["auroc"] is None and m["per_class"]["HYP"]["recall"] is None
    assert m["undefined"]["auroc"] == ["HYP"]
    assert m["per_class"]["MI"]["precision"] is None
    assert m["macro_auroc"] == pytest.approx(np.mean([m["per_class"][c]["auroc"]
                                                      for c in ["NORM", "MI", "STTC", "CD"]]))


def test_metrics_reject_nan():
    y, p = _random_problem()
    p[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        compute_metrics(y, p)


# --------------------------------------------------------------------------- thresholds

def test_threshold_grid():
    g = threshold_grid()
    assert len(g) == 99 and g[0] == 0.01 and g[-1] == 0.99


def test_calibration_finds_separating_threshold():
    y = np.zeros((20, 5))
    y[10:] = 1
    p = np.tile(np.r_[np.full(10, 0.2), np.full(10, 0.35)][:, None], (1, 5))
    res = calibrate_thresholds(y, p, threshold_grid())
    for c in res.values():
        assert c["f1"] == 1.0 and 0.2 < c["threshold"] <= 0.35
        assert c["threshold"] == pytest.approx(0.35)    # plateau (0.21..0.35): closest to 0.5
        assert (c["precision"], c["recall"], c["support"]) == (1.0, 1.0, 10)


def test_calibration_matches_brute_force_f1():
    y, p = _random_problem(seed=3)
    grid = threshold_grid()
    res = calibrate_thresholds(y, p, grid)
    for i, c in enumerate(["NORM", "MI", "STTC", "CD", "HYP"]):
        best = max(compute_metrics(y, p, t)["per_class"][c]["f1"] for t in grid)
        assert res[c]["f1"] == pytest.approx(best)


def test_calibration_requires_positives():
    y, p = _random_problem()
    y[:, 2] = 0
    with pytest.raises(ValueError, match="no positives"):
        calibrate_thresholds(y, p, threshold_grid())


# --------------------------------------------------------------------------- seeding

def test_seed_everything_is_reproducible():
    def draw():
        return random.random(), float(np.random.rand()), float(torch.rand(1))
    seed_everything(42)
    a = draw()
    seed_everything(42)
    assert draw() == a
    seed_everything(43)
    assert draw() != a


def test_rng_state_roundtrip():
    seed_everything(1)
    state = capture_rng_state()
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    restore_rng_state(state)
    assert (random.random(), float(np.random.rand()), float(torch.rand(1))) == a
    assert math.isfinite(a[0])
