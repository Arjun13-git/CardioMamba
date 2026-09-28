"""End-to-end training engine tests on CPU with a tiny synthetic problem (no PTB-XL needed)."""

import csv
import json
import logging
from dataclasses import replace

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from cardiomamba.models import CardioMamba
from cardiomamba.training import ConfigMismatchError, NonFiniteError, run_training
from cardiomamba.training import data as training_data
from cardiomamba.training.checkpoint import load_checkpoint
from cardiomamba.training.data import make_loader
from cardiomamba.training.engine import (
    RunState,
    build_optimizer,
    evaluate_validation,
    loss_fn,
    train_one_epoch,
)
from cardiomamba.training.schedule import build_scheduler
from training_helpers import TINY_MODEL, SyntheticECG, tiny_config


@pytest.fixture(scope="module")
def datasets():
    return SyntheticECG(96, seed=0), SyntheticECG(48, seed=1)


def _run(tmp_path, datasets, name="run", **kw):
    cfg_over = {k: v for k, v in kw.items() if k not in ("resume_from", "max_epochs_this_run")}
    return run_training(tiny_config(**cfg_over), *datasets, run_dir=tmp_path / name,
                        model_config=TINY_MODEL, resume_from=kw.get("resume_from"),
                        max_epochs_this_run=kw.get("max_epochs_this_run"))


def _csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def test_loss_is_bce_with_logits_on_raw_logits():
    logits = torch.tensor([[8.0, -8.0, 0.3, -0.2, 2.0]])
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0]])
    assert isinstance(loss_fn(), torch.nn.BCEWithLogitsLoss)
    expected = -(target * F.logsigmoid(logits) + (1 - target) * F.logsigmoid(-logits)).mean()
    torch.testing.assert_close(loss_fn()(logits, target), expected)


def test_weight_decay_groups():
    torch.manual_seed(0)
    model = CardioMamba(TINY_MODEL)
    opt = build_optimizer(model, tiny_config())
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.05 and no_decay["weight_decay"] == 0.0
    ids = {id(p) for p in no_decay["params"]}
    mixer = model.blocks[0].forward_mixer
    for p in (mixer.A_log, mixer.D, mixer.dt_proj.bias, model.norm_f.weight, model.head.bias):
        assert id(p) in ids
    assert id(mixer.in_proj.weight) not in ids and id(model.head.weight) not in ids


def _one_epoch(cfg, dataset, record_norms=None, model_config=TINY_MODEL):
    torch.manual_seed(0)
    model = CardioMamba(model_config)
    opt = build_optimizer(model, cfg)
    loader = make_loader(dataset, cfg.batch_size, False, 0, False)
    sched, _ = build_scheduler(opt, 100, cfg.warmup_ratio, cfg.learning_rate,
                               cfg.min_learning_rate)
    if record_norms is not None:           # total grad norm seen by each optimiser step
        def pre_step(optimizer, args, kwargs):
            grads = [p.grad for g in optimizer.param_groups for p in g["params"]
                     if p.grad is not None]
            record_norms.append(torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(g) for g in grads])).item())
        opt.register_step_pre_hook(pre_step)
    state = RunState()
    loss = train_one_epoch(model, loader, opt, sched, cfg, torch.device("cpu"), "fp32", state,
                           logging.getLogger("test"))
    return model, state, loss, sched


def test_train_step_updates_params_and_counts_optimizer_steps():
    ds = SyntheticECG(40)                                  # 5 micro-batches of 8
    cfg = tiny_config(gradient_accumulation_steps=2)
    model, state, loss, sched = _one_epoch(cfg, ds)
    assert state.global_step == 3 and sched.last_epoch == 3   # ceil(5 / 2), not 5
    assert torch.isfinite(torch.tensor(loss))
    torch.manual_seed(0)
    model0 = CardioMamba(TINY_MODEL)
    assert any(not torch.equal(a, b) for a, b in zip(model0.parameters(), model.parameters(),
                                                      strict=True))


def test_gradient_clipping_applied_before_step():
    norms = []
    _one_epoch(tiny_config(gradient_clip_norm=1e-3), SyntheticECG(32), norms)
    assert norms and max(norms) <= 1e-3 * (1 + 1e-4)


def test_accumulation_matches_large_batch():
    """Batch 16 x 1 and batch 8 x 2 give the same updates (dropout off: masks would differ)."""
    ds = SyntheticECG(32)
    no_dropout = replace(TINY_MODEL, dropout=0.0)
    m_big = _one_epoch(tiny_config(batch_size=16), ds, model_config=no_dropout)[0]
    m_acc = _one_epoch(tiny_config(batch_size=8, gradient_accumulation_steps=2), ds,
                       model_config=no_dropout)[0]
    for a, b in zip(m_big.parameters(), m_acc.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_non_finite_gradients_abort(tmp_path):
    with pytest.raises(NonFiniteError, match="Non-finite"):
        run_training(tiny_config(), SyntheticECG(32, nan_at=3), SyntheticECG(16, seed=1),
                     run_dir=tmp_path / "nan", model_config=TINY_MODEL)


def test_full_run_outputs_and_best_selection(tmp_path, datasets):
    res = _run(tmp_path, datasets, epochs=4)
    d = res.run_dir
    for f in ("config.yaml", "model_config.json", "environment.json", "train.log",
              "metrics.csv", "metrics.jsonl", "last.pt", "best.pt", "thresholds.json",
              "validation_best.json"):
        assert (d / f).exists(), f
    rows = _csv(d / "metrics.csv")
    assert [int(r["epoch"]) for r in rows] == [1, 2, 3, 4]
    aurocs = [float(r["val_macro_auroc"]) for r in rows]
    best_epoch = max(range(4), key=lambda i: (aurocs[i], -i)) + 1   # first maximum
    assert res.state.best_epoch == best_epoch
    best, last = load_checkpoint(d / "best.pt"), load_checkpoint(d / "last.pt")
    assert best["epoch"] == best_epoch and last["epoch"] == 4
    assert best["best_val_macro_auroc"] == pytest.approx(max(aurocs))
    assert last["global_step"] == 4 * 12                         # 96 / 8 per epoch
    for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
                "train_config", "model_config", "random_seed", "rng_state", "best_epoch",
                "best_val_macro_auroc", "global_step"):
        assert key in last
    thr = json.loads((d / "thresholds.json").read_text())
    assert thr["checkpoint_epoch"] == best_epoch and set(thr["per_class"]) == {
        "NORM", "MI", "STTC", "CD", "HYP"}
    assert "validation" in thr["objective"]
    # training loss decreases on this learnable problem
    assert float(rows[-1]["train_loss"]) < float(rows[0]["train_loss"])


def test_resume_reproduces_uninterrupted_run(tmp_path, datasets):
    full = _run(tmp_path, datasets, name="full", epochs=3)
    part = _run(tmp_path, datasets, name="part", epochs=3, max_epochs_this_run=1)
    assert part.state.epoch == 1 and part.thresholds is None
    resumed = _run(tmp_path, datasets, epochs=3, resume_from=tmp_path / "part" / "last.pt")
    assert resumed.run_dir == tmp_path / "part" and resumed.state.global_step == 36
    a = load_checkpoint(tmp_path / "full" / "last.pt")["model_state_dict"]
    b = load_checkpoint(tmp_path / "part" / "last.pt")["model_state_dict"]
    for k in a:
        assert torch.equal(a[k], b[k]), k                      # bit-identical on CPU
    drop = {"epoch_time_s"}
    rows_a = [{k: v for k, v in r.items() if k not in drop} for r in _csv(full.run_dir /
                                                                           "metrics.csv")]
    rows_b = [{k: v for k, v in r.items() if k not in drop} for r in _csv(resumed.run_dir /
                                                                           "metrics.csv")]
    assert rows_a == rows_b
    log = (resumed.run_dir / "train.log").read_text()
    assert "Resuming from epoch 1 | global_step = 12" in log


def test_resume_reproduces_uninterrupted_run_with_persistent_workers(tmp_path, datasets):
    """Regression: with persistent workers the resumed run creates a fresh loader iterator
    while the uninterrupted run reuses one; the shuffle order must not depend on that."""
    full = _run(tmp_path, datasets, name="full_w", epochs=2, num_workers=2)
    _run(tmp_path, datasets, name="part_w", epochs=2, num_workers=2, max_epochs_this_run=1)
    resumed = _run(tmp_path, datasets, epochs=2, num_workers=2,
                   resume_from=tmp_path / "part_w" / "last.pt")
    a = load_checkpoint(full.run_dir / "last.pt")["model_state_dict"]
    b = load_checkpoint(resumed.run_dir / "last.pt")["model_state_dict"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_shuffle_order_is_independent_of_iterator_creation():
    ds = SyntheticECG(40)
    gen = torch.Generator()
    loader = make_loader(ds, 8, True, 0, False, gen, seed=0)

    def order():
        gen.manual_seed(123)
        return [i for b in loader for i in b["ecg_id"].tolist()]
    first = order()
    for _ in range(3):
        _ = torch.empty((), dtype=torch.int64).random_(generator=loader.generator)
    assert order() == first and sorted(first) == list(range(1, 41))


def test_resume_rejects_config_mismatch(tmp_path, datasets):
    _run(tmp_path, datasets, name="r", epochs=3, max_epochs_this_run=1)
    with pytest.raises(ConfigMismatchError, match="learning_rate"):
        _run(tmp_path, datasets, epochs=3, learning_rate=1e-4,
             resume_from=tmp_path / "r" / "last.pt")


def test_early_stopping(tmp_path, datasets):
    # a (practically) frozen model cannot improve validation AUROC after epoch 1
    res = _run(tmp_path, datasets, epochs=10, early_stopping_patience=2, learning_rate=1e-12,
               min_learning_rate=1e-13)
    assert res.state.epoch == 3 and res.state.best_epoch == 1 and res.state.finished
    assert "early stopping" in (res.run_dir / "train.log").read_text()


def test_validation_only_evaluation(tmp_path, datasets):
    res = _run(tmp_path, datasets, epochs=2)
    loader = make_loader(datasets[1], 8, False, 0, False)
    out = evaluate_validation(res.run_dir / "best.pt", loader, torch.device("cpu"), "fp32",
                              tiny_config(), calibrate=True)
    best_row = res.state.history[res.state.best_epoch - 1]
    assert out["metrics_at_0.5"]["macro_auroc"] == pytest.approx(best_row["val_macro_auroc"])
    assert out["val_loss"] == pytest.approx(best_row["val_loss"])
    assert out["split"] == "validation" and "per_class" in out["thresholds"]
    saved = json.loads((res.run_dir / "thresholds.json").read_text())
    assert {c: v["threshold"] for c, v in out["thresholds"]["per_class"].items()} == \
        {c: v["threshold"] for c, v in saved["per_class"].items()}


def test_training_datasets_never_touch_test_split(monkeypatch):
    requested = []

    class FakeMeta:
        dataset_dir = "unused"

        def split(self, name, labelled_only=True):
            requested.append(name)
            if name == "test":
                raise AssertionError("test split requested during training")
            return pd.DataFrame({"filename_hr": ["x"], "n_labels": [1], "NORM": [1.0],
                                 "MI": [0.0], "STTC": [0.0], "CD": [0.0], "HYP": [0.0]},
                                index=pd.Index([1], name="ecg_id"))

    monkeypatch.setattr(training_data.PTBXLMetadata, "load", staticmethod(lambda d: FakeMeta()))
    monkeypatch.setattr(training_data.NormalizationStats, "load",
                        staticmethod(lambda p: object()))
    training_data.build_train_val_datasets(tiny_config())
    training_data.build_validation_dataset(tiny_config())
    assert requested == ["train", "val", "val"]
