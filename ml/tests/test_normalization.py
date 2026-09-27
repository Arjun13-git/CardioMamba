import json

import numpy as np
import pytest

from cardiomamba.data.preprocessing import (
    LEAD_NAMES,
    STD_FLOOR_MV,
    NormalizationStats,
    PTBXLPreprocessor,
    compute_normalization_stats,
    compute_training_stats,
    load_resampled,
)
from conftest import STATS_PATH, requires_ptbxl


def records(n=5, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(rng.normal(0, 0.1, 12), rng.uniform(0.1, 0.4, 12), size=(1000, 12))
            for _ in range(n)]


def test_stats_equal_global_per_lead_mean_std():
    recs = records()
    stats = compute_normalization_stats(recs)
    allx = np.concatenate(recs, axis=0)
    assert stats.mean.shape == (12,) and stats.std.shape == (12,)
    np.testing.assert_allclose(stats.mean, allx.mean(axis=0), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(stats.std, allx.std(axis=0), rtol=1e-12)
    assert stats.metadata["n_records"] == 5
    assert stats.metadata["n_samples_per_lead"] == 5000


def test_std_floor_applied_to_flat_lead():
    recs = records()
    for r in recs:
        r[:, 4] = 0.25  # constant lead
    stats = compute_normalization_stats(recs)
    assert stats.std[4] == STD_FLOOR_MV
    assert stats.metadata["floored_leads"] == [LEAD_NAMES[4]]
    assert np.isfinite(stats.apply(recs[0])).all()


def test_apply_uses_frozen_stats_not_per_record_zscore():
    stats = compute_normalization_stats(records())
    x = records(1, seed=99)[0]
    out = stats.apply(x)
    np.testing.assert_allclose(out, (x - stats.mean) / stats.std, rtol=1e-6, atol=1e-6)
    # Per-record z-scoring would be invariant to rescaling the record; frozen stats are not.
    np.testing.assert_allclose(stats.apply(2 * x), 2 * out + stats.mean / stats.std,
                               rtol=1e-5, atol=1e-5)
    assert not np.allclose(out.mean(axis=0), 0, atol=1e-3)


def test_save_load_roundtrip_is_exact(tmp_path):
    stats = compute_normalization_stats(records(), {"source_folds": [1, 2, 3, 4, 5, 6, 7, 8]})
    path = tmp_path / "stats.json"
    stats.save(path)
    loaded = NormalizationStats.load(path)
    np.testing.assert_array_equal(loaded.mean, stats.mean)
    np.testing.assert_array_equal(loaded.std, stats.std)
    assert loaded.metadata["source_folds"] == [1, 2, 3, 4, 5, 6, 7, 8]


def test_load_rejects_wrong_lead_order(tmp_path):
    stats = compute_normalization_stats(records())
    data = stats.to_dict() | {"lead_names": list(reversed(LEAD_NAMES))}
    (tmp_path / "bad.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Lead order"):
        NormalizationStats.load(tmp_path / "bad.json")


@requires_ptbxl
def test_training_stats_read_only_train_folds(metadata):
    """compute_training_stats must never load a validation/test record."""
    fold_of = dict(zip(metadata.records["filename_hr"], metadata.records["strat_fold"],
                       strict=True))
    seen_folds = set()
    n_loaded = 0

    def spy_loader(path):
        nonlocal n_loaded
        rel = str(path.relative_to(metadata.dataset_dir))
        seen_folds.add(fold_of[rel])
        n_loaded += 1
        return np.zeros((1000, 12)) + fold_of[rel]

    stats = compute_training_stats(metadata, loader=spy_loader)
    assert seen_folds == set(range(1, 9))
    assert n_loaded == 17084
    assert stats.metadata["source_folds"] == list(range(1, 9))


@requires_ptbxl
def test_committed_stats_artifact(metadata):
    stats = NormalizationStats.load(STATS_PATH)
    assert stats.mean.shape == (12,) and stats.std.shape == (12,)
    assert stats.metadata["source_folds"] == list(range(1, 9))
    assert stats.metadata["n_records"] == len(metadata.split("train")) == 17084
    assert stats.metadata["n_samples_per_lead"] == 17084 * 1000
    assert stats.metadata["floored_leads"] == []
    assert (stats.std > 0.05).all() and (np.abs(stats.mean) < 0.05).all()  # plausible mV


@requires_ptbxl
def test_val_and_test_use_the_same_frozen_stats(metadata, ptbxl_dir):
    stats = NormalizationStats.load(STATS_PATH)
    pre = PTBXLPreprocessor(ptbxl_dir, stats)
    for split in ("val", "test"):
        fn = metadata.split(split)["filename_hr"].iloc[0]
        raw = load_resampled(ptbxl_dir / fn)
        np.testing.assert_allclose(pre(fn), (raw - stats.mean) / stats.std, rtol=1e-5, atol=1e-6)
