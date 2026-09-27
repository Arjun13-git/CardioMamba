"""Real-data metadata tests (skipped when PTB-XL is absent). Numbers are Phase 1 facts."""

import numpy as np

from cardiomamba.data import CLASS_NAMES, SPLIT_FOLDS


def test_record_and_patient_counts(metadata):
    r = metadata.records
    assert len(r) == 21799
    assert r.index.is_unique
    assert r["patient_id"].nunique() == 18869
    # ecg_id is not contiguous (38 duplicates removed upstream)
    assert r.index.max() == 21837


def test_fold_assignment(metadata):
    r = metadata.records
    for split, folds in SPLIT_FOLDS.items():
        assert set(r.loc[r["split"] == split, "strat_fold"]) == set(folds)
    counts = r["split"].value_counts().to_dict()
    assert counts == {"train": 17418, "val": 2183, "test": 2198}


def test_patient_disjoint_splits(metadata):
    assert metadata.patient_overlap() == {"train_val": 0, "train_test": 0, "val_test": 0}


def test_unlabelled_records_excluded(metadata):
    r = metadata.records
    assert int((r["n_labels"] == 0).sum()) == 411
    sizes = {s: len(metadata.split(s)) for s in SPLIT_FOLDS}
    assert sizes == {"train": 17084, "val": 2146, "test": 2158}
    for s in SPLIT_FOLDS:
        assert (metadata.split(s)["n_labels"] > 0).all()
    assert len(metadata.split("train", labelled_only=False)) == 17418


def test_class_support(metadata):
    y = metadata.targets(metadata.records)
    assert dict(zip(CLASS_NAMES, y.sum(axis=0).astype(int).tolist(), strict=True)) == {
        "NORM": 9514, "MI": 5469, "STTC": 5235, "CD": 4898, "HYP": 2649,
    }
    train = metadata.targets(metadata.split("train")).sum(axis=0).astype(int).tolist()
    assert train == [7596, 4379, 4186, 3907, 2119]


def test_known_record_mapping(metadata):
    rec = metadata.records.loc[1]  # scp_codes {'NORM': 100.0, 'LVOLT': 0.0, 'SR': 0.0}
    assert rec["scp_codes"] == {"NORM": 100.0, "LVOLT": 0.0, "SR": 0.0}
    assert rec["filename_hr"] == "records500/00000/00001_hr"
    np.testing.assert_array_equal(metadata.targets(metadata.records.loc[[1]]), [[1, 0, 0, 0, 0]])


def test_all_diagnostic_codes_map_to_five_classes(metadata):
    assert len(metadata.code_to_superclass) == 44
    assert set(metadata.code_to_superclass.values()) == set(CLASS_NAMES)
