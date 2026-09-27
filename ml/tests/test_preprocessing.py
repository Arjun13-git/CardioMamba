import numpy as np
import pytest
from scipy.signal import resample_poly

from cardiomamba.data.preprocessing import (
    LEAD_NAMES,
    RESAMPLE_PADTYPE,
    ECGValidationError,
    load_resampled,
    load_wfdb_record,
    resample_to_target,
    validate_signal,
)
from conftest import requires_ptbxl

WFDB_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
MV = ["mV"] * 12


def synthetic(n: int = 5000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.2, size=(n, 12))


# ---------------------------------------------------------------- lead / structure validation

@pytest.mark.parametrize("names", [
    WFDB_LEADS,
    list(LEAD_NAMES),
    [n.lower() for n in WFDB_LEADS],
])
def test_augmented_leads_case_insensitive(names):
    x = synthetic()
    out = validate_signal(x, 500, names, MV)
    np.testing.assert_array_equal(out, x)  # canonical order already; unchanged


def test_permuted_leads_are_reordered_to_canonical():
    x = synthetic()
    perm = np.random.default_rng(1).permutation(12)
    out = validate_signal(x[:, perm], 500, [WFDB_LEADS[i] for i in perm], MV)
    np.testing.assert_array_equal(out, x)


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"signal": synthetic()[:, :11], "lead_names": WFDB_LEADS[:11]}, "shape"),
    ({"lead_names": WFDB_LEADS[:11] + ["V7"]}, "Expected leads"),
    ({"lead_names": WFDB_LEADS[:11] + ["V5"]}, "Expected leads"),  # duplicate lead
    ({"fs": 250}, "sampling rate"),
    ({"signal": synthetic(4999)}, "samples"),
    ({"fs": 100}, "samples"),  # 5000 samples at 100 Hz is not 10 s
    ({"units": ["uV"] * 12}, "units"),
])
def test_invalid_structure_raises(kwargs, match):
    args = {"signal": synthetic(), "fs": 500, "lead_names": WFDB_LEADS, "units": MV} | kwargs
    with pytest.raises(ECGValidationError, match=match):
        validate_signal(**args)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_raises(bad):
    x = synthetic()
    x[100, 3] = bad
    with pytest.raises(ECGValidationError, match="finite"):
        validate_signal(x, 500, WFDB_LEADS, MV)


# ---------------------------------------------------------------- resampling

def test_resample_shape_dtype_and_band_preservation():
    t = np.arange(5000) / 500
    x = np.stack([np.sin(2 * np.pi * 5 * t)] * 12, axis=1)  # 5 Hz, well below 50 Hz
    y = resample_to_target(x, 500)
    assert y.shape == (1000, 12) and y.dtype == np.float32 and y.flags.c_contiguous
    ref = np.sin(2 * np.pi * 5 * np.arange(1000) / 100)
    # Interior (away from filter edge effects) matches the analytic 100 Hz signal; the Kaiser
    # FIR passband ripple is ~0.1% at 5 Hz, so allow 0.5%.
    np.testing.assert_allclose(y[50:-50, 0], ref[50:-50], atol=5e-3)


def test_resample_edges_follow_baseline_offset():
    """padtype="line": a record with a DC offset and drift keeps correct boundary samples."""
    assert RESAMPLE_PADTYPE == "line"
    t = np.arange(5000) / 500
    x = np.stack([1.0 + 0.05 * t + 0.3 * np.sin(2 * np.pi * 5 * t)] * 12, axis=1)
    y = resample_to_target(x, 500)
    t100 = np.arange(1000) / 100
    ref = 1.0 + 0.05 * t100 + 0.3 * np.sin(2 * np.pi * 5 * t100)
    edges = np.r_[0:10, 990:1000]
    # Residual ~0.01 mV comes from the sine's slope at the boundary (not extrapolable).
    np.testing.assert_allclose(y[edges, 0], ref[edges], atol=0.02)
    # Zero padding ("constant") pulls the boundary samples towards 0 mV by ~0.4 mV.
    zero_pad = resample_poly(x[:, :1], 1, 5, axis=0, padtype="constant")[:, 0]
    assert np.abs(zero_pad[edges] - ref[edges]).max() > 0.3


def test_resample_attenuates_above_new_nyquist():
    t = np.arange(5000) / 500
    x = np.stack([np.sin(2 * np.pi * 120 * t)] * 12, axis=1)  # would alias to 20 Hz if sliced
    y = resample_to_target(x, 500)
    assert np.abs(y[50:-50]).max() < 0.01
    assert np.abs(x[::5]).max() > 0.5  # naive slicing keeps a large aliased component


def test_100hz_input_passes_through():
    x = synthetic(1000)
    np.testing.assert_allclose(resample_to_target(x, 100), x.astype(np.float32))


# ---------------------------------------------------------------- real PTB-XL records

REAL_IDS = [1, 1000, 9000, 21837]  # spread across folds, incl. the highest ecg_id


@requires_ptbxl
@pytest.mark.parametrize("ecg_id", REAL_IDS)
def test_real_record_load_and_resample(metadata, ecg_id):
    path = metadata.dataset_dir / metadata.records.loc[ecg_id, "filename_hr"]
    raw = load_wfdb_record(path)
    assert raw.shape == (5000, 12) and np.isfinite(raw).all()
    y = load_resampled(path)
    assert y.shape == (1000, 12) and y.dtype == np.float32 and np.isfinite(y).all()
    # the 100 Hz signal stays in physical units (mV) with comparable amplitude
    assert np.abs(y).max() == pytest.approx(np.abs(raw).max(), rel=0.35)


@requires_ptbxl
def test_real_record_is_deterministic(metadata):
    path = metadata.dataset_dir / metadata.records.loc[1000, "filename_hr"]
    np.testing.assert_array_equal(load_resampled(path), load_resampled(path))
