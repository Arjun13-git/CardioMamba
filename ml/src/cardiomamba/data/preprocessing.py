"""Shared ECG preprocessing (`prep-v1`) used by training, evaluation and inference.

Pipeline:
    500 Hz WFDB (physical units, mV)
    -> validate (12 leads, canonical names case-insensitively, 500 Hz, 5000 samples, mV, finite)
    -> no filtering
    -> resample 500 -> 100 Hz with scipy.signal.resample_poly
       (up=1, down=5, window=("kaiser", 5.0), padtype="line")                 [1000, 12]
    -> per-lead normalization with frozen training-fold statistics (folds 1-8)

Why resample_poly: it applies a zero-phase FIR anti-aliasing low-pass (Kaiser window,
cutoff at the new Nyquist of 50 Hz) before decimating by the integer factor 5, so content
above 50 Hz is attenuated instead of aliasing into the diagnostic band, without the
circular-boundary assumption of FFT resampling or the aliasing of plain slicing. Filter
parameters are pinned explicitly so a SciPy default change cannot alter `prep-v1`.

Why padtype="line": the FIR filter needs samples beyond the recording boundaries. SciPy's
default ("constant") pads with zeros, which for records with a baseline offset creates an
artificial step and distorted up to ~10 output samples (100 ms) at each end (Phase 2
measurement). "line" removes the straight line through the first and last samples before
filtering and adds it back afterwards, so the boundary continuation follows the signal's own
baseline instead of dropping to 0 mV.

Arrays are float32 in [T, C] layout (time, leads) with leads in LEAD_NAMES order.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import wfdb
from scipy.signal import resample_poly

if TYPE_CHECKING:
    from cardiomamba.data.metadata import PTBXLMetadata

PREPROCESSING_VERSION = "prep-v1"
LEAD_NAMES: tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
)
NUM_LEADS = len(LEAD_NAMES)
RAW_FS = 500
TARGET_FS = 100
DURATION_S = 10
RAW_SAMPLES = RAW_FS * DURATION_S        # 5000
TARGET_SAMPLES = TARGET_FS * DURATION_S  # 1000
RESAMPLE_UP, RESAMPLE_DOWN = 1, 5
RESAMPLE_WINDOW = ("kaiser", 5.0)  # SciPy's default, pinned
RESAMPLE_PADTYPE = "line"          # not SciPy's default ("constant"); see module docstring

# Std floor = one ADC step of PTB-XL (16-bit, 1 uV/LSB = 0.001 mV). A lead whose training std
# fell below this would carry no measurable signal; flooring avoids division by ~0 and the
# affected leads are recorded in the stats artifact. (Not expected to trigger on PTB-XL.)
STD_FLOOR_MV = 1e-3


class ECGValidationError(ValueError):
    """Raised when an ECG does not match the expected structure; never silently fixed."""


def canonical_lead_order(lead_names: Sequence[str]) -> np.ndarray:
    """Column indices that reorder `lead_names` into LEAD_NAMES (case-insensitive match).

    Raises ECGValidationError unless the names are exactly the 12 canonical leads.
    """
    upper = [str(n).strip().upper() for n in lead_names]
    expected = [n.upper() for n in LEAD_NAMES]
    if len(upper) != NUM_LEADS or len(set(upper)) != NUM_LEADS or set(upper) != set(expected):
        raise ECGValidationError(
            f"Expected leads {list(LEAD_NAMES)} (case-insensitive), got {list(lead_names)}")
    return np.array([upper.index(n) for n in expected])


def validate_signal(signal: np.ndarray, fs: float, lead_names: Sequence[str],
                    units: Sequence[str] | None = None) -> np.ndarray:
    """Validate a [T, 12] physical-unit ECG and return it in canonical lead order.

    Accepts 500 Hz (5000 samples) or 100 Hz (1000 samples) 10 s recordings.
    """
    signal = np.asarray(signal)
    if signal.ndim != 2 or signal.shape[1] != NUM_LEADS:
        raise ECGValidationError(f"Expected shape [T, {NUM_LEADS}], got {list(signal.shape)}")
    expected_len = {RAW_FS: RAW_SAMPLES, TARGET_FS: TARGET_SAMPLES}.get(int(fs))
    if expected_len is None or fs != int(fs):
        raise ECGValidationError(f"Unsupported sampling rate {fs} Hz (expected 500 or 100)")
    if signal.shape[0] != expected_len:
        raise ECGValidationError(
            f"Expected {expected_len} samples at {int(fs)} Hz, got {signal.shape[0]}")
    if units is not None and any(str(u).strip().lower() != "mv" for u in units):
        raise ECGValidationError(f"Expected all units to be mV, got {list(units)}")
    if not np.issubdtype(signal.dtype, np.floating) or not np.isfinite(signal).all():
        raise ECGValidationError("Signal must be floating point and contain only finite values")
    order = canonical_lead_order(lead_names)
    if np.array_equal(order, np.arange(NUM_LEADS)):
        return signal
    return signal[:, order]


def load_wfdb_record(record_path: str | Path) -> np.ndarray:
    """Read a 500 Hz PTB-XL WFDB record in physical units -> validated float64 [5000, 12].

    `record_path` is the path without extension (e.g. .../records500/00000/00001_hr).
    """
    rec = wfdb.rdrecord(str(record_path), physical=True)
    if rec.fs != RAW_FS:
        raise ECGValidationError(f"{record_path}: expected {RAW_FS} Hz, got {rec.fs}")
    if rec.n_sig != NUM_LEADS:
        raise ECGValidationError(f"{record_path}: expected {NUM_LEADS} signals, got {rec.n_sig}")
    try:
        return validate_signal(rec.p_signal, rec.fs, rec.sig_name, rec.units)
    except ECGValidationError as exc:
        raise ECGValidationError(f"{record_path}: {exc}") from exc


def resample_to_target(signal: np.ndarray, fs: int) -> np.ndarray:
    """Validated [T, 12] signal at 500 or 100 Hz -> float32 [1000, 12] at 100 Hz."""
    if fs == TARGET_FS:
        out = signal
    elif fs == RAW_FS:
        out = resample_poly(signal, RESAMPLE_UP, RESAMPLE_DOWN, axis=0,
                            window=RESAMPLE_WINDOW, padtype=RESAMPLE_PADTYPE)
    else:
        raise ECGValidationError(f"Unsupported sampling rate {fs} Hz")
    if out.shape != (TARGET_SAMPLES, NUM_LEADS):
        raise ECGValidationError(f"Resampled shape {list(out.shape)} != [1000, 12]")
    return np.ascontiguousarray(out, dtype=np.float32)


def load_resampled(record_path: str | Path) -> np.ndarray:
    """PTB-XL 500 Hz record -> validated, resampled, un-normalized float32 [1000, 12] (mV)."""
    return resample_to_target(load_wfdb_record(record_path), RAW_FS)


@dataclass(frozen=True)
class NormalizationStats:
    """Frozen per-lead statistics from training folds; applied unchanged everywhere."""

    mean: np.ndarray  # float64 [12]
    std: np.ndarray   # float64 [12], already floored at STD_FLOOR_MV
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("mean", "std"):
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.shape != (NUM_LEADS,) or not np.isfinite(arr).all():
                raise ValueError(f"{name} must be finite with shape [{NUM_LEADS}]")
            object.__setattr__(self, name, arr)
        if (self.std < STD_FLOOR_MV).any():
            raise ValueError("std values must be >= STD_FLOOR_MV")

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """(x - train_mean) / train_std per lead; input/output float32 [T, 12]."""
        return ((signal - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {**self.metadata, "lead_names": list(LEAD_NAMES),
                "mean": self.mean.tolist(), "std": self.std.tolist()}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> NormalizationStats:
        data = json.loads(Path(path).read_text())
        if data.get("lead_names") != list(LEAD_NAMES):
            raise ValueError(f"Lead order in {path} does not match {list(LEAD_NAMES)}")
        if data.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ValueError(f"{path} was built for {data.get('preprocessing_version')!r}, "
                             f"expected {PREPROCESSING_VERSION!r}")
        meta = {k: v for k, v in data.items() if k not in ("mean", "std", "lead_names")}
        return cls(mean=np.array(data["mean"]), std=np.array(data["std"]), metadata=meta)


def compute_normalization_stats(signals: Iterable[np.ndarray],
                                metadata: dict | None = None) -> NormalizationStats:
    """Per-lead mean/std (population, ddof=0) over all samples of all given [T, 12] signals.

    Streaming and order-deterministic: per-record float64 moments are merged sequentially
    with Chan et al.'s parallel-variance formula, so memory is O(one record).
    """
    n_total = 0
    mean = np.zeros(NUM_LEADS)
    m2 = np.zeros(NUM_LEADS)
    n_records = 0
    for sig in signals:
        x = np.asarray(sig, dtype=np.float64)
        n = x.shape[0]
        r_mean = x.mean(axis=0)
        r_m2 = ((x - r_mean) ** 2).sum(axis=0)
        delta = r_mean - mean
        new_n = n_total + n
        mean = mean + delta * (n / new_n)
        m2 = m2 + r_m2 + delta**2 * (n_total * n / new_n)
        n_total = new_n
        n_records += 1
    if n_records == 0:
        raise ValueError("No signals supplied for normalization statistics")
    raw_std = np.sqrt(m2 / n_total)
    floored = [LEAD_NAMES[i] for i in np.flatnonzero(raw_std < STD_FLOOR_MV)]
    meta = {
        "preprocessing_version": PREPROCESSING_VERSION,
        **(metadata or {}),
        "n_records": n_records,
        "n_samples_per_lead": int(n_total),
        "std_floor_mV": STD_FLOOR_MV,
        "floored_leads": floored,
    }
    return NormalizationStats(mean=mean, std=np.maximum(raw_std, STD_FLOOR_MV), metadata=meta)


def compute_training_stats(metadata: PTBXLMetadata,
                           loader: Callable[[Path], np.ndarray] = load_resampled,
                           workers: int = 1) -> NormalizationStats:
    """Normalization stats from the labelled **training split only** (folds 1-8).

    Validation (fold 9) and test (fold 10) records are never read. Records are loaded in
    ecg_id order (and merged in that order even when `workers > 1`), so the result is
    deterministic.
    """
    from cardiomamba.data.metadata import DATASET_VERSION, SPLIT_FOLDS

    train = metadata.split("train", labelled_only=True)
    if not set(train["strat_fold"]) <= set(SPLIT_FOLDS["train"]):
        raise AssertionError("Training split contains non-training folds")
    paths = [metadata.dataset_dir / f for f in train["filename_hr"]]
    meta = {
        "dataset_version": DATASET_VERSION,
        "source_split": "train",
        "source_folds": list(SPLIT_FOLDS["train"]),
        "labelled_records_only": True,
        "sampling_rate_hz": TARGET_FS,
    }
    if workers <= 1:
        return compute_normalization_stats(map(loader, paths), meta)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return compute_normalization_stats(pool.map(loader, paths, chunksize=64), meta)


class PTBXLPreprocessor:
    """Record path -> model-ready float32 [1000, 12] using frozen normalization stats."""

    def __init__(self, dataset_dir: str | Path, stats: NormalizationStats,
                 loader: Callable[[Path], np.ndarray] = load_resampled) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.stats = stats
        self._loader = loader

    def __call__(self, filename_hr: str) -> np.ndarray:
        return self.stats.apply(self._loader(self.dataset_dir / filename_hr))
