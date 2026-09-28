"""Decode an uploaded ECG CSV into a validated [T, 12] array (untrusted input).

Supported format (API contract): a header row with the 12 lead names
I,II,III,aVR,aVL,aVF,V1..V6 (case-insensitive), values in mV, and exactly 5000 rows (10 s at
500 Hz) or 1000 rows (10 s at 100 Hz). The sampling rate is inferred from the row count; no
client-supplied metadata is trusted. Validation itself is the production `validate_signal`.
"""

from __future__ import annotations

import csv
import io

import numpy as np

from cardiomamba.data.preprocessing import (
    RAW_FS,
    RAW_SAMPLES,
    TARGET_FS,
    TARGET_SAMPLES,
    ECGValidationError,
    validate_signal,
)

ROWS_TO_FS = {RAW_SAMPLES: RAW_FS, TARGET_SAMPLES: TARGET_FS}


def decode_csv(raw: bytes) -> tuple[np.ndarray, int]:
    """Return (signal [T, 12] float64 mV in canonical lead order, sampling rate in Hz)."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ECGValidationError("File is not UTF-8 text") from exc
    lines = text.splitlines()
    if len(lines) < 2:
        raise ECGValidationError("CSV must have a header row and signal rows")
    header = next(csv.reader([lines[0]]))
    try:
        values = np.loadtxt(io.StringIO("\n".join(lines[1:])), delimiter=",", dtype=np.float64,
                            ndmin=2)
    except ValueError as exc:
        raise ECGValidationError("CSV contains non-numeric or ragged rows") from exc
    fs = ROWS_TO_FS.get(values.shape[0])
    if fs is None:
        raise ECGValidationError(
            f"Expected {RAW_SAMPLES} rows (10 s at {RAW_FS} Hz) or {TARGET_SAMPLES} rows "
            f"(10 s at {TARGET_FS} Hz), got {values.shape[0]}")
    return validate_signal(values, fs, header), fs
