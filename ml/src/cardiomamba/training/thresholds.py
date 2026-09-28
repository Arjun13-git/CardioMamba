"""Validation-only per-class decision-threshold calibration.

Objective: for each class independently, choose the threshold t on a fixed grid (default
0.01, 0.02, ..., 0.99) that maximises the class F1 = 2TP / (2TP + FP + FN) of the rule
`probability >= t` on the **validation** predictions. Ties are broken by choosing the
threshold closest to 0.5 (then the lower one), so the choice is deterministic and does not
drift to an extreme when a plateau exists. The test split is never an input here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cardiomamba.data.labels import CLASS_NAMES


def threshold_grid(start: float = 0.01, stop: float = 0.99, step: float = 0.01) -> np.ndarray:
    n = round((stop - start) / step) + 1
    return np.round(start + step * np.arange(n), 10)


def calibrate_thresholds(y_true: np.ndarray, probs: np.ndarray, grid: np.ndarray,
                         class_names: Sequence[str] = CLASS_NAMES) -> dict[str, dict]:
    """Return {class: {threshold, f1, precision, recall, support, predicted_positive}}."""
    y_true = np.asarray(y_true).astype(bool)
    probs = np.asarray(probs, dtype=np.float64)
    result = {}
    for i, name in enumerate(class_names):
        t = y_true[:, i]
        pred = probs[:, i][:, None] >= grid[None, :]               # [N, G]
        tp = (pred & t[:, None]).sum(axis=0)
        fp = (pred & ~t[:, None]).sum(axis=0)
        fn = (~pred & t[:, None]).sum(axis=0)
        denom = 2 * tp + fp + fn
        if t.sum() == 0:
            raise ValueError(f"Class {name} has no positives in the calibration split")
        f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
        best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
        j = best[np.lexsort((grid[best], np.abs(grid[best] - 0.5)))[0]]
        result[name] = {
            "threshold": float(grid[j]),
            "f1": float(f1[j]),
            "precision": float(tp[j] / (tp[j] + fp[j])) if tp[j] + fp[j] > 0 else None,
            "recall": float(tp[j] / (tp[j] + fn[j])),
            "support": int(t.sum()),
            "predicted_positive": int(tp[j] + fp[j]),
        }
    return result
