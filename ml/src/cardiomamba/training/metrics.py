"""Multi-label classification metrics computed from sigmoid probabilities.

Undefined values are reported as None (never silently replaced):
  * AUROC / AUPRC of a class whose targets are all 0 or all 1 in the evaluated split;
  * precision when nothing is predicted positive; recall when the class has no positives;
  * F1 = 2TP / (2TP + FP + FN) is undefined only when TP + FP + FN = 0.
Macro averages are taken over classes where the metric is defined, and the list of
undefined classes is returned alongside.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from cardiomamba.data.labels import CLASS_NAMES


def _ratio(num: float, den: float) -> float | None:
    return float(num / den) if den > 0 else None


def _mean(values: list[float | None]) -> float | None:
    defined = [v for v in values if v is not None]
    return float(np.mean(defined)) if defined else None


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    """Per-class TP/FP/FN/TN for binary arrays [N, C]."""
    y_true, y_pred = y_true.astype(bool), y_pred.astype(bool)
    return {
        "tp": (y_true & y_pred).sum(axis=0),
        "fp": (~y_true & y_pred).sum(axis=0),
        "fn": (y_true & ~y_pred).sum(axis=0),
        "tn": (~y_true & ~y_pred).sum(axis=0),
    }


def compute_metrics(y_true: np.ndarray, probs: np.ndarray,
                    thresholds: float | Sequence[float] = 0.5,
                    class_names: Sequence[str] = CLASS_NAMES) -> dict:
    """AUROC/AUPRC (threshold-free) and precision/recall/F1 at `thresholds` (per class)."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=np.float64)
    if y_true.shape != probs.shape or y_true.ndim != 2 or y_true.shape[1] != len(class_names):
        raise ValueError(f"Expected [N, {len(class_names)}] arrays, got {y_true.shape}, "
                         f"{probs.shape}")
    if not np.isfinite(probs).all():
        raise ValueError("Probabilities contain NaN/Inf")
    thr = np.broadcast_to(np.asarray(thresholds, dtype=np.float64), (len(class_names),))
    y_pred = probs >= thr
    counts = confusion_counts(y_true, y_pred)

    per_class, undefined = {}, {"auroc": [], "auprc": []}
    for i, name in enumerate(class_names):
        t = y_true[:, i]
        degenerate = t.min() == t.max()
        auroc = None if degenerate else float(roc_auc_score(t, probs[:, i]))
        auprc = None if degenerate else float(average_precision_score(t, probs[:, i]))
        if degenerate:
            undefined["auroc"].append(name)
            undefined["auprc"].append(name)
        tp, fp, fn = (int(counts[k][i]) for k in ("tp", "fp", "fn"))
        per_class[name] = {
            "auroc": auroc,
            "auprc": auprc,
            "threshold": float(thr[i]),
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn),
            "support": int(t.sum()),
            "predicted_positive": tp + fp,
        }
    tp, fp, fn = (int(counts[k].sum()) for k in ("tp", "fp", "fn"))
    return {
        "macro_auroc": _mean([per_class[c]["auroc"] for c in class_names]),
        "macro_auprc": _mean([per_class[c]["auprc"] for c in class_names]),
        "macro_f1": _mean([per_class[c]["f1"] for c in class_names]),
        "micro_f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "per_class": per_class,
        "undefined": undefined,
        "n_samples": int(y_true.shape[0]),
    }
