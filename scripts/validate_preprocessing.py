"""Phase 2 evidence that the PTB-XL preprocessing pipeline (`prep-v1`) is correct.

Runs on a deterministic, fold-stratified subset (default 100 labelled records per split) and
recomputes the training-fold normalization statistics to confirm the committed artifact.

Usage:  uv run python scripts/validate_preprocessing.py [--per-split 100] [--workers 8]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from cardiomamba.data import (
    CLASS_NAMES,
    LEAD_NAMES,
    PREPROCESSING_VERSION,
    SPLIT_FOLDS,
    NormalizationStats,
    PTBXLMetadata,
    PTBXLMultilabelDataset,
    PTBXLPreprocessor,
    compute_training_stats,
    load_wfdb_record,
    resample_to_target,
)
from cardiomamba.data.labels import multi_hot, superclasses_for
from cardiomamba.data.preprocessing import (
    RESAMPLE_DOWN,
    RESAMPLE_PADTYPE,
    RESAMPLE_UP,
    RESAMPLE_WINDOW,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "ml" / "data" / "raw" / "ptb-xl-1.0.3"
STATS_PATH = REPO_ROOT / "ml" / "configs" / "normalization_stats_prep-v1.json"


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--per-split", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    meta = PTBXLMetadata.load(args.dataset_dir)
    stats = NormalizationStats.load(STATS_PATH)
    pre = PTBXLPreprocessor(args.dataset_dir, stats)
    splits = {s: meta.split(s) for s in SPLIT_FOLDS}
    checks: dict[str, bool] = {}

    print("Dataset\n-------")
    for s, df in splits.items():
        excluded = int((meta.split(s, labelled_only=False)["n_labels"] == 0).sum())
        print(f"{s:>5}: folds {list(SPLIT_FOLDS[s])}  labelled records {len(df):>6}  "
              f"patients {df['patient_id'].nunique():>6}  excluded (no superclass) {excluded}")
    overlap = meta.patient_overlap()
    checks["Patient split"] = sum(overlap.values()) == 0 and all(
        set(df["strat_fold"]) == set(SPLIT_FOLDS[s]) for s, df in splits.items())
    print(f"patient overlap: {overlap}")

    print("\nPreprocessing\n-------------")
    print(f"version: {PREPROCESSING_VERSION}; lead order: {list(LEAD_NAMES)}")
    first = meta.dataset_dir / splits["train"]["filename_hr"].iloc[0]
    raw = load_wfdb_record(first)
    out = pre(splits["train"]["filename_hr"].iloc[0])
    print("Input sampling rate: 500 Hz   Output sampling rate: 100 Hz")
    print(f"Input shape: {list(raw.shape)}   Output shape: {list(out.shape)} ({out.dtype})")
    print(f"Resampling: scipy.signal.resample_poly(up={RESAMPLE_UP}, down={RESAMPLE_DOWN}, "
          f"window={RESAMPLE_WINDOW}, padtype={RESAMPLE_PADTYPE!r})")

    print("\nNormalization\n-------------")
    print(f"Statistics source: {stats.metadata['source_split']} folds "
          f"{stats.metadata['source_folds']} ({stats.metadata['n_records']} labelled records, "
          f"{stats.metadata['n_samples_per_lead']} samples/lead)")
    print(f"Mean shape: {list(stats.mean.shape)}  Std shape: {list(stats.std.shape)}  "
          f"floored leads: {stats.metadata['floored_leads']}")
    print("mean (mV):", np.round(stats.mean, 5).tolist())
    print("std  (mV):", np.round(stats.std, 5).tolist())
    t0 = time.perf_counter()
    recomputed = compute_training_stats(meta, workers=args.workers)
    t_stats = time.perf_counter() - t0
    checks["Stats artifact reproducible"] = (
        np.array_equal(recomputed.mean, stats.mean) and np.array_equal(recomputed.std, stats.std))
    checks["Stats from train folds only"] = stats.metadata["source_folds"] == list(
        SPLIT_FOLDS["train"]) and stats.metadata["n_records"] == len(splits["train"])
    print(f"recomputed from folds 1-8 in {t_stats:.1f} s -> bit-identical: "
          f"{checks['Stats artifact reproducible']}")

    # Deterministic subset per split
    rng = np.random.default_rng(args.seed)
    shape_ok = finite_ok = label_ok = det_ok = True
    n_done = 0
    per_split_moments = {}
    t0 = time.perf_counter()
    for s, df in splits.items():
        idx = np.sort(rng.choice(len(df), size=min(args.per_split, len(df)), replace=False))
        ds = PTBXLMultilabelDataset(df.iloc[idx], pre)
        xs = []
        for i in range(len(ds)):
            item = ds[i]
            x = item["signal"].numpy()
            xs.append(x)
            shape_ok &= x.shape == (1000, 12) and x.dtype == np.float32
            finite_ok &= bool(np.isfinite(x).all())
            codes = meta.records.loc[item["ecg_id"], "scp_codes"]
            expected = multi_hot(superclasses_for(codes, meta.code_to_superclass))
            label_ok &= np.array_equal(item["target"].numpy(), expected) and expected.sum() > 0
            if i < 10:
                det_ok &= np.array_equal(x, ds[i]["signal"].numpy())
            n_done += 1
        allx = np.concatenate(xs)
        per_split_moments[s] = (allx.mean(axis=0), allx.std(axis=0))
    elapsed = time.perf_counter() - t0
    checks.update({"Shape checks": shape_ok, "NaN/Inf checks": finite_ok,
                   "Label checks": label_ok, "Determinism checks": det_ok})

    # Edge effect: resample an interior crop (samples 500-4500) and compare its first/last 10
    # outputs with the full-record resample at the same instants (far from the true edges).
    edge_err = {"line": [], "constant": []}
    for fn in splits["train"]["filename_hr"].iloc[::171][:100]:
        full = load_wfdb_record(meta.dataset_dir / fn)
        ref = resample_to_target(full, 500)[100:900]
        for pad, errs in edge_err.items():
            crop = resample_poly(full[500:4500], RESAMPLE_UP, RESAMPLE_DOWN, axis=0,
                                 window=RESAMPLE_WINDOW, padtype=pad)
            d = np.abs(crop - ref)
            errs.append(max(d[:10].max(), d[-10:].max()))
    print("\nEdge effect (100 train records; max |error| in first/last 10 samples, mV)")
    for pad, errs in edge_err.items():
        e = np.array(errs)
        tag = " (prep-v1)" if pad == RESAMPLE_PADTYPE else " (reference only)"
        print(f"{pad:>8}{tag}: median {np.median(e):.4f}  p95 {np.percentile(e, 95):.4f}  "
              f"max {e.max():.4f}")
    checks["Edge-effect check"] = (
        np.median(edge_err["line"]) < np.median(edge_err["constant"])
        and np.percentile(edge_err["line"], 95) < np.percentile(edge_err["constant"], 95))

    print("\nNormalized subset moments (sanity; not used for any decision)")
    for s, (m, sd) in per_split_moments.items():
        print(f"{s:>5}: mean range [{m.min():+.3f}, {m.max():+.3f}]  "
              f"std range [{sd.min():.3f}, {sd.max():.3f}]")

    print("\nValidation\n----------")
    print(f"Records successfully processed: {n_done} "
          f"({args.per_split}/split, seed {args.seed}; {1000 * elapsed / n_done:.2f} ms/record)")
    for name, ok in checks.items():
        print(f"{name}: {verdict(ok)}")
    print(f"Target order: {list(CLASS_NAMES)}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
