"""Build the small, version-controlled Phase 2 artifacts from PTB-XL 1.0.3.

Outputs (deterministic; safe to commit — no waveform data):
  ml/configs/normalization_stats_prep-v1.json  per-lead mean/std from training folds 1-8 only
  ml/configs/labels_superdiagnostic_v1.json    class order, SCP->superclass mapping, support

Usage:  uv run python scripts/prepare_ptbxl.py [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from cardiomamba.data import (
    CLASS_NAMES,
    DATASET_VERSION,
    LABEL_MAPPING_VERSION,
    SPLIT_FOLDS,
    PTBXLMetadata,
    compute_training_stats,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "ml" / "data" / "raw" / "ptb-xl-1.0.3"
CONFIG_DIR = REPO_ROOT / "ml" / "configs"
STATS_PATH = CONFIG_DIR / "normalization_stats_prep-v1.json"
LABELS_PATH = CONFIG_DIR / "labels_superdiagnostic_v1.json"


def label_artifact(meta: PTBXLMetadata) -> dict:
    support = {}
    for split in SPLIT_FOLDS:
        df = meta.split(split, labelled_only=True)
        support[split] = {
            "labelled_records": len(df),
            "excluded_no_superclass": int((meta.split(split, labelled_only=False)["n_labels"]
                                           == 0).sum()),
            **{c: int(df[c].sum()) for c in CLASS_NAMES},
        }
    return {
        "label_mapping_version": LABEL_MAPPING_VERSION,
        "dataset_version": DATASET_VERSION,
        "class_names": list(CLASS_NAMES),
        "rule": "scp_codes keys with scp_statements.diagnostic == 1 mapped via diagnostic_class; "
                "likelihood ignored; records with no superclass excluded",
        "split_folds": {k: list(v) for k, v in SPLIT_FOLDS.items()},
        "code_to_superclass": dict(sorted(meta.code_to_superclass.items())),
        "support": support,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    meta = PTBXLMetadata.load(args.dataset_dir)
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(json.dumps(label_artifact(meta), indent=2) + "\n")
    print(f"Wrote {LABELS_PATH.relative_to(REPO_ROOT)}")

    t0 = time.perf_counter()
    stats = compute_training_stats(meta, workers=args.workers)
    elapsed = time.perf_counter() - t0
    stats.save(STATS_PATH)
    print(f"Wrote {STATS_PATH.relative_to(REPO_ROOT)} "
          f"({stats.metadata['n_records']} train records, {elapsed:.1f} s)")
    print("mean:", [round(float(v), 5) for v in stats.mean])
    print("std: ", [round(float(v), 5) for v in stats.std])
    return 0


if __name__ == "__main__":
    sys.exit(main())
