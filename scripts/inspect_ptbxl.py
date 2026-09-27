"""Phase 1 inspection of PTB-XL v1.0.3 (read-only; no preprocessing, no normalization stats).

Checks metadata, fold/patient split, the 5-superclass label mapping, WFDB headers for every
500 Hz record, and full signals for a small fold-stratified sample. Writes a JSON report to
ml/data/reports/ (gitignored).

Usage:  uv run python scripts/inspect_ptbxl.py [--sample-per-fold 20]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "ml" / "data" / "raw" / "ptb-xl-1.0.3"
REPORT_PATH = REPO_ROOT / "ml" / "data" / "reports" / "phase1_inspection.json"

EXPECTED_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SPLITS = {"train": list(range(1, 9)), "val": [9], "test": [10]}
FS_HR, LEN_HR = 500, 5000


def load_metadata(ds: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    db = pd.read_csv(ds / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(ds / "scp_statements.csv", index_col=0)
    return db, scp


def superclass_targets(db: pd.DataFrame, scp: pd.DataFrame) -> pd.DataFrame:
    """Multi-hot [N, 5]: diagnostic==1 statements mapped via diagnostic_class; likelihood ignored."""
    diag = scp[scp["diagnostic"] == 1]
    code_to_class = diag["diagnostic_class"].to_dict()
    rows = []
    for codes in db["scp_codes"]:
        classes = {code_to_class[c] for c in codes if c in code_to_class}
        rows.append([int(c in classes) for c in SUPERCLASSES])
    return pd.DataFrame(rows, index=db.index, columns=SUPERCLASSES)


def split_of(fold: int) -> str:
    return next(name for name, folds in SPLITS.items() if fold in folds)


def check_headers(db: pd.DataFrame, ds: Path) -> dict:
    """Read every 500 Hz WFDB header (metadata only) and check shape/rate/leads/units."""
    problems: dict[str, list[int]] = {k: [] for k in
        ["missing_files", "n_sig", "fs", "sig_len", "lead_order", "units"]}
    lead_orders, units_seen = {}, {}
    for ecg_id, rel in tqdm(db["filename_hr"].items(), total=len(db), desc="headers"):
        base = ds / rel
        if not (base.with_suffix(".hea").is_file() and base.with_suffix(".dat").is_file()):
            problems["missing_files"].append(ecg_id)
            continue
        h = wfdb.rdheader(str(base))
        order = tuple(h.sig_name)
        lead_orders[order] = lead_orders.get(order, 0) + 1
        for u in h.units:
            units_seen[u] = units_seen.get(u, 0) + 1
        if h.n_sig != 12:
            problems["n_sig"].append(ecg_id)
        if h.fs != FS_HR:
            problems["fs"].append(ecg_id)
        if h.sig_len != LEN_HR:
            problems["sig_len"].append(ecg_id)
        if [s.upper() for s in h.sig_name] != EXPECTED_LEADS:
            problems["lead_order"].append(ecg_id)
        if any(u.lower() != "mv" for u in h.units):
            problems["units"].append(ecg_id)
    return {
        "records_checked": len(db),
        "lead_orders": {" ".join(k): v for k, v in lead_orders.items()},
        "units": units_seen,
        "problem_counts": {k: len(v) for k, v in problems.items()},
        "problem_ids_first20": {k: v[:20] for k, v in problems.items() if v},
    }


def check_signal_sample(db: pd.DataFrame, ds: Path, per_fold: int, seed: int) -> dict:
    """Load full 500 Hz signals for a fold-stratified random sample and validate them."""
    sample = db.groupby("strat_fold").sample(n=per_fold, random_state=seed)
    results, failures = [], []
    for ecg_id, rel in tqdm(sample["filename_hr"].items(), total=len(sample), desc="signals"):
        rec = wfdb.rdrecord(str(ds / rel), physical=True)
        x = rec.p_signal
        ok = (x.shape == (LEN_HR, 12) and rec.fs == FS_HR and np.isfinite(x).all()
              and [s.upper() for s in rec.sig_name] == EXPECTED_LEADS)
        # header/signal consistency: digital -> physical round trip via gain/baseline
        d = wfdb.rdrecord(str(ds / rel), physical=False).d_signal
        recon = (d - np.array(rec.baseline)) / np.array(rec.adc_gain)
        consistent = bool(np.allclose(recon, x, atol=1e-6, equal_nan=False))
        results.append({"ecg_id": int(ecg_id), "min_mV": float(np.nanmin(x)),
                        "max_mV": float(np.nanmax(x)), "ok": bool(ok), "consistent": consistent})
        if not (ok and consistent):
            failures.append(int(ecg_id))
    amps = np.array([[r["min_mV"], r["max_mV"]] for r in results])
    return {
        "records_loaded": len(results),
        "per_fold": per_fold,
        "seed": seed,
        "failures": failures,
        "all_shape_fs_leads_finite_ok": all(r["ok"] for r in results),
        "all_header_signal_consistent": all(r["consistent"] for r in results),
        "min_mV_over_sample": float(amps[:, 0].min()),
        "max_mV_over_sample": float(amps[:, 1].max()),
        "median_abs_peak_mV": float(np.median(np.abs(amps).max(axis=1))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--sample-per-fold", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ds: Path = args.dataset_dir

    report: dict = {"dataset_dir": str(ds.relative_to(REPO_ROOT))}
    report["structure"] = {
        name: (ds / name).exists()
        for name in ["ptbxl_database.csv", "scp_statements.csv", "records500", "records100",
                     "RECORDS", "SHA256SUMS.txt", "LICENSE.txt", "example_physionet.py"]
    }
    report["records500_dat_files"] = sum(1 for _ in (ds / "records500").rglob("*.dat"))
    report["records100_dat_files"] = sum(1 for _ in (ds / "records100").rglob("*.dat"))

    db, scp = load_metadata(ds)
    report["metadata"] = {
        "records": len(db),
        "unique_ecg_id": int(db.index.nunique()),
        "unique_patients": int(db["patient_id"].nunique()),
        "columns": list(db.columns),
        "patient_id_dtype": str(db["patient_id"].dtype),
        "strat_fold_counts": db["strat_fold"].value_counts().sort_index().to_dict(),
        "scp_codes_example": db["scp_codes"].iloc[0],
        "scp_codes_all_dicts": bool(db["scp_codes"].apply(lambda d: isinstance(d, dict)).all()),
        "records_with_empty_scp_codes": int((db["scp_codes"].apply(len) == 0).sum()),
        "validated_by_human_by_fold": db.groupby("strat_fold")["validated_by_human"].mean()
            .round(4).to_dict(),
    }
    unknown_codes = sorted({c for d in db["scp_codes"] for c in d} - set(scp.index))
    diag = scp[scp["diagnostic"] == 1]
    report["scp_statements"] = {
        "statements": len(scp),
        "columns": list(scp.columns),
        "diagnostic": int((scp["diagnostic"] == 1).sum()),
        "form": int((scp["form"] == 1).sum()),
        "rhythm": int((scp["rhythm"] == 1).sum()),
        "diagnostic_class_values": sorted(diag["diagnostic_class"].dropna().unique().tolist()),
        "diagnostic_statements_per_class": diag["diagnostic_class"].value_counts().to_dict(),
        "diagnostic_statements_missing_class": diag.index[diag["diagnostic_class"].isna()]
            .tolist(),
        "diagnostic_subclasses": int(diag["diagnostic_subclass"].nunique()),
        "codes_in_database_not_in_statements": unknown_codes,
    }

    # Patient-level split
    db["split"] = db["strat_fold"].map(split_of)
    split_patients = {s: set(db.loc[db["split"] == s, "patient_id"]) for s in SPLITS}
    fold_patients = db.groupby("strat_fold")["patient_id"].apply(set)
    cross_fold = (db.groupby("patient_id")["strat_fold"].nunique() > 1).sum()
    report["split"] = {
        s: {"folds": SPLITS[s], "records": int((db["split"] == s).sum()),
            "patients": len(split_patients[s])} for s in SPLITS
    }
    report["split"]["patients_per_fold"] = {int(k): len(v) for k, v in fold_patients.items()}
    report["split"]["overlap"] = {
        "train_val": len(split_patients["train"] & split_patients["val"]),
        "train_test": len(split_patients["train"] & split_patients["test"]),
        "val_test": len(split_patients["val"] & split_patients["test"]),
        "patients_in_more_than_one_fold": int(cross_fold),
    }

    # 5-superclass labels (inspection only; no modeling decisions from test statistics)
    y = superclass_targets(db, scp)
    empty = y.sum(axis=1) == 0
    card = y.sum(axis=1)
    report["labels"] = {
        "support": {
            "overall": y.sum().to_dict(),
            **{s: y[db["split"] == s].sum().to_dict() for s in SPLITS},
        },
        "no_superclass": {
            "overall": int(empty.sum()),
            **{s: int((empty & (db["split"] == s)).sum()) for s in SPLITS},
        },
        "labeled_records": {
            "overall": int((~empty).sum()),
            **{s: int((~empty & (db["split"] == s)).sum()) for s in SPLITS},
        },
        "labels_per_record_distribution": card.value_counts().sort_index().to_dict(),
    }

    report["headers_500hz"] = check_headers(db, ds)
    report["signal_sample_500hz"] = check_signal_sample(db, ds, args.sample_per_fold, args.seed)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"\nReport written to {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
