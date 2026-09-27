"""PTB-XL metadata loading, target construction and the official patient-disjoint split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cardiomamba.data.labels import (
    CLASS_NAMES,
    build_code_to_superclass,
    multi_hot,
    parse_scp_codes,
    superclasses_for,
)

DATASET_VERSION = "PTB-XL 1.0.3"
SPLIT_FOLDS: dict[str, tuple[int, ...]] = {
    "train": (1, 2, 3, 4, 5, 6, 7, 8),
    "val": (9,),
    "test": (10,),
}


def split_for_fold(fold: int) -> str:
    for name, folds in SPLIT_FOLDS.items():
        if fold in folds:
            return name
    raise ValueError(f"strat_fold {fold} is outside 1-10")


@dataclass(frozen=True)
class PTBXLMetadata:
    """Parsed PTB-XL metadata.

    `records` is indexed by `ecg_id` (not assumed contiguous) with columns
    `patient_id`, `strat_fold`, `split`, `filename_hr`, `scp_codes` (dict),
    one float32 column per class in CLASS_NAMES, and `n_labels`.
    """

    dataset_dir: Path
    records: pd.DataFrame
    code_to_superclass: dict[str, str]

    @classmethod
    def load(cls, dataset_dir: str | Path) -> PTBXLMetadata:
        dataset_dir = Path(dataset_dir)
        db = pd.read_csv(dataset_dir / "ptbxl_database.csv", index_col="ecg_id")
        scp = pd.read_csv(dataset_dir / "scp_statements.csv", index_col=0)
        code_to_superclass = build_code_to_superclass(scp)

        codes = db["scp_codes"].map(parse_scp_codes)
        targets = np.stack([
            multi_hot(superclasses_for(c, code_to_superclass, known_codes=scp.index))
            for c in codes
        ])
        records = pd.DataFrame({
            "patient_id": db["patient_id"],
            "strat_fold": db["strat_fold"].astype(int),
            "filename_hr": db["filename_hr"],
            "scp_codes": codes,
        }, index=db.index)
        records["split"] = records["strat_fold"].map(split_for_fold)
        for i, name in enumerate(CLASS_NAMES):
            records[name] = targets[:, i]
        records["n_labels"] = targets.sum(axis=1).astype(int)
        return cls(dataset_dir=dataset_dir, records=records,
                   code_to_superclass=code_to_superclass)

    def split(self, name: str, labelled_only: bool = True) -> pd.DataFrame:
        """Records of one split, sorted by ecg_id; unlabelled records excluded by default."""
        if name not in SPLIT_FOLDS:
            raise ValueError(f"Unknown split {name!r}; expected one of {list(SPLIT_FOLDS)}")
        df = self.records[self.records["split"] == name]
        if labelled_only:
            df = df[df["n_labels"] > 0]
        return df.sort_index()

    def targets(self, df: pd.DataFrame) -> np.ndarray:
        """float32 [N, 5] targets for the rows of `df`, in CLASS_NAMES order."""
        return df[list(CLASS_NAMES)].to_numpy(dtype=np.float32)

    def patient_overlap(self) -> dict[str, int]:
        """Number of patients shared between each pair of splits (all records)."""
        pats = {s: set(self.records.loc[self.records["split"] == s, "patient_id"])
                for s in SPLIT_FOLDS}
        return {
            "train_val": len(pats["train"] & pats["val"]),
            "train_test": len(pats["train"] & pats["test"]),
            "val_test": len(pats["val"] & pats["test"]),
        }
