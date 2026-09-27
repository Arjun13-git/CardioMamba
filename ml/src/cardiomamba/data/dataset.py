"""PyTorch Dataset over one PTB-XL split with on-demand (lazy) record loading."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from cardiomamba.data.labels import CLASS_NAMES
from cardiomamba.data.preprocessing import PTBXLPreprocessor


class PTBXLMultilabelDataset(Dataset):
    """Items: {"signal": float32 [1000, 12], "target": float32 [5], "ecg_id": int}.

    `records` is a split frame from `PTBXLMetadata.split()` (labelled records only). Each
    item is read from its 500 Hz WFDB file and preprocessed on access; nothing is held in
    memory beyond the metadata. The preprocessor must carry the frozen training-fold stats.
    """

    def __init__(self, records: pd.DataFrame, preprocessor: PTBXLPreprocessor) -> None:
        if (records["n_labels"] == 0).any():
            raise ValueError("Records without a diagnostic superclass must be excluded")
        self.ecg_ids = records.index.to_numpy()
        self.filenames = records["filename_hr"].tolist()
        self.targets = torch.from_numpy(records[list(CLASS_NAMES)].to_numpy(dtype="float32"))
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.ecg_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "signal": torch.from_numpy(self.preprocessor(self.filenames[idx])),
            "target": self.targets[idx],
            "ecg_id": int(self.ecg_ids[idx]),
        }
