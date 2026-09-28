"""PTB-XL datasets and DataLoaders for training (reuses the Phase 2 pipeline unchanged).

Training code only ever builds the train (folds 1-8) and validation (fold 9) splits. The test
split (fold 10) has its own builder for the later final-evaluation phase and is never called
from the training loop.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from cardiomamba.data import (
    NormalizationStats,
    PTBXLMetadata,
    PTBXLMultilabelDataset,
    PTBXLPreprocessor,
)
from cardiomamba.training.config import TrainConfig
from cardiomamba.training.reproducibility import seed_worker


def _subset(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Deterministic record subset for smoke runs (None = the whole split)."""
    if n is None or n >= len(df):
        return df
    return df.sample(n=n, random_state=seed).sort_index()


def _preprocessor(config: TrainConfig) -> tuple[PTBXLMetadata, PTBXLPreprocessor]:
    meta = PTBXLMetadata.load(Path(config.dataset_dir))
    stats = NormalizationStats.load(Path(config.normalization_stats))  # frozen, train-only
    return meta, PTBXLPreprocessor(meta.dataset_dir, stats)


def build_train_val_datasets(config: TrainConfig) -> tuple[Dataset, Dataset]:
    meta, pre = _preprocessor(config)
    train = _subset(meta.split("train"), config.max_train_records, config.seed)
    val = _subset(meta.split("val"), config.max_val_records, config.seed)
    return PTBXLMultilabelDataset(train, pre), PTBXLMultilabelDataset(val, pre)


def build_validation_dataset(config: TrainConfig) -> Dataset:
    meta, pre = _preprocessor(config)
    return PTBXLMultilabelDataset(_subset(meta.split("val"), config.max_val_records,
                                          config.seed), pre)


def build_test_dataset(config: TrainConfig) -> Dataset:  # pragma: no cover - later phase
    """Fold 10. For the final evaluation phase only; never used during training."""
    meta, pre = _preprocessor(config)
    return PTBXLMultilabelDataset(meta.split("test"), pre)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int,
                pin_memory: bool, shuffle_generator: torch.Generator | None = None,
                seed: int = 0) -> DataLoader:
    """DataLoader whose shuffle order depends only on `shuffle_generator`.

    The shuffle sampler and the DataLoader (which draws worker base seeds from its own
    generator whenever a *new* iterator is created) use separate generators. With persistent
    workers an uninterrupted run reuses its iterator while a resumed run creates a new one;
    sharing one generator would make the first resumed epoch's order differ. Neither touches
    the global RNG (used by dropout).
    """
    sampler = None
    if shuffle:
        if shuffle_generator is None:
            raise ValueError("shuffle=True requires a shuffle_generator")
        sampler = RandomSampler(dataset, generator=shuffle_generator)
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        # forkserver: workers are not fork()ed from the multi-threaded training process
        # (fork after threads can deadlock; Python 3.12 warns about it).
        multiprocessing_context="forkserver" if num_workers > 0 else None,
        generator=torch.Generator().manual_seed(seed), drop_last=False,
    )
