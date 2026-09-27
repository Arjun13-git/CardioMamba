import pytest
import torch
from torch.utils.data import DataLoader

from cardiomamba.data import (
    NormalizationStats,
    PTBXLMultilabelDataset,
    PTBXLPreprocessor,
)
from conftest import STATS_PATH


@pytest.fixture(scope="module")
def preprocessor(metadata, ptbxl_dir):
    return PTBXLPreprocessor(ptbxl_dir, NormalizationStats.load(STATS_PATH))


def test_dataset_lengths(metadata, preprocessor):
    sizes = {s: len(PTBXLMultilabelDataset(metadata.split(s), preprocessor))
             for s in ("train", "val", "test")}
    assert sizes == {"train": 17084, "val": 2146, "test": 2158}


def test_item_contract(metadata, preprocessor):
    ds = PTBXLMultilabelDataset(metadata.split("val"), preprocessor)
    item = ds[0]
    assert item["signal"].shape == (1000, 12) and item["signal"].dtype == torch.float32
    assert item["target"].shape == (5,) and item["target"].dtype == torch.float32
    assert isinstance(item["ecg_id"], int)
    expected = metadata.targets(metadata.records.loc[[item["ecg_id"]]])[0]
    assert item["target"].tolist() == expected.tolist()
    assert torch.isfinite(item["signal"]).all()


def test_item_is_deterministic(metadata, preprocessor):
    ds = PTBXLMultilabelDataset(metadata.split("test"), preprocessor)
    assert torch.equal(ds[3]["signal"], ds[3]["signal"])


def test_dataloader_batch_layout(metadata, preprocessor):
    ds = PTBXLMultilabelDataset(metadata.split("val").iloc[:8], preprocessor)
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
    assert batch["signal"].shape == (4, 1000, 12)  # [B, T, C]
    assert batch["target"].shape == (4, 5)
    assert batch["ecg_id"].shape == (4,)


def test_rejects_unlabelled_records(metadata, preprocessor):
    with pytest.raises(ValueError, match="excluded"):
        PTBXLMultilabelDataset(metadata.split("train", labelled_only=False), preprocessor)
