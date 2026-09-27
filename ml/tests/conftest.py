from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PTBXL_DIR = REPO_ROOT / "ml" / "data" / "raw" / "ptb-xl-1.0.3"
STATS_PATH = REPO_ROOT / "ml" / "configs" / "normalization_stats_prep-v1.json"

requires_ptbxl = pytest.mark.skipif(
    not (PTBXL_DIR / "ptbxl_database.csv").is_file(),
    reason="PTB-XL 1.0.3 not downloaded (run scripts/download_ptbxl.py)",
)


@pytest.fixture(scope="session")
def ptbxl_dir() -> Path:
    return PTBXL_DIR


@pytest.fixture(scope="session")
def metadata():
    from cardiomamba.data import PTBXLMetadata

    if not (PTBXL_DIR / "ptbxl_database.csv").is_file():
        pytest.skip("PTB-XL 1.0.3 not downloaded")
    return PTBXLMetadata.load(PTBXL_DIR)
