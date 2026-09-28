"""Environment-driven settings (prefix CARDIOMAMBA_). Paths default to the frozen run."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
FROZEN_RUN = REPO_ROOT / "outputs" / "cardiomamba" / "full-30ep-seed42"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CARDIOMAMBA_", env_file=".env", extra="ignore")

    checkpoint_path: Path = FROZEN_RUN / "best.pt"
    thresholds_path: Path = FROZEN_RUN / "thresholds.json"
    test_metrics_path: Path = FROZEN_RUN / "test_metrics.json"
    repo_root: Path = REPO_ROOT               # base for the checkpoint's relative stats path
    device: str = "cpu"                       # "cpu" | "cuda" | "auto"
    max_upload_bytes: int = 2 * 1024 * 1024
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
