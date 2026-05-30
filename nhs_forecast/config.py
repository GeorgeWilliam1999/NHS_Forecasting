"""Central configuration using pydantic-settings.

All paths and runtime knobs live here so the rest of the codebase never
hard-codes a directory. Override any value with an env var prefixed ``NHSFC_``
e.g. ``NHSFC_USE_SYNTHETIC=false``.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (nhs_forecast/config.py -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NHSFC_", env_file=".env", extra="ignore")

    # --- storage layout ---------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"          # landing zone (as-downloaded)
    lake_dir: Path = PROJECT_ROOT / "data" / "lake"        # curated Parquet data lake
    warehouse_path: Path = PROJECT_ROOT / "data" / "warehouse" / "nhs.duckdb"
    artifacts_dir: Path = PROJECT_ROOT / "data" / "artifacts"  # trained models, forecasts

    config_dir: Path = PROJECT_ROOT / "config"
    mapping_path: Path = PROJECT_ROOT / "nhs_forecast" / "mapping" / "mapping_table.csv"

    # --- behaviour --------------------------------------------------------
    # When True (default) ingestion falls back to a deterministic synthetic
    # generator so the whole pipeline runs without external network access or
    # DARS agreements. Set NHSFC_USE_SYNTHETIC=false to force live fetches.
    use_synthetic: bool = True
    synthetic_start: str = "2018-01-01"
    synthetic_end: str = "2025-03-01"
    synthetic_seed: int = 42

    # forecast horizon in months
    horizon: int = 12

    request_timeout: int = 60
    max_retries: int = 4

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.lake_dir, self.warehouse_path.parent, self.artifacts_dir):
            p.mkdir(parents=True, exist_ok=True)

    def sources(self) -> dict:
        with open(self.config_dir / "sources.yaml", "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
