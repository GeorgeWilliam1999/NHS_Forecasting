"""Central configuration using pydantic-settings.

All paths and runtime knobs live here so the rest of the codebase never
hard-codes a directory. Override any value with an env var prefixed ``NHSFC_``
e.g. ``NHSFC_USE_SYNTHETIC=false``.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_project_root() -> Path:
    """Locate a writable project root.

    Order of preference:
    1. ``NHSFC_PROJECT_ROOT`` env var, if set.
    2. The current working directory, when it looks like the project checkout
       (contains ``nhs_forecast/`` or ``data/``). This is the case on
       Streamlit Community Cloud, where the app runs from the cloned repo while
       the package itself is pip-installed into a read-only ``site-packages``.
    3. Two levels up from this file (the repo root for an editable / source
       checkout).
    """
    env_root = os.environ.get("NHSFC_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    cwd = Path.cwd()
    if (cwd / "nhs_forecast").is_dir() or (cwd / "data").is_dir():
        return cwd

    return Path(__file__).resolve().parents[1]


# Project root: prefer the working-directory checkout so paths stay writable
# even when the package is installed into a read-only site-packages location.
PROJECT_ROOT = _resolve_project_root()


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

    # --- telemetry / pay-per-use underwriting layer -----------------------
    # Device-session telemetry simulator + underwriting engine. Self-contained:
    # the aggregate demand pipeline above is unaffected by these knobs.
    telemetry_n_devices: int = 16
    telemetry_days: int = 540          # ~18 months of daily device telemetry
    telemetry_seed: int = 7
    telemetry_horizon_days: int = 28   # short-term utilisation forecast horizon
    underwriting_low_q: float = 0.05   # downside quantile (Utilisation-at-Risk)
    underwriting_high_q: float = 0.95
    risk_alpha: float = 0.05           # VaR / CVaR tail probability
    risk_n_sims: int = 4000            # Monte-Carlo portfolio draws
    min_billable_seconds: float = 180.0  # contract rule: active time to bill a session

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.lake_dir, self.warehouse_path.parent, self.artifacts_dir):
            try:
                p.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError):
                # Read-only deployment (e.g. package installed in site-packages):
                # the dashboard can still read committed artifacts. Writing
                # (running the pipeline) will surface a clearer error later.
                pass

    def sources(self) -> dict:
        with open(self.config_dir / "sources.yaml", "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
