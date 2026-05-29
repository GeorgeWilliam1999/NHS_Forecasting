"""Shared pytest fixtures: a fast synthetic settings object in a temp dir."""
from __future__ import annotations

import pytest

from nhs_forecast.config import Settings


@pytest.fixture()
def settings(tmp_path):
    s = Settings(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        lake_dir=tmp_path / "data" / "lake",
        warehouse_path=tmp_path / "data" / "warehouse" / "nhs.duckdb",
        artifacts_dir=tmp_path / "data" / "artifacts",
        use_synthetic=True,
        # short window keeps tests quick but long enough for 12-mo lags + holdout
        synthetic_start="2019-01-01",
        synthetic_end="2023-12-01",
        horizon=6,
    )
    s.ensure_dirs()
    return s
