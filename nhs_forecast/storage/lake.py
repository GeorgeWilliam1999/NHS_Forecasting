"""Parquet-based data lake writer with a simple snapshot versioning scheme.

Layout (Hive-style partitioning by ingest snapshot):

    data/lake/<dataset>/snapshot_date=YYYY-MM-DD/part-0.parquet
    data/lake/<dataset>/_latest.txt        # pointer to the current snapshot

Versioning strategy
-------------------
Each ingestion run writes an immutable snapshot keyed by date. The ``_latest``
pointer makes "current" reads trivial while preserving full history for audit
and reproducibility (you can rebuild any past warehouse state). This mirrors a
lightweight, file-system-native alternative to Delta/Iceberg suitable for a
single-node DuckDB stack; the same partition convention upgrades cleanly to S3.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from nhs_forecast.config import Settings
from nhs_forecast.logging_setup import get_logger

log = get_logger("storage.lake")


def write_snapshot(
    settings: Settings, dataset: str, df: pd.DataFrame, snapshot: str | None = None
) -> Path:
    snapshot = snapshot or date.today().isoformat()
    out_dir = settings.lake_dir / dataset / f"snapshot_date={snapshot}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "part-0.parquet"
    df.to_parquet(path, index=False)
    (settings.lake_dir / dataset / "_latest.txt").write_text(snapshot, encoding="utf-8")
    log.info("lake write %s rows=%d -> %s", dataset, len(df), path)
    return path


def latest_snapshot(settings: Settings, dataset: str) -> str | None:
    pointer = settings.lake_dir / dataset / "_latest.txt"
    return pointer.read_text(encoding="utf-8").strip() if pointer.exists() else None


def read_latest(settings: Settings, dataset: str) -> pd.DataFrame:
    snap = latest_snapshot(settings, dataset)
    if snap is None:
        raise FileNotFoundError(f"no snapshot for dataset '{dataset}' in lake")
    path = settings.lake_dir / dataset / f"snapshot_date={snap}" / "part-0.parquet"
    return pd.read_parquet(path)


def list_snapshots(settings: Settings, dataset: str) -> list[str]:
    base = settings.lake_dir / dataset
    if not base.exists():
        return []
    return sorted(p.name.split("=", 1)[1] for p in base.glob("snapshot_date=*"))
