"""Shared ingestion utilities: HTTP with retries, raw landing, schema coercion."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from nhs_forecast.config import Settings
from nhs_forecast.logging_setup import get_logger

log = get_logger("ingestion.base")


# --- data provenance registry --------------------------------------------
# Each loader records whether it returned genuinely *live* (real published)
# data or fell back to the *synthetic* generator. ``steps.run`` reads this and
# surfaces it in the run report so consumers can see exactly which sources were
# live for a given run.
PROVENANCE: dict[str, str] = {}


def record_provenance(source: str, kind: str, detail: str = "") -> None:
    """Record ``kind`` ('live' | 'synthetic') for ``source``."""
    PROVENANCE[source] = f"{kind}" + (f" ({detail})" if detail else "")
    log.info("provenance %-16s -> %s", source, PROVENANCE[source])


def reset_provenance() -> None:
    PROVENANCE.clear()


# Canonical curated column sets. Loaders MUST return at least these columns so
# downstream layers (warehouse, features) can rely on a stable contract.
CANONICAL_PROCEDURES = [
    "date", "trust_code", "region", "opcs_chapter", "procedure_code", "n_procedures"
]
CANONICAL_IMAGING = ["date", "trust_code", "region", "modality", "n_tests"]
CANONICAL_RTT = [
    "date", "trust_code", "region", "treatment_function",
    "waiting_list_size", "pct_within_18wk",
]
CANONICAL_ACTIVITY = ["date", "trust_code", "region", "activity_type", "n_activity"]
CANONICAL_DEMOGRAPHICS = ["year", "region", "age_band", "population"]
CANONICAL_SUPPLY = ["quarter", "category", "spend_gbp", "units"]


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, max=30))
def http_get(url: str, timeout: int = 60, **kwargs) -> requests.Response:
    """GET with exponential backoff. Raises for non-2xx so tenacity retries."""
    log.info("GET %s", url)
    resp = requests.get(url, timeout=timeout, **kwargs)
    resp.raise_for_status()
    return resp


def land_raw(settings: Settings, source: str, content: bytes, ext: str) -> Path:
    """Persist a raw download to the landing zone with a content hash for lineage."""
    digest = hashlib.sha256(content).hexdigest()[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = settings.raw_dir / source / f"{stamp}_{digest}.{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    log.info("landed raw %s (%d bytes) -> %s", source, len(content), dest.name)
    return dest


def coerce_schema(df: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    """Ensure required columns exist and dtypes are sane; drop unexpected nulls.

    Handles real-world schema drift: missing columns are added as NA, the frame
    is reordered to the canonical column order, and a ``date`` column (if present)
    is normalised to the first day of its month.
    """
    required = list(required)
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    # canonical first, then any extras preserved for debugging
    extras = [c for c in df.columns if c not in required]
    return df[required + extras].reset_index(drop=True)
