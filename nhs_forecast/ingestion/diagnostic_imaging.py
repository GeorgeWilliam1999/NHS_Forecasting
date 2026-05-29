"""NHS Diagnostic Imaging Dataset (DID).

PRODUCTION ACCESS
-----------------
NHS England publishes monthly DID statistics as CSV/XLS by imaging modality
(CT, MRI, Ultrasound, Plain Radiography, Fluoroscopy, Nuclear Medicine).
Bulk download from the statistical-work-areas page; no API.
Directly relevant for *scanner* equipment demand.

Returned grain: trust_code x modality x month.
Update cadence: monthly.
"""
from __future__ import annotations

import pandas as pd

from nhs_forecast.config import Settings
from nhs_forecast.ingestion import synthetic
from nhs_forecast.ingestion.base import (
    CANONICAL_IMAGING, coerce_schema, http_get, land_raw, record_provenance,
)
from nhs_forecast.logging_setup import get_logger

log = get_logger("ingestion.did")

# Map heterogeneous published modality labels to canonical names (schema drift).
MODALITY_ALIASES = {
    "computed tomography": "CT",
    "ct": "CT",
    "magnetic resonance imaging": "MRI",
    "mri": "MRI",
    "ultrasound": "Ultrasound",
    "plain radiography": "Plain Radiography",
    "x-ray": "Plain Radiography",
    "fluoroscopy": "Fluoroscopy",
    "nuclear medicine": "Nuclear Medicine",
}


def _normalise_modality(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map(MODALITY_ALIASES).fillna(s)


def _load_live(settings: Settings) -> pd.DataFrame:
    src = settings.sources()["diagnostic_imaging"]
    resp = http_get(src["url"], timeout=settings.request_timeout)
    land_raw(settings, "diagnostic_imaging", resp.content, "html")
    raise NotImplementedError("DID monthly CSV link must be resolved from the work-area page")


def load(settings: Settings) -> pd.DataFrame:
    if not settings.use_synthetic:
        try:
            df = _load_live(settings)
            if "modality" in df.columns:
                df["modality"] = _normalise_modality(df["modality"])
            out = coerce_schema(df, CANONICAL_IMAGING)
            record_provenance("diagnostic_imaging", "live")
            return out
        except Exception as exc:  # noqa: BLE001
            log.warning("DID live fetch failed (%s); using synthetic", exc)
    df = synthetic.tables(settings)["imaging"].copy()
    record_provenance("diagnostic_imaging", "synthetic", "per-release CSV link not wired")
    return coerce_schema(df, CANONICAL_IMAGING)
