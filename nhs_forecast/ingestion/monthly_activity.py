"""NHS Monthly Activity Statistics (electives, outpatients, A&E).

PRODUCTION ACCESS
-----------------
Published as multi-sheet Excel workbooks with multi-row merged headers that
drift between releases. Production ingestion uses ``openpyxl``/``pandas`` with
explicit ``skiprows`` discovered per-release, then melts wide month columns to
long form. Schema drift is handled by alias-mapping header tokens.

Returned grain: trust_code x activity_type x month.
Update cadence: monthly.
"""
from __future__ import annotations

import pandas as pd

from nhs_forecast.config import Settings
from nhs_forecast.ingestion import synthetic
from nhs_forecast.ingestion.base import (
    CANONICAL_ACTIVITY, coerce_schema, http_get, land_raw, record_provenance,
)
from nhs_forecast.logging_setup import get_logger

log = get_logger("ingestion.activity")

ACTIVITY_ALIASES = {
    "total elective admissions": "elective",
    "elective": "elective",
    "total outpatient attendances": "outpatient",
    "outpatient": "outpatient",
    "a&e attendances": "ae_attendance",
    "accident and emergency": "ae_attendance",
}


def _load_live(settings: Settings) -> pd.DataFrame:
    src = settings.sources()["monthly_activity"]
    resp = http_get(src["url"], timeout=settings.request_timeout)
    land_raw(settings, "monthly_activity", resp.content, "html")
    raise NotImplementedError("Monthly activity workbook needs per-release header mapping")


def load(settings: Settings) -> pd.DataFrame:
    if not settings.use_synthetic:
        try:
            df = _load_live(settings)
            if "activity_type" in df.columns:
                df["activity_type"] = (
                    df["activity_type"].astype(str).str.strip().str.lower()
                    .map(ACTIVITY_ALIASES).fillna(df["activity_type"])
                )
            out = coerce_schema(df, CANONICAL_ACTIVITY)
            record_provenance("monthly_activity", "live")
            return out
        except Exception as exc:  # noqa: BLE001
            log.warning("Activity live fetch failed (%s); using synthetic", exc)
    df = synthetic.tables(settings)["activity"].copy()
    record_provenance("monthly_activity", "synthetic", "per-release workbook headers not wired")
    return coerce_schema(df, CANONICAL_ACTIVITY)
