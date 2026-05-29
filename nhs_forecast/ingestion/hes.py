"""Hospital Episode Statistics (HES) — Admitted Patient Care procedures.

PRODUCTION ACCESS
-----------------
Record-level HES requires an approved DARS (Data Access Request Service)
agreement with NHS England. Openly-publishable *aggregate* activity tables are
released monthly via the NHS Digital data catalogue. There is no clean REST API;
production ingestion downloads the published CSV extracts.

Returned grain: trust_code x opcs_chapter x procedure_code x month.
Procedure codes are OPCS-4 (Office of Population Censuses and Surveys, 4th rev).
Update cadence: monthly (provisional) with an annual final refresh.
"""
from __future__ import annotations

import pandas as pd

from nhs_forecast.config import Settings
from nhs_forecast.ingestion import synthetic
from nhs_forecast.ingestion.base import (
    CANONICAL_PROCEDURES, coerce_schema, http_get, land_raw, record_provenance,
)
from nhs_forecast.logging_setup import get_logger

log = get_logger("ingestion.hes")


def _load_live(settings: Settings) -> pd.DataFrame:
    """Download + parse a published HES aggregate extract.

    The catalogue file URL changes each release; in production this is resolved
    from the NHS Digital catalogue page. We attempt a configured direct URL and
    normalise the typical HES column names to the canonical schema.
    """
    src = settings.sources()["hes"]
    resp = http_get(src["url"], timeout=settings.request_timeout)
    land_raw(settings, "hes", resp.content, "html")
    # Real parsing would follow the resolved CSV link; intentionally raise so the
    # caller falls back to synthetic when no machine-readable extract is wired.
    raise NotImplementedError("HES CSV link resolution requires DARS-gated catalogue access")


def load(settings: Settings) -> pd.DataFrame:
    if not settings.use_synthetic:
        try:
            df = coerce_schema(_load_live(settings), CANONICAL_PROCEDURES)
            record_provenance("hes", "live")
            return df
        except Exception as exc:  # noqa: BLE001 - graceful degradation by design
            log.warning("HES live fetch failed (%s); using synthetic", exc)
    df = synthetic.tables(settings)["procedures"].copy()
    record_provenance("hes", "synthetic", "record-level HES is DARS-gated")
    return coerce_schema(df, CANONICAL_PROCEDURES)
