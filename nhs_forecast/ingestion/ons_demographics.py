"""ONS mid-year population estimates by region and age band.

PRODUCTION ACCESS
-----------------
ONS exposes a Beta JSON API (api.beta.ons.gov.uk) and bulk CSV downloads.
Population by age band and region is an annual demand driver: an ageing
population raises demand for orthopaedic, cardiac and ophthalmic procedures.

Returned grain: region x age_band x year.
Update cadence: annual.
"""
from __future__ import annotations

import pandas as pd

from nhs_forecast.config import Settings
from nhs_forecast.ingestion import synthetic
from nhs_forecast.ingestion.base import (
    CANONICAL_DEMOGRAPHICS, coerce_schema, http_get, land_raw, record_provenance,
)
from nhs_forecast.logging_setup import get_logger

log = get_logger("ingestion.ons")

ONS_API_ROOT = "https://api.beta.ons.gov.uk/v1"


def _load_live(settings: Settings) -> pd.DataFrame:
    # The ONS dataset structure is multi-edition; production code resolves the
    # latest edition/version then downloads its observations CSV.
    url = f"{ONS_API_ROOT}/datasets"
    resp = http_get(url, timeout=settings.request_timeout)
    land_raw(settings, "ons_demographics", resp.content, "json")
    raise NotImplementedError("ONS dataset edition/version resolution not wired in demo")


def load(settings: Settings) -> pd.DataFrame:
    if not settings.use_synthetic:
        try:
            out = coerce_schema(_load_live(settings), CANONICAL_DEMOGRAPHICS)
            record_provenance("ons_demographics", "live")
            return out
        except Exception as exc:  # noqa: BLE001
            log.warning("ONS live fetch failed (%s); using synthetic", exc)
    df = synthetic.tables(settings)["demographics"].copy()
    record_provenance("ons_demographics", "synthetic", "full-population CSV is 126MB; not wired")
    return coerce_schema(df, CANONICAL_DEMOGRAPHICS)
