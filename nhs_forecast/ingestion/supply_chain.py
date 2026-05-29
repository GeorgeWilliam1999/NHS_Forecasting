"""NHS Supply Chain / procurement spend (optional ground-truth).

PRODUCTION ACCESS
-----------------
There is no stable open API for NHS Supply Chain consumption. Partial signals
are available via Contracts Finder and the NHS Spend Comparison Service, usually
as CSV exports requiring portal navigation (hence ``access: scrape``). When
available this gives a real consumption series to *calibrate* the procedure ->
equipment mapping weights against. Optional: the pipeline runs without it.

Returned grain: category x quarter.
Update cadence: quarterly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhs_forecast.config import Settings
from nhs_forecast.ingestion import synthetic
from nhs_forecast.ingestion.base import CANONICAL_SUPPLY, coerce_schema, record_provenance
from nhs_forecast.logging_setup import get_logger

log = get_logger("ingestion.supply")

CATEGORIES = ["endoscope", "ortho_implant", "cardiac_device", "imaging_consumable", "surgical_kit"]


def _synthetic_supply(settings: Settings) -> pd.DataFrame:
    """Derive a plausible quarterly spend series from synthetic procedure totals."""
    rng = np.random.default_rng(settings.synthetic_seed + 7)
    proc = synthetic.tables(settings)["procedures"]
    q = (proc.assign(quarter=proc["date"].dt.to_period("Q").astype(str))
            .groupby("quarter")["n_procedures"].sum())
    rows = []
    unit_cost = {"endoscope": 1800, "ortho_implant": 950, "cardiac_device": 4200,
                 "imaging_consumable": 60, "surgical_kit": 120}
    for cat in CATEGORIES:
        share = rng.uniform(0.05, 0.25)
        for quarter, total in q.items():
            units = int(total * share * rng.uniform(0.9, 1.1))
            rows.append((quarter, cat, round(units * unit_cost[cat], 2), units))
    return pd.DataFrame(rows, columns=CANONICAL_SUPPLY)


def load(settings: Settings) -> pd.DataFrame:
    if not settings.use_synthetic:
        log.warning("Live NHS Supply Chain scrape not configured; using synthetic proxy")
    record_provenance("supply_chain", "synthetic", "no stable open API")
    return coerce_schema(_synthetic_supply(settings), CANONICAL_SUPPLY)
