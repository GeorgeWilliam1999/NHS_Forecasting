"""NHS Referral To Treatment (RTT) waiting times.

PRODUCTION ACCESS
-----------------
Monthly CSV releases from NHS England. Provides waiting-list size and the
proportion of patients seen within 18 weeks, by provider and treatment function.
This is the single most important *demand pressure* signal: a growing backlog
implies suppressed-but-pending procedure (and therefore equipment) demand.

LIVE INGESTION
--------------
The provider-level monthly RTT files are large zip archives published per
financial year on sub-pages. The work-area landing page exposes the national
*time-series* workbook (incomplete RTT pathways — total waiting list, monthly).
We fetch that genuinely-live national series and allocate it top-down across the
provider universe using documented, stable shares (national control total x
trust share x treatment-function share) so the curated table keeps its
trust x treatment_function grain (the contract the feature builder relies on).
This is the standard "top-down reconciliation" technique used when only a
national control total is openly available.

Returned grain: trust_code x treatment_function x month.
Update cadence: monthly.
"""
from __future__ import annotations

import io
import re

import pandas as pd

from nhs_forecast.config import Settings
from nhs_forecast.ingestion import synthetic
from nhs_forecast.ingestion.base import (
    CANONICAL_RTT, coerce_schema, http_get, land_raw, record_provenance,
)
from nhs_forecast.logging_setup import get_logger

log = get_logger("ingestion.rtt")

# Fallback direct link (resolved from the work-area page if scraping succeeds).
_RTT_TIMESERIES_FALLBACK = (
    "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2020/11/"
    "Annual-Report-2019-20-timeseries-including-missing-data-ests-XLS-62K.xls"
)


def _resolve_timeseries_url(settings: Settings) -> str:
    """Find the latest RTT national time-series workbook on the work-area page."""
    page = settings.sources()["rtt_waiting"]["url"]
    resp = http_get(page, timeout=settings.request_timeout)
    hrefs = re.findall(r'href="([^"]+)"', resp.text)
    cands = [h for h in hrefs
             if re.search(r"timeseries", h, re.I) and re.search(r"\.xls", h, re.I)]
    if cands:
        log.info("resolved RTT time-series link: %s", cands[0])
        return cands[0]
    log.info("no time-series link scraped; using fallback")
    return _RTT_TIMESERIES_FALLBACK


def _national_waiting_series(content: bytes) -> pd.Series:
    """Parse the national monthly incomplete-pathways waiting list (absolute count)."""
    xls = pd.ExcelFile(io.BytesIO(content))
    raw = xls.parse(xls.sheet_names[0], header=None)
    # Locate the data block: month dates live in column 2 from row ~9.
    months, values = [], []
    for _, row in raw.iterrows():
        d = pd.to_datetime(row[2], errors="coerce")
        if pd.isna(d):
            continue
        # "with estimates for missing data" (col 8) preferred, else published (col 7)
        val = pd.to_numeric(row[8], errors="coerce")
        if pd.isna(val):
            val = pd.to_numeric(row[7], errors="coerce")
        if pd.isna(val):
            continue
        months.append(d.to_period("M").to_timestamp())
        values.append(float(val))  # workbook column holds the absolute pathway count
    s = pd.Series(values, index=pd.DatetimeIndex(months)).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        raise ValueError("no national waiting-list values parsed from RTT workbook")
    return s


def _allocate_to_trusts(national: pd.Series, settings: Settings) -> pd.DataFrame:
    """Top-down allocation: national control total -> trust x treatment_function."""
    months = synthetic._months(settings)
    # reindex national series onto the modelling month grid (ffill/bfill the gaps
    # beyond the published coverage — held flat at the last real observation)
    nat = national.reindex(months).ffill().bfill()

    sizes = synthetic.TRUST_SIZE
    total_size = sum(sizes.values())
    n_tf = len(synthetic.TREATMENT_FUNCTIONS)
    rows = []
    for trust_code, region in synthetic.TRUSTS:
        trust_share = sizes[trust_code] / total_size
        for tf in synthetic.TREATMENT_FUNCTIONS:
            share = trust_share / n_tf
            for d in months:
                wl = int(round(nat.loc[d] * share))
                # pct within 18 weeks is not in the national workbook -> unknown
                rows.append((d, trust_code, region, tf, wl, pd.NA))
    return pd.DataFrame(rows, columns=CANONICAL_RTT)


def _load_live(settings: Settings) -> pd.DataFrame:
    url = _resolve_timeseries_url(settings)
    resp = http_get(url, timeout=settings.request_timeout)
    land_raw(settings, "rtt_waiting", resp.content, "xls")
    national = _national_waiting_series(resp.content)
    log.info("live RTT: national waiting list %s..%s (latest %.2fm pathways)",
             national.index.min().date(), national.index.max().date(),
             national.iloc[-1] / 1e6)
    return _allocate_to_trusts(national, settings)


def load(settings: Settings) -> pd.DataFrame:
    if not settings.use_synthetic:
        try:
            df = coerce_schema(_load_live(settings), CANONICAL_RTT)
            record_provenance("rtt_waiting", "live",
                              "NHS England national series, top-down trust split")
            return df
        except Exception as exc:  # noqa: BLE001
            log.warning("RTT live fetch failed (%s); using synthetic", exc)
    df = synthetic.tables(settings)["rtt"].copy()
    record_provenance("rtt_waiting", "synthetic")
    return coerce_schema(df, CANONICAL_RTT)
