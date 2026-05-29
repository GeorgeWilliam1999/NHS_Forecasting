"""Scenario analysis on top of the baseline procedure forecast.

Scenarios are multiplicative / additive transforms applied to the *procedure*
forecast before equipment derivation, so uncertainty propagates consistently.

Implemented scenarios
----------------------
* ``baseline``        — forecast as-is.
* ``backlog_clear``   — clear the RTT backlog over N months: procedures are
                        uplifted by an extra fraction of the current waiting list
                        spread across the horizon (extra electives + equipment).
* ``capacity_cap``    — providers can only grow throughput by ``max_growth`` p.a.;
                        forecasts above that ceiling are clipped (supply-limited).
* ``demand_shock``    — uniform +/- shock (e.g. winter surge, pandemic) by ``pct``.
"""
from __future__ import annotations

import pandas as pd

from nhs_forecast.logging_setup import get_logger

log = get_logger("demand.scenarios")


def baseline(proc_fc: pd.DataFrame) -> pd.DataFrame:
    out = proc_fc.copy()
    out["scenario"] = "baseline"
    return out


def backlog_clear(
    proc_fc: pd.DataFrame, waiting: pd.DataFrame, clear_months: int = 12, clear_frac: float = 0.3
) -> pd.DataFrame:
    """Add extra activity from clearing ``clear_frac`` of the latest backlog.

    ``waiting`` is fact_rtt; the most recent waiting-list size per
    trust x treatment-function is spread evenly across ``clear_months`` and
    attributed to that family's procedure codes by their historical share.
    """
    out = proc_fc.copy()
    latest = (waiting.sort_values("date").groupby(["trust_code", "treatment_function"])
                     .tail(1))
    total_backlog = float(latest["waiting_list_size"].sum())
    horizon_months = out["date"].nunique()
    monthly_extra_total = (total_backlog * clear_frac) / max(clear_months, 1)

    # distribute the extra activity proportionally to each row's baseline yhat
    share = out["yhat"] / out["yhat"].groupby(out["date"]).transform("sum")
    uplift = share * monthly_extra_total
    applies = out["date"].rank(method="dense").le(clear_months)
    for col in ("yhat", "yhat_lower", "yhat_upper"):
        out[col] = out[col] + uplift.where(applies, 0.0)
    out["scenario"] = "backlog_clear"
    log.info("backlog_clear: total backlog %.0f, monthly extra %.0f over %d months",
             total_backlog, monthly_extra_total, clear_months)
    return out


def capacity_cap(proc_fc: pd.DataFrame, max_growth: float = 0.05) -> pd.DataFrame:
    """Cap each series at its first-forecast value grown by ``max_growth`` annually."""
    out = proc_fc.copy()
    keys = [c for c in ("trust_code", "region", "procedure_code") if c in out.columns]
    out = out.sort_values(keys + ["date"])
    first = out.groupby(keys, dropna=False)["yhat"].transform("first")
    step = out.groupby(keys, dropna=False).cumcount()
    ceiling = first * (1 + max_growth) ** (step / 12.0)
    for col in ("yhat", "yhat_lower", "yhat_upper"):
        out[col] = out[[col]].assign(c=ceiling).min(axis=1)
    out["scenario"] = "capacity_cap"
    return out


def demand_shock(proc_fc: pd.DataFrame, pct: float = 0.15) -> pd.DataFrame:
    out = proc_fc.copy()
    for col in ("yhat", "yhat_lower", "yhat_upper"):
        out[col] = out[col] * (1 + pct)
    out["scenario"] = f"demand_shock_{int(pct * 100):+d}pct"
    return out
