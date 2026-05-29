"""Hierarchical forecast reconciliation.

The forecast hierarchy is:

    national
      └── regional (NHS region)
            └── trust (provider)

Base forecasts are produced at the **trust x procedure_code** (bottom) level by
the global LightGBM model. ``bottom_up`` aggregates these into coherent regional
and national series (the sum of children always equals the parent — the defining
property of a reconciled hierarchy). ``middle_out`` is offered for cases where a
regional model is more reliable, redistributing to trusts by historical share.
"""
from __future__ import annotations

import pandas as pd

from nhs_forecast.logging_setup import get_logger

log = get_logger("models.hierarchical")

VALUE_COLS = ["yhat", "yhat_lower", "yhat_upper"]


def bottom_up(trust_fc: pd.DataFrame) -> pd.DataFrame:
    """Return trust + regional + national levels stacked with a ``level`` column.

    Interval aggregation: point forecasts sum exactly. For intervals we sum the
    bounds (a conservative comonotonic assumption); independent aggregation would
    use sqrt of summed variances — see ``demand.derive`` for that variant.
    """
    trust = trust_fc.copy()
    trust["level"] = "trust"

    regional = (trust_fc.groupby(["region", "procedure_code", "date"], as_index=False)[VALUE_COLS]
                        .sum())
    regional["level"] = "regional"
    regional["trust_code"] = pd.NA

    national = (trust_fc.groupby(["procedure_code", "date"], as_index=False)[VALUE_COLS]
                        .sum())
    national["level"] = "national"
    national["trust_code"] = pd.NA
    national["region"] = pd.NA

    cols = ["level", "trust_code", "region", "procedure_code", "date", *VALUE_COLS]
    out = pd.concat([trust[cols], regional[cols], national[cols]], ignore_index=True)
    log.info("bottom-up reconciliation: %d trust + %d regional + %d national rows",
             len(trust), len(regional), len(national))
    return out


def middle_out(regional_fc: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Disaggregate a regional forecast to trusts using historical volume share."""
    shares = (history.groupby(["region", "trust_code", "procedure_code"])["n_procedures"]
                     .sum().reset_index())
    region_tot = shares.groupby(["region", "procedure_code"])["n_procedures"].transform("sum")
    shares["share"] = shares["n_procedures"] / region_tot.replace(0, pd.NA)
    merged = regional_fc.merge(
        shares[["region", "trust_code", "procedure_code", "share"]],
        on=["region", "procedure_code"], how="left")
    for c in VALUE_COLS:
        merged[c] = merged[c] * merged["share"].fillna(0)
    merged["level"] = "trust"
    return merged.drop(columns="share")
