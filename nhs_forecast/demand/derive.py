"""Convert procedure forecasts into equipment demand with uncertainty.

Deterministic mapping
---------------------
    demand[e, t] = sum_c forecast[c, t] * weight[c, e]

Uncertainty propagation
-----------------------
Each procedure forecast carries a prediction interval. Treating the interval as
~ +/- 1.28 sigma (P10/P90), the per-procedure variance is

    var_c = ((upper_c - lower_c) / (2 * 1.28)) ** 2

Equipment demand is a weighted sum of procedure forecasts. Assuming the
procedure forecast errors are **independent** across codes, variances add:

    var[e] = sum_c (weight[c,e] ** 2) * var_c

so the equipment interval half-width is 1.28 * sqrt(var[e]). This is the
statistically correct propagation for independent inputs and is materially
tighter (and more honest) than naively summing the procedure interval bounds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhs_forecast.config import Settings
from nhs_forecast.logging_setup import get_logger
from nhs_forecast.mapping.procedure_equipment import load_mapping

log = get_logger("demand.derive")

Z = 1.2816  # P90 z-score


def derive_equipment_demand(proc_fc: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Map a (possibly hierarchical) procedure forecast to equipment demand.

    ``proc_fc`` columns: level, trust_code, region, procedure_code, date,
    yhat, yhat_lower, yhat_upper.
    """
    mapping = load_mapping(settings)
    group_keys = [c for c in ("level", "trust_code", "region", "date") if c in proc_fc.columns]

    df = proc_fc.merge(mapping[["procedure_code", "equipment_type", "weight"]],
                       on="procedure_code", how="inner")
    df["demand"] = df["yhat"] * df["weight"]
    df["var"] = (df["weight"] ** 2) * (((df["yhat_upper"] - df["yhat_lower"]) / (2 * Z)) ** 2)

    agg = (df.groupby(group_keys + ["equipment_type"], dropna=False)
             .agg(demand=("demand", "sum"), var=("var", "sum"))
             .reset_index())
    half = Z * np.sqrt(agg["var"].clip(lower=0))
    agg["demand_lower"] = (agg["demand"] - half).clip(lower=0)
    agg["demand_upper"] = agg["demand"] + half
    agg = agg.drop(columns="var")
    log.info("derived equipment demand: %d rows across %d equipment types",
             len(agg), agg["equipment_type"].nunique())
    return agg
