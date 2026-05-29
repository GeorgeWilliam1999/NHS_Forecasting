"""Build the modelling dataset for procedure-volume forecasting.

The target is monthly ``n_procedures`` at ``trust_code x procedure_code`` grain.
Feature families (Part 4 of the brief):

1. Time features      — month, quarter, fourier seasonality, linear trend.
2. Lagged variables   — y lags (1,2,3,6,12) + rolling mean/std (3,6,12).
3. Waiting-list pressure — RTT waiting-list size & % within 18wk, mapped from
                           treatment function to OPCS chapter, lagged.
4. Demographics       — regional 65+ population share (annual, forward-filled).
5. Regional effects   — region + trust categorical encodings (trust-level varia.).
6. Procedure mix      — each code's share of its trust's total monthly activity.

All joins are leakage-safe: pressure/demographic drivers are lagged so a row at
month *t* never sees information unavailable at forecast time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhs_forecast.config import Settings
from nhs_forecast.logging_setup import get_logger
from nhs_forecast.storage import warehouse

log = get_logger("features")

# OPCS chapter -> RTT treatment function, so waiting-list pressure attaches to the
# right procedure family.
CHAPTER_TO_TF = {
    "W": "Trauma & Orthopaedics",
    "H": "Gastroenterology",
    "G": "Gastroenterology",
    "K": "Cardiology",
    "M": "Urology",
    "C": "Ophthalmology",
}
LAGS = (1, 2, 3, 6, 12)
ROLL_WINDOWS = (3, 6, 12)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"].dt
    df["month"] = d.month
    df["quarter"] = d.quarter
    df["year"] = d.year
    # fourier terms capture smooth annual seasonality for tree/linear models
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["time_idx"] = (df["year"] - df["year"].min()) * 12 + (df["month"] - 1)
    return df


def _add_lags(df: pd.DataFrame, group_cols: list[str], target: str) -> pd.DataFrame:
    g = df.groupby(group_cols, observed=True)[target]
    for lag in LAGS:
        df[f"{target}_lag{lag}"] = g.shift(lag)
    for w in ROLL_WINDOWS:
        # shift(1) first so the rolling window excludes the current month, then
        # roll *within* each group via transform (keeps alignment to df index)
        df[f"{target}_rollmean{w}"] = g.transform(
            lambda s, w=w: s.shift(1).rolling(w).mean())
        df[f"{target}_rollstd{w}"] = g.transform(
            lambda s, w=w: s.shift(1).rolling(w).std())
    return df


def build_feature_table(settings: Settings) -> pd.DataFrame:
    proc = warehouse.query(settings, "SELECT * FROM fact_procedures")
    rtt = warehouse.query(settings, "SELECT * FROM fact_rtt")
    demo = warehouse.query(settings, "SELECT * FROM dim_demographics")
    proc["date"] = pd.to_datetime(proc["date"])
    rtt["date"] = pd.to_datetime(rtt["date"])

    proc = proc.sort_values(["trust_code", "procedure_code", "date"]).reset_index(drop=True)
    proc = _add_time_features(proc)
    proc = _add_lags(proc, ["trust_code", "procedure_code"], "n_procedures")

    # --- procedure mix: share of the trust's total monthly procedures ---
    trust_month_total = (proc.groupby(["trust_code", "date"], observed=True)["n_procedures"]
                             .transform("sum"))
    proc["proc_mix_share"] = proc["n_procedures"] / trust_month_total.replace(0, np.nan)

    # --- waiting-list pressure (RTT) joined via chapter -> treatment function ---
    proc["treatment_function"] = proc["opcs_chapter"].map(CHAPTER_TO_TF)
    rtt_keyed = rtt.rename(columns={
        "waiting_list_size": "wl_size", "pct_within_18wk": "wl_pct_18wk"})
    proc = proc.merge(
        rtt_keyed[["date", "trust_code", "treatment_function", "wl_size", "wl_pct_18wk"]],
        on=["date", "trust_code", "treatment_function"], how="left",
    )
    # lag pressure so it is known before the forecast month
    pg = proc.groupby(["trust_code", "treatment_function"], observed=True)
    proc["wl_size_lag1"] = pg["wl_size"].shift(1)
    proc["wl_pct_18wk_lag1"] = pg["wl_pct_18wk"].shift(1)

    # --- demographics: regional 65+ share (annual -> forward filled monthly) ---
    demo["is_old"] = demo["age_band"].isin(["65-79", "80+"])
    old_share = (demo.assign(old_pop=demo["population"] * demo["is_old"])
                     .groupby(["region", "year"])
                     .agg(old_pop=("old_pop", "sum"), tot_pop=("population", "sum"))
                     .reset_index())
    old_share["pop_65plus_share"] = old_share["old_pop"] / old_share["tot_pop"]
    proc = proc.merge(old_share[["region", "year", "pop_65plus_share"]],
                      on=["region", "year"], how="left")
    proc["pop_65plus_share"] = (proc.sort_values("date")
                                    .groupby("region")["pop_65plus_share"].ffill())

    # categorical encodings preserved as category dtype for LightGBM
    for col in ("trust_code", "region", "opcs_chapter", "procedure_code"):
        proc[col] = proc[col].astype("category")

    log.info("feature table built: %d rows x %d cols", len(proc), proc.shape[1])
    return proc


FEATURE_COLUMNS = [
    "month", "quarter", "sin_month", "cos_month", "time_idx", "proc_mix_share",
    "wl_size_lag1", "wl_pct_18wk_lag1", "pop_65plus_share",
    *[f"n_procedures_lag{l}" for l in LAGS],
    *[f"n_procedures_rollmean{w}" for w in ROLL_WINDOWS],
    *[f"n_procedures_rollstd{w}" for w in ROLL_WINDOWS],
    "trust_code", "region", "opcs_chapter", "procedure_code",
]
TARGET = "n_procedures"
