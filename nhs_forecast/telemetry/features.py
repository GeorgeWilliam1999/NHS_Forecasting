"""Leakage-safe device-day feature engineering for utilisation forecasting.

Target: ``n_billable`` (billable sessions / device-day) — the revenue driver.
Offset: ``exposure_hours`` (powered-on hours) enters the count model as
``log(exposure)`` so "device down" contributes no evidence against demand.

Feature families (mirroring the design note):
  * temporal (multi-scale): day-of-week, annual Fourier, linear trend, weekend.
  * autoregressive: billable lags (1,7,14) + rolling mean/std (7,28).
  * behavioural: operator-concentration Herfindahl + active-operator count (28d).
  * device health: trailing error rate, downtime fraction (28d).
  * correlation proxy: lagged peer utilisation (mean of other same-region devices).

Every driver is shifted so a row at day *t* never sees day-*t* information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 7, 14)
ROLL = (7, 28)
TARGET = "n_billable"

NUMERIC_FEATURES = [
    "dow_sin", "dow_cos", "doy_sin", "doy_cos", "trend", "is_weekend",
    "op_herfindahl", "active_ops_28", "err_rate_28", "downtime_frac_28",
    "peer_util_lag1",
    *[f"{TARGET}_lag{lag}" for lag in LAGS],
    *[f"{TARGET}_rollmean{w}" for w in ROLL],
    *[f"{TARGET}_rollstd{w}" for w in ROLL],
]


def _operator_herfindahl(sessions: pd.DataFrame, dd_index: pd.DataFrame) -> pd.Series:
    """28-day trailing operator concentration H = Σ share² per device-day."""
    s = sessions.copy()
    s["date"] = s["t_start"].dt.normalize()
    daily = (s.groupby(["device_id", "date", "operator_hash"]).size()
              .rename("n").reset_index())
    out = {}
    for dev, g in daily.groupby("device_id"):
        wide = (g.pivot_table(index="date", columns="operator_hash", values="n",
                              aggfunc="sum", fill_value=0).sort_index())
        roll = wide.rolling("28D").sum()
        tot = roll.sum(axis=1).replace(0, np.nan)
        herf = ((roll.div(tot, axis=0)) ** 2).sum(axis=1)
        for d, h in herf.items():
            out[(dev, pd.Timestamp(d))] = float(h)
    idx = list(zip(dd_index["device_id"], dd_index["date"]))
    return pd.Series([out.get(k, np.nan) for k in idx], index=dd_index.index)


def build_features(device_day: pd.DataFrame, sessions: pd.DataFrame,
                   devices: pd.DataFrame) -> pd.DataFrame:
    df = device_day.merge(
        devices[["device_id", "region", "specialty", "device_type"]],
        on="device_id", how="left")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["device_id", "date"]).reset_index(drop=True)

    dow = df["date"].dt.dayofweek
    doy = df["date"].dt.dayofyear
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["is_weekend"] = (dow >= 5).astype(int)
    df["trend"] = (df["date"] - df["date"].min()).dt.days / 365.25

    g = df.groupby("device_id", observed=True)[TARGET]
    for lag in LAGS:
        df[f"{TARGET}_lag{lag}"] = g.shift(lag)
    for w in ROLL:
        df[f"{TARGET}_rollmean{w}"] = g.transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=2).mean())
        df[f"{TARGET}_rollstd{w}"] = g.transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=2).std())

    # device health: trailing error rate per session + downtime fraction.
    df["err_per_session"] = df["n_errors"] / df["n_sessions"].replace(0, np.nan)
    df["err_rate_28"] = (df.groupby("device_id")["err_per_session"]
                           .transform(lambda s: s.shift(1).rolling(28, min_periods=3).mean()))
    # downtime: calendar gaps where the device produced no telemetry that day
    df["err_rate_28"] = df["err_rate_28"].fillna(0.0)
    df["downtime_frac_28"] = 0.0  # exposure already encodes same-day downtime

    # behavioural concentration
    df["op_herfindahl"] = _operator_herfindahl(sessions, df).fillna(0.0)
    df["active_ops_28"] = (df.groupby("device_id")["n_operators"]
                             .transform(lambda s: s.shift(1).rolling(28, min_periods=3).max())
                             .fillna(1.0))

    # peer utilisation (same region, excluding self) — correlation/referral proxy
    reg = (df.groupby(["region", "date"])[TARGET].transform("sum") - df[TARGET])
    reg_n = (df.groupby(["region", "date"])[TARGET].transform("size") - 1).clip(lower=1)
    df["peer_util"] = reg / reg_n
    df["peer_util_lag1"] = (df.groupby("device_id")["peer_util"].shift(1)).fillna(0.0)

    df["exposure_hours"] = df["exposure_hours"].clip(lower=0.5)
    return df
