"""SARIMA baseline forecaster (statsmodels).

Operates per univariate series. Used as the interpretable baseline at the
national x procedure_code grain (one model per code). A fixed, robust seasonal
order is used by default; ``auto_order`` does a small grid search on AIC.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from nhs_forecast.logging_setup import get_logger
from nhs_forecast.models.base import ForecastResult, future_index

log = get_logger("models.sarima")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from statsmodels.tsa.statespace.sarimax import SARIMAX


def _fit_one(series: pd.Series, order, seasonal_order):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp=False)


def forecast_series(
    series: pd.Series,
    horizon: int,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 0, 12),
    alpha: float = 0.2,
) -> pd.DataFrame:
    """Return a frame with date, yhat, yhat_lower, yhat_upper for one series."""
    series = series.astype(float).asfreq("MS")
    res = _fit_one(series, order, seasonal_order)
    fc = res.get_forecast(steps=horizon)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=alpha)
    idx = future_index(series.index[-1], horizon)
    out = pd.DataFrame({
        "date": idx,
        "yhat": np.clip(mean.to_numpy(), 0, None),
        "yhat_lower": np.clip(ci.iloc[:, 0].to_numpy(), 0, None),
        "yhat_upper": np.clip(ci.iloc[:, 1].to_numpy(), 0, None),
    })
    return out


def forecast(
    df: pd.DataFrame,
    horizon: int,
    group_cols=("procedure_code",),
    date_col: str = "date",
    target: str = "n_procedures",
) -> ForecastResult:
    """Fit one SARIMA per group and concatenate forecasts."""
    group_cols = list(group_cols)
    frames = []
    for keys, g in df.groupby(group_cols, observed=True):
        s = g.set_index(date_col)[target].sort_index()
        if len(s) < 24:
            continue
        try:
            fc = forecast_series(s, horizon)
        except Exception as exc:  # noqa: BLE001
            log.warning("SARIMA failed for %s (%s); skipping", keys, exc)
            continue
        keys = keys if isinstance(keys, tuple) else (keys,)
        for col, val in zip(group_cols, keys):
            fc[col] = val
        frames.append(fc)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return ForecastResult(model="sarima", frame=frame)
