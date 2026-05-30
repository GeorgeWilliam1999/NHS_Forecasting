"""Global LightGBM forecaster with recursive multi-step prediction.

A single model is trained across *all* ``trust_code x procedure_code`` series
using the engineered feature table. Three quantile regressors (P10/P50/P90)
provide point forecasts and prediction intervals.

Multi-step forecasting is *recursive*: at each future month the autoregressive
features (lags, rolling stats) are recomputed from the series' growing history
(predicted values feed the next step). Exogenous drivers (waiting-list pressure,
demographic share, procedure mix) are persisted at their last observed value —
a transparent assumption that can be replaced by driver-specific forecasts.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from nhs_forecast.features.build_features import FEATURE_COLUMNS, LAGS, ROLL_WINDOWS, TARGET
from nhs_forecast.logging_setup import get_logger
from nhs_forecast.models.base import ForecastResult, future_index

log = get_logger("models.lgbm")

CATEGORICAL = ["trust_code", "region", "opcs_chapter", "procedure_code"]
NUMERIC = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL]
QUANTILES = {"yhat_lower": 0.1, "yhat": 0.5, "yhat_upper": 0.9}


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


def train(train_df: pd.DataFrame, params: dict | None = None) -> dict[str, lgb.Booster]:
    train_df = _prep(train_df).dropna(subset=[TARGET])
    # rows with NaN lags (series warm-up) are dropped from training
    train_df = train_df.dropna(subset=[f"{TARGET}_lag{max(LAGS)}"])
    X = train_df[FEATURE_COLUMNS]
    y = train_df[TARGET]
    base = dict(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        verbosity=-1,
    )
    base.update(params or {})
    models: dict[str, lgb.Booster] = {}
    for name, q in QUANTILES.items():
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **base)
        m.fit(X, y, categorical_feature=CATEGORICAL)
        models[name] = m
    log.info("trained LightGBM quantile models on %d rows", len(train_df))
    return models


def _ar_features(hist: list[float]) -> dict[str, float]:
    """Compute lag and rolling features from a target history list."""
    arr = np.asarray(hist, float)
    feat: dict[str, float] = {}
    for lag in LAGS:
        feat[f"{TARGET}_lag{lag}"] = arr[-lag] if len(arr) >= lag else np.nan
    for w in ROLL_WINDOWS:
        window = arr[-w:] if len(arr) >= 1 else arr
        feat[f"{TARGET}_rollmean{w}"] = window.mean() if len(window) else np.nan
        feat[f"{TARGET}_rollstd{w}"] = window.std(ddof=1) if len(window) > 1 else 0.0
    return feat


def recursive_forecast(history: pd.DataFrame, models: dict, horizon: int) -> ForecastResult:
    """Forecast every series ``horizon`` months ahead.

    ``history`` is the full engineered feature table (historical rows only).
    """
    history = history.sort_values("date")
    keys = ["trust_code", "procedure_code"]
    cat_dtypes = {c: history[c].astype("category").cat.categories for c in CATEGORICAL}
    rows = []

    for (trust, code), g in history.groupby(keys, observed=True):
        g = g.sort_values("date")
        last_row = g.iloc[-1]
        hist_vals = g[TARGET].tolist()
        last_date = pd.Timestamp(last_row["date"])
        # exogenous drivers persisted at last observed value
        exog = {
            "proc_mix_share": last_row["proc_mix_share"],
            "wl_size_lag1": last_row.get("wl_size", last_row.get("wl_size_lag1", np.nan)),
            "wl_pct_18wk_lag1": last_row.get("wl_pct_18wk", last_row.get("wl_pct_18wk_lag1", np.nan)),
            "pop_65plus_share": last_row["pop_65plus_share"],
        }
        region, chapter = last_row["region"], last_row["opcs_chapter"]
        for d in future_index(last_date, horizon):
            feat = {
                "month": d.month, "quarter": d.quarter,
                "sin_month": np.sin(2 * np.pi * d.month / 12),
                "cos_month": np.cos(2 * np.pi * d.month / 12),
                "time_idx": (d.year - history["date"].dt.year.min()) * 12 + (d.month - 1),
                "trust_code": trust, "region": region,
                "opcs_chapter": chapter, "procedure_code": code,
                **exog, **_ar_features(hist_vals),
            }
            Xrow = pd.DataFrame([feat])[FEATURE_COLUMNS]
            for c in CATEGORICAL:
                Xrow[c] = pd.Categorical(Xrow[c], categories=cat_dtypes[c])
            preds = {name: float(max(0.0, m.predict(Xrow)[0])) for name, m in models.items()}
            # enforce monotone interval ordering
            lo, mid, hi = sorted((preds["yhat_lower"], preds["yhat"], preds["yhat_upper"]))
            rows.append({
                "trust_code": trust, "region": region, "procedure_code": code, "date": d,
                "yhat": mid, "yhat_lower": lo, "yhat_upper": hi,
            })
            hist_vals.append(mid)  # recursive feedback

    return ForecastResult(model="lgbm", frame=pd.DataFrame(rows))
