"""Forecast accuracy metrics and a backtest splitter.

MAPE is computed with a small denominator floor so months with near-zero volume
do not blow the metric up (a standard, defensible variant for intermittent NHS
sub-series).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, floor: float = 1.0) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.maximum(np.abs(y_true), floor)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def evaluate(y_true, y_pred) -> dict[str, float]:
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "mape": mape(y_true, y_pred)}


def temporal_split(df: pd.DataFrame, date_col: str, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last ``horizon`` months as the test set (no shuffling)."""
    cutoff = df[date_col].sort_values().unique()[-horizon]
    train = df[df[date_col] < cutoff].copy()
    test = df[df[date_col] >= cutoff].copy()
    return train, test
