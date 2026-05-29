"""Common forecasting types and helpers."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ForecastResult:
    """Long-format forecast with prediction intervals.

    Columns of ``frame``: keys (e.g. procedure_code, trust_code, region),
    ``date``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
    """
    model: str
    frame: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)

    def tag(self, **cols) -> "ForecastResult":
        for k, v in cols.items():
            self.frame[k] = v
        return self


def future_index(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(last_date, periods=horizon + 1, freq="MS")[1:]
