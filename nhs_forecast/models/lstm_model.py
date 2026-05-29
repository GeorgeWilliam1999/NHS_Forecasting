"""LSTM sequence forecaster (optional — requires the ``deep`` extra / torch).

A compact global LSTM trained on sliding windows pooled across all series. If
torch is not installed the module exposes ``HAS_TORCH = False`` and the pipeline
silently skips it. This stands in for a Temporal Fusion Transformer; the data
plumbing (windowing, scaling, recursive decode) is identical and a TFT can be
dropped in by swapping the ``nn.Module``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhs_forecast.logging_setup import get_logger
from nhs_forecast.models.base import ForecastResult, future_index

log = get_logger("models.lstm")

try:
    import torch
    from torch import nn

    HAS_TORCH = True
except Exception:  # noqa: BLE001
    HAS_TORCH = False


if HAS_TORCH:

    class _LSTM(nn.Module):
        def __init__(self, n_features: int = 1, hidden: int = 32, layers: int = 1):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    def _windows(series: np.ndarray, lookback: int):
        xs, ys = [], []
        for i in range(len(series) - lookback):
            xs.append(series[i : i + lookback])
            ys.append(series[i + lookback])
        return np.array(xs), np.array(ys)


def forecast(
    df: pd.DataFrame,
    horizon: int,
    group_cols=("trust_code", "procedure_code"),
    date_col: str = "date",
    target: str = "n_procedures",
    lookback: int = 12,
    epochs: int = 30,
) -> ForecastResult:
    if not HAS_TORCH:
        log.warning("torch unavailable; LSTM skipped")
        return ForecastResult(model="lstm", frame=pd.DataFrame())

    group_cols = list(group_cols)
    torch.manual_seed(0)
    series_map: dict[tuple, tuple[np.ndarray, float, float]] = {}
    X_all, y_all = [], []
    for keys, g in df.groupby(group_cols, observed=True):
        s = g.sort_values(date_col)[target].to_numpy(float)
        if len(s) < lookback + 6:
            continue
        mu, sd = s.mean(), s.std() + 1e-6
        sn = (s - mu) / sd
        keys = keys if isinstance(keys, tuple) else (keys,)
        series_map[keys] = (sn, mu, sd)
        xw, yw = _windows(sn, lookback)
        X_all.append(xw)
        y_all.append(yw)
    if not X_all:
        return ForecastResult(model="lstm", frame=pd.DataFrame())

    X = torch.tensor(np.concatenate(X_all)[..., None], dtype=torch.float32)
    y = torch.tensor(np.concatenate(y_all)[:, None], dtype=torch.float32)
    model = _LSTM()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
    log.info("LSTM trained: final loss %.4f over %d windows", float(loss.detach()), len(X))

    model.eval()
    rows = []
    last_dates = df.groupby(group_cols, observed=True)[date_col].max()
    with torch.no_grad():
        for keys, (sn, mu, sd) in series_map.items():
            window = list(sn[-lookback:])
            kd = keys if len(group_cols) > 1 else keys[0]
            last_date = pd.Timestamp(last_dates.loc[kd])
            preds = []
            for _ in range(horizon):
                xin = torch.tensor(np.array(window[-lookback:])[None, :, None], dtype=torch.float32)
                p = float(model(xin)[0, 0])
                preds.append(p)
                window.append(p)
            yhat = np.clip(np.array(preds) * sd + mu, 0, None)
            resid = sd  # crude interval = 1 std of the series
            for d, v in zip(future_index(last_date, horizon), yhat):
                row = {col: k for col, k in zip(group_cols, keys)}
                row.update({"date": d, "yhat": v,
                            "yhat_lower": max(0.0, v - 1.28 * resid),
                            "yhat_upper": v + 1.28 * resid})
                rows.append(row)
    return ForecastResult(model="lstm", frame=pd.DataFrame(rows))
