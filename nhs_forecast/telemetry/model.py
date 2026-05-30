"""Censored Negative-Binomial utilisation model + pinball-loss backtest.

Why NegBin: device-day session counts are over-dispersed (var > mean); Poisson
under-states tail risk, which is exactly the risk we must price. The mean is
log-linear in the features with an **exposure offset**:

    n_billable_dt ~ NegBin(mu_dt, alpha),   log mu_dt = x_dt·beta + log(exposure_dt)

Censoring is handled pragmatically for a deployable baseline:
  * downtime  -> excluded via the exposure offset (no spurious zeros),
  * cap days  -> dropped from the *fit* so the contract ceiling does not bias the
                 latent-demand coefficients (right-censoring), then predicted
                 unclipped. (A full Tobit-NegBin likelihood is the next step.)

Dispersion ``alpha`` is estimated by the standard Cameron–Trivedi auxiliary
regression on a first-stage Poisson fit. The whole thing degrades gracefully:
NegBin -> Poisson -> seasonal-naive empirical quantiles, so the hosted app never
hard-fails on a thin or singular device series.

Forecasts are full predictive distributions (NegBin), summarised at the
underwriting quantiles and carried — with mean + dispersion — into the risk
engine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nhs_forecast.logging_setup import get_logger
from nhs_forecast.telemetry.features import NUMERIC_FEATURES, ROLL, TARGET

log = get_logger("telemetry.model")

try:
    import statsmodels.api as sm
    from scipy import stats
    _HAS_SM = True
except Exception:  # pragma: no cover - optional hard dependency guard
    _HAS_SM = False


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def pinball_loss(y_true, y_pred, q: float) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    d = y_true - y_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def coverage(y_true, lo, hi) -> float:
    y_true = np.asarray(y_true, float)
    return float(np.mean((y_true >= np.asarray(lo)) & (y_true <= np.asarray(hi))))


@dataclass
class FittedModel:
    kind: str                      # negbin | poisson | naive
    params: np.ndarray | None
    alpha: float                   # NegBin dispersion (0 => Poisson)
    feature_cols: list[str]
    fallback_rate: float           # per-exposure-hour rate for the naive path


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def _design(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = df[cols].to_numpy(float)
    return np.column_stack([np.ones(len(df)), X])


def _fit_negbin(df: pd.DataFrame, cols: list[str]):
    y = df[TARGET].to_numpy(float)
    offset = np.log(df["exposure_hours"].to_numpy(float))
    X = _design(df, cols)
    poi = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit(maxiter=100)
    mu = np.clip(poi.mu, 1e-6, None)
    # Cameron–Trivedi: ((y-mu)^2 - y)/mu = alpha * mu  (regress through origin)
    aux_y = ((y - mu) ** 2 - y) / mu
    alpha = float(np.clip(np.dot(mu, aux_y) / np.dot(mu, mu), 1e-4, 10.0))
    nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha),
                offset=offset).fit(maxiter=100)
    return nb.params, alpha


def fit(features: pd.DataFrame) -> FittedModel:
    """Fit the pooled censored NegBin model across the fleet."""
    cols = [c for c in NUMERIC_FEATURES if c in features.columns]
    train = features.dropna(subset=[TARGET, f"{TARGET}_rollmean{max(ROLL)}"]).copy()
    # right-censoring: drop device-days that hit the contractual/physical cap
    if "cap_sessions_day" in train.columns:
        train = train[train["n_sessions"] < train["cap_sessions_day"]]
    train = train[np.isfinite(train[cols].to_numpy(float)).all(axis=1)]

    fallback_rate = float((train[TARGET].sum() /
                           max(train["exposure_hours"].sum(), 1.0)))
    if not _HAS_SM or len(train) < 50:
        log.warning("model: falling back to seasonal-naive (n=%d, sm=%s)",
                    len(train), _HAS_SM)
        return FittedModel("naive", None, 0.0, cols, fallback_rate)
    try:
        params, alpha = _fit_negbin(train, cols)
        log.info("fitted NegBin on %d device-days (alpha=%.3f)", len(train), alpha)
        return FittedModel("negbin", params, alpha, cols, fallback_rate)
    except Exception as exc:  # pragma: no cover - numerical guard
        log.warning("NegBin failed (%s); using Poisson/naive fallback", exc)
        try:
            y = train[TARGET].to_numpy(float)
            off = np.log(train["exposure_hours"].to_numpy(float))
            poi = sm.GLM(y, _design(train, cols),
                         family=sm.families.Poisson(), offset=off).fit(maxiter=100)
            return FittedModel("poisson", poi.params, 0.0, cols, fallback_rate)
        except Exception:
            return FittedModel("naive", None, 0.0, cols, fallback_rate)


def predict_mu(model: FittedModel, df: pd.DataFrame) -> np.ndarray:
    exposure = df["exposure_hours"].to_numpy(float)
    if model.kind in ("negbin", "poisson") and model.params is not None:
        X = _design(df, model.feature_cols)
        eta = X @ model.params + np.log(exposure)
        return np.clip(np.exp(eta), 1e-6, None)
    return np.clip(model.fallback_rate * exposure, 1e-6, None)


def predict_quantiles(model: FittedModel, mu: np.ndarray,
                      quantiles: tuple[float, ...]) -> dict[float, np.ndarray]:
    """NegBin (or Poisson) predictive quantiles for each row."""
    out: dict[float, np.ndarray] = {}
    if not _HAS_SM:
        for q in quantiles:
            out[q] = mu * (0.5 + q)  # crude symmetric fallback
        return out
    if model.alpha > 0:
        r = 1.0 / model.alpha
        p = r / (r + mu)
        for q in quantiles:
            out[q] = stats.nbinom.ppf(q, r, p)
    else:
        for q in quantiles:
            out[q] = stats.poisson.ppf(q, mu)
    return out


def dispersion_var(model: FittedModel, mu: np.ndarray) -> np.ndarray:
    """Predictive variance: NegBin var = mu + alpha*mu^2 (Poisson if alpha=0)."""
    return mu + model.alpha * mu ** 2


# --------------------------------------------------------------------------- #
# backtest
# --------------------------------------------------------------------------- #
def backtest(features: pd.DataFrame, horizon_days: int,
             quantiles: tuple[float, float, float]) -> dict:
    """Rolling-origin holdout: last ``horizon_days`` per device held out."""
    cutoff = features["date"].max() - pd.Timedelta(days=horizon_days)
    train = features[features["date"] <= cutoff]
    test = features[features["date"] > cutoff].copy()
    test = test.dropna(subset=[f"{TARGET}_rollmean{max(ROLL)}"])
    if len(test) == 0 or len(train) == 0:
        return {}

    qlo, qmid, qhi = quantiles
    model = fit(train)
    mu = predict_mu(model, test)
    qs = predict_quantiles(model, mu, quantiles)
    y = test[TARGET].to_numpy(float)

    # seasonal-naive benchmark: last week's same-day billable count
    naive = test[f"{TARGET}_lag7"].fillna(test[f"{TARGET}_rollmean7"]).to_numpy(float)

    return {
        "model_kind": model.kind,
        "alpha": round(model.alpha, 4),
        "n_test": int(len(test)),
        "pinball_p05": round(pinball_loss(y, qs[qlo], qlo), 4),
        "pinball_p50": round(pinball_loss(y, qs[qmid], qmid), 4),
        "pinball_p95": round(pinball_loss(y, qs[qhi], qhi), 4),
        "pinball_p50_naive": round(pinball_loss(y, naive, qmid), 4),
        "mae": round(float(np.mean(np.abs(y - mu))), 4),
        "coverage_90": round(coverage(y, qs[qlo], qs[qhi]), 4),
    }
