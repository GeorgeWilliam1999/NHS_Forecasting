"""Horizon utilisation forecast + portfolio risk & underwriting.

Pipeline of this module:
  1. ``forecast_horizon``     — recursive device-day mean forecast over H days.
  2. ``correlation_matrix``   — Ledoit–Wolf-shrunk utilisation correlation across
                                devices (the factor that breaks diversification).
  3. ``simulate_portfolio``   — Monte-Carlo horizon revenue via a Gaussian copula
                                with per-device NegBin marginals.
  4. ``underwrite``           — per-device pricing, tail metrics, key-person risk;
                                portfolio VaR / CVaR / effective diversification.

The risk engine deliberately models the *joint* distribution. The aggregate
demand pipeline's variance-summing shortcut assumes independence and therefore
under-prices the tail; here correlation enters explicitly through the copula.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhs_forecast.config import Settings
from nhs_forecast.logging_setup import get_logger
from nhs_forecast.telemetry import model as M
from nhs_forecast.telemetry.features import LAGS, ROLL, TARGET

log = get_logger("telemetry.risk")

try:
    from scipy import stats
    from sklearn.covariance import LedoitWolf
    _HAS = True
except Exception:  # pragma: no cover
    _HAS = False

_KAPPA = 0.5   # idiosyncratic (CV) loading on price
_RHO = 0.3     # systematic (beta-to-book) loading on price


# --------------------------------------------------------------------------- #
# 1. horizon forecast
# --------------------------------------------------------------------------- #
def forecast_horizon(model: M.FittedModel, features: pd.DataFrame,
                     devices: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Recursive device-day mean forecast; slow features persisted, AR fed back."""
    cap = devices.set_index("device_id")["cap_sessions_day"]
    rows = []
    for dev, g in features.groupby("device_id", observed=True):
        g = g.sort_values("date")
        if g.empty:
            continue
        last = g.iloc[-1]
        hist = list(g[TARGET].fillna(0).to_numpy(float))
        slow = {c: last[c] for c in (
            "op_herfindahl", "active_ops_28", "err_rate_28", "downtime_frac_28",
            "peer_util_lag1") if c in g.columns}
        start = pd.Timestamp(last["date"])
        trend0 = float(last["trend"])
        for h in range(1, horizon_days + 1):
            day = start + pd.Timedelta(days=h)
            dow, doy = day.dayofweek, day.dayofyear
            is_we = int(dow >= 5)
            row = dict(slow)
            row.update({
                "device_id": dev, "date": day, "is_weekend": is_we,
                "dow_sin": np.sin(2 * np.pi * dow / 7),
                "dow_cos": np.cos(2 * np.pi * dow / 7),
                "doy_sin": np.sin(2 * np.pi * doy / 365.25),
                "doy_cos": np.cos(2 * np.pi * doy / 365.25),
                "trend": trend0 + h / 365.25,
                "exposure_hours": 4.0 if is_we else 10.0,
            })
            arr = np.asarray(hist, float)
            for lag in LAGS:
                row[f"{TARGET}_lag{lag}"] = arr[-lag] if len(arr) >= lag else arr.mean()
            for w in ROLL:
                tail = arr[-w:] if len(arr) >= 2 else arr
                row[f"{TARGET}_rollmean{w}"] = float(np.mean(tail))
                row[f"{TARGET}_rollstd{w}"] = float(np.std(tail)) if len(tail) > 1 else 0.0
            mu = float(M.predict_mu(model, pd.DataFrame([row]))[0])
            row["mu"] = mu
            row["cap_sessions_day"] = int(cap.get(dev, 99))
            rows.append(row)
            hist.append(mu)  # feed the mean back for the next step
    fc = pd.DataFrame(rows)
    log.info("horizon forecast: %d device-days (H=%d)", len(fc), horizon_days)
    return fc


# --------------------------------------------------------------------------- #
# 2. correlation
# --------------------------------------------------------------------------- #
def correlation_matrix(device_day: pd.DataFrame, lookback: int = 120):
    """Ledoit–Wolf-shrunk correlation of recent daily billable utilisation."""
    dd = device_day.copy()
    dd["date"] = pd.to_datetime(dd["date"])
    last = dd["date"].max() - pd.Timedelta(days=lookback)
    wide = (dd[dd["date"] >= last]
            .pivot_table(index="date", columns="device_id", values="n_billable",
                         aggfunc="sum", fill_value=0).sort_index())
    devices = list(wide.columns)
    if not _HAS or wide.shape[0] < 10 or len(devices) < 2:
        return np.eye(len(devices)), devices
    cov = LedoitWolf().fit(wide.to_numpy(float)).covariance_
    d = np.sqrt(np.clip(np.diag(cov), 1e-9, None))
    corr = cov / np.outer(d, d)
    corr = np.clip(corr, -0.999, 0.999)
    np.fill_diagonal(corr, 1.0)
    return corr, devices


# --------------------------------------------------------------------------- #
# 3. Monte-Carlo portfolio simulation
# --------------------------------------------------------------------------- #
def _nb_params(mean: float, var: float):
    var = max(var, mean + 1e-6)
    r = mean ** 2 / (var - mean)
    p = r / (r + mean)
    return max(r, 1e-3), min(max(p, 1e-6), 1 - 1e-6)


def simulate_portfolio(forecast: pd.DataFrame, model: M.FittedModel,
                       devices: pd.DataFrame, corr: np.ndarray,
                       corr_devices: list[str], settings: Settings) -> dict:
    """Gaussian-copula Monte-Carlo of horizon revenue per device + portfolio."""
    rng = np.random.default_rng(settings.telemetry_seed + 1)
    n_sims = settings.risk_n_sims
    horizon_frac = settings.telemetry_horizon_days / 30.4

    # per-device horizon mean & variance of billable sessions
    agg = forecast.groupby("device_id").agg(
        exp_sessions=("mu", "sum"),
        cap_day=("cap_sessions_day", "first")).reset_index()
    agg["var_sessions"] = forecast.assign(
        v=lambda d: M.dispersion_var(model, d["mu"].to_numpy(float))
    ).groupby("device_id")["v"].sum().values

    dev_ids = list(agg["device_id"])
    econ = devices.set_index("device_id")
    # correlation aligned to forecast device order (identity for the rest)
    idx = {d: i for i, d in enumerate(corr_devices)}
    D = len(dev_ids)
    R = np.eye(D)
    for a in range(D):
        for b in range(D):
            ia, ib = idx.get(dev_ids[a]), idx.get(dev_ids[b])
            if ia is not None and ib is not None:
                R[a, b] = corr[ia, ib]
    # ensure positive-definite for Cholesky
    R = 0.5 * (R + R.T)
    eig = np.linalg.eigvalsh(R)
    if eig.min() < 1e-6:
        R += (1e-6 - eig.min()) * np.eye(D)
    L = np.linalg.cholesky(R)

    Z = rng.standard_normal((n_sims, D)) @ L.T
    U = stats.norm.cdf(Z) if _HAS else (Z - Z.min()) / (Z.ptp() + 1e-9)

    rev = np.zeros((n_sims, D))
    sessions = np.zeros((n_sims, D))
    horizon_working = int(np.sum(pd.to_datetime(
        forecast["date"].unique()).dayofweek < 5)) or settings.telemetry_horizon_days
    for j, dev in enumerate(dev_ids):
        r, p = _nb_params(float(agg.loc[j, "exp_sessions"]),
                          float(agg.loc[j, "var_sessions"]))
        draws = stats.nbinom.ppf(U[:, j], r, p) if _HAS else \
            np.random.poisson(agg.loc[j, "exp_sessions"], n_sims)
        draws = np.minimum(draws, agg.loc[j, "cap_day"] * horizon_working)
        sessions[:, j] = draws
        price = float(econ.loc[dev, "price_per_session_gbp"])
        floor = float(econ.loc[dev, "min_monthly_floor_gbp"]) * horizon_frac
        rev[:, j] = np.maximum(draws * price, floor)  # contractual floor

    fixed = np.array([float(econ.loc[d, "monthly_fixed_cost_gbp"]) * horizon_frac
                      for d in dev_ids])
    port_rev = rev.sum(axis=1)
    loss = fixed.sum() - port_rev            # platform shortfall
    alpha = settings.risk_alpha
    var_a = float(np.quantile(loss, 1 - alpha))
    cvar_a = float(loss[loss >= var_a].mean()) if np.any(loss >= var_a) else var_a

    return {
        "dev_ids": dev_ids, "rev": rev, "sessions": sessions, "fixed": fixed,
        "agg": agg, "horizon_working": horizon_working, "horizon_frac": horizon_frac,
        "portfolio": {
            "expected_revenue_gbp": float(port_rev.mean()),
            "revenue_std_gbp": float(port_rev.std()),
            "fixed_cost_gbp": float(fixed.sum()),
            "expected_margin_gbp": float(port_rev.mean() - fixed.sum()),
            "var_loss_gbp": var_a,
            "cvar_loss_gbp": cvar_a,
            "prob_book_loss": float((loss > 0).mean()),
            "alpha": alpha,
        },
    }


# --------------------------------------------------------------------------- #
# 4. underwriting
# --------------------------------------------------------------------------- #
def underwrite(sim: dict, features: pd.DataFrame, devices: pd.DataFrame,
               run_id: str) -> tuple[pd.DataFrame, dict]:
    rev, sessions, fixed = sim["rev"], sim["sessions"], sim["fixed"]
    dev_ids, econ = sim["dev_ids"], devices.set_index("device_id")
    port_rev = rev.sum(axis=1)
    # latest operator concentration per device
    herf = (features.sort_values("date").groupby("device_id")["op_herfindahl"]
            .last().to_dict())

    rows = []
    sigma = rev.std(axis=0)
    for j, dev in enumerate(dev_ids):
        r_d = rev[:, j]
        exp_rev = float(r_d.mean())
        cv = float(r_d.std() / exp_rev) if exp_rev > 0 else 0.0
        rest = port_rev - r_d
        var_rest = rest.var()
        beta = float(np.cov(r_d, rest)[0, 1] / var_rest) if var_rest > 0 else 0.0
        exp_sess = float(sessions[:, j].mean())
        fixed_h = float(fixed[j])
        # portfolio-aware price: cover carry, load idiosyncratic CV + systematic beta
        base_price = fixed_h / max(exp_sess, 1.0)
        beta_norm = beta / max(sigma.mean(), 1e-9) * sigma[j]
        suggested = base_price * (1 + _KAPPA * cv + _RHO * np.tanh(beta_norm))
        floor_h = float(econ.loc[dev, "min_monthly_floor_gbp"]) * sim["horizon_frac"]
        rows.append({
            "run_id": run_id, "device_id": dev,
            "site_id": econ.loc[dev, "site_id"], "region": econ.loc[dev, "region"],
            "specialty": econ.loc[dev, "specialty"],
            "expected_sessions": round(exp_sess, 2),
            "expected_revenue_gbp": round(exp_rev, 2),
            "cv": round(cv, 4),
            "beta_book": round(beta, 4),
            "suggested_price_gbp": round(float(suggested), 2),
            "current_price_gbp": round(float(econ.loc[dev, "price_per_session_gbp"]), 2),
            "uar_p5_sessions": round(float(np.quantile(sessions[:, j], 0.05)), 2),
            "floor_breach_prob": round(float((r_d <= floor_h + 1e-6).mean()), 4),
            "op_herfindahl": round(float(herf.get(dev, np.nan) or 0.0), 4),
            "expected_margin_gbp": round(exp_rev - fixed_h, 2),
        })
    book = pd.DataFrame(rows)

    # effective diversification: N_eff = (Σσ)² / (σ' R σ) via the realised sims
    if sigma.sum() > 0:
        port_var = port_rev.var()
        n_eff = float((sigma.sum() ** 2) / max(port_var, 1e-9))
    else:
        n_eff = float(len(dev_ids))
    portfolio = dict(sim["portfolio"])
    portfolio.update({
        "n_devices": len(dev_ids),
        "effective_n_independent": round(min(n_eff, len(dev_ids)), 2),
        "diversification_ratio": round(min(n_eff, len(dev_ids)) / max(len(dev_ids), 1), 3),
        "mean_operator_herfindahl": round(float(book["op_herfindahl"].mean()), 4),
        "n_devices_negative_margin": int((book["expected_margin_gbp"] < 0).sum()),
    })
    return book, portfolio
