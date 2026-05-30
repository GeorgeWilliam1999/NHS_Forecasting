"""Tests for the pay-per-use telemetry → underwriting subsystem.

These run on a deliberately small fleet/window so the full generate → sessionise
→ model → risk flow stays fast while still exercising every stage.
"""
from __future__ import annotations

import numpy as np
import pytest

from nhs_forecast.config import Settings
from nhs_forecast.telemetry import features as F
from nhs_forecast.telemetry import model as M
from nhs_forecast.telemetry import risk as R
from nhs_forecast.telemetry import sessionise as S
from nhs_forecast.telemetry import synthetic


@pytest.fixture()
def tsettings(tmp_path):
    s = Settings(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        lake_dir=tmp_path / "data" / "lake",
        warehouse_path=tmp_path / "data" / "warehouse" / "nhs.duckdb",
        artifacts_dir=tmp_path / "data" / "artifacts",
        telemetry_n_devices=6,
        telemetry_days=120,
        telemetry_horizon_days=14,
        risk_n_sims=500,
    )
    s.ensure_dirs()
    return s


@pytest.fixture()
def built(tsettings):
    frames = synthetic.generate(tsettings)
    sessions = S.sessionise(frames["events"], tsettings)
    sessions = S.attach_billing(sessions, frames["devices"])
    dd = S.device_day(frames["events"], sessions)
    feats = F.build_features(dd, sessions, frames["devices"])
    feats = feats.merge(frames["devices"][["device_id", "cap_sessions_day"]],
                        on="device_id", how="left")
    return tsettings, frames, sessions, dd, feats


def test_generator_is_deterministic(tsettings):
    a = synthetic.generate(tsettings)["events"]
    b = synthetic.generate(tsettings)["events"]
    assert a.shape == b.shape
    assert a["event_type"].tolist() == b["event_type"].tolist()


def test_sessionise_roundtrip(built):
    _, frames, sessions, _, _ = built
    assert not sessions.empty
    # every session belongs to a known device and has non-negative duration
    assert set(sessions["device_id"]).issubset(set(frames["devices"]["device_id"]))
    assert (sessions["active_seconds"] >= 0).all()
    assert sessions["billable"].dtype == bool


def test_device_day_aggregation(built):
    _, _, sessions, dd, _ = built
    assert (dd["exposure_hours"] >= 0).all()
    assert (dd["n_billable"] >= 0).all()
    assert (dd["n_billable"] <= dd["n_sessions"]).all()
    # billable count reconciles with session-level flags per device-day
    sdate = sessions["t_start"].dt.normalize()
    by_day = sessions.groupby([sessions["device_id"], sdate])["billable"].sum()
    assert int(by_day.sum()) == int(dd["n_billable"].sum())


def test_model_backtest_finite(built):
    _, _, _, _, feats = built
    metrics = M.backtest(feats, 14, (0.05, 0.5, 0.95))
    assert np.isfinite(metrics["pinball_p50"])
    assert np.isfinite(metrics["pinball_p05"])
    assert np.isfinite(metrics["pinball_p95"])
    assert 0.0 <= metrics["coverage_90"] <= 1.0


def test_risk_portfolio_well_formed(built):
    tsettings, frames, _, dd, feats = built
    model = M.fit(feats)
    fc = R.forecast_horizon(model, feats, frames["devices"], tsettings.telemetry_horizon_days)
    corr, corr_devs = R.correlation_matrix(dd)
    # correlation matrix is symmetric PSD with unit diagonal
    assert np.allclose(corr, corr.T, atol=1e-8)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-6)
    assert np.linalg.eigvalsh(corr).min() > -1e-8
    sim = R.simulate_portfolio(fc, model, frames["devices"], corr, corr_devs, tsettings)
    book, portfolio = R.underwrite(sim, feats, frames["devices"], "test")
    n = frames["devices"].shape[0]
    assert 1.0 <= portfolio["effective_n_independent"] <= n + 1e-6
    assert portfolio["cvar_loss_gbp"] >= portfolio["var_loss_gbp"]
    # devices with no observed activity in the (short) window are cold-start and
    # cannot be priced; the book covers the active subset of the fleet.
    assert set(book["device_id"]).issubset(set(frames["devices"]["device_id"]))
    assert not book.empty
