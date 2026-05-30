"""Telemetry → underwriting orchestration.

Runs the device-session underwriting flow end-to-end and persists artifacts that
the dashboard / API read. Writes are best-effort: on a read-only deployment the
in-memory report is still returned, and the warehouse load is skipped cleanly.

Artifacts (under ``settings.artifacts_dir``):
  telemetry_device_day_latest.parquet
  telemetry_forecast_latest.parquet
  underwriting_latest.parquet
  telemetry_report_latest.json
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from nhs_forecast.config import Settings, get_settings
from nhs_forecast.logging_setup import get_logger
from nhs_forecast.storage import warehouse
from nhs_forecast.telemetry import features as F
from nhs_forecast.telemetry import model as M
from nhs_forecast.telemetry import risk as R
from nhs_forecast.telemetry import sessionise as S
from nhs_forecast.telemetry import synthetic

log = get_logger("telemetry.pipeline")


def run(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    log.info("=== telemetry underwriting run %s ===", run_id)

    frames = synthetic.generate(settings)
    devices, events = frames["devices"], frames["events"]

    sessions = S.sessionise(events, settings)
    sessions = S.attach_billing(sessions, devices)
    dd = S.device_day(events, sessions)
    feats = F.build_features(dd, sessions, devices)
    feats = feats.merge(devices[["device_id", "cap_sessions_day"]], on="device_id", how="left")

    metrics = M.backtest(
        feats, settings.telemetry_horizon_days,
        (settings.underwriting_low_q, 0.5, settings.underwriting_high_q))
    model = M.fit(feats)
    fc = R.forecast_horizon(model, feats, devices, settings.telemetry_horizon_days)

    corr, corr_devs = R.correlation_matrix(dd)
    sim = R.simulate_portfolio(fc, model, devices, corr, corr_devs, settings)
    book, portfolio = R.underwrite(sim, feats, devices, run_id)

    _persist(settings, run_id, dd, fc, book)
    report = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_devices": int(devices.shape[0]),
        "n_events": int(events.shape[0]),
        "n_sessions": int(sessions.shape[0]),
        "n_device_days": int(dd.shape[0]),
        "horizon_days": settings.telemetry_horizon_days,
        "backtest": metrics,
        "portfolio_risk": portfolio,
    }
    try:
        (settings.artifacts_dir / "telemetry_report_latest.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
    except (PermissionError, OSError):
        log.warning("artifacts dir not writable; report not persisted")
    log.info("=== telemetry run %s complete (CVaR=£%.0f, N_eff=%.1f) ===",
             run_id, portfolio["cvar_loss_gbp"], portfolio["effective_n_independent"])
    return report


def _persist(settings, run_id, dd, fc, book) -> None:
    try:
        dd.to_parquet(settings.artifacts_dir / "telemetry_device_day_latest.parquet",
                      index=False)
        fc.to_parquet(settings.artifacts_dir / "telemetry_forecast_latest.parquet",
                      index=False)
        book.to_parquet(settings.artifacts_dir / "underwriting_latest.parquet",
                        index=False)
    except (PermissionError, OSError):
        log.warning("artifacts dir not writable; parquet outputs skipped")

    # warehouse load is optional (skipped cleanly on read-only deployments)
    try:
        warehouse.init_schema(settings)
        with warehouse.connect(settings) as con:
            con.register("uw", book)
            con.execute("DELETE FROM underwriting_device WHERE run_id = ?", [run_id])
            con.execute(
                """INSERT INTO underwriting_device
                   (run_id, device_id, site_id, region, specialty, expected_sessions,
                    expected_revenue_gbp, cv, beta_book, suggested_price_gbp,
                    current_price_gbp, uar_p5_sessions, floor_breach_prob,
                    op_herfindahl, expected_margin_gbp)
                   SELECT run_id, device_id, site_id, region, specialty, expected_sessions,
                          expected_revenue_gbp, cv, beta_book, suggested_price_gbp,
                          current_price_gbp, uar_p5_sessions, floor_breach_prob,
                          op_herfindahl, expected_margin_gbp FROM uw""")
            con.unregister("uw")
    except Exception as exc:  # pragma: no cover - warehouse optional
        log.warning("warehouse load skipped: %s", exc)
