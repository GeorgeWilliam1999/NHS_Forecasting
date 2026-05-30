"""End-to-end pipeline stages.

Stages (each is independently callable and testable):

    1. ingest      — pull all sources -> canonical frames
    2. validate    — declarative checks; abort on error severity
    3. land        — write immutable Parquet snapshots to the data lake
    4. warehouse   — load facts + build conformed dimensions in DuckDB
    5. features    — build the modelling table
    6. backtest    — temporal holdout; compare SARIMA / LightGBM / LSTM
    7. forecast    — train on full history, forecast `horizon` months (LightGBM)
    8. hierarchy   — bottom-up reconcile trust -> regional -> national
    9. equipment   — map to equipment demand + uncertainty for each scenario
    10. persist    — write forecasts to the warehouse + artifacts

``run`` wires them together and returns a structured report.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pandas as pd

from nhs_forecast.config import Settings, get_settings
from nhs_forecast.demand import scenarios
from nhs_forecast.demand.derive import derive_equipment_demand
from nhs_forecast.features.build_features import TARGET, build_feature_table
from nhs_forecast.ingestion import (
    base, diagnostic_imaging, hes, monthly_activity, ons_demographics, rtt_waiting, supply_chain,
)
from nhs_forecast.logging_setup import get_logger
from nhs_forecast.models import hierarchical, lgbm_model, lstm_model, sarima_model
from nhs_forecast.models.evaluate import evaluate, temporal_split
from nhs_forecast.storage import lake, warehouse
from nhs_forecast.validation import checks

log = get_logger("pipeline")

INGESTORS = {
    "procedures": hes.load,
    "imaging": diagnostic_imaging.load,
    "rtt": rtt_waiting.load,
    "activity": monthly_activity.load,
    "demographics": ons_demographics.load,
    "supply": supply_chain.load,
}


def ingest(settings: Settings) -> dict[str, pd.DataFrame]:
    base.reset_provenance()
    return {name: fn(settings) for name, fn in INGESTORS.items()}


def validate(frames: dict[str, pd.DataFrame]) -> list[checks.CheckResult]:
    results: list[checks.CheckResult] = []
    for name, df in frames.items():
        if name in checks.SPECS:
            results += checks.validate(name, df, checks.SPECS[name])
    checks.assert_no_errors(results)
    return results


def land(settings: Settings, frames: dict[str, pd.DataFrame]) -> None:
    for name, df in frames.items():
        lake.write_snapshot(settings, name, df)


def load_warehouse(settings: Settings, frames: dict[str, pd.DataFrame]) -> None:
    warehouse.init_schema(settings)
    for name, df in frames.items():
        table = warehouse.LOAD_MAP.get(name)
        if table:
            warehouse.replace_table(settings, table, df)
    warehouse.build_dimensions(settings)


def backtest(features: pd.DataFrame, settings: Settings) -> dict[str, dict[str, float]]:
    """Compare models on a temporal holdout of length ``horizon``."""
    horizon = settings.horizon
    train_df, test_df = temporal_split(features, "date", horizon)
    metrics: dict[str, dict[str, float]] = {}

    # LightGBM (global, recursive)
    models = lgbm_model.train(train_df)
    fc = lgbm_model.recursive_forecast(train_df, models, horizon).frame
    merged = fc.merge(test_df[["trust_code", "procedure_code", "date", TARGET]],
                      on=["trust_code", "procedure_code", "date"], how="inner")
    if len(merged):
        metrics["lgbm"] = evaluate(merged[TARGET], merged["yhat"])

    # SARIMA baseline at national x code level
    nat = (features.groupby(["procedure_code", "date"], as_index=False)[TARGET].sum())
    nat_train = nat[nat["date"] < test_df["date"].min()]
    sar = sarima_model.forecast(nat_train, horizon, group_cols=("procedure_code",), target=TARGET)
    nat_test = nat[nat["date"] >= test_df["date"].min()]
    if len(sar.frame):
        sm = sar.frame.merge(nat_test, on=["procedure_code", "date"], how="inner")
        if len(sm):
            metrics["sarima"] = evaluate(sm[TARGET], sm["yhat"])

    # LSTM (optional)
    if lstm_model.HAS_TORCH:
        lf = lstm_model.forecast(train_df, horizon, target=TARGET)
        if len(lf.frame):
            lm = lf.frame.merge(test_df[["trust_code", "procedure_code", "date", TARGET]],
                                on=["trust_code", "procedure_code", "date"], how="inner")
            if len(lm):
                metrics["lstm"] = evaluate(lm[TARGET], lm["yhat"])

    for model, m in metrics.items():
        log.info("backtest %-7s MAE=%.1f RMSE=%.1f MAPE=%.1f%%",
                 model, m["mae"], m["rmse"], m["mape"])
    return metrics


def forecast(features: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Train LightGBM on full history and forecast the horizon (trust x code)."""
    models = lgbm_model.train(features)
    return lgbm_model.recursive_forecast(features, models, settings.horizon).frame


def run(settings: Settings | None = None, run_scenarios: bool = True) -> dict:
    settings = settings or get_settings()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    log.info("=== pipeline run %s (synthetic=%s) ===", run_id, settings.use_synthetic)

    frames = ingest(settings)
    validate(frames)
    land(settings, frames)
    load_warehouse(settings, frames)

    features = build_feature_table(settings)
    metrics = backtest(features, settings)
    proc_fc = forecast(features, settings)

    # hierarchy
    hier = hierarchical.bottom_up(proc_fc)

    # scenarios at trust level then re-reconcile + derive equipment demand
    rtt = warehouse.query(settings, "SELECT * FROM fact_rtt")
    rtt["date"] = pd.to_datetime(rtt["date"])
    scenario_fns = {
        "baseline": lambda d: scenarios.baseline(d),
        "backlog_clear": lambda d: scenarios.backlog_clear(d, rtt),
        "capacity_cap": lambda d: scenarios.capacity_cap(d),
    } if run_scenarios else {"baseline": lambda d: scenarios.baseline(d)}

    equip_frames = []
    for name, fn in scenario_fns.items():
        scen_proc = fn(proc_fc.copy())
        scen_hier = hierarchical.bottom_up(scen_proc)
        equip = derive_equipment_demand(scen_hier, settings)
        equip["scenario"] = name
        equip_frames.append(equip)
    equipment = pd.concat(equip_frames, ignore_index=True)

    _persist(settings, run_id, hier, equipment)
    report = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "use_synthetic": settings.use_synthetic,
        "horizon": settings.horizon,
        "data_provenance": dict(base.PROVENANCE),
        "backtest_metrics": metrics,
        "n_procedure_forecast_rows": int(len(hier)),
        "n_equipment_forecast_rows": int(len(equipment)),
        "scenarios": list(scenario_fns),
    }
    (settings.artifacts_dir / f"report_{run_id}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    (settings.artifacts_dir / "latest_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    log.info("=== run %s complete ===", run_id)
    return report


def _persist(settings, run_id, hier, equipment) -> None:
    proc_out = hier.copy()
    proc_out["run_id"] = run_id
    proc_out["model"] = "lgbm"
    equip_out = equipment.copy()
    equip_out["run_id"] = run_id
    with warehouse.connect(settings) as con:
        con.register("proc_out", proc_out)
        con.execute(
            """INSERT INTO forecast_procedures
               (run_id, model, level, trust_code, region, procedure_code, date,
                yhat, yhat_lower, yhat_upper)
               SELECT run_id, model, level, trust_code, region, procedure_code, date,
                      yhat, yhat_lower, yhat_upper FROM proc_out""")
        con.register("equip_out", equip_out)
        con.execute(
            """INSERT INTO forecast_equipment
               (run_id, scenario, level, trust_code, region, equipment_type, date,
                demand, demand_lower, demand_upper)
               SELECT run_id, scenario, level, trust_code, region, equipment_type, date,
                      demand, demand_lower, demand_upper FROM equip_out""")
        con.unregister("proc_out")
        con.unregister("equip_out")
    # parquet artifacts for the dashboard / API
    hier.to_parquet(settings.artifacts_dir / f"procedure_forecast_{run_id}.parquet", index=False)
    equipment.to_parquet(settings.artifacts_dir / f"equipment_forecast_{run_id}.parquet", index=False)
    equipment.to_parquet(settings.artifacts_dir / "equipment_forecast_latest.parquet", index=False)
    hier.to_parquet(settings.artifacts_dir / "procedure_forecast_latest.parquet", index=False)
