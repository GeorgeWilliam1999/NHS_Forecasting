from nhs_forecast.models import hierarchical
from nhs_forecast.pipeline import steps
from nhs_forecast.storage import warehouse


def test_end_to_end_pipeline_runs(settings):
    report = steps.run(settings, run_scenarios=True)
    assert report["n_procedure_forecast_rows"] > 0
    assert report["n_equipment_forecast_rows"] > 0
    assert "lgbm" in report["backtest_metrics"]

    # forecasts persisted to the warehouse
    eq = warehouse.query(settings, "SELECT COUNT(*) AS n FROM forecast_equipment")
    assert eq.iloc[0]["n"] > 0


def test_hierarchy_is_coherent(settings):
    steps.ingest(settings)
    frames = steps.ingest(settings)
    steps.validate(frames)
    steps.load_warehouse(settings, frames)
    from nhs_forecast.features.build_features import build_feature_table

    feats = build_feature_table(settings)
    proc_fc = steps.forecast(feats, settings)
    hier = hierarchical.bottom_up(proc_fc)

    # national must equal the sum of trust-level for each procedure_code/date
    trust = (hier[hier["level"] == "trust"]
             .groupby(["procedure_code", "date"])["yhat"].sum().round(3))
    national = (hier[hier["level"] == "national"]
                .set_index(["procedure_code", "date"])["yhat"].round(3))
    aligned = national.reindex(trust.index)
    assert (abs(aligned - trust) < 1e-3).all()
