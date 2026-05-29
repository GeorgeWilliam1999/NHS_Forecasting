"""Prefect orchestration flow (optional — requires the ``orchestration`` extra).

Wraps the pipeline stages as Prefect tasks so the same logic runs locally and on
a Prefect deployment with retries, logging and a monthly schedule. If Prefect is
not installed this module is simply not imported by the pipeline.

Deploy with a monthly cron (NHS data refreshes monthly):

    prefect deployment build nhs_forecast/pipeline/flows.py:monthly_flow \
        -n nhs-monthly --cron "0 6 5 * *"   # 06:00 on the 5th each month
"""
from __future__ import annotations

from prefect import flow, get_run_logger, task

from nhs_forecast.config import get_settings
from nhs_forecast.pipeline import steps


@task(retries=3, retry_delay_seconds=60)
def _ingest(settings):
    return steps.ingest(settings)


@task
def _validate(frames):
    return steps.validate(frames)


@task
def _land_and_warehouse(settings, frames):
    steps.land(settings, frames)
    steps.load_warehouse(settings, frames)


@task
def _model_and_persist(settings):
    from nhs_forecast.features.build_features import build_feature_table

    feats = build_feature_table(settings)
    steps.backtest(feats, settings)
    # reuse the high-level run for the modelling + persistence half
    return steps.run(settings)


@flow(name="nhs-equipment-monthly")
def monthly_flow(use_synthetic: bool = True):
    logger = get_run_logger()
    settings = get_settings()
    settings.use_synthetic = use_synthetic
    frames = _ingest(settings)
    _validate(frames)
    _land_and_warehouse(settings, frames)
    report = _model_and_persist(settings)
    logger.info("pipeline complete: run_id=%s", report["run_id"])
    return report


if __name__ == "__main__":
    monthly_flow()
