"""Typer CLI: ``nhsfc <command>``.

Commands:
    run         run the full end-to-end pipeline
    ingest      ingest + validate + land only (no modelling)
    backtest    build features and print model comparison
    report      print the latest run report
"""
from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from nhs_forecast.config import get_settings
from nhs_forecast.features.build_features import build_feature_table
from nhs_forecast.pipeline import steps

app = typer.Typer(add_completion=False, help="NHS equipment demand forecasting pipeline")
console = Console()


@app.command()
def run(synthetic: bool = True, scenarios: bool = True):
    """Run the full pipeline (ingest -> forecast -> equipment demand)."""
    settings = get_settings()
    settings.use_synthetic = synthetic
    report = steps.run(settings, run_scenarios=scenarios)
    _print_report(report)


@app.command()
def ingest(synthetic: bool = True):
    """Ingest, validate and land all sources to the lake + warehouse."""
    settings = get_settings()
    settings.use_synthetic = synthetic
    frames = steps.ingest(settings)
    steps.validate(frames)
    steps.land(settings, frames)
    steps.load_warehouse(settings, frames)
    for name, df in frames.items():
        console.print(f"[green]{name}[/]: {len(df):,} rows")


@app.command()
def backtest(synthetic: bool = True):
    """Build features and compare models on a temporal holdout."""
    settings = get_settings()
    settings.use_synthetic = synthetic
    frames = steps.ingest(settings)
    steps.validate(frames)
    steps.load_warehouse(settings, frames)
    feats = build_feature_table(settings)
    metrics = steps.backtest(feats, settings)
    _print_metrics(metrics)


@app.command()
def report():
    """Print the latest pipeline run report."""
    settings = get_settings()
    path = settings.artifacts_dir / "latest_report.json"
    if not path.exists():
        console.print("[red]no run report found; run `nhsfc run` first[/]")
        raise typer.Exit(1)
    _print_report(json.loads(path.read_text(encoding="utf-8")))


def _print_metrics(metrics: dict):
    table = Table(title="Backtest model comparison")
    table.add_column("model")
    for col in ("MAE", "RMSE", "MAPE %"):
        table.add_column(col, justify="right")
    for model, m in sorted(metrics.items(), key=lambda kv: kv[1]["rmse"]):
        table.add_row(model, f"{m['mae']:.1f}", f"{m['rmse']:.1f}", f"{m['mape']:.1f}")
    console.print(table)


def _print_report(report: dict):
    console.print(f"[bold]run_id[/]: {report['run_id']}")
    console.print(f"horizon: {report['horizon']} months | synthetic: {report['use_synthetic']}")
    console.print(f"procedure forecast rows: {report['n_procedure_forecast_rows']:,}")
    console.print(f"equipment forecast rows: {report['n_equipment_forecast_rows']:,}")
    _print_metrics(report.get("backtest_metrics", {}))


if __name__ == "__main__":
    app()
