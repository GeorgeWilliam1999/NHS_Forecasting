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


@app.command()
def underwrite(devices: int = 0, days: int = 0):
    """Run the pay-per-use telemetry → utilisation underwriting pipeline.

    Generates device-session telemetry, fits the censored Negative-Binomial
    utilisation model, forecasts the horizon and produces per-device pricing +
    portfolio risk (VaR / CVaR / effective diversification).
    """
    from nhs_forecast.telemetry import pipeline as tpipe

    settings = get_settings()
    if devices:
        settings.telemetry_n_devices = devices
    if days:
        settings.telemetry_days = days
    report = tpipe.run(settings)
    _print_underwriting(report)


def _print_underwriting(report: dict) -> None:
    p = report.get("portfolio_risk", {})
    b = report.get("backtest", {})
    console.print(f"[bold]telemetry run[/]: {report['run_id']}")
    console.print(f"devices: {report['n_devices']} | sessions: {report['n_sessions']:,} | "
                  f"device-days: {report['n_device_days']:,}")
    console.print(f"model: [cyan]{b.get('model_kind', 'n/a')}[/] "
                  f"(alpha={b.get('alpha', 'n/a')}) | "
                  f"pinball P50={b.get('pinball_p50', 'n/a')} "
                  f"vs naive {b.get('pinball_p50_naive', 'n/a')} | "
                  f"90% coverage={b.get('coverage_90', 'n/a')}")
    table = Table(title="Portfolio risk")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k in ("expected_revenue_gbp", "fixed_cost_gbp", "expected_margin_gbp",
              "var_loss_gbp", "cvar_loss_gbp", "prob_book_loss",
              "effective_n_independent", "diversification_ratio",
              "n_devices_negative_margin"):
        if k in p:
            v = p[k]
            table.add_row(k, f"{v:,.2f}" if isinstance(v, (int, float)) else str(v))
    console.print(table)


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
