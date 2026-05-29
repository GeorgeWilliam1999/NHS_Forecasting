"""FastAPI service exposing forecasts (optional — requires the ``api`` extra).

Run:  uvicorn nhs_forecast.api.main:app --reload --port 8000

Endpoints:
    GET /health
    GET /equipment-demand   ?scenario=&equipment_type=&level=&region=&trust_code=
    GET /procedure-forecast ?procedure_code=&level=&trust_code=
    GET /runs               latest run report
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from nhs_forecast.config import get_settings
from nhs_forecast.storage import warehouse

app = FastAPI(title="NHS Equipment Demand API", version="0.1.0")


def _latest_run_id() -> str:
    settings = get_settings()
    df = warehouse.query(
        settings, "SELECT run_id FROM forecast_equipment ORDER BY run_id DESC LIMIT 1")
    if df.empty:
        raise HTTPException(404, "no forecast runs found; run the pipeline first")
    return df.iloc[0]["run_id"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/runs")
def runs():
    settings = get_settings()
    path = settings.artifacts_dir / "latest_report.json"
    if not path.exists():
        raise HTTPException(404, "no run report")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/equipment-demand")
def equipment_demand(
    scenario: str = "baseline",
    level: str = "national",
    equipment_type: Optional[str] = None,
    region: Optional[str] = None,
    trust_code: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: int = Query(1000, le=20000),
):
    settings = get_settings()
    run_id = run_id or _latest_run_id()
    where = ["run_id = ?", "scenario = ?", "level = ?"]
    params: list = [run_id, scenario, level]
    for col, val in (("equipment_type", equipment_type), ("region", region),
                     ("trust_code", trust_code)):
        if val is not None:
            where.append(f"{col} = ?")
            params.append(val)
    sql = (f"SELECT * FROM forecast_equipment WHERE {' AND '.join(where)} "
           f"ORDER BY date LIMIT {limit}")
    with warehouse.connect(settings) as con:
        df = con.execute(sql, params).df()
    return {"run_id": run_id, "rows": len(df), "data": df.to_dict(orient="records")}


@app.get("/procedure-forecast")
def procedure_forecast(
    level: str = "national",
    procedure_code: Optional[str] = None,
    trust_code: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: int = Query(1000, le=20000),
):
    settings = get_settings()
    run_id = run_id or _latest_run_id()
    where = ["run_id = ?", "level = ?"]
    params: list = [run_id, level]
    for col, val in (("procedure_code", procedure_code), ("trust_code", trust_code)):
        if val is not None:
            where.append(f"{col} = ?")
            params.append(val)
    sql = (f"SELECT * FROM forecast_procedures WHERE {' AND '.join(where)} "
           f"ORDER BY date LIMIT {limit}")
    with warehouse.connect(settings) as con:
        df = con.execute(sql, params).df()
    return {"run_id": run_id, "rows": len(df), "data": df.to_dict(orient="records")}
