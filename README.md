# NHS Medical Equipment Demand Forecasting

A production-grade pipeline that ingests UK NHS healthcare activity data, forecasts
**procedure volumes**, and converts those forecasts into **medical equipment demand**
with uncertainty and scenario analysis.

The system runs end-to-end **offline** out of the box using a deterministic synthetic
data generator that mimics real NHS structure (trends, seasonality, the COVID shock and
the elective backlog). Every ingestion module also contains the real production access
code path; set `NHSFC_USE_SYNTHETIC=false` to switch to live fetches.

---

## Quick start

```powershell
# from this folder
python -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -e ".[dev]"            # core + tests; add ".[all]" for torch/api/dashboard

# run the whole pipeline (ingest -> forecast -> equipment demand)
python -m nhs_forecast.pipeline.cli run --synthetic

# inspect results
python -m nhs_forecast.pipeline.cli report
pytest -q
```

Serve / visualise (needs the optional extras):

```powershell
pip install -e ".[api,dashboard]"
uvicorn nhs_forecast.api.main:app --port 8000
streamlit run nhs_forecast/dashboard/app.py
```

Containerised:

```powershell
docker compose up --build              # api on :8000, dashboard on :8501
docker compose run pipeline            # one-shot batch forecast
```

---

## Architecture

```
sources ─▶ ingestion ─▶ validation ─▶ Parquet data lake ─▶ DuckDB warehouse
                                                                  │
                                              feature engineering ▼
                                            ┌──────────────────────────────┐
                                            │ SARIMA · LightGBM · LSTM/TFT  │  backtest (MAE/RMSE/MAPE)
                                            └──────────────────────────────┘
                                                                  │ trust-level forecast
                                            bottom-up reconciliation ▼ (trust→region→national)
                                            procedure→equipment mapping (rule + probabilistic)
                                                                  │ + uncertainty propagation
                                              scenario analysis ▼ (baseline / backlog-clear / capacity-cap)
                                            DuckDB + Parquet ─▶ FastAPI + Streamlit
```

### Folder structure

```
nhs_forecast/
  config.py            # pydantic-settings; all paths & knobs (env NHSFC_*)
  ingestion/           # one loader per source + synthetic generator
  storage/             # Parquet lake (snapshot versioning) + DuckDB warehouse + schema.sql
  mapping/             # procedure→equipment mapping table + applier
  features/            # leakage-safe feature engineering
  models/              # sarima · lgbm (global, recursive) · lstm · hierarchical · evaluate
  demand/              # equipment derivation, uncertainty, scenarios
  validation/          # declarative data checks + monitoring
  pipeline/            # stages, Typer CLI, Prefect flow
  api/                 # FastAPI
  dashboard/           # Streamlit
config/sources.yaml    # data source registry (access method, cadence, caveats)
tests/                 # unit + end-to-end pipeline tests
```

---

## Part 1 — Data sources

| Source | Access | Cadence | Relevance |
|--------|--------|---------|-----------|
| Hospital Episode Statistics (HES) | bulk CSV (DARS for record-level) | monthly | OPCS-4 procedure volumes (core target) |
| Diagnostic Imaging Dataset (DID) | bulk CSV/XLS | monthly | imaging scanner demand by modality |
| Monthly Activity Statistics | bulk XLSX (drifting headers) | monthly | electives / outpatients / A&E context |
| RTT Waiting Times | bulk CSV | monthly | demand-pressure driver (backlog) |
| NHS Supply Chain / procurement | scrape (no open API) | quarterly | optional consumption ground-truth |
| ONS population estimates | Beta API + bulk CSV | annual | demographic demand driver (ageing) |

Each loader returns a **canonical curated schema**, handles schema drift via alias maps,
lands the raw download with a content hash for lineage, and falls back to synthetic data
if a live fetch fails. See `config/sources.yaml` and `nhs_forecast/ingestion/`.

## Part 3 — Procedure → equipment mapping

`nhs_forecast/mapping/mapping_table.csv` holds an extensible table:

| procedure_code | equipment_type | weight | map_type |
|----------------|----------------|--------|----------|
| W37 | ortho_implant | 1.0 | rule |
| W93 | ortho_implant | 0.8 | probabilistic |
| H22 | endoscope | 1.0 | rule |
| K49 | cardiac_device | 0.35 | probabilistic |

`weight` is the **expected units of equipment per procedure**, so probabilistic
1-to-many mappings emit several rows. Equipment demand is the matrix product
`demand[e,t] = Σ_c forecast[c,t]·weight[c,e]`.

## Part 5 — Forecasting models

* **SARIMA** baseline (per national×code series).
* **LightGBM** global quantile model with recursive multi-step forecasting (the workhorse, trust×code).
* **LSTM** (optional, torch) standing in for a Temporal Fusion Transformer.
* **Hierarchical** bottom-up reconciliation: trust → regional → national.

Compared on a temporal holdout with **MAE / RMSE / MAPE** (`nhsfc backtest`).

## Part 6 — Equipment demand & uncertainty

Procedure prediction intervals are propagated to equipment demand assuming
independent procedure errors (`var[e] = Σ_c weight²·var_c`), which is tighter and
more honest than summing interval bounds. Scenarios (`backlog_clear`,
`capacity_cap`, `demand_shock`) transform the procedure forecast before derivation.

---

## Scaling to real NHS data

* **Access**: apply for a DARS agreement for record-level HES; until then the
  openly-published aggregates wired here are sufficient for trust-level forecasting.
* **Storage**: the Parquet snapshot convention (`snapshot_date=…`) upgrades directly to
  S3 + Athena or GCS + BigQuery; swap `storage/warehouse.py` for a BigQuery client.
* **Orchestration**: `pipeline/flows.py` is a ready Prefect flow; schedule monthly on the
  5th (after NHS releases). Airflow equivalent: one task per pipeline stage.
* **Models**: replace the LSTM stub with `pytorch-forecasting`'s TFT for covariate-rich,
  quantile, multi-horizon forecasts; calibrate mapping weights against NHS Supply Chain
  consumption.
* **Governance**: all data here is aggregate/OGL; record-level data must stay inside an
  approved Secure Data Environment.
