"""Streamlit dashboard (optional — requires the ``dashboard`` extra).

Run:  streamlit run nhs_forecast/dashboard/app.py

Visualises equipment demand forecasts by scenario / equipment type / level and
the procedure forecasts that drive them, reading the latest parquet artifacts.
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from nhs_forecast.config import Settings, get_settings

st.set_page_config(page_title="NHS Equipment Demand", layout="wide")
settings = get_settings()


@st.cache_data
def _load():
    eq = settings.artifacts_dir / "equipment_forecast_latest.parquet"
    pr = settings.artifacts_dir / "procedure_forecast_latest.parquet"
    if not eq.exists():
        return None, None
    return pd.read_parquet(eq), pd.read_parquet(pr)


def _latest_report() -> dict | None:
    path = settings.artifacts_dir / "latest_report.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _run_pipeline(use_live: bool) -> None:
    """Run the end-to-end pipeline from within the app (full-pipeline deploy)."""
    from nhs_forecast.pipeline import steps

    with st.spinner("Running pipeline — ingest, model, forecast… this can take a minute"):
        steps.run(Settings(use_synthetic=not use_live))
    _load.clear()


# --- sidebar controls -----------------------------------------------------
with st.sidebar:
    st.header("Pipeline")
    use_live = st.checkbox(
        "Use live NHS RTT data",
        value=False,
        help="When ticked, fetches the real NHS England national RTT waiting-list "
             "series and allocates it top-down to trusts. Other sources stay synthetic.",
    )
    if st.button("Run / refresh forecast", type="primary"):
        _run_pipeline(use_live)
        st.success("Pipeline complete.")
        st.rerun()

    report = _latest_report()
    if report:
        st.caption(f"Latest run: `{report.get('run_id', 'n/a')}`")
        prov = report.get("data_provenance", {})
        if prov:
            st.markdown("**Data provenance**")
            for src, detail in prov.items():
                icon = "🟢" if str(detail).startswith("live") else "⚪"
                st.markdown(f"{icon} `{src}` — {detail}")


equipment, procedures = _load()
st.title("NHS Medical Equipment Demand Forecast")

if equipment is None:
    st.warning("No forecast artifacts found. Use **Run / refresh forecast** in the "
               "sidebar to generate forecasts.")
    st.stop()


col1, col2, col3 = st.columns(3)
scenario = col1.selectbox("Scenario", sorted(equipment["scenario"].unique()))
level = col2.selectbox("Level", ["national", "regional", "trust"])
eq_type = col3.selectbox("Equipment type", ["(all)"] + sorted(equipment["equipment_type"].unique()))

df = equipment[(equipment["scenario"] == scenario) & (equipment["level"] == level)]
if eq_type != "(all)":
    df = df[df["equipment_type"] == eq_type]

st.subheader("Forecast equipment demand")
agg = (df.groupby(["date", "equipment_type"], as_index=False)
         .agg(demand=("demand", "sum"),
              lower=("demand_lower", "sum"),
              upper=("demand_upper", "sum")))
fig = px.line(agg, x="date", y="demand", color="equipment_type",
              title=f"{scenario} — {level} equipment demand")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Demand by equipment type (horizon total)")
totals = df.groupby("equipment_type", as_index=False)["demand"].sum().sort_values("demand")
st.plotly_chart(px.bar(totals, x="demand", y="equipment_type", orientation="h"),
                use_container_width=True)

with st.expander("Underlying procedure forecast"):
    pf = procedures[procedures["level"] == level]
    st.dataframe(pf.head(500), use_container_width=True)
