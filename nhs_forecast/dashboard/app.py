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


@st.cache_data
def _load_underwriting():
    uw = settings.artifacts_dir / "underwriting_latest.parquet"
    rep = settings.artifacts_dir / "telemetry_report_latest.json"
    if not uw.exists() or not rep.exists():
        return None, None
    return pd.read_parquet(uw), json.loads(rep.read_text(encoding="utf-8"))


def _run_underwriting() -> None:
    from nhs_forecast.telemetry import pipeline as tpipe

    with st.spinner("Generating telemetry, fitting NegBin, simulating portfolio…"):
        tpipe.run(settings)
    _load_underwriting.clear()


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

    st.divider()
    st.header("Underwriting")
    st.caption("Pay-per-use device-session telemetry → portfolio risk.")
    if st.button("Run / refresh underwriting"):
        _run_underwriting()
        st.success("Underwriting run complete.")
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


st.title("NHS MedTech Forecasting & Utilisation Underwriting")
tab_demand, tab_uw = st.tabs(["Equipment demand", "Utilisation underwriting"])

# ===================== TAB 1: aggregate equipment demand ==================
with tab_demand:
    equipment, procedures = _load()
    if equipment is None:
        st.warning("No forecast artifacts found. Use **Run / refresh forecast** in "
                   "the sidebar to generate forecasts.")
    else:
        col1, col2, col3 = st.columns(3)
        scenario = col1.selectbox("Scenario", sorted(equipment["scenario"].unique()))
        level = col2.selectbox("Level", ["national", "regional", "trust"])
        eq_type = col3.selectbox("Equipment type",
                                 ["(all)"] + sorted(equipment["equipment_type"].unique()))

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
        totals = (df.groupby("equipment_type", as_index=False)["demand"].sum()
                    .sort_values("demand"))
        st.plotly_chart(px.bar(totals, x="demand", y="equipment_type", orientation="h"),
                        use_container_width=True)

        with st.expander("Underlying procedure forecast"):
            pf = procedures[procedures["level"] == level]
            st.dataframe(pf.head(500), use_container_width=True)

# ===================== TAB 2: utilisation underwriting ====================
with tab_uw:
    book, trep = _load_underwriting()
    if book is None:
        st.info("No underwriting artifacts yet. Use **Run / refresh underwriting** "
                "in the sidebar to simulate the pay-per-use device fleet.")
    else:
        p = trep.get("portfolio_risk", {})
        b = trep.get("backtest", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected margin (horizon)", f"£{p.get('expected_margin_gbp', 0):,.0f}")
        m2.metric(f"CVaR loss ({int(p.get('alpha', 0.05) * 100)}%)",
                  f"£{p.get('cvar_loss_gbp', 0):,.0f}")
        m3.metric("Effective independent devices",
                  f"{p.get('effective_n_independent', 0):.1f} / {p.get('n_devices', 0)}")
        m4.metric("P(book loss)", f"{p.get('prob_book_loss', 0) * 100:.0f}%")
        st.caption(
            f"Model: {b.get('model_kind', 'n/a')} (α={b.get('alpha', 'n/a')}) · "
            f"pinball P50 {b.get('pinball_p50', 'n/a')} vs naive "
            f"{b.get('pinball_p50_naive', 'n/a')} · 90% coverage "
            f"{b.get('coverage_90', 'n/a')}")

        st.subheader("Price adequacy: current vs risk-based suggested")
        sc = px.scatter(
            book, x="current_price_gbp", y="suggested_price_gbp",
            size="expected_revenue_gbp", color="specialty",
            hover_data=["device_id", "cv", "beta_book", "op_herfindahl"],
            title="Above the diagonal ⇒ under-priced for its risk")
        lim = float(max(book["current_price_gbp"].max(),
                        book["suggested_price_gbp"].max()))
        sc.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                     line=dict(dash="dash", color="grey"))
        st.plotly_chart(sc, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Key-person risk")
            st.plotly_chart(
                px.bar(book.sort_values("op_herfindahl", ascending=False).head(15),
                       x="op_herfindahl", y="device_id", color="specialty",
                       orientation="h", title="Operator concentration (Herfindahl)"),
                use_container_width=True)
        with c2:
            st.subheader("Utilisation-at-Risk (P5 sessions)")
            st.plotly_chart(
                px.bar(book.sort_values("uar_p5_sessions").head(15),
                       x="uar_p5_sessions", y="device_id", color="specialty",
                       orientation="h", title="Lowest downside utilisation"),
                use_container_width=True)

        st.subheader("Per-device underwriting book")
        st.dataframe(book.sort_values("expected_margin_gbp"), use_container_width=True)
