"""Generate static visualisations from the latest pipeline run.

Reads historical facts from the DuckDB warehouse and forecast artifacts
(parquet + report json), then writes PNGs to data/artifacts/figures/.

Run:  python make_figures.py
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from nhs_forecast.config import get_settings
from nhs_forecast.storage import warehouse

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

settings = get_settings()
FIG_DIR = settings.artifacts_dir / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- load data ----------
proc_hist = warehouse.query(settings, "SELECT * FROM fact_procedures")
proc_hist["date"] = pd.to_datetime(proc_hist["date"])
rtt = warehouse.query(settings, "SELECT * FROM fact_rtt")
rtt["date"] = pd.to_datetime(rtt["date"])

proc_fc = pd.read_parquet(settings.artifacts_dir / "procedure_forecast_latest.parquet")
equip_fc = pd.read_parquet(settings.artifacts_dir / "equipment_forecast_latest.parquet")
report = json.loads((settings.artifacts_dir / "latest_report.json").read_text(encoding="utf-8"))


def save(fig, name):
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------- 1. national procedure history (shows trend, seasonality, COVID) ----------
nat_hist = proc_hist.groupby("date")["n_procedures"].sum()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(nat_hist.index, nat_hist.values, color="#1f77b4")
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-03-01"),
           color="red", alpha=0.08, label="COVID-19 disruption")
ax.set(title="National monthly procedure volume (history)",
       xlabel="month", ylabel="procedures")
ax.legend()
save(fig, "01_national_procedure_history.png")


# ---------- 2. national procedure forecast with intervals (top 4 codes) ----------
nat_fc = proc_fc[proc_fc["level"] == "national"].copy()
top_codes = (proc_hist.groupby("procedure_code")["n_procedures"].sum()
                       .sort_values(ascending=False).head(4).index.tolist())
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for ax, code in zip(axes.ravel(), top_codes):
    h = proc_hist[proc_hist["procedure_code"] == code].groupby("date")["n_procedures"].sum()
    f = nat_fc[nat_fc["procedure_code"] == code].sort_values("date")
    ax.plot(h.index, h.values, color="#333", label="history")
    ax.plot(f["date"], f["yhat"], color="#d62728", label="forecast")
    ax.fill_between(f["date"], f["yhat_lower"], f["yhat_upper"],
                    color="#d62728", alpha=0.2, label="80% PI")
    ax.set_title(f"OPCS {code}")
axes[0, 0].legend(fontsize=8)
fig.suptitle("National procedure forecast (12 months, LightGBM)")
save(fig, "02_procedure_forecast_topcodes.png")


# ---------- 3. equipment demand forecast by type (baseline, national) ----------
base = equip_fc[(equip_fc["scenario"] == "baseline") & (equip_fc["level"] == "national")]
piv = base.pivot_table(index="date", columns="equipment_type", values="demand", aggfunc="sum")
fig, ax = plt.subplots(figsize=(11, 5))
piv.plot(ax=ax, marker="o", ms=3)
ax.set(title="Forecast equipment demand by type (baseline, national)",
       xlabel="month", ylabel="expected units")
ax.legend(title="equipment", bbox_to_anchor=(1.02, 1), loc="upper left")
save(fig, "03_equipment_demand_by_type.png")


# ---------- 4. scenario comparison (total equipment demand) ----------
scen = (equip_fc[equip_fc["level"] == "national"]
        .groupby(["scenario", "date"])["demand"].sum().reset_index())
fig, ax = plt.subplots(figsize=(10, 5))
for name, g in scen.groupby("scenario"):
    ax.plot(g["date"], g["demand"], marker="o", ms=3, label=name)
ax.set(title="Total equipment demand by scenario (national)",
       xlabel="month", ylabel="expected units (all equipment)")
ax.legend(title="scenario")
save(fig, "04_scenario_comparison.png")


# ---------- 5. backtest model comparison ----------
metrics = report.get("backtest_metrics", {})
if metrics:
    mdf = pd.DataFrame(metrics).T  # rows=models, cols=mae/rmse/mape
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, title in zip(axes, ["mae", "rmse", "mape"],
                              ["MAE", "RMSE", "MAPE (%)"]):
        mdf[col].sort_values().plot.bar(ax=ax, color="#2ca02c")
        ax.set_title(title)
        ax.set_xlabel("")
    fig.suptitle("Backtest accuracy (lower is better)")
    save(fig, "05_model_backtest.png")


# ---------- 6. RTT waiting-list pressure (national) ----------
nat_wl = rtt.groupby("date")["waiting_list_size"].sum() / 1e6
nat_pct = rtt.groupby("date")["pct_within_18wk"].mean()
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(nat_wl.index, nat_wl.values, color="#9467bd", label="waiting list (m)")
ax1.set(xlabel="month", ylabel="waiting list size (millions)")
ax2 = ax1.twinx()
ax2.plot(nat_pct.index, nat_pct.values, color="#ff7f0e", label="% within 18 wks")
ax2.set_ylabel("% treated within 18 weeks")
ax2.grid(False)
ax1.set_title("RTT waiting-list pressure (national)")
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))
save(fig, "06_rtt_waiting_pressure.png")


# ---------- 7. regional equipment demand (baseline, all types) ----------
reg = equip_fc[(equip_fc["scenario"] == "baseline") & (equip_fc["level"] == "regional")]
reg_tot = reg.groupby(["region", "date"])["demand"].sum().reset_index()
fig, ax = plt.subplots(figsize=(11, 5))
for name, g in reg_tot.groupby("region"):
    ax.plot(g["date"], g["demand"], marker="o", ms=3, label=name)
ax.set(title="Forecast equipment demand by region (baseline, all equipment)",
       xlabel="month", ylabel="expected units")
ax.legend(title="region", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
save(fig, "07_regional_equipment_demand.png")


# ---------- 8. region x equipment-type heatmap (mean monthly demand) ----------
reg_type = (reg.groupby(["region", "equipment_type"])["demand"].mean()
               .unstack("equipment_type"))
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(reg_type.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(reg_type.columns)))
ax.set_xticklabels(reg_type.columns, rotation=30, ha="right")
ax.set_yticks(range(len(reg_type.index)))
ax.set_yticklabels(reg_type.index)
for i in range(reg_type.shape[0]):
    for j in range(reg_type.shape[1]):
        ax.text(j, i, f"{reg_type.values[i, j]:.0f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, label="mean monthly units")
ax.set_title("Mean monthly equipment demand: region x type (baseline)")
ax.grid(False)
save(fig, "08_region_equipment_heatmap.png")


# ---------- 9. trust-level ranking (total horizon demand, baseline) ----------
trust = equip_fc[(equip_fc["scenario"] == "baseline") & (equip_fc["level"] == "trust")]
trust_tot = (trust.groupby(["trust_code", "region"])["demand"].sum()
                  .reset_index().sort_values("demand", ascending=True))
fig, ax = plt.subplots(figsize=(10, 5))
colors = {r: c for r, c in zip(sorted(trust_tot["region"].unique()),
                               plt.cm.tab10.colors)}
ax.barh(trust_tot["trust_code"], trust_tot["demand"],
        color=[colors[r] for r in trust_tot["region"]])
ax.set(title="Total forecast equipment demand by trust (12-month horizon, baseline)",
       xlabel="total expected units", ylabel="trust")
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[r]) for r in colors]
ax.legend(handles, list(colors), title="region", fontsize=8,
          bbox_to_anchor=(1.02, 1), loc="upper left")
save(fig, "09_trust_demand_ranking.png")


# ---------- 10. trust procedure forecast small-multiples (top 6 trusts) ----------
proc_trust = proc_fc[proc_fc["level"] == "trust"].copy()
top_trusts = (proc_hist.groupby("trust_code")["n_procedures"].sum()
                       .sort_values(ascending=False).head(6).index.tolist())
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
for ax, tc in zip(axes.ravel(), top_trusts):
    h = proc_hist[proc_hist["trust_code"] == tc].groupby("date")["n_procedures"].sum()
    f = (proc_trust[proc_trust["trust_code"] == tc]
         .groupby("date")[["yhat", "yhat_lower", "yhat_upper"]].sum().reset_index())
    ax.plot(h.index, h.values, color="#333", lw=0.9, label="history")
    ax.plot(f["date"], f["yhat"], color="#1f77b4", label="forecast")
    ax.fill_between(f["date"], f["yhat_lower"], f["yhat_upper"],
                    color="#1f77b4", alpha=0.2)
    ax.set_title(tc, fontsize=9)
axes[0, 0].legend(fontsize=8)
fig.suptitle("Trust-level total procedure forecast (top 6 trusts by volume)")
save(fig, "10_trust_procedure_forecast.png")


# ---------- 11. regional procedure history (stacked area) ----------
reg_hist = (proc_hist.groupby(["date", "region"])["n_procedures"].sum()
                     .unstack("region").fillna(0))
fig, ax = plt.subplots(figsize=(11, 5))
ax.stackplot(reg_hist.index, [reg_hist[c].values for c in reg_hist.columns],
             labels=list(reg_hist.columns), alpha=0.85)
ax.set(title="Regional contribution to national procedure volume (history)",
       xlabel="month", ylabel="procedures")
ax.legend(title="region", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
save(fig, "11_regional_procedure_history.png")


# ---------- combined overview ----------
print("\nFigures written to:", FIG_DIR)
