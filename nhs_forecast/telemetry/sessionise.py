"""Reconstruct sessions and device-day aggregates from the raw event log.

``sessionise`` pairs ``active_start`` / ``active_end`` events per device (in
sequence order) into billable sessions. ``device_day`` rolls sessions and
power events up to the modelling grain, computing the **exposure** offset
(powered-on hours) that separates "device idle" from "device down".

These functions are pure (event log in, frames out) so they are trivially
testable and re-runnable from the immutable log.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nhs_forecast.config import Settings
from nhs_forecast.logging_setup import get_logger

log = get_logger("telemetry.sessionise")


def sessionise(events: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Pair active_start/active_end into sessions; apply the billing rule."""
    ev = events.sort_values(["device_id", "seq_no"]).copy()
    active = ev[ev["event_type"].isin(["active_start", "active_end"])]
    starts = active[active["event_type"] == "active_start"].reset_index(drop=True)
    ends = active[active["event_type"] == "active_end"].reset_index(drop=True)

    # In this generator every start has its matching end next in sequence; align
    # defensively by device + order so a dropped packet can't silently mis-pair.
    rows = []
    for dev, s_grp in starts.groupby("device_id", sort=False):
        e_grp = ends[ends["device_id"] == dev].reset_index(drop=True)
        s_grp = s_grp.reset_index(drop=True)
        m = min(len(s_grp), len(e_grp))
        for i in range(m):
            s, e = s_grp.iloc[i], e_grp.iloc[i]
            dur = float(e["active_seconds"]) if e["active_seconds"] > 0 else \
                (e["ts_device"] - s["ts_device"]).total_seconds()
            billable = (dur >= settings.min_billable_seconds) and (e["ingest_quality"] == "ok")
            rows.append((
                e["event_id"], dev, s["site_id"], s["operator_hash"],
                s["procedure_code"], s["ts_device"], e["ts_device"], dur,
                int(e["n_errors"]), bool(billable)))

    sess = pd.DataFrame(rows, columns=[
        "session_id", "device_id", "site_id", "operator_hash", "procedure_code",
        "t_start", "t_end", "active_seconds", "n_errors", "billable"])
    # billed amount attached from the device price downstream (needs dim_device)
    log.info("sessionised %d sessions from %d events", len(sess), len(events))
    return sess


def attach_billing(sessions: pd.DataFrame, devices: pd.DataFrame) -> pd.DataFrame:
    price = devices.set_index("device_id")["price_per_session_gbp"]
    s = sessions.copy()
    s["billed_amount_gbp"] = np.where(
        s["billable"], s["device_id"].map(price).astype(float), 0.0)
    return s


def device_day(events: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """Roll up to one row per device per *available* day (exposure > 0)."""
    pe = events[events["event_type"].isin(["power_on", "power_off"])].copy()
    pe["date"] = pe["ts_device"].dt.normalize()
    # exposure hours = power_off - power_on per device-day
    piv = (pe.pivot_table(index=["device_id", "date"], columns="event_type",
                          values="ts_device", aggfunc="first"))
    piv = piv.dropna(subset=["power_on", "power_off"])
    exposure = ((piv["power_off"] - piv["power_on"]).dt.total_seconds() / 3600.0)
    exposure = exposure.rename("exposure_hours").reset_index()

    s = sessions.copy()
    s["date"] = s["t_start"].dt.normalize()
    agg = (s.groupby(["device_id", "date"])
            .agg(n_sessions=("session_id", "size"),
                 n_billable=("billable", "sum"),
                 active_seconds=("active_seconds", "sum"),
                 n_errors=("n_errors", "sum"),
                 n_operators=("operator_hash", "nunique"),
                 billed_gbp=("billed_amount_gbp", "sum"))
            .reset_index())

    dd = exposure.merge(agg, on=["device_id", "date"], how="left")
    fill = {"n_sessions": 0, "n_billable": 0, "active_seconds": 0.0,
            "n_errors": 0, "n_operators": 0, "billed_gbp": 0.0}
    dd = dd.fillna(fill)
    for c in ("n_sessions", "n_billable", "n_errors", "n_operators"):
        dd[c] = dd[c].astype(int)
    dd = dd.sort_values(["device_id", "date"]).reset_index(drop=True)
    log.info("device-day grain: %d rows across %d devices",
             len(dd), dd["device_id"].nunique())
    return dd
