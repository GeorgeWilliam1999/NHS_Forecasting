"""Deterministic device-session telemetry generator.

Produces a raw, append-only telemetry **event log** for a fleet of pay-per-use
medical devices, plus the device / operator / contract dimensions. The event log
is the single source of truth; sessions and device-day aggregates are
reconstructed downstream (see :mod:`sessionise`).

The generative model is deliberately structured so the downstream risk engine
has something real to find:

* a **hierarchical** latent intensity  site → specialty → device,
* **common factors** (regional + national) that induce *correlated* utilisation
  across devices — the thing that breaks naive diversification,
* a per-device **health** random walk driving downtime (censoring),
* **operator concentration** (key-person risk) via a Dirichlet allocation,
* staggered **install dates** so some devices are sparse / cold-start,
* contract **floors and caps** that warp billed vs latent utilisation.

Everything is seeded, so two runs with the same settings are identical.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nhs_forecast.config import Settings
from nhs_forecast.logging_setup import get_logger

log = get_logger("telemetry.synthetic")

# --- reference universe ---------------------------------------------------
SITES = [
    ("STH01", "London", "Neurosurgery"),
    ("STH02", "London", "Cardiology"),
    ("STH03", "North West", "Orthopaedics"),
    ("STH04", "Midlands", "General Surgery"),
    ("STH05", "South East", "Ophthalmology"),
    ("STH06", "South West", "Gastroenterology"),
]
DEVICE_TYPES = {
    "Neurosurgery": ("surgical_microscope", ["W37", "A57"]),
    "Cardiology": ("cardiac_robotic_platform", ["K49", "K63"]),
    "Orthopaedics": ("ortho_imaging_arm", ["W37", "W93"]),
    "General Surgery": ("laparoscopic_stack", ["H22", "T43"]),
    "Ophthalmology": ("ophthalmic_microscope", ["C71", "C75"]),
    "Gastroenterology": ("endoscopy_stack", ["H22", "H25"]),
}
# typical billable price and platform-carried monthly fixed cost by type
TYPE_ECONOMICS = {
    "surgical_microscope": (1400.0, 26000.0),
    "cardiac_robotic_platform": (2600.0, 52000.0),
    "ortho_imaging_arm": (900.0, 18000.0),
    "laparoscopic_stack": (650.0, 12000.0),
    "ophthalmic_microscope": (700.0, 14000.0),
    "endoscopy_stack": (520.0, 9000.0),
}


def _hash(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


@dataclass
class _DeviceSpec:
    device_id: str
    device_type: str
    site_id: str
    region: str
    specialty: str
    install_offset_days: int
    base_daily: float          # baseline expected sessions / working day
    price: float
    fixed_cost: float
    floor_gbp: float
    cap_sessions: int
    region_beta: float         # loading on the regional common factor
    national_beta: float       # loading on the national common factor
    operators: list[str]
    op_weights: np.ndarray


def _build_universe(settings: Settings, rng: np.random.Generator):
    devices: list[_DeviceSpec] = []
    operators: list[tuple] = []
    n = settings.telemetry_n_devices

    # operators per site (one dominant => key-person concentration)
    site_ops: dict[str, list[str]] = {}
    for site_id, region, specialty in SITES:
        k = int(rng.integers(2, 5))
        ops = [_hash("op", site_id, i) for i in range(k)]
        site_ops[site_id] = ops
        for oh in ops:
            role = "consultant" if oh == ops[0] else rng.choice(["consultant", "registrar"])
            operators.append((oh, site_id, specialty, role))

    for i in range(n):
        site_id, region, specialty = SITES[i % len(SITES)]
        dtype, _codes = DEVICE_TYPES[specialty]
        price, fixed = TYPE_ECONOMICS[dtype]
        # stagger installs: ~a third of the fleet starts late (cold-start regime)
        install_offset = int(rng.choice([0, 0, 0, 90, 210, 330], p=[.4, .2, .1, .12, .1, .08]))
        base_daily = float(np.clip(rng.normal(4.5, 1.6), 1.0, 9.0))
        ops = site_ops[site_id]
        # Dirichlet with small concentration => one surgeon dominates the device
        op_weights = rng.dirichlet(np.full(len(ops), 0.4))
        devices.append(_DeviceSpec(
            device_id=_hash("dev", site_id, i),
            device_type=dtype,
            site_id=site_id,
            region=region,
            specialty=specialty,
            install_offset_days=install_offset,
            base_daily=base_daily,
            price=price * float(rng.uniform(0.9, 1.1)),
            fixed_cost=fixed * float(rng.uniform(0.9, 1.1)),
            floor_gbp=fixed * 0.35,          # floor covers ~35% of carry
            cap_sessions=int(rng.integers(9, 14)),
            region_beta=float(np.clip(rng.normal(0.6, 0.25), 0.0, 1.2)),
            national_beta=float(np.clip(rng.normal(0.4, 0.2), 0.0, 1.0)),
            operators=ops,
            op_weights=op_weights,
        ))
    return devices, operators


def _common_factors(settings: Settings, dates: pd.DatetimeIndex, rng: np.random.Generator):
    """AR(1) regional + national factors (log-multiplicative shocks)."""
    regions = sorted({r for _, r, _ in SITES})
    T = len(dates)
    national = np.zeros(T)
    for t in range(1, T):
        national[t] = 0.95 * national[t - 1] + rng.normal(0, 0.05)
    region_f = {}
    for r in regions:
        f = np.zeros(T)
        for t in range(1, T):
            f[t] = 0.9 * f[t - 1] + rng.normal(0, 0.07)
        region_f[r] = f
    return national, region_f


def generate(settings: Settings) -> dict[str, pd.DataFrame]:
    """Return ``{devices, operators, events}`` frames (canonical schema)."""
    rng = np.random.default_rng(settings.telemetry_seed)
    dates = pd.date_range(end=pd.Timestamp("2025-03-31"),
                          periods=settings.telemetry_days, freq="D")
    devices, operators = _build_universe(settings, rng)
    national, region_f = _common_factors(settings, dates, rng)
    t_idx = np.arange(len(dates))
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()
    # smooth annual seasonality (winter emergency uplift, summer elective dip)
    seasonal = 1.0 + 0.10 * np.sin(2 * np.pi * (doy / 365.25) + 1.3)
    trend = 1.0 + 0.0004 * t_idx

    ev: list[tuple] = []
    seq = {d.device_id: 0 for d in devices}

    for dev in devices:
        health = 1.0
        rbeta_f = region_f[dev.region]
        for ti, day in enumerate(dates):
            if ti < dev.install_offset_days:
                continue
            # health random walk -> downtime hazard
            health = float(np.clip(health + rng.normal(-0.002, 0.03), 0.3, 1.0))
            is_weekend = dow[ti] >= 5
            # multi-day downtime when health is poor or random maintenance
            down = (rng.random() < (0.02 + 0.06 * (1 - health)))
            if down:
                continue  # no telemetry at all (censored) — informative absence
            exposure_hours = 4.0 if is_weekend else 10.0
            # latent intensity (log-linear, hierarchical + common factors)
            log_lam = (
                np.log(dev.base_daily)
                + np.log(seasonal[ti]) + np.log(trend[ti])
                + (-0.9 if is_weekend else 0.0)
                + dev.region_beta * rbeta_f[ti]
                + dev.national_beta * national[ti]
                + 0.4 * (health - 1.0)
            )
            lam = float(np.exp(log_lam))
            n_sessions = int(rng.poisson(lam))
            n_sessions = min(n_sessions, dev.cap_sessions)  # physical/contract cap

            # power-on / power-off frame the available window
            open_ts = day + pd.Timedelta(hours=8)
            close_ts = day + pd.Timedelta(hours=8 + exposure_hours)
            seq[dev.device_id] += 1
            ev.append((_hash("e", dev.device_id, ti, "on"), dev.device_id, dev.site_id,
                       None, "power_on", open_ts, seq[dev.device_id], 0.0, 0, None, "ok"))

            # sessions through the day
            slot = exposure_hours * 3600 / max(n_sessions + 1, 1)
            for s in range(n_sessions):
                oh = rng.choice(dev.operators, p=dev.op_weights)
                code = rng.choice(DEVICE_TYPES[dev.specialty][1])
                # active duration: lognormal by type, degraded by poor health
                dur = float(np.clip(rng.lognormal(mean=7.0, sigma=0.5) / max(health, 0.5),
                                    60, 4 * 3600))
                n_err = int(rng.poisson(0.15 + 0.6 * (1 - health)))
                start = open_ts + pd.Timedelta(seconds=slot * (s + 0.3))
                end = start + pd.Timedelta(seconds=dur)
                quality = "ok" if rng.random() > 0.03 else "suspect"
                seq[dev.device_id] += 1
                ev.append((_hash("e", dev.device_id, ti, f"as{s}"), dev.device_id,
                           dev.site_id, oh, "active_start", start, seq[dev.device_id],
                           0.0, 0, code, quality))
                seq[dev.device_id] += 1
                ev.append((_hash("e", dev.device_id, ti, f"ae{s}"), dev.device_id,
                           dev.site_id, oh, "active_end", end, seq[dev.device_id],
                           dur, n_err, code, quality))

            seq[dev.device_id] += 1
            ev.append((_hash("e", dev.device_id, ti, "off"), dev.device_id, dev.site_id,
                       None, "power_off", close_ts, seq[dev.device_id], 0.0, 0, None, "ok"))

    events = pd.DataFrame(ev, columns=[
        "event_id", "device_id", "site_id", "operator_hash", "event_type",
        "ts_device", "seq_no", "active_seconds", "n_errors", "procedure_code",
        "ingest_quality"])
    events["ts_device"] = pd.to_datetime(events["ts_device"])

    dev_df = pd.DataFrame([{
        "device_id": d.device_id, "device_type": d.device_type, "site_id": d.site_id,
        "region": d.region, "specialty": d.specialty,
        "install_date": (dates[d.install_offset_days] if d.install_offset_days < len(dates)
                         else dates[0]).date(),
        "contract_id": _hash("ctr", d.device_id),
        "price_per_session_gbp": round(d.price, 2),
        "monthly_fixed_cost_gbp": round(d.fixed_cost, 2),
        "min_monthly_floor_gbp": round(d.floor_gbp, 2),
        "cap_sessions_day": d.cap_sessions,
    } for d in devices])
    op_df = pd.DataFrame(operators, columns=["operator_hash", "site_id", "specialty", "role"])

    log.info("telemetry generated: %d devices, %d operators, %d events",
             len(dev_df), len(op_df), len(events))
    return {"devices": dev_df, "operators": op_df, "events": events}
