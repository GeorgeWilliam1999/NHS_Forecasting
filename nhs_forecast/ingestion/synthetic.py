"""Deterministic synthetic data generator.

Produces internally-consistent monthly NHS-shaped data so the full pipeline runs
offline. The generator encodes realistic structure:

* trend growth in activity + demographic drift,
* strong 12-month seasonality (winter pressure, summer dip),
* a COVID-19 shock (sharp drop spring 2020, slow recovery),
* a post-COVID elective backlog that inflates RTT waiting lists,
* procedures -> imaging and procedures -> waiting lists are causally linked.

All randomness is seeded so outputs are reproducible across runs.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from nhs_forecast.config import Settings

# --- reference universe ---------------------------------------------------
TRUSTS = [
    ("RGT", "East of England"),
    ("R1H", "London"),
    ("RJZ", "London"),
    ("RM3", "North West"),
    ("RTH", "South East"),
    ("RWE", "Midlands"),
    ("RHQ", "North East and Yorkshire"),
    ("RA7", "South West"),
]

# OPCS-4 chapter -> representative 3-char codes used in the demo universe.
OPCS = {
    "W": ["W37", "W93", "W40"],   # bones / joints (ortho)
    "H": ["H22", "H25"],          # lower digestive (endoscopy)
    "G": ["G45", "G16"],          # upper digestive (endoscopy)
    "K": ["K49", "K63"],          # heart (cardiac cath / pacing)
    "M": ["M47", "M65"],          # urinary (cystoscopy)
    "C": ["C71", "C75"],          # eye (cataract)
}
MODALITIES = ["CT", "MRI", "Ultrasound", "Plain Radiography", "Fluoroscopy", "Nuclear Medicine"]
TREATMENT_FUNCTIONS = [
    "Trauma & Orthopaedics", "General Surgery", "Cardiology",
    "Ophthalmology", "Urology", "Gastroenterology",
]
AGE_BANDS = ["0-17", "18-39", "40-64", "65-79", "80+"]

# relative size weights per trust (some trusts are much larger)
TRUST_SIZE = {t[0]: s for t, s in zip(TRUSTS, [1.0, 1.4, 0.9, 1.2, 1.1, 1.3, 1.0, 0.7])}


def _months(settings: Settings) -> pd.DatetimeIndex:
    return pd.date_range(settings.synthetic_start, settings.synthetic_end, freq="MS")


def _covid_factor(dates: pd.DatetimeIndex) -> np.ndarray:
    """Multiplicative activity factor: ~1.0 normally, deep dip in 2020, slow recovery."""
    f = np.ones(len(dates))
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2020-03-01"):
            months_since = (d.year - 2020) * 12 + (d.month - 3)
            dip = 0.45 * np.exp(-months_since / 8.0)   # 55% drop, recovering over ~8mo
            f[i] = 1.0 - dip
    return f


def _seasonal(dates: pd.DatetimeIndex, amp: float, phase: float = 0.0) -> np.ndarray:
    m = dates.month.to_numpy()
    return 1.0 + amp * np.sin(2 * np.pi * (m / 12.0) + phase)


def generate(settings: Settings) -> dict[str, pd.DataFrame]:
    """Return a dict of canonical curated frames keyed by table name."""
    rng = np.random.default_rng(settings.synthetic_seed)
    dates = _months(settings)
    t = np.arange(len(dates))
    trend = 1.0 + 0.0025 * t            # ~3% annual growth
    covid = _covid_factor(dates)

    proc_rows, img_rows, rtt_rows, act_rows = [], [], [], []

    for trust_code, region in TRUSTS:
        size = TRUST_SIZE[trust_code]
        # ---- procedures by OPCS chapter / code ----
        trust_proc_total = np.zeros(len(dates))
        for chapter, codes in OPCS.items():
            seasonal = _seasonal(dates, amp=0.12, phase=rng.uniform(0, 1))
            base = 220 * size * rng.uniform(0.7, 1.3)
            for code in codes:
                code_base = base * rng.uniform(0.5, 1.0)
                mean = code_base * trend * seasonal * covid
                noise = rng.normal(1.0, 0.06, len(dates))
                vals = np.clip(mean * noise, 0, None).round().astype(int)
                trust_proc_total += vals
                for d, v in zip(dates, vals):
                    proc_rows.append((d, trust_code, region, chapter, code, int(v)))

        # ---- imaging: partly driven by procedure volume + own seasonality ----
        for modality in MODALITIES:
            seasonal = _seasonal(dates, amp=0.08, phase=rng.uniform(0, 1))
            base = 900 * size * rng.uniform(0.4, 1.4)
            mean = base * trend * seasonal * covid + 0.15 * trust_proc_total
            noise = rng.normal(1.0, 0.05, len(dates))
            vals = np.clip(mean * noise, 0, None).round().astype(int)
            for d, v in zip(dates, vals):
                img_rows.append((d, trust_code, region, modality, int(v)))

        # ---- RTT waiting list: backlog accumulates when capacity < demand ----
        for tf in TREATMENT_FUNCTIONS:
            demand = 800 * size * rng.uniform(0.6, 1.2) * trend
            capacity = demand * covid * 0.98          # capacity tracks demand but lags in covid
            backlog = np.maximum.accumulate(np.cumsum(demand - capacity)) * 0.5
            wl = (3000 * size + backlog).clip(min=0)
            wl = (wl * rng.normal(1.0, 0.03, len(dates))).round().astype(int)
            # % within 18 weeks falls as waiting list grows
            pct = (92 - (wl - wl.min()) / max(wl.max() - wl.min(), 1) * 45).clip(40, 95)
            for d, w, p in zip(dates, wl, pct):
                rtt_rows.append((d, trust_code, region, tf, int(w), round(float(p), 1)))

        # ---- monthly activity (electives / outpatients / A&E) ----
        for activity_type, mult, amp in [
            ("elective", 0.55, 0.10), ("outpatient", 3.2, 0.07), ("ae_attendance", 1.8, 0.15)
        ]:
            seasonal = _seasonal(dates, amp=amp, phase=0.4 if activity_type == "ae_attendance" else 0)
            base = 1500 * size * mult
            mean = base * trend * seasonal * (covid if activity_type != "ae_attendance" else 1.0)
            vals = np.clip(mean * rng.normal(1.0, 0.05, len(dates)), 0, None).round().astype(int)
            for d, v in zip(dates, vals):
                act_rows.append((d, trust_code, region, activity_type, int(v)))

    # ---- demographics by region/age band/year ----
    demo_rows = []
    regions = sorted({r for _, r in TRUSTS})
    years = range(pd.Timestamp(settings.synthetic_start).year,
                  pd.Timestamp(settings.synthetic_end).year + 1)
    base_pop = {ab: p for ab, p in zip(AGE_BANDS, [1.1e6, 1.8e6, 2.2e6, 1.0e6, 0.45e6])}
    for region in regions:
        rscale = rng.uniform(0.6, 1.6)
        for ab in AGE_BANDS:
            for yr in years:
                # population ages: 65+ grows faster
                growth = 1.0 + (0.012 if ab in ("65-79", "80+") else 0.004) * (yr - min(years))
                pop = base_pop[ab] * rscale * growth * rng.uniform(0.97, 1.03)
                demo_rows.append((yr, region, ab, int(pop)))

    return {
        "procedures": pd.DataFrame(proc_rows, columns=[
            "date", "trust_code", "region", "opcs_chapter", "procedure_code", "n_procedures"]),
        "imaging": pd.DataFrame(img_rows, columns=[
            "date", "trust_code", "region", "modality", "n_tests"]),
        "rtt": pd.DataFrame(rtt_rows, columns=[
            "date", "trust_code", "region", "treatment_function",
            "waiting_list_size", "pct_within_18wk"]),
        "activity": pd.DataFrame(act_rows, columns=[
            "date", "trust_code", "region", "activity_type", "n_activity"]),
        "demographics": pd.DataFrame(demo_rows, columns=[
            "year", "region", "age_band", "population"]),
    }


@lru_cache(maxsize=4)
def _cached(seed: int, start: str, end: str) -> dict[str, pd.DataFrame]:
    # build a throwaway Settings-like object via the real one but override knobs
    s = Settings(synthetic_seed=seed, synthetic_start=start, synthetic_end=end)
    return generate(s)


def tables(settings: Settings) -> dict[str, pd.DataFrame]:
    """Cached access so the five loaders share one generated universe per run."""
    return _cached(settings.synthetic_seed, settings.synthetic_start, settings.synthetic_end)
