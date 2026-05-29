"""Procedure -> equipment mapping.

Two complementary mapping styles, both stored in one extensible table
(`mapping_table.csv`) and distinguished by the ``map_type`` column:

* **rule**  — deterministic 1:1 / 1:N relationships where every procedure of a
  code consumes a fixed expected quantity of an equipment type
  (e.g. every colonoscopy ``H22`` uses one endoscope).

* **probabilistic** — only a *fraction* of procedures consume the equipment, so
  ``weight`` is the expected units per procedure (e.g. ``K49`` PCI uses a stent
  ~35% of the time -> weight 0.35). This naturally handles "multiple equipment
  types per procedure" by emitting several rows whose weights need not sum to 1.

The table is extensible: add rows, add ``equipment_type``s, or override weights
from calibration against NHS Supply Chain consumption without code changes.

Equipment demand for a procedure forecast is the matrix product:

    demand[equipment, t] = sum_over_codes( forecast[code, t] * weight[code, equipment] )
"""
from __future__ import annotations

import pandas as pd

from nhs_forecast.config import Settings
from nhs_forecast.logging_setup import get_logger

log = get_logger("mapping")


def load_mapping(settings: Settings) -> pd.DataFrame:
    df = pd.read_csv(settings.mapping_path)
    expected = {"procedure_code", "equipment_type", "weight", "map_type", "source"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"mapping table missing columns: {missing}")
    df["weight"] = df["weight"].astype(float)
    # drop explicit not-applicable zero-weight rows (kept in CSV for documentation)
    return df[df["weight"] > 0].reset_index(drop=True)


def load_equipment_dim(settings: Settings) -> pd.DataFrame:
    path = settings.mapping_path.with_name("equipment_dim.csv")
    return pd.read_csv(path)


def mapping_matrix(settings: Settings) -> pd.DataFrame:
    """Return a dense procedure_code x equipment_type weight matrix."""
    m = load_mapping(settings)
    return m.pivot_table(
        index="procedure_code", columns="equipment_type", values="weight",
        aggfunc="sum", fill_value=0.0,
    )


def apply_mapping(forecast: pd.DataFrame, settings: Settings, value_col: str = "yhat") -> pd.DataFrame:
    """Convert a long procedure forecast into a long equipment-demand frame.

    ``forecast`` must contain ``procedure_code``, ``date``, ``value_col`` and any
    grouping keys (e.g. ``trust_code``/``region``/``level``) which are preserved.
    """
    mapping = load_mapping(settings)
    group_keys = [c for c in ("run_id", "model", "level", "trust_code", "region", "date")
                  if c in forecast.columns]
    merged = forecast.merge(mapping, on="procedure_code", how="inner")
    merged["demand"] = merged[value_col] * merged["weight"]
    out = (merged.groupby(group_keys + ["equipment_type"], dropna=False)["demand"]
                 .sum().reset_index())
    log.info("mapped %d procedure rows -> %d equipment-demand rows",
             len(forecast), len(out))
    return out
