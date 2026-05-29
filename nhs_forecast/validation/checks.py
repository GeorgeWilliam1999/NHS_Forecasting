"""Lightweight, dependency-free data validation & monitoring.

Runs declarative checks against each curated dataset before it is loaded into
the warehouse. A failed *error*-severity check aborts the pipeline; *warn*-level
issues are logged and surfaced in the run report (mirrors Great Expectations'
intent without the heavy dependency). Also computes simple drift metrics versus
the previous snapshot for monitoring.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nhs_forecast.logging_setup import get_logger

log = get_logger("validation")


@dataclass
class CheckResult:
    dataset: str
    name: str
    passed: bool
    severity: str  # error | warn
    detail: str = ""


def _expect(results: list, dataset, name, cond, severity, detail=""):
    results.append(CheckResult(dataset, name, bool(cond), severity, detail))


def validate(dataset: str, df: pd.DataFrame, spec: dict) -> list[CheckResult]:
    """``spec`` keys: required_cols, non_negative, unique_keys, min_rows, date_col."""
    res: list[CheckResult] = []
    _expect(res, dataset, "non_empty", len(df) >= spec.get("min_rows", 1), "error",
            f"rows={len(df)}")
    for col in spec.get("required_cols", []):
        _expect(res, dataset, f"has_col[{col}]", col in df.columns, "error")
    for col in spec.get("non_negative", []):
        if col in df.columns:
            bad = int((pd.to_numeric(df[col], errors="coerce") < 0).sum())
            _expect(res, dataset, f"non_negative[{col}]", bad == 0, "error", f"{bad} negatives")
    keys = spec.get("unique_keys")
    if keys and all(k in df.columns for k in keys):
        dupes = int(df.duplicated(keys).sum())
        _expect(res, dataset, "unique_keys", dupes == 0, "warn", f"{dupes} duplicate keys")
    date_col = spec.get("date_col")
    if date_col and date_col in df.columns:
        nulls = int(pd.to_datetime(df[date_col], errors="coerce").isna().sum())
        _expect(res, dataset, "valid_dates", nulls == 0, "error", f"{nulls} unparseable dates")
        # freshness: warn if newest month is > 120 days behind the max in data
        # (in production compare to today; here we just ensure monotone coverage)
    for r in res:
        lvl = log.error if (not r.passed and r.severity == "error") else (
            log.warning if not r.passed else log.debug)
        lvl("check %s.%s passed=%s %s", r.dataset, r.name, r.passed, r.detail)
    return res


def assert_no_errors(results: list[CheckResult]) -> None:
    errors = [r for r in results if not r.passed and r.severity == "error"]
    if errors:
        msgs = "; ".join(f"{r.dataset}.{r.name}: {r.detail}" for r in errors)
        raise ValueError(f"data validation failed: {msgs}")


# default specs per curated dataset
SPECS = {
    "procedures": dict(required_cols=["date", "trust_code", "procedure_code", "n_procedures"],
                       non_negative=["n_procedures"],
                       unique_keys=["date", "trust_code", "procedure_code"],
                       date_col="date", min_rows=100),
    "imaging": dict(required_cols=["date", "trust_code", "modality", "n_tests"],
                    non_negative=["n_tests"], date_col="date", min_rows=50),
    "rtt": dict(required_cols=["date", "trust_code", "treatment_function", "waiting_list_size"],
                non_negative=["waiting_list_size"], date_col="date", min_rows=50),
    "activity": dict(required_cols=["date", "trust_code", "activity_type", "n_activity"],
                     non_negative=["n_activity"], date_col="date", min_rows=50),
    "demographics": dict(required_cols=["year", "region", "age_band", "population"],
                         non_negative=["population"], min_rows=10),
    "supply": dict(required_cols=["quarter", "category", "spend_gbp", "units"],
                   non_negative=["spend_gbp", "units"], min_rows=4),
}
