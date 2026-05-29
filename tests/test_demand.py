import numpy as np
import pandas as pd

from nhs_forecast.demand.derive import derive_equipment_demand


def test_uncertainty_propagation_independent_sum(settings):
    # two procedures both mapping to endoscope (weight 1.0): H22 and G45.
    proc = pd.DataFrame({
        "level": ["national", "national"],
        "trust_code": [pd.NA, pd.NA],
        "region": [pd.NA, pd.NA],
        "procedure_code": ["H22", "G45"],
        "date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        "yhat": [100.0, 100.0],
        "yhat_lower": [80.0, 80.0],
        "yhat_upper": [120.0, 120.0],
    })
    out = derive_equipment_demand(proc, settings)
    endo = out[out["equipment_type"] == "endoscope"].iloc[0]
    assert endo["demand"] == 200.0
    # independent variances add: half-width = sqrt(2) * single half-width (~20)
    single_half = (120 - 80) / 2
    expected_half = np.sqrt(2) * single_half
    got_half = (endo["demand_upper"] - endo["demand_lower"]) / 2
    assert abs(got_half - expected_half) < 1e-6


def test_demand_non_negative(settings):
    proc = pd.DataFrame({
        "level": ["national"],
        "procedure_code": ["H22"],
        "date": pd.to_datetime(["2025-01-01"]),
        "yhat": [10.0], "yhat_lower": [0.0], "yhat_upper": [50.0],
    })
    out = derive_equipment_demand(proc, settings)
    assert (out["demand_lower"] >= 0).all()
