import pandas as pd

from nhs_forecast.mapping.procedure_equipment import (
    apply_mapping, load_mapping, mapping_matrix,
)


def test_mapping_loads_and_filters_zero_weights(settings):
    m = load_mapping(settings)
    assert {"procedure_code", "equipment_type", "weight"} <= set(m.columns)
    assert (m["weight"] > 0).all()  # zero-weight not-applicable rows dropped


def test_mapping_matrix_shape(settings):
    mat = mapping_matrix(settings)
    assert "endoscope" in mat.columns
    assert "W37" in mat.index


def test_apply_mapping_weighted_sum(settings):
    fc = pd.DataFrame({
        "level": ["national", "national"],
        "date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        "procedure_code": ["H22", "G45"],
        "yhat": [100.0, 50.0],
    })
    out = apply_mapping(fc, settings)
    endo = out[out["equipment_type"] == "endoscope"]["demand"].sum()
    # H22 endoscope weight 1.0 (100) + G45 endoscope 1.0 (50) = 150
    assert endo == 150.0
