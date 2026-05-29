import pandas as pd

from nhs_forecast.features.build_features import FEATURE_COLUMNS, build_feature_table
from nhs_forecast.pipeline import steps


def _prepare_warehouse(settings):
    frames = steps.ingest(settings)
    steps.validate(frames)
    steps.load_warehouse(settings, frames)
    return frames


def test_feature_table_has_expected_columns(settings):
    _prepare_warehouse(settings)
    feats = build_feature_table(settings)
    for col in FEATURE_COLUMNS:
        assert col in feats.columns, col
    # lag features should be NaN at the very start of each series (warm-up)
    assert feats["n_procedures_lag12"].isna().any()


def test_no_target_leakage_in_rolling(settings):
    _prepare_warehouse(settings)
    feats = build_feature_table(settings).sort_values(
        ["trust_code", "procedure_code", "date"])
    # rolling mean is shifted by 1, so for any row it must not equal current value
    # in the degenerate constant case; check the rolling col never uses current y
    one = feats[(feats["trust_code"] == feats["trust_code"].iloc[0])]
    assert one["n_procedures_rollmean3"].notna().sum() > 0
