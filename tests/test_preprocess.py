"""Unit tests for preprocessing functions (no real data needed)."""
import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    build_preprocessor,
    clean_total_charges,
    encode_target,
    prepare_data,
    validate_columns,
)

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["Contract", "PaymentMethod"]


# ---------- missing values ----------

def test_clean_total_charges_converts_blanks_to_nan(small_df):
    out = clean_total_charges(small_df)
    assert pd.api.types.is_float_dtype(out["TotalCharges"])
    # " " and "" are both blank -> NaN
    assert out["TotalCharges"].isna().sum() == 2
    assert out.loc[0, "TotalCharges"] == pytest.approx(29.85)


def test_preprocessor_imputes_all_missing_values(small_df):
    X = clean_total_charges(small_df)[NUMERIC + CATEGORICAL]
    assert X.isna().sum().sum() > 0  # sanity: input has gaps
    transformed = build_preprocessor(NUMERIC, CATEGORICAL).fit_transform(X)
    assert not np.isnan(transformed).any()


def test_numeric_imputation_uses_median():
    X = pd.DataFrame(
        {"tenure": [1.0, 3.0, np.nan, 100.0], "Contract": ["a", "b", "a", "b"]})
    pre = build_preprocessor(["tenure"], ["Contract"], scale_numeric=False)
    out = pre.fit_transform(X)
    assert out[2, 0] == pytest.approx(3.0)  # median of [1, 3, 100]


# ---------- categorical encoding ----------

def test_preprocessor_one_hot_encodes_categoricals(small_df):
    X = clean_total_charges(small_df)[NUMERIC + CATEGORICAL]
    pre = build_preprocessor(NUMERIC, CATEGORICAL).fit(X)
    out = pre.transform(X)
    n_categories = X["Contract"].nunique() + X["PaymentMethod"].nunique()
    assert out.shape == (len(X), len(NUMERIC) + n_categories)
    # each row has exactly one active category per categorical column
    cat_block = out[:, len(NUMERIC):]
    assert set(np.unique(cat_block)) <= {0.0, 1.0}
    assert (cat_block.sum(axis=1) == len(CATEGORICAL)).all()


def test_preprocessor_ignores_unseen_categories(small_df):
    X = clean_total_charges(small_df)[NUMERIC + CATEGORICAL]
    pre = build_preprocessor(NUMERIC, CATEGORICAL).fit(X)
    new = X.head(1).copy()
    new["Contract"] = "Ten year"  # never seen in training
    out = pre.transform(new)
    contract_cols = out[:, len(NUMERIC):len(NUMERIC) + X["Contract"].nunique()]
    assert contract_cols.sum() == 0


def test_encode_target_maps_yes_no_to_1_0():
    y = encode_target(pd.Series(["Yes", "No", "No", "Yes"]))
    assert y.tolist() == [1, 0, 0, 1]
    assert y.dtype.kind == "i"


# ---------- immutability ----------

def test_clean_total_charges_does_not_modify_input(small_df):
    original = small_df.copy(deep=True)
    clean_total_charges(small_df)
    pd.testing.assert_frame_equal(small_df, original)


def test_prepare_data_does_not_modify_input(small_df, config):
    cfg = {**config, "features": {"numeric": NUMERIC, "categorical": CATEGORICAL}}
    original = small_df.copy(deep=True)
    prepare_data(small_df, cfg)
    pd.testing.assert_frame_equal(small_df, original)


# ---------- invalid input ----------

def test_encode_target_rejects_unexpected_labels():
    with pytest.raises(ValueError, match="Unexpected target labels"):
        encode_target(pd.Series(["Yes", "No", "Maybe"]))


def test_validate_columns_rejects_missing_columns(small_df):
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_columns(small_df, ["tenure", "not_a_column"])


def test_validate_columns_rejects_non_dataframe():
    with pytest.raises(TypeError):
        validate_columns([1, 2, 3], ["tenure"])


def test_validate_columns_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty"):
        validate_columns(pd.DataFrame(columns=["tenure"]), ["tenure"])


def test_build_preprocessor_requires_features():
    with pytest.raises(ValueError):
        build_preprocessor([], [])
