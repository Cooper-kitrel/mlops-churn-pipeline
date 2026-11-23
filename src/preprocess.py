"""Preprocessing: cleaning, target encoding, and the sklearn feature transformer.

Imputation and one-hot encoding live inside an sklearn ColumnTransformer so they
are fit on the training split only (no leakage) and are saved with the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise if df is not a DataFrame, is empty, or lacks required columns."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_total_charges(df: pd.DataFrame, column: str = "TotalCharges") -> pd.DataFrame:
    """Convert TotalCharges to numeric. Blank strings become NaN.

    Returns a new DataFrame; the input is not modified.
    """
    validate_columns(df, [column])
    out = df.copy()
    out[column] = pd.to_numeric(out[column].replace(r"^\s*$", np.nan, regex=True),
                                errors="coerce")
    return out


def encode_target(y: pd.Series, positive_label: str = "Yes") -> pd.Series:
    """Map a Yes/No target to 1/0. Raises on unexpected labels."""
    allowed = {positive_label, "No"}
    unexpected = set(y.dropna().unique()) - allowed
    if unexpected:
        raise ValueError(f"Unexpected target labels: {sorted(unexpected)}")
    if y.isna().any():
        raise ValueError("Target contains missing values")
    return (y == positive_label).astype(int)


def prepare_data(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Clean raw data and split into features X and encoded target y."""
    target = config["data"]["target"]
    numeric = config["features"]["numeric"]
    categorical = config["features"]["categorical"]
    validate_columns(df, numeric + categorical + [target])

    cleaned = clean_total_charges(df)
    X = cleaned[numeric + categorical].copy()
    # Float numerics so missing values are representable at inference time
    X[numeric] = X[numeric].astype(float)
    y = encode_target(cleaned[target], config["data"]["positive_label"])
    return X, y


def build_preprocessor(numeric: list[str], categorical: list[str],
                       numeric_strategy: str = "median",
                       categorical_strategy: str = "most_frequent",
                       scale_numeric: bool = True) -> ColumnTransformer:
    """Build a ColumnTransformer that imputes missing values and one-hot encodes."""
    if not numeric and not categorical:
        raise ValueError(
            "At least one numeric or categorical feature is required")

    num_steps = [("imputer", SimpleImputer(strategy=numeric_strategy))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    cat_steps = [
        ("imputer", SimpleImputer(strategy=categorical_strategy)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", Pipeline(cat_steps), categorical),
    ])


def build_preprocessor_from_config(config: dict) -> ColumnTransformer:
    p = config["preprocessing"]
    return build_preprocessor(
        config["features"]["numeric"],
        config["features"]["categorical"],
        numeric_strategy=p["numeric_impute_strategy"],
        categorical_strategy=p["categorical_impute_strategy"],
        scale_numeric=p["scale_numeric"],
    )
