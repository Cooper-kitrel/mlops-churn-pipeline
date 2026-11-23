"""Data validation tests that run against the real dataset."""
import pandas as pd

from src.preprocess import clean_total_charges


def test_expected_columns_present(raw_df, config):
    expected = (
        config["features"]["numeric"]
        + config["features"]["categorical"]
        + [config["data"]["target"], config["data"]["id_column"]]
    )
    missing = set(expected) - set(raw_df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_dataset_meets_size_requirements(raw_df):
    assert len(raw_df) >= 1000
    assert raw_df.shape[1] - 2 >= 8  # minus id + target


def test_target_has_only_expected_values(raw_df, config):
    values = set(raw_df[config["data"]["target"]].unique())
    assert values == {"Yes", "No"}


def test_target_is_not_degenerate(raw_df, config):
    churn_rate = (raw_df[config["data"]["target"]] == "Yes").mean()
    assert 0.05 < churn_rate < 0.95


def test_numeric_features_within_expected_ranges(raw_df):
    df = clean_total_charges(raw_df)
    assert df["tenure"].between(0, 100).all()          # months
    assert df["MonthlyCharges"].between(0, 500).all()  # dollars
    assert df["TotalCharges"].dropna().between(0, 20000).all()
    assert set(df["SeniorCitizen"].unique()) <= {0, 1}


def test_total_charges_missing_rate_is_small(raw_df):
    df = clean_total_charges(raw_df)
    assert df["TotalCharges"].isna().mean() < 0.01


def test_categorical_values_are_known(raw_df):
    assert set(raw_df["Contract"].unique()) == {
        "Month-to-month", "One year", "Two year"}
    assert set(raw_df["InternetService"].unique()) <= {
        "DSL", "Fiber optic", "No"}


def test_customer_ids_are_unique(raw_df, config):
    assert raw_df[config["data"]["id_column"]].is_unique
    assert isinstance(raw_df, pd.DataFrame)
