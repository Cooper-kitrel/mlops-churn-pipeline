"""Shared pytest fixtures."""
import numpy as np
import pandas as pd
import pytest

from src.data import download_data, load_raw_data
from src.utils import load_config, resolve_path


@pytest.fixture(scope="session")
def config():
    return load_config("configs/config.yaml")


@pytest.fixture(scope="session")
def raw_df(config):
    """The real dataset. Pulled by DVC locally; downloaded as a fallback (e.g. in CI)."""
    if not resolve_path(config["data"]["raw_path"]).exists():
        try:
            download_data(config)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"Dataset unavailable: run `dvc pull` ({exc})")
    return load_raw_data(config)


@pytest.fixture
def small_df():
    """A tiny hand-built frame with known missing values and categories."""
    return pd.DataFrame({
        "tenure": [1.0, 24.0, np.nan, 60.0, 12.0],
        "MonthlyCharges": [29.85, 56.95, 53.85, np.nan, 70.70],
        "TotalCharges": ["29.85", "1889.5", " ", "6000", ""],
        "Contract": ["Month-to-month", "One year", np.nan, "Two year", "Month-to-month"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Mailed check",
                          "Bank transfer (automatic)", np.nan],
        "Churn": ["No", "No", "Yes", "No", "Yes"],
    })
