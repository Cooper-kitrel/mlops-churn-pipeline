"""Data loading (and a download fallback for environments without the DVC remote)."""
from __future__ import annotations

import argparse
import urllib.request

import pandas as pd

from src.utils import load_config, resolve_path


def load_raw_data(config: dict) -> pd.DataFrame:
    """Load the raw CSV defined in config['data']['raw_path']."""
    path = resolve_path(config["data"]["raw_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `dvc pull` "
            "(or `python -m src.data --download` if you don't have the DVC remote)."
        )
    return pd.read_csv(path)


def download_data(config: dict) -> None:
    """Download the dataset from its public source URL.

    Used by CI, because GitHub runners can't reach a *local* DVC remote.
    """
    path = resolve_path(config["data"]["raw_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(config["data"]["source_url"], path)
    print(f"Downloaded dataset to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--download", action="store_true",
                        help="Download raw data")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.download:
        download_data(cfg)
    df = load_raw_data(cfg)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
