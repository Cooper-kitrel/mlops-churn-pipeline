"""Query MLflow with mlflow.search_runs() and identify the best run.

Usage:
    python -m src.compare_experiments
"""
from __future__ import annotations

import argparse

import mlflow
import pandas as pd

from src.utils import load_config


def get_runs(config: dict) -> pd.DataFrame:
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    primary = config["mlflow"]["primary_metric"]
    return mlflow.search_runs(
        experiment_names=[config["mlflow"]["experiment_name"]],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{primary} DESC"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    primary = config["mlflow"]["primary_metric"]

    runs = get_runs(config)
    if runs.empty:
        print("No finished runs found. Run `python -m src.run_experiments` first.")
        return

    cols = {
        "tags.mlflow.runName": "run_name",
        "params.model.type": "model",
        "metrics.roc_auc": "roc_auc",
        "metrics.f1": "f1",
        "metrics.precision": "precision",
        "metrics.recall": "recall",
        "metrics.accuracy": "accuracy",
        "params.data_version": "data_version",
    }
    table = runs[[c for c in cols if c in runs.columns]].rename(columns=cols)
    pd.set_option("display.width", 200)
    print(f"Found {len(table)} runs (sorted by {primary}):\n")
    print(table.round(4).to_string(index=False))

    best = runs.iloc[0]
    print(f"\nBest run by {primary}:")
    print(f"  run_id   : {best['run_id']}")
    print(f"  run_name : {best.get('tags.mlflow.runName')}")
    print(f"  model    : {best.get('params.model.type')}")
    print(f"  {primary:<9}: {best[f'metrics.{primary}']:.4f}")
    print(f"  model URI: runs:/{best['run_id']}/model")


if __name__ == "__main__":
    main()
