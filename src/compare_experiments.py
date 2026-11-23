"""Pick the best run by VALIDATION score, then evaluate it once on the TEST set.

Usage:
    python -m src.compare_experiments
"""
from __future__ import annotations

import argparse

import mlflow
import mlflow.sklearn
import pandas as pd

from src.data import load_raw_data
from src.evaluate import compute_metrics
from src.preprocess import prepare_data
from src.train import split_data
from src.utils import load_config


def get_runs(config: dict) -> pd.DataFrame:
    """All runs that have a validation score, best first."""
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    primary = config["mlflow"]["primary_metric"]
    return mlflow.search_runs(
        experiment_names=[config["mlflow"]["experiment_name"]],
        filter_string=f"attributes.status = 'FINISHED' and metrics.val_{primary} > 0",
        order_by=[f"metrics.val_{primary} DESC"],
    )


def evaluate_on_test(config: dict, run_id: str) -> dict[str, float]:
    """Score the selected run's model on the held-out test set (used only here)."""
    X, y = prepare_data(load_raw_data(config), config)
    _X_train, X_test, _y_train, y_test = split_data(X, y, config)
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    metrics = compute_metrics(y_test, model.predict(
        X_test), model.predict_proba(X_test)[:, 1])

    # Record the final test metrics on the selected run in MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(
            {f"test_{name}": value for name, value in metrics.items()})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    primary = config["mlflow"]["primary_metric"]

    runs = get_runs(config)
    if runs.empty:
        print("No runs found. Run `python -m src.run_experiments` first.")
        return

    # 1. Compare the runs on the VALIDATION set
    cols = {
        "tags.mlflow.runName": "run_name",
        "params.model.type": "model",
        "metrics.val_roc_auc": "val_roc_auc",
        "metrics.val_f1": "val_f1",
        "metrics.val_precision": "val_precision",
        "metrics.val_recall": "val_recall",
        "metrics.val_accuracy": "val_accuracy",
        "params.data_version": "data_version",
    }
    table = runs[[c for c in cols if c in runs.columns]].rename(columns=cols)
    pd.set_option("display.width", 200)
    print(f"Found {len(table)} runs (sorted by validation {primary}):\n")
    print(table.round(4).to_string(index=False))

    best = runs.iloc[0]
    print(f"\nBest run by validation {primary}:")
    print(f"  run_id   : {best['run_id']}")
    print(f"  run_name : {best.get('tags.mlflow.runName')}")
    print(f"  model    : {best.get('params.model.type')}")
    print(f"  val_{primary:<5}: {best[f'metrics.val_{primary}']:.4f}")
    print(f"  model URI: runs:/{best['run_id']}/model")

    # 2. Evaluate ONLY the selected model on the untouched TEST set
    test_metrics = evaluate_on_test(config, best["run_id"])
    print("\nFinal performance of the selected model on the TEST set:")
    for name, value in test_metrics.items():
        print(f"  {name:<10} {value:.4f}")


if __name__ == "__main__":
    main()
