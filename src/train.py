"""Train a churn model, log everything to MLflow, and enforce metric thresholds.

Usage:
    python -m src.train --config configs/config.yaml
    python -m src.train --config configs/config.yaml --run-name my_run

Exits with code 1 if the model misses any threshold in config['thresholds'].
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

import joblib
import mlflow
import mlflow.sklearn
import skops.io as sio
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_raw_data
from src.evaluate import check_thresholds, compute_metrics
from src.preprocess import build_preprocessor_from_config, prepare_data
from src.utils import get_data_version, load_config, resolve_path

# The model outputs 0/1 integers; MLflow warns about integer schemas, which is
# irrelevant for a classifier's predicted labels.
warnings.filterwarnings(
    "ignore", message=".*Inferred schema contains integer.*")

MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def build_model(config: dict) -> Pipeline:
    """Build preprocessing + classifier pipeline from config."""
    model_type = config["model"]["type"]
    if model_type not in MODELS:
        raise ValueError(
            f"Unknown model type '{model_type}'. Choose from {list(MODELS)}")
    params = dict(config["model"].get("params") or {})
    params["random_state"] = config["project"]["random_state"]
    clf = MODELS[model_type](**params)
    return Pipeline([("preprocessor", build_preprocessor_from_config(config)),
                     ("classifier", clf)])


def split_data(X, y, config: dict):
    return train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["project"]["random_state"],
        stratify=y,
    )


def split_train_val(X_train, y_train, config: dict):
    """Split the training data again: part for fitting, part for validation.

    Model selection uses the validation set, so the test set stays untouched.
    """
    return train_test_split(
        X_train, y_train,
        test_size=config["data"]["val_size"],
        random_state=config["project"]["random_state"],
        stratify=y_train,
    )


def flatten_params(config: dict) -> dict:
    """Flatten the config sections we want logged as MLflow params."""
    params = {"model.type": config["model"]["type"]}
    for k, v in (config["model"].get("params") or {}).items():
        params[f"model.{k}"] = v
    for k, v in config["preprocessing"].items():
        params[f"preprocessing.{k}"] = v
    params["data.test_size"] = config["data"]["test_size"]
    params["data.val_size"] = config["data"]["val_size"]
    params["random_state"] = config["project"]["random_state"]
    params["n_numeric_features"] = len(config["features"]["numeric"])
    params["n_categorical_features"] = len(config["features"]["categorical"])
    return params


def run_training(config: dict, run_name: str | None = None) -> tuple[dict, list[str]]:
    """Train, evaluate, and log one run. Returns (metrics, threshold_failures)."""
    df = load_raw_data(config)
    X, y = prepare_data(df, config)
    X_train_full, _X_test, y_train_full, _y_test = split_data(
        X, y, config)  # test set held out
    X_train, X_val, y_train, y_val = split_train_val(
        X_train_full, y_train_full, config)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=run_name) as run:
        # 1. Hyperparameters + data version
        mlflow.log_params(flatten_params(config))
        mlflow.log_param("data_version", get_data_version(config))
        mlflow.log_param("n_train_rows", len(X_train))
        mlflow.set_tag("features.categorical", ",".join(
            config["features"]["categorical"]))

        # 2. Train
        model = build_model(config)
        model.fit(X_train, y_train)

        # 3. Evaluate
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_pred, y_proba)
        mlflow.log_metrics(
            {f"val_{name}": value for name, value in metrics.items()})

        failures = check_thresholds(metrics, config["thresholds"])
        mlflow.set_tag("thresholds_passed", str(not failures))

        # 4. Log the model as an MLflow artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train.head(5),
            # MLflow 3 serializes sklearn models with skops (safe loading).
            # We only trust types found in a model WE just trained.
            skops_trusted_types=sio.get_untrusted_types(data=sio.dumps(model)),
        )
        print(f"MLflow run id: {run.info.run_id}")

    # Save local copies for CI artifacts
    model_path = resolve_path(config["output"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path = resolve_path(config["output"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return metrics, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Train churn model")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    metrics, failures = run_training(config, run_name=args.run_name)

    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name:<10} {value:.4f}")

    if failures:
        print("\nFAILED performance thresholds:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll performance thresholds met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
