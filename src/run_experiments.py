"""Run every experiment defined in configs/experiments.yaml.

Usage:
    python -m src.run_experiments
"""
from __future__ import annotations

import argparse

from src.train import run_training
from src.utils import apply_overrides, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--experiments", default="configs/experiments.yaml")
    args = parser.parse_args()

    base = load_config(args.config)
    experiments = load_config(args.experiments)["experiments"]

    for exp in experiments:
        print(f"\n=== Running experiment: {exp['name']} ===")
        config = apply_overrides(base, exp.get("overrides", {}))
        metrics, failures = run_training(config, run_name=exp["name"])
        status = "PASS" if not failures else "FAIL: " + "; ".join(failures)
        print(
            f"{exp['name']}: roc_auc={metrics['roc_auc']:.4f} f1={metrics['f1']:.4f} [{status}]")


if __name__ == "__main__":
    main()
