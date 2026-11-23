"""Drift monitoring with Evidently.

Reference  = the training split (the data the model learned from).
Production = the held-out split, optionally with simulated post-deployment drift.

Usage:
    python -m src.monitor_drift                  # simulated drift (expect exit 1)
    python -m src.monitor_drift --no-simulate    # clean holdout (expect exit 0)
    python -m src.monitor_drift --threshold 0.5  # override drift-share threshold

Exits with code 1 if the share of drifted features exceeds the threshold.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.data import load_raw_data
from src.preprocess import prepare_data
from src.train import split_data
from src.utils import load_config, resolve_path

warnings.filterwarnings("ignore")


def build_reference_and_production(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reference = training features; production = held-out features (same split as train.py)."""
    X, y = prepare_data(load_raw_data(config), config)
    X_train, X_test, _, _ = split_data(X, y, config)
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True)


def simulate_drift(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Simulate realistic post-deployment changes. Returns a new DataFrame.

    - Price increase: MonthlyCharges up ~15% plus noise.
    - Acquisition campaign: 40% of customers are brand-new (tenure 0-6 months).
    - Promotion pushes customers onto month-to-month contracts.
    - Upstream pipeline bug: 5% of TotalCharges values arrive missing.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)

    out["MonthlyCharges"] = out["MonthlyCharges"] * 1.15 + rng.normal(0, 3, n)

    new_customers = rng.random(n) < 0.40
    out.loc[new_customers, "tenure"] = rng.integers(
        0, 7, new_customers.sum()).astype(float)
    out["TotalCharges"] = out["tenure"] * out["MonthlyCharges"]

    to_m2m = (out["Contract"] != "Month-to-month") & (rng.random(n) < 0.5)
    out.loc[to_m2m, "Contract"] = "Month-to-month"

    missing = rng.random(n) < 0.05
    out.loc[missing, "TotalCharges"] = np.nan
    return out


def column_drifted(method: str, value: float, threshold: float) -> bool:
    """Evidently returns p-values for statistical tests (drift if p < threshold)
    and distances for large samples (drift if distance >= threshold)."""
    if "p_value" in method:
        return value < threshold
    return value >= threshold


def summarize(snapshot) -> tuple[pd.DataFrame, float, int]:
    """Extract per-column drift results and the overall drift share."""
    rows, share, count = [], None, None
    for m in snapshot.dict()["metrics"]:
        cfg = m["config"]
        if cfg["type"].endswith("DriftedColumnsCount"):
            share, count = m["value"]["share"], int(m["value"]["count"])
        elif cfg["type"].endswith("ValueDrift"):
            rows.append({
                "feature": cfg["column"],
                "method": cfg["method"],
                "score": m["value"],
                "threshold": cfg["threshold"],
                "drifted": column_drifted(cfg["method"], m["value"], cfg["threshold"]),
            })
    return pd.DataFrame(rows), share, count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Evidently drift monitoring")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Max allowed drift share (overrides config)")
    parser.add_argument("--no-simulate", action="store_true",
                        help="Use the clean holdout as production data")
    args = parser.parse_args()

    config = load_config(args.config)
    dcfg = config["drift"]
    threshold = args.threshold if args.threshold is not None else dcfg["drift_share_threshold"]
    simulate = dcfg["simulate_drift"] and not args.no_simulate

    reference, production = build_reference_and_production(config)
    if simulate:
        production = simulate_drift(
            production, seed=config["project"]["random_state"])

    print(f"Reference rows: {len(reference)} | Production rows: {len(production)} | "
          f"Simulated drift: {simulate}")

    report = Report([DataDriftPreset(drift_share=threshold)])
    snapshot = report.run(current_data=production, reference_data=reference)

    report_dir = resolve_path(dcfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    tag = "simulated" if simulate else "clean"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = report_dir / f"drift_report_{tag}_{stamp}.html"
    snapshot.save_html(str(html_path))

    results, share, count = summarize(snapshot)
    results = results.sort_values("drifted", ascending=False)

    print("\nPer-feature drift:")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    drifted = results.loc[results["drifted"], "feature"].tolist()
    print(f"\nDrifted features ({count}/{len(results)}): {drifted or 'none'}")
    print(f"Overall drift share: {share:.2%} (threshold {threshold:.0%})")
    print(f"HTML report saved to: {html_path}")

    if share > threshold:
        print("\nALERT: drift share exceeds threshold -> exit code 1")
        return 1
    print("\nOK: drift within threshold -> exit code 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
