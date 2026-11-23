"""Shared helpers: config loading, overrides, and data-version lookup."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of config with dotted-key overrides applied.

    Example: {"model.type": "random_forest"} sets config["model"]["type"].
    """
    new = copy.deepcopy(config)
    for dotted_key, value in (overrides or {}).items():
        keys = dotted_key.split(".")
        node = new
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return new


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_data_version(config: dict[str, Any]) -> str:
    """Return the DVC md5 hash of the dataset (from the .dvc pointer file).

    Falls back to hashing the file directly if the .dvc file is missing.
    """
    dvc_file = resolve_path(config["data"]["dvc_file"])
    if dvc_file.exists():
        with open(dvc_file) as f:
            meta = yaml.safe_load(f)
        return f"dvc:{meta['outs'][0]['md5']}"

    data_file = resolve_path(config["data"]["raw_path"])
    md5 = hashlib.md5(data_file.read_bytes()).hexdigest()
    return f"file-md5:{md5}"
