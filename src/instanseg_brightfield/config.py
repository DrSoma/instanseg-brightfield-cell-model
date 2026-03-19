"""Configuration loading with environment variable expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

_ENV_PATTERN = re.compile(r"\$\{(\w+):=([^}]*)\}")

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


def _expand_env_vars(value: str) -> str:
    """Expand ${VAR:=default} patterns using environment variables."""
    def _replace(match: re.Match) -> str:
        var_name, default = match.group(1), match.group(2)
        return os.environ.get(var_name, default)
    return _ENV_PATTERN.sub(_replace, value)


def _walk_and_expand(obj):
    """Recursively expand environment variables in all string values."""
    if isinstance(obj, str):
        return _expand_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _walk_and_expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_expand(item) for item in obj]
    return obj


def load_config(config_path: Path | str | None = None) -> dict:
    """Load and validate the pipeline configuration.

    Args:
        config_path: Path to YAML config. Defaults to config/default.yaml.

    Returns:
        Configuration dictionary with environment variables expanded.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _walk_and_expand(raw)


def get_config_hash(cfg: dict) -> str:
    """Compute a short hash of the config for artifact tagging."""
    import hashlib
    import json
    content = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:12]
