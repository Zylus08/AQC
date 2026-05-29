"""
aqc/utils/config_loader.py
==========================
YAML configuration loader with defaults and validation.

Author: AQC Team
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration (merged with user config)
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "backtest": {
        "symbols": ["AAPL"],
        "start_date": None,
        "end_date": None,
        "initial_capital": 100_000.0,
    },
    "data": {
        "source": "csv",
        "data_dir": "data/raw",
        "cache_dir": "data/cache",
        "use_cache": False,
        "datetime_format": None,
        "timezone": None,
        "fill_method": "ffill",
    },
    "strategy": {
        "name": "sma_crossover",
        "params": {
            "fast_period": 20,
            "slow_period": 50,
            "rsi_period": 14,
            "oversold": 30,
            "overbought": 70,
            "allow_short": False,
            "short_period": 9,
            "medium_period": 21,
            "long_period": 50,
        },
    },
    "broker": {
        "commission_model": "percentage",
        "commission_rate": 0.001,
        "commission_flat_fee": 5.0,
        "slippage_model": "fixed_bps",
        "slippage_bps": 5,
        "exchange": "SIMULATED",
    },
    "portfolio": {
        "default_quantity": 100.0,
        "currency": "USD",
    },
    "risk": {
        "max_position_size": float("inf"),
        "max_position_pct_equity": 0.20,
        "max_gross_exposure_pct": 1.0,
        "max_daily_loss_pct": 0.05,
        "max_open_positions": 10,
    },
    "logging": {
        "level": "INFO",
        "log_dir": "logs",
        "log_to_file": True,
        "log_filename": "backtest.log",
        "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    },
    "output": {
        "reports_dir": "reports",
        "save_report": True,
        "export_equity_curve": True,
        "export_trade_log": True,
    },
}


def load_config(path: Optional[str] = None) -> dict[str, Any]:
    """Load and validate the AQC configuration.

    Merges the user-supplied YAML with :data:`_DEFAULTS`.  Missing keys
    fall back to defaults without raising errors, making partial
    configuration files valid.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.  If ``None`` or the file does
        not exist, only defaults are used.

    Returns
    -------
    dict
        Fully merged configuration dictionary.

    Examples
    --------
    >>> config = load_config("configs/config.yaml")
    >>> config["backtest"]["initial_capital"]
    100000.0
    """
    config = _deep_copy_defaults()

    if path is not None:
        yaml_path = Path(path)
        if yaml_path.exists():
            with yaml_path.open("r", encoding="utf-8") as fh:
                user_config = yaml.safe_load(fh) or {}
            _deep_merge(config, user_config)
            logger.info("Configuration loaded from %s", yaml_path)
        else:
            logger.warning(
                "Config file not found: %s — using defaults.", yaml_path
            )
    else:
        logger.info("No config path provided — using defaults.")

    # Handle .inf in YAML (PyYAML loads .inf as float('inf'))
    risk = config.get("risk", {})
    if risk.get("max_position_size") == float("inf"):
        risk["max_position_size"] = float("inf")

    return config


def _deep_copy_defaults() -> dict[str, Any]:
    """Return a deep copy of the default configuration."""
    return copy.deepcopy(_DEFAULTS)


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in-place.

    Nested dictionaries are merged; all other values are overwritten.

    Parameters
    ----------
    base:
        The base dictionary (mutated in-place).
    override:
        The dictionary whose values take precedence.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
