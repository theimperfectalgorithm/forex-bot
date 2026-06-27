"""
strategy_loader -- maps a pair's `strategy:` name string to the correct
strategy class, instantiates it with that pair's config, and returns it.

Usage:
    pair_config = load_yaml('pairs/GBPJPY.yaml')
    strategy    = load(pair_config)   # -> LondonBreakout instance
"""

from __future__ import annotations

from pathlib import Path

import yaml

from strategies.registry import STRATEGY_REGISTRY


def load_yaml(path) -> dict:
    """Read one pair YAML file and return its contents as a dict."""
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_strategy_class(strategy_name: str):
    """Return the class registered for strategy_name, or raise a clear error."""
    cls = STRATEGY_REGISTRY.get(strategy_name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy name '{strategy_name}' -- registered strategies "
            f"are: {sorted(STRATEGY_REGISTRY.keys())}"
        )
    return cls


def load(pair_config_or_path):
    """
    Accepts either an already-parsed pair_config dict, or a path to a pair
    YAML file. Returns an initialized instance of the correct strategy class.
    """
    if isinstance(pair_config_or_path, (str, Path)):
        pair_config = load_yaml(pair_config_or_path)
    else:
        pair_config = pair_config_or_path

    strategy_name = pair_config.get('strategy')
    if not strategy_name:
        raise ValueError(f"pair_config missing 'strategy' key: {pair_config}")

    cls = get_strategy_class(strategy_name)
    return cls(pair_config)
