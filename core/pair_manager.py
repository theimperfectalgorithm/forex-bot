"""
pair_manager -- scans pairs/*.yaml on startup, loads only pairs marked
active: true, and hands back ready-to-use strategy instances.

Usage:
    from core.pair_manager import get_active_pairs
    active_pairs = get_active_pairs()
    # -> [(pair_name, strategy_instance, pair_config), ...]
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.strategy_loader import load, load_yaml

PAIRS_DIR = Path(__file__).parent.parent / 'pairs'


def _validate_compatible_pair(pair_name: str, pair_config: dict, strategy_instance) -> None:
    """
    Raise a clear ValueError if pair_name is not listed in the strategy
    class's COMPATIBLE_PAIRS. Prevents accidentally loading e.g. a London
    breakout strategy on a USDJPY Asian-session pair via a YAML typo.
    """
    compatible = getattr(type(strategy_instance), 'COMPATIBLE_PAIRS', None)
    if compatible is None:
        return   # strategy class declares no compatibility restriction
    if pair_name not in compatible:
        raise ValueError(
            f"{pair_name} is configured with strategy="
            f"'{pair_config.get('strategy')}' ({type(strategy_instance).__name__}), "
            f"but {pair_name} is not in that strategy's COMPATIBLE_PAIRS "
            f"{compatible}. Check pairs/{pair_name}.yaml for a mismatched "
            f"'strategy:' value."
        )


def get_active_pairs(pairs_dir: Path | str | None = None, log: logging.Logger | None = None) -> list:
    """
    Scans all YAML files in pairs_dir (defaults to <repo_root>/pairs),
    loads only pairs where active: true, and returns:
        [(pair_name, strategy_instance, pair_config), ...]

    Logs which pairs are active and which are inactive on startup.
    """
    if log is None:
        log = logging.getLogger('PAIR_MANAGER')
        if not log.handlers:
            logging.basicConfig(level=logging.INFO)

    directory = Path(pairs_dir) if pairs_dir else PAIRS_DIR
    yaml_files = sorted(directory.glob('*.yaml'))

    active_pairs  = []
    inactive_pairs = []

    for yaml_path in yaml_files:
        pair_config = load_yaml(yaml_path)
        pair_name   = pair_config.get('pair', yaml_path.stem)

        if not pair_config.get('active', False):
            inactive_pairs.append(pair_name)
            continue

        try:
            strategy_instance = load(pair_config)
            _validate_compatible_pair(pair_name, pair_config, strategy_instance)
        except Exception as e:
            log.error(f"pair_manager: failed to load strategy for {pair_name}: {e}")
            continue

        active_pairs.append((pair_name, strategy_instance, pair_config))

    log.info(f"pair_manager: active pairs   = {[p[0] for p in active_pairs]}")
    log.info(f"pair_manager: inactive pairs = {inactive_pairs}")

    return active_pairs
