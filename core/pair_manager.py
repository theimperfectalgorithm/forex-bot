"""
pair_manager -- scans pairs/*.yaml on startup, loads only pairs marked
active: true, and hands back ready-to-use strategy instances. Optionally
restricted further by a `locked_pairs` allowlist in the gitignored
per-instance config/local_config.yaml (used on the prop-firm clone to
stay isolated from demo-only pairs/*.yaml changes pulled from git).

Usage:
    from core.pair_manager import get_active_pairs
    active_pairs = get_active_pairs()
    # -> [(pair_name, strategy_instance, pair_config), ...]
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml as _yaml

from core.strategy_loader import load, load_yaml

PAIRS_DIR = Path(__file__).parent.parent / 'pairs'
_LOCAL_CONFIG_FILE = Path(__file__).parent.parent / 'config' / 'local_config.yaml'


def _load_locked_pairs() -> set[tuple[str, str]] | None:
    """
    Reads the optional top-level `locked_pairs` key from the gitignored
    config/local_config.yaml (sibling to the `global:` block, since this
    is a list of {pair, strategy} entries, not scalar config). Returns
    None if the file or key is absent (no restriction -- today's
    behavior, e.g. the demo clone), else a set of (pair, strategy)
    tuples for O(1) membership checks.
    """
    try:
        with open(_LOCAL_CONFIG_FILE, encoding='utf-8') as f:
            cfg = _yaml.safe_load(f) or {}
    except FileNotFoundError:
        return None
    except Exception:
        return None

    entries = cfg.get('locked_pairs')
    if entries is None:
        return None
    return {(e['pair'], e['strategy']) for e in entries}


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


def get_active_pairs(pairs_dir: Path | str | None = None, log: logging.Logger | None = None,
                      locked_pairs: set[tuple[str, str]] | None = 'unset') -> list:
    """
    Scans all YAML files in pairs_dir (defaults to <repo_root>/pairs),
    loads only pairs where active: true, and returns:
        [(pair_name, strategy_instance, pair_config), ...]

    If this instance's config/local_config.yaml defines a `locked_pairs`
    allowlist, the result is further intersected against it -- a pair
    must be BOTH active: true in its YAML AND listed in locked_pairs to
    be returned. This never forces a pair active; it only ever removes
    ones the YAML already turned on. `locked_pairs` can be passed
    explicitly (e.g. by tests) to skip the config file read; leave unset
    to read config/local_config.yaml normally.

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

    locked = _load_locked_pairs() if locked_pairs == 'unset' else locked_pairs
    if locked is not None:
        kept, excluded = [], []
        for entry in active_pairs:
            key = (entry[0], entry[2].get('strategy'))
            (kept if key in locked else excluded).append(entry)
        if excluded:
            log.warning(
                "pair_manager: LOCKED instance -- excluded pairs not in "
                "config/local_config.yaml's locked_pairs allowlist: "
                f"{[(p, c.get('strategy')) for p, _i, c in excluded]}"
            )
        active_pairs = kept

    log.info(f"pair_manager: active pairs   = {[p[0] for p in active_pairs]}")
    log.info(f"pair_manager: inactive pairs = {inactive_pairs}")

    return active_pairs
