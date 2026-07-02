"""
AsianBreakout strategy -- STUB.

Placeholder for future USDJPY / AUDJPY / NZDJPY Asian-session breakout
strategies. Not yet implemented.

Design note: validate_config() performs REAL (not stubbed) validation so
that pair_manager can successfully load this strategy when a pair's YAML
sets `active: true` -- the pair will correctly appear in the active pairs
list. Only the actual decision-making methods (generate_signal,
calculate_sl, calculate_tp, get_session_windows) raise NotImplementedError,
so the failure happens loudly the moment the strategy is actually asked to
do something, not at load time. This is why pairs/USDJPY.yaml ships with
`active: false` -- flip it to `true` only once this strategy is built out.
"""

from __future__ import annotations

from strategies.base_strategy import BaseStrategy

NOT_IMPLEMENTED_MSG = (
    "Asian breakout strategy not yet implemented — set active: false in pair YAML"
)

REQUIRED_KEYS = ['pair', 'strategy', 'active', 'timeframe', 'risk_percent',
                  'sl_pips', 'tp_pips', 'h4_filter', 'session', 'friday_close']


class AsianBreakout(BaseStrategy):

    COMPATIBLE_PAIRS = ["USDJPY", "AUDJPY", "NZDJPY"]

    def __init__(self, pair_config: dict):
        super().__init__(pair_config)
        self.validate_config(pair_config)

    def validate_config(self, pair_config: dict) -> None:
        missing = [k for k in REQUIRED_KEYS if k not in pair_config]
        if missing:
            raise ValueError(
                f"AsianBreakout config for {pair_config.get('pair', '?')} "
                f"missing required key(s): {missing}"
            )

    def generate_signal(self, pair: str, current_price: float, h4_trend: int, **context):
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)

    def calculate_sl(self, entry_price: float, direction: str, pair_config: dict) -> float:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)

    def calculate_tp(self, entry_price: float, direction: str, pair_config: dict) -> float:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)

    def get_session_windows(self) -> list:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)
