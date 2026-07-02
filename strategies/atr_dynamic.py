"""
AtrDynamic -- STUB. Not yet implemented -- build in Episode 15.
"""

from __future__ import annotations

from strategies.base_strategy import BaseStrategy

NOT_IMPLEMENTED_MSG = "AtrDynamic not yet implemented — build in Episode 15"


class AtrDynamic(BaseStrategy):

    NAME = "atr_dynamic"
    SESSION = "TBD"
    COMPATIBLE_PAIRS = []  # TBD -- assign when this strategy is built out

    def generate_signal(self, pair: str, current_price: float, h4_trend: int, **context):
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)

    def calculate_sl(self, entry_price: float, direction: str, pair_config: dict) -> float:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)

    def calculate_tp(self, entry_price: float, direction: str, pair_config: dict) -> float:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)

    def get_session_windows(self) -> list:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)

    def validate_config(self, pair_config: dict) -> None:
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)
