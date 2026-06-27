"""
BaseStrategy -- abstract contract every trading strategy must implement.

This is intentionally NOT an abc.ABC: subclasses can be instantiated even
if they haven't implemented every method yet (see strategies/asian_breakout.py,
a stub that loads successfully via pair_manager but raises NotImplementedError
the moment any of its real decision-making methods is called). Any method
left unoverridden raises NotImplementedError with a clear message naming
the class and method, so a missing implementation fails loudly the first
time it's used rather than silently doing nothing.

Required interface:
  generate_signal(pair, current_price, h4_trend)  -> 'BUY' | 'SELL' | None
  calculate_sl(entry_price, direction, pair_config) -> float (SL price)
  calculate_tp(entry_price, direction, pair_config) -> float (TP price)
  get_session_windows()                            -> list of session windows
  validate_config(pair_config)                     -> None, raises ValueError
                                                        if required keys missing
"""

from __future__ import annotations


class BaseStrategy:

    def __init__(self, pair_config: dict):
        self.pair_config = pair_config
        self.pair = pair_config.get('pair')

    def generate_signal(self, pair: str, current_price: float, h4_trend: int):
        """Return 'BUY', 'SELL', or None given the pair, current price, and
        H4 trend direction (+1 bullish, -1 bearish, 0 neutral)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement generate_signal()"
        )

    def calculate_sl(self, entry_price: float, direction: str, pair_config: dict) -> float:
        """Return the stop-loss price for a trade opened at entry_price."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement calculate_sl()"
        )

    def calculate_tp(self, entry_price: float, direction: str, pair_config: dict) -> float:
        """Return the take-profit price for a trade opened at entry_price."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement calculate_tp()"
        )

    def get_session_windows(self) -> list:
        """Return the list of valid UTC trading session windows for this strategy."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_session_windows()"
        )

    def validate_config(self, pair_config: dict) -> None:
        """Raise ValueError if pair_config is missing any required key for
        this strategy. Must return None (not raise) when config is valid."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement validate_config()"
        )
