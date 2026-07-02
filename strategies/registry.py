"""
STRATEGY_REGISTRY -- single source of truth mapping a pair YAML's
`strategy:` name string to its implementing class.

To add a new strategy in future: write the class, then add one line here.
Nothing else in the architecture needs to change -- strategy_loader.py
imports this registry rather than maintaining its own import list.
"""

from strategies.london_breakout import LondonBreakout
from strategies.sma_ema_combined import SmaEmaCombined
from strategies.asian_breakout import AsianBreakout
from strategies.breakout.ny_open_breakout import NyOpenBreakout
from strategies.trend_following.h4_trend_pullback import H4TrendPullback

# -- stubs, not yet implemented, NOT activated in any pairs/*.yaml
from strategies.trend_following.ema_crossover import EmaCrossover
from strategies.trend_following.macd_trend import MacdTrend
from strategies.breakout.asian_range_breakout import AsianRangeBreakout
from strategies.breakout.consolidation_breakout import ConsolidationBreakout
from strategies.reversal.rsi_reversal import RsiReversal
from strategies.reversal.mean_reversion import MeanReversion
from strategies.volatility.atr_dynamic import AtrDynamic
from strategies.volatility.bollinger_squeeze import BollingerSqueeze

STRATEGY_REGISTRY = {
    "london_breakout": LondonBreakout,
    "sma_ema_combined": SmaEmaCombined,
    "asian_breakout": AsianBreakout,

    # -- fully implemented, but backtest-FAILED (see each class's docstring) --
    # do not set active: true on a pairs/*.yaml without re-validating
    "ny_open_breakout": NyOpenBreakout,
    "h4_trend_pullback": H4TrendPullback,

    # -- stubs (raise NotImplementedError on use -- see each class's docstring)
    "ema_crossover": EmaCrossover,
    "macd_trend": MacdTrend,
    "asian_range_breakout": AsianRangeBreakout,
    "consolidation_breakout": ConsolidationBreakout,
    "rsi_reversal": RsiReversal,
    "mean_reversion": MeanReversion,
    "atr_dynamic": AtrDynamic,
    "bollinger_squeeze": BollingerSqueeze,
}
