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
from strategies.ny_open_breakout import NyOpenBreakout
from strategies.h4_trend_pullback import H4TrendPullback
from strategies.asian_range_breakout import AsianRangeBreakout
from strategies.mean_reversion import MeanReversion
from strategies.volatility_regime_trend import VolatilityRegimeTrend
from strategies.momentum_divergence_session import MomentumDivergenceSession
from strategies.regime_filtered_ma_cross import RegimeFilteredMACross

# -- stubs, not yet implemented, NOT activated in any pairs/*.yaml
from strategies.ema_crossover import EmaCrossover
from strategies.macd_trend import MacdTrend
from strategies.consolidation_breakout import ConsolidationBreakout
from strategies.rsi_reversal import RsiReversal
from strategies.atr_dynamic import AtrDynamic
from strategies.bollinger_squeeze import BollingerSqueeze

STRATEGY_REGISTRY = {
    "london_breakout": LondonBreakout,
    "sma_ema_combined": SmaEmaCombined,
    "asian_breakout": AsianBreakout,

    # -- fully implemented, but backtest-FAILED (see each class's docstring) --
    # do not set active: true on a pairs/*.yaml without re-validating
    "ny_open_breakout": NyOpenBreakout,
    "h4_trend_pullback": H4TrendPullback,

    # -- fully implemented, NOT YET backtested (keep active: false until
    # a walk-forward validation run is complete and passes criteria)
    "asian_range_breakout": AsianRangeBreakout,
    "mean_reversion": MeanReversion,
    "volatility_regime_trend": VolatilityRegimeTrend,
    "momentum_divergence_session": MomentumDivergenceSession,
    "regime_filtered_ma_cross": RegimeFilteredMACross,

    # -- stubs (raise NotImplementedError on use -- see each class's docstring)
    "ema_crossover": EmaCrossover,
    "macd_trend": MacdTrend,
    "consolidation_breakout": ConsolidationBreakout,
    "rsi_reversal": RsiReversal,
    "atr_dynamic": AtrDynamic,
    "bollinger_squeeze": BollingerSqueeze,
}
