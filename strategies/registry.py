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
from strategies.asian_hours_reversion import AsianHoursReversion

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

    # -- VALIDATED 2026-07-04 on GBPJPY ONLY (tp_multiplier=2.0,
    # h4_filter=false). ACTIVE since 2026-07-05 (demo forward-test) via
    # the '@arb' dispatch path in agent_strategy. Other pairs FAILED.
    "asian_range_breakout": AsianRangeBreakout,

    # -- FORWARD-TEST CANDIDATE (2026-07-05): all 36 backtest variants
    # OOS-positive on GBPJPY/EURJPY/AUDJPY but IS PF 1.10-1.17 < 1.3 bar
    # (regime-strengthening edge). Keep active: false until the
    # orchestrator gains Asian-hours polling + a 07:00 time-exit step --
    # see the class docstring for the integration spec.
    "asian_hours_reversion": AsianHoursReversion,

    # -- backtest-FAILED 2026-07-04 walk-forward matrix across all 9
    # majors (data/strategy_matrix_results.csv, data/phase2_results.csv).
    # Do not set active: true without a structural rework:
    #   mean_reversion             : PF < 1.1 everywhere, incl. GBPUSD
    #   volatility_regime_trend    : trades ~daily, IS PF 0.8-1.1, DD to 28%
    #   momentum_divergence_session: lone GBPUSD in-sample pass collapsed
    #                                out-of-sample (PF 0.63) across every
    #                                parameter neighbourhood -- overfit
    #   regime_filtered_ma_cross   : 0 IS passes; OOS negative on 8/9 pairs
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
