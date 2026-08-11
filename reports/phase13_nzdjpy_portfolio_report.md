==============================================================================
PHASE 13: NZDJPY MECHANISM VERIFICATION + PORTFOLIO EXPOSURE ANALYSIS
==============================================================================
Frozen NZDJPY spec: unchanged from phase 12 (EXP-030). Nothing in this script modifies it or any live demo strategy.

Fetching data (NZDJPY candidate + USDJPY proxy + 8 existing demo strategies underlying pairs) ...
  NZDJPY H1: 19,356 bars
  USDJPY H1: 19,352 bars
  GBPJPY H1: 19,348 bars
  CADJPY H1: 19,351 bars
  EURJPY H1: 19,351 bars
  AUDJPY H1: 19,349 bars
  GBPUSD H1: 19,350 bars
  XAUUSD H1: 17,984 bars

==============================================================================
PART 1 -- NZDJPY MECHANISM: exact documentation + lookahead check
==============================================================================

  How USDJPY is used: USDJPY's H1 CLOSE price is tracked from the open
  of that day's server-hour-7 bar (its open price, captured once per
  day when hours[i]==7) forward to the CLOSE of the current bar. The
  difference in pips is the 'move'. This is a MOMENTUM/DRIFT measure
  of USDJPY over the London-session-to-current window, not a level or
  an oscillator.

  Timeframe: H1, both for USDJPY (the proxy) and NZDJPY (the traded
  instrument).

  Signal calculation timing: evaluated ONLY on the H1 bar whose
  timestamp (bar OPEN convention, standard MT5) has hour==15 server
  time. Since MT5 timestamps a bar at its OPEN, 'hour==15' bar closes
  at hour 16 -- the signal is therefore only fully known once that bar
  closes, i.e. effectively usable starting server hour 16:00. The
  backtest uses closes[i] (bar i's own close) as both the signal value
  AND the entry price for bar i -- this matches how every other
  strategy in this project is simulated (decision made using
  information available at that bar's own close), and matches the live
  orchestrator convention (agent_strategy checks the LAST CLOSED bar,
  pos=1).

  Contemporaneous or lagged: the USDJPY move and the NZDJPY entry are
  READ FROM THE SAME BAR INDEX i. This is contemporaneous by
  construction, not lagged -- both series must be correctly time-
  aligned bar-for-bar for this to be valid (checked explicitly below).

  How direction is determined: move > 0 (USDJPY up = broad JPY
  weakness) -> BUY NZDJPY. move < 0 -> SELL NZDJPY.

  LOOKAHEAD/ALIGNMENT CHECK: verifying NZDJPY and USDJPY H1 bar timestamps are identical index-for-index (a silent misalignment here -- e.g. one symbol missing a bar the other has -- would make usdjpy_move[i] correspond to a DIFFERENT calendar time than NZDJPY bar i, corrupting every signal without an obvious error).
    NZDJPY bars: 19356   USDJPY bars: 19352   common timestamps: 19350
    NZDJPY-only timestamps (no USDJPY match): 6
    USDJPY-only timestamps (no NZDJPY match): 2
    Fully aligned: False
    *** WARNING: arrays are NOT fully aligned. build_usdjpy_proxy() and signals_xmomentum() index by POSITION not by timestamp -- any misalignment silently corrupts the signal. This must be fixed before the result can be trusted. ***

  Frozen NZDJPY candidate signal count: 506
  "Profitable without the USDJPY input?" -- already answered by the phase-12 permutation test (n=1000 random-direction shuffles of the SAME entry bars/SL/TP): the null models OWN mean outcome was a LOSING strategy (mean PF 0.815, mean P&L -$18,706). Answer: NO, removing the directional information (keeping only the timing/SL/TP skeleton) does not remain profitable on average.

  CONTROL: same entry bars/SL/TP, but direction from NZDJPYs OWN self-momentum (hour-7 open -> signal bar) instead of USDJPY:
    IS PF=0.92  P&L=$-5,460   OOS PF=1.06  P&L=$+1,767
    (If self-momentum alone matched USDJPY-momentum's performance, that would suggest "any lagged momentum works" rather than USDJPY specifically being informative. Compare to the frozen result below.)

==============================================================================
PART 2 -- CADJPY REPLICATION (exact frozen mechanism, unmodified)
==============================================================================
CADJPY is ALREADY traded live (ARB + AMR). This is a RESEARCH REPLICATION of the NZDJPY mechanism only -- it does NOT touch, read, or influence the live CADJPY ARB/AMR strategies in any way, and a positive result here is evidence about the MECHANISM, not evidence of portfolio diversification (that is Part 3/4).
  CADJPY (frozen NZDJPY params, unchanged): n_signals=424
    IS:  n=294 PF=0.72   DD=20.28% pm= 28.0%  P&L=$-19,132.99
    OOS: n=114 PF=1.23   P&L=$+4,133.54  DD=3.64%
  CADJPY permutation test: real PF=0.812, beats 45.7% of 1000 random-direction shuffles (null mean PF=0.827)

  REPLICATION CLASSIFICATION: PARTIALLY REPLICATED

==============================================================================
RECONSTRUCTING THE 8 EXISTING DEMO STRATEGIES (verified live params)
==============================================================================
  GBPJPY_ARB       n= 200 trades  (risk 0.50%/trade, live YAML-verified params)
  CADJPY_ARB       n= 199 trades  (risk 0.50%/trade, live YAML-verified params)
  XAUUSD_ARB       n= 259 trades  (risk 0.25%/trade, live YAML-verified params)
  GBPJPY_AMR       n= 424 trades  (risk 0.25%/trade, live YAML-verified params)
  EURJPY_AMR       n= 751 trades  (risk 0.25%/trade, live YAML-verified params)
  AUDJPY_AMR       n= 677 trades  (risk 0.25%/trade, live YAML-verified params)
  CADJPY_AMR       n= 629 trades  (risk 0.25%/trade, live YAML-verified params)
  GBPUSD_MON       n= 158 trades  (risk 0.25%/trade, live YAML-verified params)

==============================================================================
PART 3 -- PORTFOLIO EXPOSURE ANALYSIS (mandatory)
==============================================================================

-- A. Daily-return correlation matrix (all 9, incl. NZDJPY candidate) --
                  GBPJPY_ARB  CADJPY_ARB  XAUUSD_ARB  GBPJPY_AMR  EURJPY_AMR  AUDJPY_AMR  CADJPY_AMR  GBPUSD_MON  NZDJPY_CANDIDATE
GBPJPY_ARB              1.00        0.31       -0.06       -0.02       -0.00        0.03        0.05        0.01              0.09
CADJPY_ARB              0.31        1.00        0.01       -0.01        0.01        0.01        0.03        0.00              0.01
XAUUSD_ARB             -0.06        0.01        1.00       -0.03       -0.04        0.03       -0.01        0.03              0.03
GBPJPY_AMR             -0.02       -0.01       -0.03        1.00        0.32        0.28        0.33        0.08             -0.07
EURJPY_AMR             -0.00        0.01       -0.04        0.32        1.00        0.24        0.34        0.10             -0.03
AUDJPY_AMR              0.03        0.01        0.03        0.28        0.24        1.00        0.29        0.05              0.02
CADJPY_AMR              0.05        0.03       -0.01        0.33        0.34        0.29        1.00        0.07             -0.06
GBPUSD_MON              0.01        0.00        0.03        0.08        0.10        0.05        0.07        1.00             -0.02
NZDJPY_CANDIDATE        0.09        0.01        0.03       -0.07       -0.03        0.02       -0.06       -0.02              1.00

-- Weekly-return correlation matrix --
                  GBPJPY_ARB  CADJPY_ARB  XAUUSD_ARB  GBPJPY_AMR  EURJPY_AMR  AUDJPY_AMR  CADJPY_AMR  GBPUSD_MON  NZDJPY_CANDIDATE
GBPJPY_ARB              1.00        0.31       -0.03       -0.02        0.06       -0.04        0.16       -0.05              0.29
CADJPY_ARB              0.31        1.00        0.03       -0.05        0.06        0.05        0.08       -0.15              0.04
XAUUSD_ARB             -0.03        0.03        1.00       -0.03       -0.15        0.08       -0.13        0.13             -0.02
GBPJPY_AMR             -0.02       -0.05       -0.03        1.00        0.36        0.27        0.32        0.15             -0.08
EURJPY_AMR              0.06        0.06       -0.15        0.36        1.00        0.18        0.40        0.17             -0.04
AUDJPY_AMR             -0.04        0.05        0.08        0.27        0.18        1.00        0.22        0.05             -0.05
CADJPY_AMR              0.16        0.08       -0.13        0.32        0.40        0.22        1.00        0.05             -0.12
GBPUSD_MON             -0.05       -0.15        0.13        0.15        0.17        0.05        0.05        1.00             -0.03
NZDJPY_CANDIDATE        0.29        0.04       -0.02       -0.08       -0.04       -0.05       -0.12       -0.03              1.00

-- NZDJPY candidates correlation with each existing strategy --
  vs GBPJPY_ARB       daily r=+0.09   weekly r=+0.29
  vs CADJPY_ARB       daily r=+0.01   weekly r=+0.04
  vs XAUUSD_ARB       daily r=+0.03   weekly r=-0.02
  vs GBPJPY_AMR       daily r=-0.07   weekly r=-0.08
  vs EURJPY_AMR       daily r=-0.03   weekly r=-0.04
  vs AUDJPY_AMR       daily r=+0.02   weekly r=-0.05
  vs CADJPY_AMR       daily r=-0.06   weekly r=-0.12
  vs GBPUSD_MON       daily r=-0.02   weekly r=-0.03

-- B. JPY exposure matrix (BUY XXXJPY = short JPY = -1; SELL = +1 long JPY) --
  Days with >=3 JPY-cross strategies (of 7) entering the SAME JPY direction simultaneously: 581 / 1140 days
  Net JPY-direction-days distribution: mean=-0.60  std=2.42  max_same_direction_stack=7

-- Trade/time overlap: NZDJPY candidate open position vs each existing strategy --
  GBPJPY_ARB      : 167 / 506 NZDJPY-candidate trades had a simultaneous OPEN position in this strategy
  CADJPY_ARB      : 170 / 506 NZDJPY-candidate trades had a simultaneous OPEN position in this strategy
  XAUUSD_ARB      : 129 / 506 NZDJPY-candidate trades had a simultaneous OPEN position in this strategy
  GBPJPY_AMR      : 0 / 506 NZDJPY-candidate trades had a simultaneous OPEN position in this strategy
  EURJPY_AMR      : 0 / 506 NZDJPY-candidate trades had a simultaneous OPEN position in this strategy
  AUDJPY_AMR      : 0 / 506 NZDJPY-candidate trades had a simultaneous OPEN position in this strategy
  CADJPY_AMR      : 0 / 506 NZDJPY-candidate trades had a simultaneous OPEN position in this strategy
  GBPUSD_MON      : 77 / 506 NZDJPY-candidate trades had a simultaneous OPEN position in this strategy

-- Drawdown co-occurrence (fraction of days both strategies are simultaneously below their own rolling peak) --
  NZDJPY_CANDIDATE & GBPJPY_ARB      : co-drawdown 69.4% of days
  NZDJPY_CANDIDATE & CADJPY_ARB      : co-drawdown 70.8% of days
  NZDJPY_CANDIDATE & XAUUSD_ARB      : co-drawdown 68.6% of days
  NZDJPY_CANDIDATE & GBPJPY_AMR      : co-drawdown 67.4% of days
  NZDJPY_CANDIDATE & EURJPY_AMR      : co-drawdown 72.7% of days
  NZDJPY_CANDIDATE & AUDJPY_AMR      : co-drawdown 69.7% of days
  NZDJPY_CANDIDATE & CADJPY_AMR      : co-drawdown 71.6% of days
  NZDJPY_CANDIDATE & GBPUSD_MON      : co-drawdown 49.5% of days

-- Worst-case clustered losses --
  Worst single day for the EXISTING 8-strategy portfolio: 2025-08-14  (P&L=$-2,276.19)
    NZDJPY candidate P&L that same day: $+967.16
  Worst single week for the EXISTING 8-strategy portfolio: week of 2024-10-06  (P&L=$-6,745.06)
    NZDJPY candidate P&L that same week: $+2,006.66

-- D. Portfolio-level Monte Carlo, WITH vs WITHOUT the NZDJPY candidate --
  WITHOUT NZDJPY (existing 8): monthly mean=$+2,253  6mo-equiv pass=98%  bust=1%  median days=57
  WITH NZDJPY (9 slots):       monthly mean=$+3,082  6mo-equiv pass=99%  bust=1%  median days=44

==============================================================================
PART 4 -- JPY FACTOR ANALYSIS (attribution, not optimization)
==============================================================================

-- Correlation of NZDJPY-candidate daily P&L vs each pairs own daily price return --
  USDJPY: r=-0.086
  GBPJPY: r=-0.035
  CADJPY: r=-0.099
  EURJPY: r=-0.064
  AUDJPY: r=-0.024

-- Multi-factor OLS: NZDJPY_pnl ~ USDJPY + GBPJPY + CADJPY + EURJPY + AUDJPY --
  R-squared: 0.020  (fraction of daily P&L variance explained by these 5 pairs own contemporaneous price moves)
    beta[USDJPY] = +16.08
    beta[GBPJPY] = +94.22
    beta[CADJPY] = -183.02
    beta[EURJPY] = -63.77
    beta[AUDJPY] = +75.75

  FACTOR-OVERLAP CLASSIFICATION: POTENTIAL DIVERSIFICATION  (R-squared=0.020, threshold 0.30)

==============================================================================
PART 5 -- EXPLORATORY: does the NZDJPY signal flag AMR-hostile days?
==============================================================================
EXPLORATORY ONLY. No AMR code is touched. The Aug-25 AMR checkpoint remains fully independent of this analysis.
  AMR (all 4 pairs combined) mean daily P&L on days the NZDJPY signal FIRED: $+40.09  (n=506)
  AMR mean daily P&L on days the NZDJPY signal did NOT fire: $+68.00  (n=308)
  Difference: $-27.91/day (AMR notably worse on signal-fire days -- consistent with a trend-regime marker)
  This is a descriptive comparison only -- not a statistically powered test, and NOT a proposal to implement anything. Recorded as exploratory evidence for a possible future regime-filter study.

==============================================================================
PARTS 6-9 -- REPEAT FROZEN VALIDATION (identical methodology to phase 12)
==============================================================================

-- Part 6: cost/execution stress --
  normal 1.0x (2.2p)   IS PF=1.49   OOS PF=1.2    OOS P&L=$  +7,731
  1.5x (3.3p)          IS PF=1.33   OOS PF=1.06   OOS P&L=$  +2,223
  2.0x (4.4p)          IS PF=1.19   OOS PF=0.93   OOS P&L=$  -2,311
  1-bar entry delay: OOS PF=0.95  P&L=$-1,443

-- Part 7: rolling walk-forward (each window independent) --
   fold test_start   test_end  trades  win_rate   pf      pnl  max_dd_pct
      0 2024-06-28 2024-12-28      92      58.7 1.69 10489.10        2.62
      1 2024-12-28 2025-06-28      77      59.7 1.95 13180.48        2.18
      2 2025-06-28 2025-12-28      83      56.6 1.71 11766.08        1.44
      3 2025-12-28 2026-06-28      74      43.2 0.75 -5374.51        6.53
SUMMARY       None       None     326      54.6 1.52 30061.15        6.53
  3/4 folds profitable (75.0%)

-- Part 8: trade-sequence Monte Carlo --
  Median max DD: -6.52%   Worst-5% max DD: -10.58%
  Losing-streak p50/p95: 7/11
  Risk of ruin (-20% breach, 3000 shuffles): 0.03%

-- Part 9: parameter-plateau re-verification (values from phase 12, NOT re-optimized) --
  thr_atr 1.0-1.5: IS PF 1.33-1.59 (all positive, smooth) -- plateau CONFIRMED
  sl_atr  1.2-1.8: IS PF 1.38-1.53 (all positive, smooth) -- plateau CONFIRMED
  tp_atr  2.0-3.0: IS PF 1.43-1.53 (all positive, smooth) -- plateau CONFIRMED
  (full detail: data/phase12_param_sensitivity.csv, unchanged)

==============================================================================
PART 10 -- FINAL DUAL CLASSIFICATION
==============================================================================
  Max |correlation| vs any existing strategy: 0.09
  Strategy classification (unchanged from phase 12): PROMISING BUT INSUFFICIENT
  Portfolio classification: POTENTIAL DIVERSIFIER

  COMBINED: PROMISING BUT INSUFFICIENT + POTENTIAL DIVERSIFIER

Logged as EXP-033