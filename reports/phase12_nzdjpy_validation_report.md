==============================================================================
PHASE 12: NZDJPY FROZEN VALIDATION GATE
==============================================================================

Frozen spec logged as EXP-030 in experiments/experiments.csv
  instrument: NZDJPY
  proxy: USDJPY
  timeframe: H1
  check_hour_server: 15
  thr_atr: 1.25
  sl_atr_mult: 1.5
  tp_atr_mult: 2.5
  time_exit_server_hour: 21
  friday_close_server_hour: 20
  risk_pct_per_trade: 0.005
  spread_pips: 2.2
  slippage_pips: 0
  news_filter: none (offline backtest)
  max_trades_per_day: 1
  sizing: compounding balance x risk_pct / sl_pips
  lot_clamp_modeled: False
  entry_price: signal-bar close, no delay
  exit_priority: SL > TP > TimeExit(21) > FridayClose(Fri 20:00)
  dataset_months: 36
  is_months: 24
  oos_months: 12
  source_experiments: phase10 EXP set + phase10b refinement (best cell)

Fetching NZDJPY + USDJPY H1 ...
  NZDJPY: 19,356 bars (2023-06-28 -> 2026-08-11)
  Frozen candidate signal count (full 36mo): 506

==============================================================================
A/B/C. IN-SAMPLE / INTERNAL SPLIT / OUT-OF-SAMPLE
==============================================================================
NOTE: only ~3 years of history exist, so a true 3-way independent split (dev/validation/final-OOS) is not honestly available. IS-early and IS-late below are BOTH inside the original discovery window (not independent) -- an internal consistency check only. OOS is the genuinely held-out final 12 months, touched exactly once, here.
  IS-early (internal, first 12mo of IS): n=164  PF=1.26    DD= 5.86%  pm= 69.2%  P&L=$+7,225.23  WR=55.5%
  IS-late (internal, second 12mo of IS): n=168  PF=1.72    DD= 2.24%  pm= 92.3%  P&L=$+21,599.60  WR=58.9%
  IS-FULL (24mo, original selection window): n=332  PF=1.49    DD= 5.86%  pm= 84.0%  P&L=$+28,824.83  WR=57.2%
  OOS (final 12mo, held out, touched once): n=158  PF=1.2     DD= 7.05%  pm= 53.8%  P&L=$+7,731.07  WR=50.0%

==============================================================================
D. ROLLING WALK-FORWARD (each window reported independently)
==============================================================================
   fold test_start   test_end  trades  win_rate   pf      pnl  max_dd_pct
      0 2024-06-28 2024-12-28      92      58.7 1.69 10489.10        2.62
      1 2024-12-28 2025-06-28      77      59.7 1.95 13180.48        2.18
      2 2025-06-28 2025-12-28      83      56.6 1.71 11766.08        1.44
      3 2025-12-28 2026-06-28      74      43.2 0.75 -5374.51        6.53
SUMMARY       None       None     326      54.6 1.52 30061.15        6.53
  4 fold(s) achievable from available history -- adequate fold count
  Consistency: 3/4 folds profitable (75.0%)

==============================================================================
E. TRADE-SEQUENCE MONTE CARLO (order-shuffle, drawdown/streak only -- not meaningless shuffled-P&L "confidence intervals")
==============================================================================
  Historical (actual, single path) max DD: -6.85%
  Historical (actual) longest losing streak: 9 trades
  MC median max DD:     -6.52%
  MC worst-5% max DD:   -10.58%
  MC losing-streak p50/p95: 7/11
  Risk of ruin (breach -20% at any point, 3000 shuffles): 0.03%

==============================================================================
F/12. COST + EXECUTION STRESS (spread multiples, entry delay)
==============================================================================
  normal (1.0x = 2.2p)         IS PF=1.49   P&L=$  +28,825  OOS PF=1.2    P&L=$   +7,731
  conservative (1.5x = 3.3p)   IS PF=1.33   P&L=$  +19,655  OOS PF=1.06   P&L=$   +2,223
  stress (2.0x = 4.4p)         IS PF=1.19   P&L=$  +11,203  OOS PF=0.93   P&L=$   -2,311
  extreme (3.0x = 6.6p)        IS PF=0.94   P&L=$   -3,761  OOS PF=0.73   P&L=$   -8,983
  1-bar entry delay (execute next bar's close instead of signal bar's close):
    IS PF=1.22   P&L=$  +10,434  OOS PF=0.95   P&L=$   -1,443

==============================================================================
G/11. PARAMETER SENSITIVITY -- neighbors reported, NONE selected
==============================================================================

  -- check_hour neighbors (thr/sl/tp held at frozen values) --
    check_hour=13: IS PF=1.36   DD=12.87%  OOS PF=1.0    P&L=$    -181
    check_hour=14: IS PF=1.38   DD= 7.66%  OOS PF=1.09   P&L=$  +3,670
    check_hour=15: IS PF=1.49   DD= 5.86%  OOS PF=1.2    P&L=$  +7,731 <== FROZEN
    check_hour=16: IS PF=1.25   DD= 3.53%  OOS PF=0.86   P&L=$  -4,602

  -- thr_atr neighbors (+/-10%/20%, others held at frozen values) --
    thr_atr=1.000: IS PF=1.33   DD= 6.76%  OOS PF=1.16   P&L=$  +6,604
    thr_atr=1.125: IS PF=1.41   DD= 6.89%  OOS PF=1.2    P&L=$  +7,931
    thr_atr=1.250: IS PF=1.49   DD= 5.86%  OOS PF=1.2    P&L=$  +7,731 <== FROZEN
    thr_atr=1.375: IS PF=1.59   DD= 5.80%  OOS PF=1.18   P&L=$  +6,759
    thr_atr=1.500: IS PF=1.53   DD= 6.38%  OOS PF=1.13   P&L=$  +4,404

  -- sl_atr neighbors (+/-10%/20%, others held at frozen values) --
    sl_atr=1.200: IS PF=1.38   DD= 7.45%  OOS PF=1.21   P&L=$  +9,341
    sl_atr=1.350: IS PF=1.48   DD= 6.04%  OOS PF=1.22   P&L=$  +9,375
    sl_atr=1.500: IS PF=1.49   DD= 5.86%  OOS PF=1.2    P&L=$  +7,731 <== FROZEN
    sl_atr=1.650: IS PF=1.53   DD= 5.18%  OOS PF=1.24   P&L=$  +7,856
    sl_atr=1.800: IS PF=1.51   DD= 5.12%  OOS PF=1.22   P&L=$  +6,680

  -- tp_atr neighbors (+/-10%/20%, others held at frozen values) --
    tp_atr=2.000: IS PF=1.43   DD= 6.24%  OOS PF=1.14   P&L=$  +5,151
    tp_atr=2.250: IS PF=1.45   DD= 6.17%  OOS PF=1.19   P&L=$  +7,030
    tp_atr=2.500: IS PF=1.49   DD= 5.86%  OOS PF=1.2    P&L=$  +7,731 <== FROZEN
    tp_atr=2.750: IS PF=1.53   DD= 5.94%  OOS PF=1.13   P&L=$  +5,113
    tp_atr=3.000: IS PF=1.53   DD= 6.28%  OOS PF=1.13   P&L=$  +4,910

==============================================================================
H. YEAR-BY-YEAR (concentration check)
==============================================================================
  2023: n= 75  WR= 53.3%  PF=1.08    P&L=$  +949.54
  2024: n=180  WR= 53.9%  PF=1.31    P&L=$+10,123.32
  2025: n=160  WR= 58.1%  PF=1.77    P&L=$+23,647.13
  2026: n= 91  WR= 46.2%  PF=0.90    P&L=$-2,367.54
  Single best year contributes 73.1% of total P&L -- CONCENTRATION FLAG

==============================================================================
I. MONTHLY PERFORMANCE
==============================================================================
                           trades      pnl
entry_time                                
2023-07-31 00:00:00+00:00      13 -4050.92
2023-08-31 00:00:00+00:00      11  1039.64
2023-09-30 00:00:00+00:00      13   489.96
2023-10-31 00:00:00+00:00      13  2003.57
2023-11-30 00:00:00+00:00      14  1232.60
2023-12-31 00:00:00+00:00      11   234.69
2024-01-31 00:00:00+00:00      16  2997.44
2024-02-29 00:00:00+00:00      11   133.82
2024-03-31 00:00:00+00:00      15 -2435.80
2024-04-30 00:00:00+00:00      17 -1320.18
2024-05-31 00:00:00+00:00      14   306.42
2024-06-30 00:00:00+00:00      13  -717.51
2024-07-31 00:00:00+00:00      15  3654.99
2024-08-31 00:00:00+00:00      15 -1520.66
2024-09-30 00:00:00+00:00      15  2239.00
2024-10-31 00:00:00+00:00      19  4135.58
2024-11-30 00:00:00+00:00      12  2165.89
2024-12-31 00:00:00+00:00      18   484.33
2025-01-31 00:00:00+00:00      12  5127.03
2025-02-28 00:00:00+00:00      13     8.17
2025-03-31 00:00:00+00:00      11  2309.54
2025-04-30 00:00:00+00:00      10  4150.72
2025-05-31 00:00:00+00:00      14   864.39
2025-06-30 00:00:00+00:00      15    50.60
2025-07-31 00:00:00+00:00      14   173.81
2025-08-31 00:00:00+00:00      15  2514.67
2025-09-30 00:00:00+00:00      16  1672.34
2025-10-31 00:00:00+00:00      12  2356.96
2025-11-30 00:00:00+00:00      13  3341.97
2025-12-31 00:00:00+00:00      15  1076.93
2026-01-31 00:00:00+00:00      10  -235.65
2026-02-28 00:00:00+00:00      14  3308.28
2026-03-31 00:00:00+00:00      17 -3947.95
2026-04-30 00:00:00+00:00       9  -702.87
2026-05-31 00:00:00+00:00      11 -1282.60
2026-06-30 00:00:00+00:00      13  -798.20
2026-07-31 00:00:00+00:00      13  -644.43
2026-08-31 00:00:00+00:00       4  1935.88
  Months with zero trades: 0 / 38

==============================================================================
J. DRAWDOWN ANALYSIS (historical single path)
==============================================================================
  Max drawdown: -6.85%  ($41,996.07 at worst)
  Avg drawdown duration: 10.7 trades
  Max drawdown duration: 103 trades
  Recovery factor (net P&L / max peak-to-trough $): 7.70

==============================================================================
K. MFE / MAE ANALYSIS
==============================================================================
  Median MFE: 0.89R   Median MAE: 0.68R
  Winners: median MFE captured = 62% of the best available excursion
  Losers: median MAE vs 1R stop = 1.04R (should be ~1.0R if stops are the binding constraint)

==============================================================================
L. LOSING-STREAK ANALYSIS (see section E for historical + MC figures)
==============================================================================
  Historical longest losing streak: 9 consecutive trades
  Monte Carlo p50/p95 across 3000 order-shuffles: 7/11

==============================================================================
M. TRADE FREQUENCY
==============================================================================
  Total trades: 506 over 1133 days (3.13/week, 13.40/month)
  Entry hour distribution: {15: 506} (should be 100% at hour 15 by construction)
  Days between trades: median=1.0  max=13

==============================================================================
BENCHMARK: random-direction null model (permutation test)
==============================================================================
Buy-and-hold is not a meaningful null model for an intraday FX directional strategy. The correct null here is: same entry bars, same ATR-based SL/TP construction, but the BUY/SELL direction is a coin flip instead of the USDJPY-proxy signal. This isolates whether the DIRECTION-PICKING logic adds value beyond generic session-time trading with these SL/TP mechanics.
  real_pf: 1.322
  real_pnl: 32352.45
  null_pf_mean: 0.815
  null_pf_p50: 0.812
  null_pf_p95: 0.965
  null_pnl_mean: -18706.21
  real_pf_percentile_vs_null: 100.0
  real_pnl_percentile_vs_null: 100.0
  n_perm: 1000

  Interpretation: the real strategy's profit factor exceeded 100.0% of 1000 random-direction shuffles of the SAME entry bars/SL/TP construction.

==============================================================================
FINAL VALIDATION SCORECARD
==============================================================================
Test                                   Result                                             Pass/Fail
In-sample (24mo)                       PF=1.49 DD=5.86% pm=84.0%                          PASS
Out-of-sample (12mo, held out)         PF=1.2 P&L=$+7,731                                 PASS
Walk-forward                           3/4 folds profitable                               PASS
Monte Carlo (risk of ruin)             0.03%                                              PASS
Cost stress (2x spread)                IS PF=1.19 OOS PF=0.93                             FAIL
Parameter sensitivity (plateau)        thr/sl/tp PF range < 0.6 spread: True              PASS
Year consistency                       best year = 73% of total P&L                       FAIL
Direction vs random (permutation)      beats 100% of random-direction shuffles            PASS
Drawdown                               -6.85%                                             PASS

7/9 checks pass.  FINAL STATUS: VALIDATED

Final summary logged as EXP-031