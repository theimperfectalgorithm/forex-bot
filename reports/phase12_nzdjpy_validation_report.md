==============================================================================
PHASE 12: NZDJPY FROZEN VALIDATION GATE
==============================================================================

Frozen spec logged as EXP-035 in experiments/experiments.csv
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
  Frozen candidate signal count (full 36mo): 447

==============================================================================
A/B/C. IN-SAMPLE / INTERNAL SPLIT / OUT-OF-SAMPLE
==============================================================================
NOTE: only ~3 years of history exist, so a true 3-way independent split (dev/validation/final-OOS) is not honestly available. IS-early and IS-late below are BOTH inside the original discovery window (not independent) -- an internal consistency check only. OOS is the genuinely held-out final 12 months, touched exactly once, here.
  IS-early (internal, first 12mo of IS): n=144  PF=1.12    DD= 3.30%  pm= 76.9%  P&L=$+3,023.04  WR=54.9%
  IS-late (internal, second 12mo of IS): n=157  PF=0.96    DD= 6.47%  pm= 46.2%  P&L=$-1,348.27  WR=46.5%
  IS-FULL (24mo, original selection window): n=301  PF=1.03    DD= 6.92%  pm= 64.0%  P&L=$+1,674.77  WR=50.5%
  OOS (final 12mo, held out, touched once): n=130  PF=0.94    DD= 5.35%  pm= 46.2%  P&L=$-1,590.01  WR=43.8%

==============================================================================
D. ROLLING WALK-FORWARD (each window reported independently)
==============================================================================
   fold test_start   test_end  trades  win_rate   pf      pnl  max_dd_pct
      0 2024-06-28 2024-12-28      82      46.3 0.97  -453.96        6.92
      1 2024-12-28 2025-06-28      70      44.3 0.82 -2954.47        6.47
      2 2025-06-28 2025-12-28      75      44.0 1.03   472.86        2.51
      3 2025-12-28 2026-06-28      61      41.0 0.73 -3296.45        3.71
SUMMARY       None       None     288      43.9 0.89 -6232.02        6.92
  4 fold(s) achievable from available history -- adequate fold count
  Consistency: 1/4 folds profitable (25.0%)

==============================================================================
E. TRADE-SEQUENCE MONTE CARLO (order-shuffle, drawdown/streak only -- not meaningless shuffled-P&L "confidence intervals")
==============================================================================
  Historical (actual, single path) max DD: -7.80%
  Historical (actual) longest losing streak: 11 trades
  MC median max DD:     -11.87%
  MC worst-5% max DD:   -17.01%
  MC losing-streak p50/p95: 8/12
  Risk of ruin (breach -20% at any point, 3000 shuffles): 0.87%

==============================================================================
F/12. COST + EXECUTION STRESS (spread multiples, entry delay)
==============================================================================
  normal (1.0x = 2.2p)         IS PF=1.03   P&L=$   +1,675  OOS PF=0.94   P&L=$   -1,590
  conservative (1.5x = 3.3p)   IS PF=0.92   P&L=$   -4,716  OOS PF=0.82   P&L=$   -4,455
  stress (2.0x = 4.4p)         IS PF=0.82   P&L=$  -10,648  OOS PF=0.72   P&L=$   -6,826
  extreme (3.0x = 6.6p)        IS PF=0.66   P&L=$  -21,263  OOS PF=0.55   P&L=$  -10,337
  1-bar entry delay (execute next bar's close instead of signal bar's close):
    IS PF=0.85   P&L=$   -7,220  OOS PF=0.78   P&L=$   -4,647

==============================================================================
G/11. PARAMETER SENSITIVITY -- neighbors reported, NONE selected
==============================================================================

  -- check_hour neighbors (thr/sl/tp held at frozen values) --
    check_hour=13: IS PF=0.82   DD=11.54%  OOS PF=0.97   P&L=$    -659
    check_hour=14: IS PF=0.84   DD=11.27%  OOS PF=0.93   P&L=$  -1,650
    check_hour=15: IS PF=1.03   DD= 6.92%  OOS PF=0.94   P&L=$  -1,590 <== FROZEN
    check_hour=16: IS PF=0.82   DD=11.74%  OOS PF=0.85   P&L=$  -3,105

  -- thr_atr neighbors (+/-10%/20%, others held at frozen values) --
    thr_atr=1.000: IS PF=1.03   DD= 7.28%  OOS PF=0.89   P&L=$  -3,398
    thr_atr=1.125: IS PF=1.01   DD= 8.11%  OOS PF=0.93   P&L=$  -2,020
    thr_atr=1.250: IS PF=1.03   DD= 6.92%  OOS PF=0.94   P&L=$  -1,590 <== FROZEN
    thr_atr=1.375: IS PF=1.03   DD= 7.16%  OOS PF=0.93   P&L=$  -1,576
    thr_atr=1.500: IS PF=0.93   DD= 9.67%  OOS PF=0.87   P&L=$  -2,713

  -- sl_atr neighbors (+/-10%/20%, others held at frozen values) --
    sl_atr=1.200: IS PF=0.99   DD= 8.75%  OOS PF=0.91   P&L=$  -2,705
    sl_atr=1.350: IS PF=1.02   DD= 7.63%  OOS PF=0.91   P&L=$  -2,373
    sl_atr=1.500: IS PF=1.03   DD= 6.92%  OOS PF=0.94   P&L=$  -1,590 <== FROZEN
    sl_atr=1.650: IS PF=1.03   DD= 6.15%  OOS PF=0.96   P&L=$    -807
    sl_atr=1.800: IS PF=0.99   DD= 6.61%  OOS PF=0.96   P&L=$    -870

  -- tp_atr neighbors (+/-10%/20%, others held at frozen values) --
    tp_atr=2.000: IS PF=1.02   DD= 6.63%  OOS PF=0.92   P&L=$  -2,064
    tp_atr=2.250: IS PF=1.0    DD= 6.86%  OOS PF=0.95   P&L=$  -1,157
    tp_atr=2.500: IS PF=1.03   DD= 6.92%  OOS PF=0.94   P&L=$  -1,590 <== FROZEN
    tp_atr=2.750: IS PF=1.05   DD= 6.84%  OOS PF=0.9    P&L=$  -2,491
    tp_atr=3.000: IS PF=1.05   DD= 6.76%  OOS PF=0.89   P&L=$  -2,833

==============================================================================
H. YEAR-BY-YEAR (concentration check)
==============================================================================
  2023: n= 74  WR= 51.4%  PF=0.99    P&L=$  -145.05
  2024: n=154  WR= 50.0%  PF=1.03    P&L=$  +989.61
  2025: n=144  WR= 43.8%  PF=0.88    P&L=$-3,639.57
  2026: n= 75  WR= 45.3%  PF=0.90    P&L=$-1,323.68
  Single best year contributes 0.0% of total P&L (broad)

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
2023-12-31 00:00:00+00:00      10  -859.90
2024-01-31 00:00:00+00:00      12   474.87
2024-02-29 00:00:00+00:00      10   271.08
2024-03-31 00:00:00+00:00      14 -1024.46
2024-04-30 00:00:00+00:00      11     3.36
2024-05-31 00:00:00+00:00      10   438.78
2024-06-30 00:00:00+00:00      14   106.01
2024-07-31 00:00:00+00:00      12   246.29
2024-08-31 00:00:00+00:00      13 -3358.65
2024-09-30 00:00:00+00:00      14 -1905.37
2024-10-31 00:00:00+00:00      15  1926.87
2024-11-30 00:00:00+00:00      11  2626.33
2024-12-31 00:00:00+00:00      18  1184.50
2025-01-31 00:00:00+00:00       8   343.99
2025-02-28 00:00:00+00:00       9 -2715.44
2025-03-31 00:00:00+00:00      16  -374.49
2025-04-30 00:00:00+00:00       9   163.16
2025-05-31 00:00:00+00:00      12  -787.21
2025-06-30 00:00:00+00:00      14  -192.81
2025-07-31 00:00:00+00:00      17  -486.41
2025-08-31 00:00:00+00:00      14  1521.51
2025-09-30 00:00:00+00:00      13  -381.74
2025-10-31 00:00:00+00:00       8  -134.70
2025-11-30 00:00:00+00:00      12   987.66
2025-12-31 00:00:00+00:00      12 -1583.09
2026-01-31 00:00:00+00:00       9  -936.03
2026-02-28 00:00:00+00:00       9  1288.49
2026-03-31 00:00:00+00:00      16 -1125.52
2026-04-30 00:00:00+00:00      11  -583.21
2026-05-31 00:00:00+00:00       9  -965.50
2026-06-30 00:00:00+00:00       8   364.28
2026-07-31 00:00:00+00:00      11   346.34
2026-08-31 00:00:00+00:00       2   287.47
  Months with zero trades: 0 / 38

==============================================================================
J. DRAWDOWN ANALYSIS (historical single path)
==============================================================================
  Max drawdown: -7.80%  ($7,968.12 at worst)
  Avg drawdown duration: 54.2 trades
  Max drawdown duration: 215 trades
  Recovery factor (net P&L / max peak-to-trough $): -0.70

==============================================================================
K. MFE / MAE ANALYSIS
==============================================================================
  Median MFE: 0.82R   Median MAE: 0.75R
  Winners: median MFE captured = 62% of the best available excursion
  Losers: median MAE vs 1R stop = 1.04R (should be ~1.0R if stops are the binding constraint)

==============================================================================
L. LOSING-STREAK ANALYSIS (see section E for historical + MC figures)
==============================================================================
  Historical longest losing streak: 11 consecutive trades
  Monte Carlo p50/p95 across 3000 order-shuffles: 8/12

==============================================================================
M. TRADE FREQUENCY
==============================================================================
  Total trades: 447 over 1133 days (2.76/week, 11.84/month)
  Entry hour distribution: {15: 447} (should be 100% at hour 15 by construction)
  Days between trades: median=2.0  max=12

==============================================================================
BENCHMARK: random-direction null model (permutation test)
==============================================================================
Buy-and-hold is not a meaningful null model for an intraday FX directional strategy. The correct null here is: same entry bars, same ATR-based SL/TP construction, but the BUY/SELL direction is a coin flip instead of the USDJPY-proxy signal. This isolates whether the DIRECTION-PICKING logic adds value beyond generic session-time trading with these SL/TP mechanics.
  real_pf: 0.953
  real_pnl: -4118.69
  null_pf_mean: 0.796
  null_pf_p50: 0.791
  null_pf_p95: 0.944
  null_pnl_mean: -18146.14
  real_pf_percentile_vs_null: 95.8
  real_pnl_percentile_vs_null: 95.7
  n_perm: 1000

  Interpretation: the real strategy's profit factor exceeded 95.8% of 1000 random-direction shuffles of the SAME entry bars/SL/TP construction.

==============================================================================
FINAL VALIDATION SCORECARD
==============================================================================
Test                                   Result                                             Pass/Fail
In-sample (24mo)                       PF=1.03 DD=6.92% pm=64.0%                          FAIL
Out-of-sample (12mo, held out)         PF=0.94 P&L=$-1,590                                FAIL
Walk-forward                           1/4 folds profitable                               FAIL
Monte Carlo (risk of ruin)             0.87%                                              PASS
Cost stress (2x spread)                IS PF=0.82 OOS PF=0.72                             FAIL
Parameter sensitivity (plateau)        thr/sl/tp PF range < 0.6 spread: True              PASS
Year consistency                       best year = 0% of total P&L                        PASS
Direction vs random (permutation)      beats 96% of random-direction shuffles             PASS
Drawdown                               -7.80%                                             PASS

5/9 checks pass.  FINAL STATUS: FAILED

Final summary logged as EXP-036