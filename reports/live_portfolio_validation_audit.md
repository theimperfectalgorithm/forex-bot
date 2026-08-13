# Forensic Research Audit — Live Portfolio Validation

**AUDIT ONLY. No strategy, entry/exit logic, risk, live configuration,
or account was modified, deployed, paused, or optimized in the
production of this report.** Every major conclusion cites an exact
source file, commit hash, or experiment ID. Where the audit trail does
not support a conclusion, this report says so rather than filling the
gap.

**Central question:** on what evidence did we decide each current
strategy was good enough to trade with real money, and did that
evidence meet a rigorous standard **at the time the decision was made**
— not judged with hindsight from later research?

---

## 0. Deployment timeline (established once, applies to all 8)

| Event | Commit | Date | Source |
|---|---|---|---|
| Risk/orchestrator infrastructure for the new book | a17520f, edbe008 | 2026-07-05 10:22:17–10:22:34 +0530 | `git log` |
| Strategies parameterized (ARB generalized, AMR added) | 8d4ddf5 | 2026-07-05 10:22:48 +0530 | `git log` |
| **ARB+AMR book deployed to demo (7 slots)** | **e1f3ec9** | **2026-07-05 10:23:03 +0530** | `git log`, commit message |
| Research scripts (phases 1-7) + status report committed | 6d3ddc9 | 2026-07-05 10:23:22 +0530 | `git log` |
| **Monday Drift deployed to demo (8th slot)** | **43284a6** | **2026-07-06 08:15:18 +0530** | `git log`, commit message |
| News blackout gate added | 708f12c | 2026-07-06 08:15:18 +0530 | `git log` |
| Server-time session-gating bug fixed (ARB/Monday were "dead live," AMR truncated, before this fix) | e75d680 | 2026-07-07 10:01:35 +0530 | `git log`, project memory "Server-time fix" |
| Multi-account (demo + prop clone) infrastructure added | 84358fd | 2026-07-09 09:46:40 +0530 | `git log` |
| 5ers $5,000 challenge account goes live | (exact commit not identified; PROJECT_REPORT.md §"rev 3" states it happened after 2026-07-15 and before 2026-08-11) | between 2026-07-15 and 2026-08-11 | `PROJECT_REPORT.md`, commit 3abd9a7 message |
| locked_pairs allowlist added (isolates prop clone pair set) | 95492ed | 2026-07-21 15:59:54 +0530 | `git log` |
| **Book diverges: GBPJPY ARB + XAUUSD ARB demoted on 5ers, risk_scale cut to 0.5** | (manual `local_config.yaml` edit, not a git commit — per PROJECT_REPORT.md) | 2026-07-31 | `PROJECT_REPORT.md` §3 |

**Important forensic caveat on strict sequencing:** the deployment
commit (e1f3ec9, 10:23:03) and the research-documentation commit
(6d3ddc9, 10:23:22) are **19 seconds apart**, with deployment committed
*first*. This means the git record alone cannot prove the research was
committed to version control before the deployment config was. The most
defensible reading, based on the commit messages' own claims ("walk-forward
validated 2026-07-04/05, ~500 backtests") and the existence of the
phase 1-7 scripts and their referenced output files
(`data/strategy_matrix_results.csv`, `data/phase2_results.csv`,
`data/phase3b_results.csv`), is that the **actual analysis work** was
done over 2026-07-04/05 and **all resulting code was committed in one
end-of-session batch** — a common and forgivable pattern for
single-operator local development, not evidence of research being
fabricated after the fact. But it means this audit cannot independently
verify strict temporal ordering from git alone; the exact validation
numbers embedded in the YAML comments and PROJECT_REPORT.md are treated
as the authoritative decision record instead.

---

## 1. GBPJPY ARB

**1. First proposed:** phase 2 (`src/phase2_meanrev_arb_search.py`), committed 2026-07-05, but the analysis itself is dated 2026-07-04 per the strategy's own YAML comment.
**2. Hypothesis:** Asian-session (00:00-07:00) range breakout — first H1 close beyond the Asian range edge during London-open hours (7,8) continues in that direction.
**3. Entry rules:** H1 close beyond Asian-range edge, hours {7,8}, `signals_arb_p` (`src/phase2_meanrev_arb_search.py`).
**4. Exit rules:** SL at opposite range edge; TP = `tp_multiplier × range`; runs to SL/TP, Friday close.
**5. Stop-loss methodology:** opposite edge of the measured Asian range.
**6. Take-profit methodology:** `tp_multiplier` (2.0, selected in-sample) × range width.
**7. Position sizing/risk:** 0.5% per trade, compounding balance × risk_pct / SL distance.
**8. Session/time window:** Asian range 00:00-07:00 server; breakout check hours 7-8 server.
**9. Timeframe:** H1.
**10. Pairs tested:** GBPJPY specifically selected out of a 95-combination grid across 9+ majors/crosses (`src/strategy_matrix_backtest.py` + `phase2_meanrev_arb_search.py`) — **GBPJPY was the ONLY pair-strategy combination that passed the full grid**, per `pairs/GBPJPY_asianrange.yaml`'s own comment.
**11. Years tested:** 3 (IS Jul 2023-Jun 2025, OOS Jul 2025-Jun 2026).
**12. Data source:** MT5 historical H1 bars (per project-wide convention).
**13-16. Spread/slippage/swap/execution realism:** spread paid per this project's stated backtest convention ("spread paid" — `PROJECT_REPORT.md` §4 methodology line); explicit slippage model, swap/commission modeling: **NOT AVAILABLE** in the original phase-2 artifact (no report file exists for phase 2 specifically — only the YAML comment and PROJECT_REPORT.md summary survive as artifacts; the original script's console output was not archived as a report).
**17. Lookahead bias:** the frozen ARB signal function (`signals_arb_p`) computes the Asian range from bars strictly before the breakout check hour — consistent with the no-lookahead convention this project later formalized in `src/alignment_utils.py` (built 2026-08-11, i.e. **after** this strategy's deployment — see §"Data integrity" below for what this means for pre-live assurance).
**18. Survivorship/selection bias:** real risk here — GBPJPY was selected as "the ONLY combination that passed" out of 95 tested. This is disclosed openly in the strategy's own comment, which is good practice, but a 1-in-95 selection is exactly the kind of result multiple-testing discipline should treat cautiously. No FDR/multiple-testing correction is documented as having been applied at the time.
**19-20. Optimization:** yes — `tp_multiplier` was searched (grid: 1.5/2.0/2.5, per the YAML comment's "Robustness: OOS-positive across the whole TP grid"); H4 filter on/off was also compared. Small parameter space (roughly 2×3), not an extensive search.
**21. IS/OOS split:** yes — Jul23-Jun25 IS / Jul25-Jun26 OOS, standard project convention.
**22. Walk-forward (rolling multi-fold):** **NOT AVAILABLE** — the original validation was a single IS/OOS split, not a rolling walk-forward. (Note: PROJECT_REPORT.md's overall project narrative refers to "~530 walk-forward backtests," which in this project's usage means the aggregate count of parameter-grid IS/OOS runs across all research phases, not a rolling multi-fold test of GBPJPY ARB specifically.)
**23. Monte Carlo before live:** portfolio-level Monte Carlo was run (phase 6/7, "Book B MC 63% pass/0% bust," "Book B+ MC 83%/2%") but this is a **portfolio-level** MC, not a strategy-specific trade-sequence MC for GBPJPY ARB alone.
**24. Parameter sensitivity:** yes, for `tp_multiplier` (3-point grid, OOS-positive across all three) — this is the one dimension explicitly tested.
**25. Cost stress (1.5x/2x spread):** **NOT AVAILABLE** for the original validation.
**26. Execution-delay sensitivity:** **NOT AVAILABLE** for the original validation.
**27. Multi-year/regime test:** partially — 3 years span IS+OOS, but no explicit year-by-year breakdown is preserved in any surviving artifact from the original validation.
**28. Multi-pair test:** yes, implicitly (95-combination grid), but GBPJPY was the only pass — see survivorship note above.
**29. Genuine OOS evidence:** yes — 63 OOS trades, PF 1.19, +$3,676 (per YAML comment), a real held-out period not touched during selection.
**30. Live version = backtested version:** **materially, yes, at the parameter level** (`tp_multiplier: 2.0, h4_filter: false, risk_percent: 0.5` in `pairs/GBPJPY_asianrange.yaml` matches the validated variant exactly) — but see the server-time bug (2026-07-05 to 2026-07-07) and the current risk_scale 0.5 cut on 5ers (since 2026-07-31) as documented deviations from the pure backtested version.

### Evidence classification

- **PRE-LIVE:** phase 2 discovery + IS/OOS split + TP-multiplier sensitivity grid + H4-filter comparison + portfolio-level MC (phase 6/7) + explicit disclosure of "only 1-of-95 combination passed."
- **POST-LIVE:** phase 20 diagnostics (EXP-066, this session) — full-history reconstruction, regime classification "C. STABLE," max drawdown -$6,732.75, max losing streak 8. **This is POST-LIVE evidence and does not retroactively justify the original 2026-07-05 deployment decision**, even though it is useful for current confidence.
- **PROSPECTIVE LIVE:** actual demo/prop trade history since 2026-07-05 (0 wins/3 losses on the funded 5ers account specifically as of the 2026-07-31 demotion, per `PROJECT_REPORT.md` §3).

### PRE-LIVE STATUS: **PROMISING**

Real IS/OOS split, real out-of-sample evidence, a disclosed and honest
survivorship caveat, and a small but real parameter-sensitivity check.
Missing: walk-forward, cost stress, execution-delay sensitivity,
year-by-year breakdown, strategy-specific Monte Carlo. This clears a
meaningful bar but not a complete one — "PROMISING," not "WELL
VALIDATED," is the accurate pre-live characterization.

---

## 2. CADJPY ARB

**1-9.** Identical mechanism to GBPJPY ARB (same `signals_arb_p`, same params: `tp_multiplier 2.0`, no H4, 0.5% risk, Asian range, H1). First tested phase 6 (`src/phase6_portfolio_model.py`), dated 2026-07-05 per `pairs/CADJPY_asianrange.yaml`.
**10. Pairs tested:** this is the key distinction the user's question asked about — **CADJPY was NOT part of the original 95-combination grid search that selected GBPJPY.** Per the YAML comment: "validated 2026-07-05 (phase 6)... The JPY-cross session thesis **replicating** on a pair it was never tuned on." This is a genuinely different, and in one sense *stronger*, kind of evidence than GBPJPY's: it is an out-of-family replication test (does the already-fixed, already-selected specification work on a new instrument with zero re-tuning?), not another draw from the same selection-biased search. **CADJPY ARB's inclusion is independently evidenced, not merely "included because GBPJPY worked" — but it is a lighter-weight validation (one confirmatory test, not a full grid search) than GBPJPY's original discovery process.**
**11-21.** Same 3-year IS/OOS split; IS PF 1.15/DD 7.7%, OOS PF 1.38/+$6,424 (YAML comment, `PROJECT_REPORT.md` §3 row 2).
**19-20. Optimization:** **no new optimization** — the specification (`tp_multiplier=2.0`, no H4) was carried over unchanged from GBPJPY's already-selected parameters. This is actually a point in CADJPY's favor from an overfitting standpoint (zero fresh parameter search on this pair) but means it inherits any weakness in the original GBPJPY selection if the underlying mechanism doesn't generalize.
**22-26.** Walk-forward, strategy-specific Monte Carlo, cost stress, execution-delay: **NOT AVAILABLE**, same as GBPJPY.
**29. Genuine OOS evidence:** yes, real held-out OOS period, PF 1.38.
**30. Live match:** yes — YAML params match exactly.

### Evidence classification

- **PRE-LIVE:** phase 6 replication test, IS/OOS split, portfolio-level MC (as part of Book B/B+).
- **POST-LIVE:** phase 20 diagnostics (EXP-067) — inverted-U regime pattern shared with GBPJPY ARB, max drawdown -$7,933.62 (the deepest of any of the 8 strategies), max losing streak 10.
- **PROSPECTIVE LIVE:** actual demo/prop history; CADJPY ARB was **not** part of the 2026-07-31 demotion — it remains active on both demo and 5ers.

### PRE-LIVE STATUS: **PROMISING**

A genuine, independent (not parameter-searched) replication result with
real OOS evidence — methodologically clean in the sense that no new
optimization occurred, but the underlying validation depth (no
walk-forward, no cost stress, single confirmatory test rather than a
full search) is lighter than GBPJPY's. Same missing-evidence profile as
GBPJPY ARB.

---

## 3. XAUUSD ARB

**This is the strategy the audit brief specifically flagged, and the
evidence here is unambiguous and already self-documented in the
project's own files — no reconstruction needed.**

**1-9.** Same ARB mechanism, phase 7 (`src/phase7_exits_calendar_gold.py`), dated 2026-07-05. `tp_multiplier=1.5`, `min_range_pips=30` (gold-specific), 0.25% risk (half of the JPY-cross ARB slots).
**10-21.** IS PF 1.45/DD 2.9%; **OOS PF 1.05 ("flat," +$607)** — per `pairs/XAUUSD_asianrange.yaml`'s own comment: **"PROVISIONAL slot... IS PF 1.45/DD 2.9% but OOS only flat-positive (PF 1.05, +$607)."**

**Was the provisional status known BEFORE deployment?** **Yes, unambiguously.** The word "PROVISIONAL" is in the YAML file's own comment, committed in the same commit (e1f3ec9, 2026-07-05) that activated the strategy. This was not a later discovery — it was disclosed, in writing, at the moment of deployment.

**Was there sufficient evidence to justify even 0.25% risk?** This is a judgment call the audit can characterize but not resolve definitively. The evidence that existed: IS-strong (PF 1.45), OOS-flat-but-not-negative (PF 1.05, marginally profitable). The project's own response to this weak evidence was **proportionate, not reckless**: (a) risk was set at 0.25%, the smallest slot size in the book, specifically because of the weaker evidence; (b) the YAML comment includes a **built-in review trigger**: "Review after 2-3 months of demo -- drop if live tracks the flat OOS year." This is disciplined risk management around a known-weak signal, not an undisclosed risk.

**Was XAUUSD actually validated, or included because ARB worked elsewhere?** Based on the evidence trail, it was **tested independently** (phase 7, its own IS/OOS run with gold-specific parameters: different TP multiplier, gold-specific `min_range_pips`) — it is not a copy-paste inclusion. But the OOS result (PF 1.05, "flat") does not clear this project's own stated acceptance bar used elsewhere (PF > 1.3, per `PROJECT_REPORT.md` §4's stated methodology: "criteria PF>1.3 / DD<8% / ≥60% profitable months / positive OOS"). **By the project's own documented acceptance criteria, XAUUSD ARB's OOS result (PF 1.05) does NOT meet the bar that was applied elsewhere in the same research program.** It was deployed anyway, explicitly labeled provisional, at reduced size, with a review trigger — a conscious exception, not an oversight.

**Post-deployment, real-world outcome (prospective live evidence, reported for completeness — not counted toward the pre-live judgment):** XAUUSD ARB was demoted from the 5ers account on 2026-07-31 — **not for performance reasons**, but because the 5ers broker has no H1 gold data at the bot's 04:45 UTC daily prep check (confirmed via a week of `WARNING: no H1 data for today` log lines) and gold's SL distance at 0.25% risk on a $5,000 account requires a lot size below the broker's 0.01 minimum. **This is a pre-deployment infrastructure/compatibility gap that should ideally have been checked before risking capital on the funded account** — the backtest never modeled "does this specific broker have this data, and can this account size this trade at all" — but it is a distinct issue from the strategy's statistical edge being wrong. (`PROJECT_REPORT.md` §3.)

### PRE-LIVE STATUS: **PROVISIONAL / WEAK EVIDENCE BEFORE LIVE**

The project's own label ("PROVISIONAL") is the correct classification —
this report is not overriding it, only confirming it against the
project's own stated acceptance criteria. The OOS PF (1.05) does not
meet the >1.3 bar used elsewhere in this project's research. Deployment
proceeded anyway, with disclosed awareness, reduced size, and a review
trigger — a defensible risk-management posture around a known-weak
result, but the underlying evidence itself was insufficient by the
project's own standard.

---

## 4. GBPJPY AMR

**1. First proposed:** phase 3 (`src/phase3_session_structure_search.py`), the original AMR discovery — "AMR-JPY discovery (all 36 variants OOS-positive)" per `PROJECT_REPORT.md` §4. Refined in phase 3b (`src/phase3b_amr_jpy_refine.py`), both dated 2026-07-05.
**2. Hypothesis:** mean reversion during quiet Asian hours — M15 z-score vs. SMA20; |z| ≥ threshold fades back toward the mean.
**3-4. Entry/exit:** `signals_amr_v` — z ≤ -threshold → BUY (TP = SMA20, SL = `sl_multiplier` × distance to SMA20); z ≥ +threshold → SELL, mirrored. Force-flat at server 07:00.
**5-6. SL/TP methodology:** SL = `sl_multiplier × |close - SMA20|` at signal time; TP = SMA20 itself (mean-reversion target).
**7. Risk:** 0.25%.
**8-9. Session/timeframe:** Asian hours, entries 00:00 to `entry_end_hour` (4 for GBPJPY), M15.
**10. Pairs tested:** phase 3's discovery ran **36 variants, all OOS-positive** across the JPY-cross family; phase 3b refined the grid (36 more variants) specifically on GBPJPY/EURJPY/AUDJPY/CADJPY.
**11-18.** 3-year IS/OOS; IS PF 1.16/DD 4.53%/68% profitable months, OOS PF 2.03/+$17,529 (YAML comment, `data/phase3b_results.csv`). Spread paid per project convention; slippage/commission: **NOT AVAILABLE** in a surviving artifact.
**19-20. Optimization:** yes — `z_threshold` and `sl_multiplier` were searched across a grid (36 variants in phase 3b) — this is a real, non-trivial parameter search, larger than ARB's.
**21. IS/OOS split:** yes.
**22-26. Walk-forward/MC/cost-stress/execution-delay:** **NOT AVAILABLE** for the original validation (portfolio-level MC only, via phase 6/7).
**27-28. Multi-year/multi-pair:** multi-pair yes (4 JPY crosses); year-by-year breakdown from the original validation: **NOT AVAILABLE** as a surviving artifact.
**29. OOS evidence:** yes, real.
**30. Live match:** parameters match (`z_threshold: 2.5, sl_multiplier: 1.25, entry_end_hour: 4` in `pairs/GBPJPY_asianrev.yaml`) — **but with one explicit, self-documented material gap**: the YAML comment states *"the phase-7 BE@0.75R exit refinement is backtest-only for now -- live breakeven handling stays with monitor_positions' existing 25-pip logic until demo data justifies it."* **This means the live exit logic does not fully match what a later research phase (phase 7) explored — the live version uses an older, different breakeven rule than the one investigated in research.** This is disclosed, not hidden, but it is exactly the kind of live-vs-backtest divergence this audit was asked to surface.

### Evidence classification

- **PRE-LIVE:** phase 3 discovery (36 OOS-positive variants) + phase 3b refinement grid (36 more) + IS/OOS split + portfolio MC.
- **POST-LIVE:** phase 20 diagnostics (EXP-069) — pooled 4-bin regime table looked "C. STABLE" at first; **phase 21 mechanism research (EXP-078) then downgraded this to "E. OTHER / INCONCLUSIVE"** after finding the volatility effect reverses sign under high-trend conditions. **Per the audit brief's explicit instruction, this downgrade affects CURRENT confidence, not the original justification for deployment** — the original deployment decision was made in July on the phase 3/3b evidence above, which did not include (and could not have included) the phase 21 finding.
- **PROSPECTIVE LIVE:** ongoing demo/5ers trade history.

### PRE-LIVE STATUS: **PROMISING**

A real, sizable discovery-and-refinement process (36+36 variants, all
OOS-checked) with genuine multi-pair replication, but no walk-forward,
no strategy-specific Monte Carlo, no cost stress, and — like all AMR
pairs — the strategy's own live-deployment comment explicitly frames it
as a **"DEMO FORWARD-TEST candidate"** requiring 2-3 months of demo
confirmation before challenge use (see §5 "Deployment discipline vs.
actual timeline" below).

---

## 5. EURJPY AMR

Same family, same phase 3/3b provenance, same 36+36-variant discovery/
refinement process. Params: `z_threshold: 2.0, sl_multiplier: 1.5,
entry_end_hour: 6`. IS PF 1.10/DD 9.24%/60% profitable months, OOS PF
1.47/+$15,734 (YAML comment).

**Same evidence profile and same gaps as GBPJPY AMR** (no walk-forward,
no strategy-specific MC, no cost stress, same "DEMO FORWARD-TEST
candidate" framing, same live-BE-logic gap per the shared comment
structure across all AMR YAMLs).

**Post-live evidence, explicitly not blended into the pre-live judgment
per instruction:** phase 20 (EXP-070) found an inverted-U regime
pattern (not monotonic); phase 21 (EXP-079) found the volatility effect
is sign-unstable across trend terciles — same downgrade to "E. OTHER /
INCONCLUSIVE" as GBPJPY AMR, discovered entirely after the July
deployment.

### PRE-LIVE STATUS: **PROMISING**

Same reasoning as GBPJPY AMR — real IS/OOS evidence and a genuine
multi-pair discovery process, but missing the deeper robustness testing
(walk-forward, MC, cost stress) that this project only started applying
in phases 20+, all of which postdate deployment.

---

## 6. AUDJPY AMR

**This pair received the deepest post-live scrutiny of any strategy in
this portfolio, so the pre-live/post-live separation matters most here.**

### What existed BEFORE deployment (2026-07-05)

Same phase 3/3b provenance as the other AMR pairs. Params:
`z_threshold: 2.0, sl_multiplier: 1.5, entry_end_hour: 4`. IS PF
1.17/DD 4.80%/60% profitable months, OOS PF 1.23/+$8,616 (YAML comment).
**This is the weakest original OOS PF of the four AMR pairs at
deployment time** (1.23, vs. GBPJPY's 2.03, EURJPY's 1.47, CADJPY's
1.35) — a fact the pre-live record itself shows, not something later
research revealed.

No walk-forward, no strategy-specific Monte Carlo, no cost stress, no
execution-delay sensitivity existed before deployment — same gap
profile as the sibling AMR pairs.

### What was discovered AFTER deployment (all POST-LIVE, does not
retroactively justify the July decision)

- **Phase 20 (EXP-071, this session, 2026-08-11):** full-history
  reconstruction found a clean monotonic volatility-regime decline
  (LOW PF 1.35 → HIGH PF 0.85, net losing), confirmed in 3 of 3
  testable years — the strongest regime finding of any of the 8
  strategies.
- **Phase 21 (EXP-076):** this relationship **survives conditioning on
  trend** in all three trend terciles — the strongest causal (not just
  correlational) evidence produced anywhere in this project. Also found
  AUDJPY AMR's SELL leg is independently net-losing (PF 0.699 across
  240 trades) while the BUY leg is strong (PF 1.591 across 412 trades).
- **Phase 22 (EXP-082-086):** a frozen "BUY-only" candidate was
  confirmatory-tested with a genuine chronological TRAIN/VALIDATION/OOS
  split and classified **SUPPORTED** (not VALIDATED) — large,
  consistent OOS improvement, survives 2x spread stress, but the OOS
  bootstrap confidence interval still crosses zero.
- **Phase 23 (EXP-087-089):** a final historical validation gate was
  attempted and correctly classified **"B. INSUFFICIENT FRESH DATA"**
  — no calendar time had passed since phase 22's own OOS window closed,
  so no further historical validation was possible without reusing
  already-used evidence.
- **Phase 24 (EXP-090-091):** a prospective forward-validation tracker
  for the BUY-only candidate was built and started (0 trades collected
  as of this audit).

**Was AUDJPY AMR sufficiently validated before it entered the live
portfolio, or did we only discover its strongest evidence after it was
already trading?** **The latter, unambiguously.** The pre-live evidence
(OOS PF 1.23, the weakest of the four AMR pairs) was real but modest.
**The strongest, most rigorous evidence this project has ever produced
about AUDJPY AMR — the trend-conditioning result, the directional
asymmetry, the confirmatory experiment — was all generated after
deployment, in this current session (phase 20-24, 2026-08-11), roughly
5 weeks after the strategy went live.** This is exactly the pattern the
audit brief asked to be surfaced: the original deployment decision was
made on comparatively thin evidence; the deep evidence came later and
should inform *current* confidence and the upcoming August 25
checkpoint, but it cannot be retroactively credited to the original
July decision.

**Exact status of the BUY-only hypothesis on the live account, verified
against actual code — not assumed:** `pairs/AUDJPY_asianrev.yaml`
(inspected directly for this audit) contains no BUY-only restriction,
filter, or direction constraint of any kind. `strategies/asian_hours_reversion.py`
implements both BUY and SELL branches per the frozen `signals_amr_v`
logic, matching the original 2026-07-05 specification. **The BUY-only
candidate exists only as research code
(`src/phase22_audjpy_amr_confirmatory.py` `Model B`) and a data-
collection tracker (`src/amr_forward_tracker.py`) — it has NOT been
implemented in the live strategy class, live config, or demo/prop
account.** The currently-trading AUDJPY AMR is the original, unmodified,
both-directions strategy.

### PRE-LIVE STATUS: **PROMISING BUT INSUFFICIENT**

Weakest original OOS PF of the AMR family (1.23), same missing-tests
profile as its siblings, and — as this audit's own forensic
reconstruction shows — the evidence that would make a stronger case for
this specific pair's edge did not exist at deployment time. This is not
a claim that AUDJPY AMR is currently a bad strategy (current evidence,
per phase 20-22, is actually the strongest in the AMR family) — it is a
claim that the **original July decision** to deploy it rested on
thinner evidence than the AMR family average, a fact only visible in
hindsight.

---

## 7. CADJPY AMR

Same phase 3/3b provenance. Params: `z_threshold: 2.0, sl_multiplier:
1.5, entry_end_hour: 4` (identical to AUDJPY AMR's parameters). IS PF
1.10, OOS PF 1.35/+$4,792 (`pairs/CADJPY_asianrev.yaml` comment,
`PROJECT_REPORT.md` §3 row 7).

**Original pre-live evidence:** same gap profile as the rest of the AMR
family — real IS/OOS split, real 36+36-variant discovery/refinement
process, no walk-forward/MC/cost-stress.

**Post-live evidence (explicitly not blended into the pre-live
judgment):** Phase 20 (EXP-072) found a clean monotonic regime decline
(LOW PF 1.85 → HIGH PF 0.76, net losing), the largest LOW-vs-HIGH swing
of any strategy studied, confirmed in its 2 testable years. Phase 21
(EXP-077) found this is **NOT a pure volatility effect but a
volatility×trend INTERACTION** — the effect is near-zero within
low-trend trades and only appears within normal/high-trend trades.

**On the distinction the audit brief specifically asked about — "volatility
filter" vs. "volatility × trend interaction":** this report treats them
as materially different findings and has not treated the interaction
observation as a validated trading filter. No filter of any kind — volatility,
trend, or combined — has been implemented for CADJPY AMR. Phase 21's own
report (`reports/amr_regime_mechanism.md` §19) explicitly recommended a
confirmatory experiment **only for AUDJPY**, not CADJPY, specifically
because "CADJPY's interaction structure means a simple threshold filter
would be mis-specified" — no confirmatory filter experiment has been
run for CADJPY AMR, and none should be inferred from this audit.

### PRE-LIVE STATUS: **PROMISING**

Same original-evidence profile as the other AMR pairs — real but
missing deeper robustness tests. The later interaction finding is
genuinely new information (not merely restating known weakness) but,
per the brief's instruction, is not being treated as a validated filter
or as retroactive justification/condemnation of the original deployment.

---

## 8. GBPUSD Monday Drift

**1. First proposed:** discovered via a calendar screen in phase 7
("Monday drift t=+3.3 IS / +4.0 OOS"), bounded and validated as its own
strategy in phase 8 (`src/phase8_monday_validation.py`), 2026-07-05.
**2. Hypothesis:** GBPUSD exhibits a positive Monday-session drift.
**3-6. Entry/exit/SL/TP:** BUY at the close of Monday's 00:00 H1 bar
(server time); SL = `sl_atr_mult × ATR20d`; TP = `tp_atr_mult × ATR20d`;
force-flat 21:00 UTC Monday.
**7. Risk:** 0.25%.
**8-9. Session/timeframe:** Monday only, H1 (with a daily-ATR lookback).
**10. Pairs tested:** GBPUSD (selected), **EURUSD run explicitly as a
control** — "EURUSD control weak → GBPUSD-specific effect," a genuine
falsification attempt, not just a single-pair fishing result.
**11. Years tested:** same 3-year window.
**12-16.** Same data-source/spread convention as the rest of the project;
slippage/commission modeling: **NOT AVAILABLE** as a surviving artifact.
**17-18. Lookahead/selection bias:** the EURUSD control is a real
mitigant against pure data-mining (a spurious pattern would likely also
show up, or show up differently, on a similar pair — here it did not
replicate, strengthening the GBPUSD-specific case).
**19-20. Optimization:** yes — `sl_atr_mult`/`tp_atr_mult` were grid-searched ("Robust across the grid; OOS PF 2.9-3.1" per the YAML comment) — a small grid, and the result was reported as stable across it, not an isolated peak.
**21. IS/OOS split:** yes — **103 Mondays IS, 52 Mondays OOS.**
**22-26. Walk-forward/MC/cost-stress/execution-delay:** **NOT AVAILABLE** for the original validation, same as every other strategy in this portfolio.
**27. Multi-year:** span-tested (3 years), no year-by-year breakdown preserved from the original validation.
**28. Multi-pair:** EURUSD control only (as a falsification check, not a search for more winners).
**29. OOS evidence:** yes — IS PF 1.97/DD 0.66%/66.7% profitable months (103 Mondays); OOS PF 3.08/+$2,573/DD 0.42% (52 Mondays) — **the strongest pass of the entire project**, per both the YAML comment and `PROJECT_REPORT.md` §3 ("**strongest pass: IS PF 1.97/DD 0.66%/66.7%pm; OOS PF 3.08/DD 0.42%**").
**30. Live match:** yes — `sl_atr_mult: 1.25, tp_atr_mult: 1.0` in `pairs/GBPUSD_monday.yaml` matches exactly.

### Sample size — the explicit concern the audit brief raised

**52 OOS trades (Mondays) is the smallest OOS sample of any strategy in
this portfolio** — smaller than every ARB and AMR slot. At ~52
trades/year, roughly one trade per week, this is an inherently
low-frequency edge. The OOS profit factor (3.08) and drawdown (0.42%)
are exceptionally strong numbers, but they rest on a sample size that
is genuinely thin by conventional statistical standards (no explicit
significance test, e.g. a t-test or bootstrap CI, is preserved from the
original validation as a surviving artifact — the commit message cites
"t=+3.3 IS / +4.0 OOS," which are real reported t-statistics from the
phase 7 calendar screen, but no confidence interval or Monte Carlo
result specific to Monday Drift survives as an artifact). **This
audit's judgment: the sample size is a genuine, disclosed limitation —
52 OOS trades is not "insufficient to justify deployment" outright
(the t-statistics reported are large, and the EURUSD control adds real
falsification value), but it is thinner evidence than a naive read of
"strongest pass of the project" would suggest, and it means Monday
Drift's results are more sensitive to a small number of individual
trades than any other strategy in the book.**

### PRE-LIVE STATUS: **PROMISING**

Strong OOS statistics, a genuine cross-pair falsification control, and
a stable parameter grid — but, like every strategy in this portfolio,
missing walk-forward, strategy-specific Monte Carlo, and cost stress,
and additionally carrying the smallest trade sample of the book. "The
strongest pass of the project" is an accurate characterization of the
*numbers*, but the *sample size* means this report does not upgrade it
to "WELL VALIDATED" on the strength of those numbers alone.

---

## Cross-strategy pattern: NONE of the 8 original validations included

- Walk-forward (rolling multi-fold) testing
- Strategy-specific Monte Carlo (only portfolio-level MC existed)
- Cost stress (1.5x/2x spread)
- Execution-delay sensitivity
- A preserved year-by-year performance breakdown

**This is the single most important portfolio-level finding of this
audit.** Every one of these five test types was introduced into this
project's methodology only in phases 15+ (this project's later research,
almost entirely from 2026-08-11 onward per `experiments/experiments.csv`,
which itself only begins at EXP-001 around the same time) — **all of it
postdates the entire live portfolio's deployment by roughly five
weeks.** The original validation standard actually used in July was: IS/OOS
split, a modest parameter grid, portfolio-level Monte Carlo, and (for
2 of 8 strategies) a cross-pair replication or control test. That was
the real bar applied at the time — not the much more rigorous standard
(null tests, bootstrap CIs, walk-forward, trend-conditioning) this
project built up over the following month of research.

---

## Live implementation vs. backtested version — comparison table

| Strategy | Pair/TF/session/entry/SL/TP/risk | Verdict | Notes |
|---|---|---|---|
| GBPJPY ARB | IDENTICAL | **IDENTICAL** | `tp_multiplier: 2.0, h4_filter: false, risk_percent: 0.5` matches validated variant exactly. |
| CADJPY ARB | IDENTICAL | **IDENTICAL** | Same signal function, unmodified params. |
| XAUUSD ARB | IDENTICAL config; **broker/execution reality diverges materially** | **MATERIAL DIFFERENCE (execution, not logic)** | Strategy logic matches; the 5ers broker cannot supply the data or minimum lot size the backtest assumed — see §3 above. |
| GBPJPY AMR | Entry/SL/TP config identical; **live BE handling ≠ researched BE handling** | **MINOR-TO-MATERIAL DIFFERENCE** | YAML comment: "the phase-7 BE@0.75R exit refinement is backtest-only for now -- live breakeven handling stays with monitor_positions' existing 25-pip logic." |
| EURJPY AMR | Same as GBPJPY AMR | **MINOR-TO-MATERIAL DIFFERENCE** | Same BE-logic gap (shared `monitor_positions` code path). |
| AUDJPY AMR | Config identical; **BUY-only research candidate NOT implemented live** | **IDENTICAL (confirmed by direct code inspection)** | `strategies/asian_hours_reversion.py` and `pairs/AUDJPY_asianrev.yaml` show both-direction trading, unchanged from the original spec — verified directly for this audit, not assumed. |
| CADJPY AMR | Same as GBPJPY AMR | **MINOR-TO-MATERIAL DIFFERENCE** | Same BE-logic gap. |
| GBPUSD Monday Drift | IDENTICAL | **IDENTICAL** | `sl_atr_mult: 1.25, tp_atr_mult: 1.0` matches exactly. |

**All 8 strategies also carry two portfolio-wide, dated execution
deviations from their pure backtested form**, both already found and
fixed, both worth restating here for completeness:
1. **2026-07-05 to 2026-07-07:** session-gating used real UTC instead of
   MT5 server time — per project memory, "ARB/MON were dead live, AMR
   truncated" during this ~2-day window (commit e75d680 fixed it).
   **Any live trades in this ~48-hour window are not representative of
   the validated strategy's actual session logic.**
2. **Since 2026-07-31 (5ers only):** `risk_scale: 0.5` — every 5ers
   trade is sized at half the validated risk_percent. This is a
   deliberate, disclosed risk-management decision, not a bug, but it
   means 5ers position sizes do not match the backtested/demo sizing.

---

## Data integrity audit

Per the brief's explicit instruction to check whether any current
strategy's validation depends on cross-symbol or positional-alignment
logic similar to the NZDJPY bug (`reports/data_integrity_audit.md`,
2026-08-11, commit aca58e0):

- **ARB (`signals_arb_p`) and AMR (`signals_amr_v`)** both operate on a
  **single symbol's own OHLC series** — no second symbol's array is
  read, joined, or indexed anywhere in either function
  (`src/phase2_meanrev_arb_search.py`, `src/phase3b_amr_jpy_refine.py`).
  There is no cross-symbol positional-alignment surface for either
  family to be vulnerable to. This is a structural fact, verified by
  reading the signal functions directly, not inferred from the earlier
  audit's summary.
- **Monday Drift (`signals_monday`)** similarly operates on GBPUSD's own
  H1 series plus its own daily-resampled ATR20d — again single-symbol.
- **The one prior finding this project made in this exact bug class
  (NZDJPY cross-asset momentum, EXP-034) was never part of this live
  portfolio** — it was explicitly rejected before deployment (phase
  13b, `reports/phase13b_alignment_fix_report.md`) and does not appear
  in any of the 8 `pairs/*.yaml` files.
- **The general data-integrity audit (`reports/data_integrity_audit.md`)
  covers `src/phase2_meanrev_arb_search.py`, `phase3_session_structure_search.py`,
  and `phase3b_amr_jpy_refine.py`** (all three are single-symbol,
  classified NOT APPLICABLE for the positional-alignment bug class in
  that audit) — **but that audit was run 2026-08-11, over a month after
  this portfolio was deployed.** It is valid, current evidence about
  data integrity, but it too is POST-LIVE — the original July decision
  was made without the benefit of that audit's assurance, even though
  the code it examined has not changed since deployment.
- **Timestamp/session-boundary correctness specifically for this live
  portfolio** was checked, and found broken, by the server-time bug
  (e75d680) described above — this is a concrete, real instance of
  exactly the kind of "incorrect session boundaries" the brief asked
  about, and it affected live trading for ~2 days before being fixed.

**Conclusion: no cross-symbol alignment risk exists in any of the 8
live strategies' signal logic** (all single-symbol). **A real
session-boundary/timezone bug did affect live execution for a ~48-hour
window immediately after deployment**, since fixed. No other data-
integrity defect specific to these 8 strategies' backtests has been
found in this audit.

---

## Current 5ers performance vs. validated historical distribution

**Dashboard snapshot (as supplied, not independently re-queried in this
audit):** ~33 trades, balance ≈ $4,797.42 (from $5,000), win rate 30.3%,
PF 0.30, expectancy -0.33R, max drawdown ≈3.8%, best trade +1.48R, worst
trade -1.65R.

**Framing, per the brief's explicit instruction — do NOT conclude
invalidity from a losing 33-trade sample:** the prior portfolio
drawdown audit (`reports/portfolio_drawdown_distribution_audit.md`,
2026-08-11, EXP-092-095) found the current prop drawdown (-3.4% to
-3.8%, consistent with the -3.8% figure here) sits at only the
**3rd-9th percentile** of a 10,000-run Monte Carlo drawdown distribution
built from this exact portfolio's own historical trade sequence — i.e.
**shallower than 90%+ of what this portfolio's own trade-order variance
could plausibly produce.** On the drawdown dimension, this is
**statistically unremarkable, not evidence of strategy failure.**

**A real complication for that comparison, surfaced by this audit and
not previously accounted for:** the 5ers account currently runs **only
6 of the 8 strategies** (GBPJPY ARB and XAUUSD ARB excluded since
2026-07-31) **at `risk_scale: 0.5`**, while the historical Monte Carlo
benchmark in the prior audit was built from **all 8 strategies at full
weight.** This means the prior audit's percentile comparison, while
directionally still informative (the current drawdown is mild in
absolute terms), is not a strict apples-to-apples comparison against
the *actual currently-running* 6-slot, half-risk configuration. **This
is a specific, actionable gap this audit is flagging: a corrected
Monte Carlo benchmark using only the 6 currently-active strategies at
half risk would be a more precise comparison than the one already
produced.** No such corrected benchmark has been built in this audit
(that would be new analysis, out of scope for a forensic review).

**On "MANUAL/OTHER" exit reasons, per the brief's clarification:** any
dashboard row showing `MANUAL/OTHER` should be read as the strategy's
own **scheduled London-open closure** (a designed time-exit, not a
discretionary human intervention) — this audit treats it as **SCHEDULED
STRATEGY EXIT** throughout, consistent with the brief's instruction, and
did not have reason to question this classification given the
consistent orchestrator-driven time-exit design (`step_asian_time_exit`,
`step_monday_time_exit`) documented across every AMR/Monday YAML file.

**GBPJPY ARB's own demotion record is directly relevant context:** the
worst trade cited (-1.65R against a 1R design) matches the exact pattern
documented as the reason for GBPJPY ARB's 2026-07-31 demotion — "min-lot
granularity + live spread on a $5K account inflates loss magnitude vs.
backtest assumptions" (`PROJECT_REPORT.md` §3). **If this specific worst
trade belongs to GBPJPY ARB, it occurred before that strategy's
demotion and is not representative of the currently-running 6-slot
book** — but this audit cannot confirm which specific dashboard rows
belong to which strategy without the underlying per-trade data
(pair + strategy tag), which was not supplied for this audit.

---

## Demo vs. 5ers comparison

**Data insufficiency, stated plainly per instruction:** this audit was
supplied a dashboard *summary* for 5ers (aggregate stats only) and,
separately, the demo account's raw 50-trade history was pulled in an
earlier session turn (not reproduced here). **No trade-level 5ers data
(pair, direction, entry/exit timestamp, entry price, spread, exit
reason, R) was supplied to this audit** — only the aggregate dashboard
numbers listed above. **A signal-by-signal demo-vs-5ers comparison
(same pair, same strategy, same entry timestamp) is therefore NOT
AVAILABLE in this audit.** If that comparison is wanted, it requires
either the 5ers dashboard's `/api/trades` or `/api/journal` output
(both exist per `mcp/server.py`, see the earlier conversation turn on
how to fetch them) or direct MT5 history from the 5ers account,
neither of which was provided here.

---

## Portfolio-level synthesis

**1. Did we actually conduct extensive research before deploying this
portfolio?** Yes, in volume — six research phases (1,2,3,3b,6,7,8)
covering roughly 300+ backtested combinations across the project's own
count. But the *depth* of that research (per-strategy) was IS/OOS-split-plus-modest-grid,
not the walk-forward/Monte-Carlo/cost-stress standard this project later
adopted.

**2. Which strategies were genuinely validated before live deployment
(by the standard actually used at the time)?** All 8 cleared the IS/OOS
bar with the exception of XAUUSD ARB's OOS profit factor (1.05, below
the project's own >1.3 threshold). Monday Drift's numbers were the
strongest; GBPJPY/CADJPY ARB and the AMR family were solid-but-unremarkable
passes.

**3. Which strategies were deployed while still provisional?** One,
explicitly labeled as such in its own config: **XAUUSD ARB.**

**4. Which strategies had only in-sample evidence?** None — all 8 have
a genuine, disclosed OOS result.

**5. Which strategies lacked genuine OOS testing?** None.

**6. Which strategies lacked cost stress?** All 8.

**7. Which strategies lacked walk-forward testing?** All 8.

**8. Which strategies lacked Monte Carlo testing (strategy-specific,
not portfolio-level)?** All 8.

**9. Which strategies were added based on evidence discovered after
deployment?** None were *added* post-deployment — all 8 were deployed
in the same two-day window (2026-07-05/06). However, **AUDJPY AMR's
strongest evidentiary support was discovered after deployment** (see
§6), and the entire post-live research program (phases 15-24) has since
generated a much deeper evidence base for the AMR family specifically
that did not exist when the deployment decision was made.

**10. Is the current portfolio actually research-backed, or did we
gradually assemble it from promising candidates?** **Both, honestly.**
It is research-backed in the sense that every slot has a real,
documented IS/OOS backtest with a disclosed methodology — this was not
an ad hoc assembly of untested ideas. But it was assembled to a
**"promising," not "rigorously validated,"** standard: no walk-forward,
no strategy-level Monte Carlo, no cost stress existed for any of the 8
strategies at deployment time, and one slot (XAUUSD ARB) was
consciously deployed below the project's own stated acceptance bar.

**11. What percentage of the current portfolio was genuinely validated
before deployment (to a standard including walk-forward/MC/cost-stress)?**
**0%** — none of the 8 strategies had any of those three test types
before deployment. If the bar is instead "real IS/OOS split with
positive, disclosed out-of-sample results," **100%** (8/8) clear it.
The honest answer depends entirely on which standard is applied, which
is exactly why this audit reports both rather than picking one.

**12. What percentage should be considered provisional?** **12.5%**
(1/8, XAUUSD ARB) by the project's own explicit self-labeling.
Additionally, **all 6 AMR slots (50% of the book) were explicitly
labeled "DEMO FORWARD-TEST candidate"** in their own YAML comments at
deployment time, with an intended 2-3 month demo confirmation period
before "challenge use" — see the honesty-rule finding immediately below.

**13. Does the current 5ers losing period suggest strategy failure,
normal variance, execution differences, or insufficient evidence?**
Based on the prior Monte Carlo audit and this audit's own findings: **primarily
normal variance** on the drawdown dimension (3rd-9th percentile, mild)
and **partially execution/configuration differences** (6-of-8 strategies
at half risk, not the full 8-slot book the historical benchmark was
built from; GBPJPY ARB's worst trade pattern is consistent with the
known min-lot/spread execution gap that led to its demotion). There is
**not currently sufficient evidence to conclude strategy failure** —
but there is also not yet a benchmark built specifically for the
*actual currently-running* 6-slot, half-risk configuration, which this
audit recommends as a concrete next step (research, not a strategy
change).

**14. What evidence is still missing before we should trust each
strategy with meaningful prop-firm risk?** Uniformly, across all 8:
walk-forward testing, strategy-specific Monte Carlo, and cost stress —
none of which existed at deployment and none of which have since been
run for GBPJPY ARB, CADJPY ARB, XAUUSD ARB, GBPJPY AMR, EURJPY AMR, or
Monday Drift (only AUDJPY AMR has since received this level of scrutiny,
in phases 20-22). XAUUSD ARB specifically needs its broker-data and
minimum-lot-size problem resolved before it could be reconsidered for
the funded account at all, independent of its statistical evidence.

---

## Honesty-rule finding — the single most important fact this audit surfaced

**Every one of the 8 strategy configuration files, at the moment they
were activated (2026-07-05/06), explicitly documented that they were
intended for demo forward-testing before prop/challenge use** — direct
quotes: *"Compare demo results to the walk-forward stats above before
using on a challenge account"* (GBPJPY ARB), *"ACTIVE for demo
forward-testing"* (CADJPY ARB), *"forward-test 2-3 months on demo before
any challenge use"* (GBPJPY AMR, and materially identical language on
EURJPY/AUDJPY/CADJPY AMR), *"demo-confirm before challenge use"* (CADJPY
AMR), *"Review after 2-3 months of demo"* (XAUUSD ARB).

Per project memory (`project_multi_account_setup.md`, this session's
own prior record): the decision to buy the 5ers challenge and begin
funded trading was made on a timeline driven by **content/YouTube
production needs, explicitly prioritized over the self-imposed demo-gate
period** ("user buying 5ers NOW (YouTube timeline > demo gate; demo =
permanent control)"). The 5ers account went live between 2026-07-15 and
2026-08-11 — **at most about 5 weeks after the 2026-07-05/06 deployment,
short of the 2-3 months every AMR/ARB slot's own documentation called
for.**

**This report states this plainly, per the explicit instruction not to
protect previous conclusions: the project's own pre-specified demo
forward-test period was not completed before six of the eight
strategies (all four AMR pairs plus, implicitly, the "review after 2-3
months" language on XAUUSD ARB) were exposed to real prop-firm capital.
This was a disclosed, intentional trade-off — not a hidden one — but it
is a real gap between the project's own stated evidence standard and
what was actually observed before funded deployment.**

---

## Final classifications

| Strategy | Classification |
|---|---|
| GBPJPY ARB | **B. REASONABLY VALIDATED BUT MISSING IMPORTANT TESTS** |
| CADJPY ARB | **B. REASONABLY VALIDATED BUT MISSING IMPORTANT TESTS** |
| XAUUSD ARB | **D. PROVISIONAL / WEAK EVIDENCE BEFORE LIVE** |
| GBPJPY AMR | **C. PROMISING BUT INSUFFICIENT BEFORE LIVE** |
| EURJPY AMR | **C. PROMISING BUT INSUFFICIENT BEFORE LIVE** |
| AUDJPY AMR | **C. PROMISING BUT INSUFFICIENT BEFORE LIVE** (weakest pre-live OOS PF of the AMR family; strongest evidence discovered post-deployment) |
| CADJPY AMR | **C. PROMISING BUT INSUFFICIENT BEFORE LIVE** |
| GBPUSD Monday Drift | **B. REASONABLY VALIDATED BUT MISSING IMPORTANT TESTS** (strongest numbers in the project, but smallest sample and same missing-test profile as the rest of the book) |

No strategy in this portfolio reaches **A. WELL VALIDATED BEFORE LIVE**
by a standard that includes walk-forward, Monte Carlo, and cost stress
— because none of the 8 had any of those three tests before deployment.
None reach **E. NOT VALIDATED / DEPLOYMENT NOT JUSTIFIED** — every
strategy has a real, disclosed, positive OOS result from a genuine
held-out period. **F. UNKNOWN** does not apply to any of the 8 — the
audit trail, while imperfect, was traceable for all 8 via YAML comments,
`PROJECT_REPORT.md`, and git history.

---

## Sources cited in this audit

`PROJECT_REPORT.md` §3-4; `pairs/GBPJPY_asianrange.yaml`,
`pairs/CADJPY_asianrange.yaml`, `pairs/XAUUSD_asianrange.yaml`,
`pairs/GBPJPY_asianrev.yaml`, `pairs/EURJPY_asianrev.yaml`,
`pairs/AUDJPY_asianrev.yaml`, `pairs/CADJPY_asianrev.yaml`,
`pairs/GBPUSD_monday.yaml`; `strategies/asian_hours_reversion.py`,
`strategies/asian_range_breakout.py`; git commits a17520f, edbe008,
8d4ddf5, e1f3ec9, 6d3ddc9, 43284a6, 708f12c, e75d680, 84358fd, 95492ed,
3abd9a7; `reports/data_integrity_audit.md` (EXP-037, aca58e0);
`reports/portfolio_drawdown_distribution_audit.md` (EXP-092-095,
32fe299); `reports/amr_regime_mechanism.md` (EXP-076-081, e10d189);
`reports/audjpy_amr_confirmatory_filter.md` (EXP-082-086, 55e301e);
`reports/audjpy_amr_final_validation.md` (EXP-087-089, fe22c56);
`reports/volatility_regime_strategy_diagnostics.md` (EXP-066-075,
d98c401); `data/phase20_trades.csv`; project memory
`project_multi_account_setup.md`.

---

## What I did NOT do (per instructions)

- Did not modify any strategy, entry/exit logic, risk, or live
  configuration.
- Did not deploy, pause, add, or remove anything.
- Did not run a parameter search or optimize anything.
- Did not claim any strategy is validated because it is currently
  profitable, or invalid because it is currently losing.
- Did not allow post-live research (phases 15-24) to retroactively
  strengthen or weaken the original July deployment judgment — every
  post-live finding is explicitly labeled as affecting current
  confidence only.
- Did not build a corrected 6-slot/half-risk Monte Carlo benchmark
  (flagged as a useful next step, not performed here, as it would
  constitute new analysis beyond a forensic audit of existing evidence).

See `reports/live_portfolio_validation_audit.csv` for the
machine-readable summary.
