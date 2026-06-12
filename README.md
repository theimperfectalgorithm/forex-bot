# Forex Trading Bot

**A fully autonomous 5-agent Python system trading forex — built to remove emotional decision-making from trading entirely.**

---

## Why this exists

I lost multiple prop firm challenge accounts. Not to the market — to myself. Same pattern every time: solid strategy, good start, then one emotional decision (moved a stop loss, doubled down after a loss, closed a winner early out of fear) that undid weeks of disciplined trading.

The conclusion wasn't "trade better." It was that no amount of discipline reliably survives contact with real money. The only real fix was architectural — build a system where the human physically cannot intervene in the moment.

This started as a basic Python script that just downloaded EURUSD data and plotted it. It's now a 5-agent autonomous trading system running 24/5 — and the lessons from building this fed directly into [AEGIS](https://github.com/theimperfectalgorithm/aegis), a larger multi-market system for Indian equity and forex.

The entire build is documented on [YouTube — TheImPerfectAlgorithm](https://www.youtube.com/@TheImPerfectAlgorithm), including the failures along the way.

---

## What it does

Trades **GBPJPY, EURJPY, and EURUSD** on MetaTrader 5, fully autonomously — no manual intervention required.

| Agent | Role |
|---|---|
| **Main Orchestrator** | Runs 24/5, wakes every 15 minutes, calls each agent in sequence, makes the final trade decision, handles all errors so the system never crashes |
| **Agent 1 — Market Intelligence** | Runs at 00:00 UTC. Checks account balance against the hard floor, scans for high-impact news, returns TRADE_DAY or AVOID for the day |
| **Agent 2 — Strategy** | Calculates the Asian session range, checks H4 trend, detects London/NY breakout signals for GBPJPY/EURJPY. Runs SMA + EMA strategies on EURUSD |
| **Agent 3 — Risk Management** | Runs before every trade. Five sequential gates — hard floor, daily loss limit, consecutive losses, pair pause status, SL validity. Calculates dynamic lot sizing |
| **Agent 4 — Execution** | Places orders, sets SL/TP, moves SL to breakeven at 25 pips profit, monitors all open trades every 15 minutes, logs every trade to CSV |
| **Agent 5 — Reporting** | Runs at 21:00 UTC. Compiles daily summary, updates equity curve, flags anomalies, tracks progress toward the prop firm challenge target |

---

## Risk gates (every trade, no exceptions)

1. **Hard floor check** — balance above minimum, or all trading halts permanently
2. **Daily loss limit** — 5% max daily drawdown
3. **Consecutive losses** — 2 losses on a pair today pauses that pair
4. **Pair pause status** — checks if a pair is already paused
5. **Stop loss validity** — rejects the trade rather than trading blind

---

## Stack

- **Python** — core language
- **Claude Code** — entire system built through it, no prior formal programming background
- **MetaTrader 5 API** — execution and account data
- **Asian session range breakout** strategy (GBPJPY/EURJPY), SMA/EMA strategies (EURUSD)

---

## Status

Running on a MetaTrader 5 demo account, forward-testing toward a prop firm challenge target. Build process — including every failed attempt — documented on YouTube.

---

## Disclaimer

This is a personal trading system shared for educational and portfolio purposes. Nothing here is financial advice. Trading involves substantial risk of loss. Use at your own risk.

---

## Get in touch

Building something similar? I take on freelance projects for trading systems, automation, and AI workflows.

→ [zerohand.dev](https://zerohand.dev)
→ hello@zerohand.dev
