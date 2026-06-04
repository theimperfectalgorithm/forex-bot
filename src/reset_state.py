"""Reset daily_state.json to today's date with correct fresh structure."""
import json
from datetime import datetime, timezone
from pathlib import Path

PAIRS = ["GBPJPY", "EURJPY"]
today = datetime.now(timezone.utc).date().isoformat()

fresh = {
    "date"             : today,
    "market_ran"       : False,
    "london_prep_done" : False,
    "ny_prep_done"     : False,
    "report_ran"       : False,
    "trade_allowed"    : None,
    "avoid_reason"     : None,
    "london_news_flag" : False,
    "ny_news_flag"     : False,
    "session_data"     : {},
    "london_traded"    : {p: False for p in PAIRS},
    "ny_traded"        : {p: False for p in PAIRS},
    "open_trades"      : [],
    "closed_today"     : [],
    "daily_pnl"        : 0.0,
    "consec_losses"    : {p: 0 for p in PAIRS},
    "pair_paused"      : {p: False for p in PAIRS},
    "eurusd": {
        "sma_daily_trades"    : 0,
        "ema_daily_trades"    : 0,
        "sma_consec_losses"   : 0,
        "ema_consec_losses"   : 0,
        "ema_pullback_pending": False,
        "ema_pullback_dir"    : "",
    },
}

sf = Path("data/state/daily_state.json")
sf.parent.mkdir(parents=True, exist_ok=True)
with open(sf, "w", encoding="utf-8") as f:
    json.dump(fresh, f, indent=2)

with open(sf, encoding="utf-8") as f:
    d = json.load(f)

print("Reset OK  date=" + d["date"])
print("Top-level keys: " + str(sorted(d.keys())))
print("eurusd keys: " + str(sorted(d["eurusd"].keys())))
