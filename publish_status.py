"""
Read the paper-trade log from Supabase and write state/status.json — a small
summary the weekly-check routine can read from GitHub (it can't reach Supabase
directly). Run by .github/workflows/status.yml every hour.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pandas as pd

URL = os.environ["WADDLE_SUPABASE_URL"].rstrip("/")
KEY = os.environ["WADDLE_SUPABASE_KEY"]
START = 50.0
OUT = Path(__file__).parent / "state" / "status.json"


def _get(path: str):
    r = httpx.get(f"{URL}/rest/v1/{path}",
                  headers={"apikey": KEY, "Authorization": f"Bearer {KEY}"}, timeout=15)
    r.raise_for_status()
    return r.json()


def main():
    trades = _get("waddle_paper_trades?select=*&order=id.asc")
    state_rows = _get("waddle_bot_state?id=eq.1&select=state")
    state = state_rows[0]["state"] if state_rows else {}

    opens: dict = {}
    closed = []
    for e in trades:
        if e["event"] == "open":
            opens[e["symbol"]] = e
        elif e["event"] == "close":
            o = opens.pop(e["symbol"], None)
            closed.append({
                "symbol": e["symbol"], "side": e.get("side"),
                "pnl": e.get("pnl") or 0.0, "reason": e.get("reason"),
                "opened": (o or {}).get("candle_ts"), "closed": e.get("candle_ts"),
            })

    wins = sum(1 for t in closed if t["pnl"] > 0)
    net = round(sum(t["pnl"] for t in closed), 4)
    last_ts = max([t["closed"] for t in closed] + [None] if closed else [None])

    status = {
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        "start_equity": START,
        "closed_trades": len(closed),
        "wins": wins,
        "win_rate_pct": round(wins / len(closed) * 100, 1) if closed else None,
        "net_pnl": net,
        "equity": round(START + net, 2),
        "return_pct": round(net / START * 100, 1),
        "open_positions": [
            {"symbol": s, "side": o.get("side"), "entry": o.get("price"),
             "opened": o.get("candle_ts")}
            for s, o in opens.items()
        ],
        "last_trade_at": last_ts,
        "recent_trades": closed[-10:],
        "bot_state_seen_candles": state.get("seen_closed_ts"),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
