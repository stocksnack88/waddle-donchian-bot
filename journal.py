"""
Summarise the bot's paper_trades.jsonl - account curve, win rate, per-symbol,
and how the live/paper record compares to the backtest expectation.

    python3 journal.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
TRADES = HERE / "paper_trades.jsonl"
START = 50.0
# backtest expectation (XLM+XRP 4h Donchian, 0.5%/leg, 2.7yr)
EXP_WIN_RATE = 40.0
EXP_TRADES_PER_MONTH = 11


def main():
    if not TRADES.exists():
        print("no paper_trades.jsonl yet - run the bot first"); return
    recs = [json.loads(l) for l in TRADES.read_text().splitlines() if l.strip()]
    closes = [r for r in recs if r["event"] == "close"]
    opens = [r for r in recs if r["event"] == "open"]
    print(f"log: {len(opens)} entries, {len(closes)} closed trades\n")
    if not closes:
        print("open positions:", [r["symbol"] for r in opens]); return

    df = pd.DataFrame(closes)
    df["ts"] = pd.to_datetime(df["ts"])
    wins = df[df["pnl"] > 0]
    span_days = max((df["ts"].max() - df["ts"].min()).days, 1)

    print(f"{'':16s}{'trades':>7s}{'win%':>7s}{'pnl $':>10s}{'avg $':>9s}")
    print(f"{'ALL':16s}{len(df):>7d}{len(wins)/len(df)*100:>7.0f}"
          f"{df['pnl'].sum():>10.3f}{df['pnl'].mean():>9.3f}")
    for sym, g in df.groupby("symbol"):
        gw = (g["pnl"] > 0).mean() * 100
        print(f"{sym:16s}{len(g):>7d}{gw:>7.0f}{g['pnl'].sum():>10.3f}{g['pnl'].mean():>9.3f}")

    final = START + df["pnl"].sum()
    print(f"\naccount: ${START:.2f} -> ${final:.2f}  ({(final/START-1)*100:+.1f}%) over {span_days}d")
    print(f"trades/month: {len(df)/span_days*30:.1f}  (backtest ~{EXP_TRADES_PER_MONTH})")
    print(f"win rate: {len(wins)/len(df)*100:.0f}%  (backtest ~{EXP_WIN_RATE:.0f}%)")
    by_reason = df.groupby("reason")["pnl"].agg(["count", "sum"])
    print("\nby exit reason:")
    print(by_reason.to_string())
    if len(df) < 20:
        print("\n(<20 trades - too early to judge; the edge needs months to show)")


if __name__ == "__main__":
    main()
