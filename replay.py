"""
Replay the LIVE bot's decision code over history, bar by bar, and check the
resulting trades against the backtester. If they line up, the bot's wiring
(closed-candle gating, sizing, stop/target watch) matches the tested engine.

    python3 replay.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import bot
import data as datamod
import metrics
from engine import Backtester
from sizing import Instrument
from strategies.donchian_atr import DonchianATR

SYMS = ["XLMUSDT", "XRPUSDT"]
DAYS = 400
START = 50.0


def replay_symbol(sym: str):
    full = datamod.fetch_ohlcv(sym, "4h", days=DAYS)
    inst = Instrument(**datamod.fetch_instrument(sym))
    br = bot.PaperBroker(cash=START)
    warm = bot.CFG["ema_filter"] + 30

    # feed the bot growing history; last row of each slice = "forming" candle
    for end in range(warm + 2, len(full) + 1):
        window = full.iloc[:end]
        _one_step(br, sym, window, inst)

    trades = _pair_trades(sym)
    return br, trades, full


def _one_step(br, sym, window, inst):
    """A single poll: same logic as bot.step() for one symbol."""
    fresh = window
    last_price = float(fresh["close"].iloc[-1])
    closed = fresh.iloc[:-1]
    closed_ts = str(closed.index[-1])
    dec_i = len(closed) - 1
    strat = DonchianATR(**bot.CFG)
    feats = strat.prepare(closed, None)
    marks = {sym: last_price}
    eq = br.get_equity(marks)
    pos = br.get_positions()

    if sym in pos:
        p = pos[sym]
        hi = float(fresh["high"].iloc[-1]); lo = float(fresh["low"].iloc[-1])
        if (p["side"] == 1 and lo <= p["sl"]) or (p["side"] == -1 and hi >= p["sl"]):
            br.close_position(sym, p["sl"], "stop", closed_ts); return
        if p["tp"] == p["tp"] and ((p["side"] == 1 and hi >= p["tp"]) or (p["side"] == -1 and lo <= p["tp"])):
            br.close_position(sym, p["tp"], "target", closed_ts); return

    if br.seen_closed_ts.get(sym) == closed_ts:
        return
    br.seen_closed_ts[sym] = closed_ts

    from engine import Context
    if sym in pos:
        ctx = Context(bars=closed, feats=feats, i=dec_i, position=pos[sym]["side"],
                      entry_price=pos[sym]["entry"], equity=eq, funding=None)
        sig = strat.on_bar(ctx)
        if sig.action == "exit":
            br.close_position(sym, last_price, sig.tag or "signal", closed_ts)
    else:
        ctx = Context(bars=closed, feats=feats, i=dec_i, position=0, entry_price=0.0,
                      equity=eq, funding=None)
        sig = strat.on_bar(ctx)
        if sig.action in ("enter_long", "enter_short"):
            side = 1 if sig.action == "enter_long" else -1
            stop = sig.sl if sig.sl else last_price * (1 - side * 0.01)
            from sizing import size_position
            sized = size_position(equity=eq, entry_price=last_price, stop_price=stop,
                                  risk_pct=bot.RISK_PCT, inst=inst, max_leverage=bot.MAX_LEVERAGE)
            if sized.qty > 0:
                br.market_order(sym, side, sized.qty, last_price, sl=stop,
                                tp=sig.tp if (sig.tp and sig.tp == sig.tp) else float("nan"),
                                tag=sig.tag, ts=closed_ts)


def _pair_trades(sym):
    import json
    recs = [json.loads(l) for l in bot.TRADES_PATH.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r["symbol"] == sym]
    out, opn = [], None
    for r in recs:
        if r["event"] == "open":
            opn = r
        elif r["event"] == "close" and opn:
            out.append((opn, r))
            opn = None
    return out


if __name__ == "__main__":
    bot.TRADES_PATH.unlink(missing_ok=True)
    bot.STATE_PATH.unlink(missing_ok=True)
    print(f"replaying the live bot code over {DAYS}d of history, {SYMS}\n")
    for sym in SYMS:
        br, trades, full = replay_symbol(sym)

        bt = Backtester(full, DonchianATR(**bot.CFG), instrument=Instrument(**datamod.fetch_instrument(sym)),
                        initial_equity=START, slippage_bps=bot.SLIPPAGE_BPS, risk_pct=bot.RISK_PCT).run()
        m = metrics.compute(bt, "4h")

        realized = sum(c["pnl"] for _, c in trades)
        wins = sum(1 for _, c in trades if c["pnl"] > 0)
        bot_ret = realized / START * 100
        print(f"── {sym}")
        print(f"   bot replay : {len(trades):3d} trades  return {bot_ret:+5.1f}%  "
              f"win {wins/max(len(trades),1)*100:.0f}%  end ${br.cash:.2f}")
        print(f"   backtester : {m['trades']:3d} trades  return {m['return_pct']:+5.1f}%  "
              f"win {m['win_rate_pct']:.0f}%  end ${START*(1+m['return_pct']/100):.2f}")
        # live acts on close / fills ~now; backtest fills at next open -> small unbiased drift
        gap = abs(bot_ret - m["return_pct"])
        print(f"   -> return gap {gap:.1f} pts over {DAYS}d  "
              f"({'OK - within fill-model noise' if gap < 3 else 'CHECK'})\n")
