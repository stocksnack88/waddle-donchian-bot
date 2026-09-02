"""
Risk / reward / leverage sweep for the XLM+XRP 4h Donchian basket.

Shows how the dials actually move the outcome:
  - risk_pct   : fraction of equity risked to the stop per trade   <- THE main dial
  - max_concurrent : how many legs can be open at once
  - rr         : reward:risk ratio (target distance vs stop distance)
  - leverage   : only a safety cap; risk-based sizing rarely needs >2x
"""
import numpy as np
import pandas as pd

import data, metrics
from portfolio import PortfolioBacktester, Leg
from sizing import Instrument
from strategies.donchian_atr import DonchianATR

SYMS = ["XLMUSDT", "XRPUSDT"]
BASE = dict(lookback=20, atr_period=14, ema_filter=200)


def make_legs(rr, days=1000):
    legs = []
    for s in SYMS:
        legs.append(Leg(symbol=s, bars=data.fetch_ohlcv(s, "4h", days=days),
                        strategy=DonchianATR(atr_mult=2.0, rr=rr, **BASE),
                        funding=data.fetch_funding(s, days=days),
                        instrument=Instrument(**data.fetch_instrument(s))))
    return legs


def stats(res):
    m = metrics.compute(res, "4h")
    mt = metrics.monthly_table(res)
    tr = res.trades
    # Kelly from realised trades: f* = W - (1-W)/R_win  (R_win = avg win / avg loss)
    wins = tr[tr.pnl > 0]["pnl"]
    losses = tr[tr.pnl <= 0]["pnl"]
    W = len(wins) / len(tr) if len(tr) else 0
    Rw = (wins.mean() / -losses.mean()) if len(wins) and len(losses) else np.nan
    kelly = W - (1 - W) / Rw if Rw and Rw == Rw else np.nan
    lev_used = (tr["notional"] / (res.initial_equity)).max() if len(tr) else 0
    worst_month = mt["pnl"].min() / res.initial_equity * 100 if len(mt) else 0
    return dict(final=res.equity_curve.iloc[-1], cagr=m["cagr_pct"], sharpe=m["sharpe"],
               maxdd=m["max_dd_pct"], trades=m["trades"], win=m["win_rate_pct"],
               posmonths=(mt["pnl"] > 0).mean() * 100 if len(mt) else 0,
               worst_month_pct=worst_month, kelly=kelly, lev_used=lev_used,
               skipped=res.skipped_trades)


print("### 1. RISK PER TRADE  (rr=2.0, max_concurrent=2, leverage cap 5x)")
print(f"{'risk/leg':>9s} {'$50->':>8s} {'CAGR%':>7s} {'Sharpe':>7s} {'maxDD%':>7s} "
      f"{'worstMo%':>9s} {'+months':>8s} {'peakLev':>8s} {'skip':>5s}")
legs = make_legs(rr=2.0)
for rp in [0.0025, 0.005, 0.01, 0.015, 0.02, 0.03]:
    res = PortfolioBacktester([Leg(l.symbol, l.bars, DonchianATR(atr_mult=2.0, rr=2.0, **BASE),
                                   l.funding, l.instrument) for l in legs],
                              initial_equity=50, risk_pct=rp, slippage_bps=5.0,
                              max_leverage=5.0, max_concurrent=2).run()
    s = stats(res)
    print(f"{rp*100:8.2f}% ${s['final']:7.2f} {s['cagr']:7.1f} {s['sharpe']:7.2f} {s['maxdd']:7.1f} "
          f"{s['worst_month_pct']:9.1f} {s['posmonths']:7.0f}% {s['lev_used']:7.2f}x {s['skipped']:5d}")
print(f"   (realised Kelly fraction ~ {stats(res)['kelly']:.2f}  => full Kelly would risk "
      f"{stats(res)['kelly']*100:.0f}% per trade; use 1/4 to 1/2 of that)\n")

print("### 2. MAX CONCURRENT POSITIONS  (risk 0.5%/leg, rr=2.0)")
print(f"{'concur':>7s} {'$50->':>8s} {'CAGR%':>7s} {'Sharpe':>7s} {'maxDD%':>7s} {'+months':>8s}")
for mc in [1, 2]:
    res = PortfolioBacktester([Leg(l.symbol, l.bars, DonchianATR(atr_mult=2.0, rr=2.0, **BASE),
                                   l.funding, l.instrument) for l in legs],
                              initial_equity=50, risk_pct=0.005, slippage_bps=5.0,
                              max_leverage=5.0, max_concurrent=mc).run()
    s = stats(res)
    print(f"{mc:7d} ${s['final']:7.2f} {s['cagr']:7.1f} {s['sharpe']:7.2f} {s['maxdd']:7.1f} {s['posmonths']:7.0f}%")
print()

print("### 3. REWARD:RISK RATIO  (risk 0.5%/leg, max_concurrent=2)")
print(f"{'rr':>5s} {'$50->':>8s} {'CAGR%':>7s} {'Sharpe':>7s} {'maxDD%':>7s} {'win%':>5s} {'trades':>6s}")
for rr in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    lg = make_legs(rr=rr)
    res = PortfolioBacktester(lg, initial_equity=50, risk_pct=0.005, slippage_bps=5.0,
                              max_leverage=5.0, max_concurrent=2).run()
    s = stats(res)
    print(f"{rr:5.1f} ${s['final']:7.2f} {s['cagr']:7.1f} {s['sharpe']:7.2f} {s['maxdd']:7.1f} {s['win']:5.0f} {s['trades']:6d}")
print()

print("### 4. LEVERAGE CAP  (risk 1%/leg, rr=2.0) - does raising the cap change anything?")
print(f"{'cap':>5s} {'$50->':>8s} {'CAGR%':>7s} {'maxDD%':>7s} {'peakLev used':>12s} {'skip':>5s}")
for lev in [1, 2, 3, 5, 10]:
    res = PortfolioBacktester([Leg(l.symbol, l.bars, DonchianATR(atr_mult=2.0, rr=2.0, **BASE),
                                   l.funding, l.instrument) for l in legs],
                              initial_equity=50, risk_pct=0.01, slippage_bps=5.0,
                              max_leverage=float(lev), max_concurrent=2).run()
    s = stats(res)
    print(f"{lev:5d} ${s['final']:7.2f} {s['cagr']:7.1f} {s['maxdd']:7.1f} {s['lev_used']:11.2f}x {s['skipped']:5d}")
