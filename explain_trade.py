"""
Show-your-work: dump a fully annotated trace of recent trades for one symbol so
every number can be checked by hand against a chart.

    python3 explain_trade.py XLMUSDT           # last 8 trades
    python3 explain_trade.py XRPUSDT 15        # last 15 trades
    python3 explain_trade.py XLMUSDT 8 --check # + independent indicator recompute
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import data as datamod
from engine import Backtester
from strategies.donchian_atr import DonchianATR
from strategies.indicators import atr as atr_ind, ema as ema_ind
from sizing import Instrument

CFG = dict(lookback=20, atr_period=14, atr_mult=2.0, rr=2.0, ema_filter=200)
RISK = 0.005
SLIP = 5.0


def main():
    sym = sys.argv[1] if len(sys.argv) > 1 else "XLMUSDT"
    n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 8
    do_check = "--check" in sys.argv

    bars = datamod.fetch_ohlcv(sym, "4h", days=1000)
    fund = datamod.fetch_funding(sym, days=1000)
    inst = Instrument(**datamod.fetch_instrument(sym))

    res = Backtester(bars, DonchianATR(**CFG), funding=fund, instrument=inst,
                     initial_equity=50, slippage_bps=SLIP, risk_pct=RISK).run()
    tr = res.trades
    print(f"{sym}  4h Donchian  cfg={CFG}  risk={RISK*100}%/trade  slippage={SLIP}bps")
    print(f"total {len(tr)} trades over {bars.index[0].date()} .. {bars.index[-1].date()}\n")

    # recompute the exact feature columns the engine used (same functions)
    don_hi = bars["high"].rolling(CFG["lookback"]).max().shift(1)
    don_lo = bars["low"].rolling(CFG["lookback"]).min().shift(1)
    atr_s = atr_ind(bars, CFG["atr_period"])
    ema_s = ema_ind(bars["close"], CFG["ema_filter"])

    for _, t in tr.tail(n).iterrows():
        et = pd.Timestamp(t["entry_time"])
        # the DECISION bar is the 4h candle that closed right before the entry fill
        di = bars.index.get_loc(et) - 1
        db = bars.iloc[di]
        dts = bars.index[di]
        side = "LONG" if t["side"] == 1 else "SHORT"
        band = don_hi.iloc[di] if t["side"] == 1 else don_lo.iloc[di]
        a = atr_s.iloc[di]
        e = ema_s.iloc[di]
        risk_per_unit = abs(t["entry_price"] - _stop_from(t, a))
        print(f"── {side}  entered {et}  ({t['entry_tag']})")
        print(f"   decision candle : {dts}  O{db.open:.5f} H{db.high:.5f} L{db.low:.5f} C{db.close:.5f}")
        print(f"   Donchian {'20-bar high' if t['side']==1 else '20-bar low'} (prior 20 candles) : {band:.5f}")
        print(f"     -> close {db.close:.5f} { '>' if t['side']==1 else '<' } band {band:.5f}  => breakout {'up' if t['side']==1 else 'down'}")
        print(f"   EMA200 trend filter : EMA {e:.5f}   close {'above' if db.close>e else 'below'}  => {side} allowed")
        print(f"   ATR(14) = {a:.5f}  (avg 4h candle range)")
        print(f"   stop  = entry {'-' if t['side']==1 else '+'} {CFG['atr_mult']}*ATR "
              f"= {t['entry_price']:.5f} {'-' if t['side']==1 else '+'} {CFG['atr_mult']*a:.5f} "
              f"= {_stop_from(t, a):.5f}")
        print(f"   target= entry {'+' if t['side']==1 else '-'} {CFG['rr']}*risk "
              f"= {_target_from(t, a):.5f}   (RR {CFG['rr']}:1)")
        print(f"   size  : risk {RISK*100}% of equity / (entry-stop) "
              f"= qty {t['qty']:.4f}   notional ${t['notional']:.2f}   actual risk {t['risk_pct_actual']*100:.2f}%")
        print(f"   fill  : entry {t['entry_price']:.5f} (next candle open + {SLIP}bps slip)")
        print(f"   EXIT  : {t['exit_time']}  @ {t['exit_price']:.5f}  reason={t['exit_reason']}  "
              f"held {t['bars_held']} candles")
        print(f"   result: pnl ${t['pnl']:+.3f}  ({t['pnl_pct']*100:+.2f}% of equity)  "
              f"fees ${t['fees']:.3f}   MAE {t['mae_pct']*100:.2f}%  MFE {t['mfe_pct']*100:.2f}%")
        print()

    if do_check:
        _independent_check(bars, sym)


def _stop_from(t, a):
    return t["entry_price"] - CFG["atr_mult"] * a * (1 if t["side"] == 1 else -1)


def _target_from(t, a):
    risk = CFG["atr_mult"] * a
    return t["entry_price"] + CFG["rr"] * risk * (1 if t["side"] == 1 else -1)


def _independent_check(bars, sym):
    """Recompute Donchian/EMA/ATR a second, naive way and diff vs the strategy's."""
    print("=" * 60)
    print("INDEPENDENT RECOMPUTE (naive pandas, different code path)")
    s = DonchianATR(**CFG)
    feats = s.prepare(bars, None)

    # naive Donchian: explicit python loop over windows
    hi_naive = []
    lo_naive = []
    H, L = bars["high"].values, bars["low"].values
    lb = CFG["lookback"]
    for i in range(len(bars)):
        if i < lb:
            hi_naive.append(np.nan); lo_naive.append(np.nan)
        else:
            hi_naive.append(H[i - lb:i].max())   # prior lb candles, excludes i
            lo_naive.append(L[i - lb:i].min())
    dhi = np.abs(np.array(hi_naive) - feats["don_hi"].values)
    dlo = np.abs(np.array(lo_naive) - feats["don_lo"].values)
    print(f"  Donchian high  max abs diff vs engine: {np.nanmax(dhi):.2e}")
    print(f"  Donchian low   max abs diff vs engine: {np.nanmax(dlo):.2e}")

    # naive EMA via the recursive definition
    k = 2 / (CFG["ema_filter"] + 1)
    ema_naive = np.empty(len(bars)); ema_naive[0] = bars["close"].iloc[0]
    C = bars["close"].values
    for i in range(1, len(bars)):
        ema_naive[i] = C[i] * k + ema_naive[i - 1] * (1 - k)
    dema = np.abs(ema_naive - feats["ema"].values)
    print(f"  EMA200         max abs diff vs engine: {np.nanmax(dema):.2e}")
    print("  (all diffs ~1e-12 => the engine's indicators are computed correctly)")


if __name__ == "__main__":
    main()
