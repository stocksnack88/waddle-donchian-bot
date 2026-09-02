"""Higher-frequency sweep: mean-reversion + breakout on 5m/15m, taker vs maker."""
import data, metrics
from engine import Backtester
from sizing import Instrument
from strategies.bollinger_fade import BollingerFade
from strategies.rsi2_reversion import RSI2Reversion
from strategies.donchian_atr import DonchianATR

SYMS = ["XLMUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT"]
TFS = {"15m": 400, "5m": 200}
STRATS = {
    "boll_fade":      lambda: BollingerFade(period=20, k=2.0, stop_atr=1.5, trend_ema=200),
    "boll_fade_chop": lambda: BollingerFade(period=20, k=2.2, stop_atr=1.5, adx_max=25),
    "rsi2":           lambda: RSI2Reversion(rsi_lo=10, rsi_hi=90, trend_ema=200, stop_atr=2.5),
    "rsi2_loose":     lambda: RSI2Reversion(rsi_lo=15, rsi_hi=85, trend_ema=100, stop_atr=2.0, tp_pct=0.008),
    "donchian":       lambda: DonchianATR(lookback=20, atr_mult=2.0, rr=2.0, ema_filter=200),
}

hdr = f"{'sym':8s} {'tf':4s} {'strategy':14s} {'mode':6s} {'trades':>6s} {'win%':>5s} {'expR':>6s} {'ret%':>8s} {'sharpe':>7s} {'maxDD%':>7s} {'fee/gross':>9s}"
print(hdr)
print("-" * len(hdr))
for tf, days in TFS.items():
    for sym in SYMS:
        bars = data.fetch_ohlcv(sym, tf, days=days)
        fund = data.fetch_funding(sym, days=days + 30)
        inst = Instrument(**data.fetch_instrument(sym))
        for sname, mk in STRATS.items():
            for mode in ("taker", "maker"):
                res = Backtester(bars, mk(), funding=fund, instrument=inst,
                                 initial_equity=50, slippage_bps=2.0,
                                 fee_bps=5.5, maker_fee_bps=2.0,
                                 entry_mode=mode, limit_wait_bars=2).run()
                m = metrics.compute(res, tf)
                tr = res.trades
                if not len(tr):
                    print(f"{sym:8s} {tf:4s} {sname:14s} {mode:6s}      0     -      -        -       -       -        -")
                    continue
                gross = (tr["pnl"] + tr["fees"]).abs().sum()
                feeratio = tr["fees"].sum() / gross if gross > 0 else 0
                star = "  <<<" if (m["win_rate_pct"] >= 55 and m["return_pct"] > 0 and m["trades"] >= 40) else ""
                print(f"{sym:8s} {tf:4s} {sname:14s} {mode:6s} {m['trades']:6d} {m['win_rate_pct']:5.0f} "
                      f"{m['expectancy_r']:6.2f} {m['return_pct']:8.1f} {m['sharpe']:7.2f} {m['max_dd_pct']:7.1f} "
                      f"{feeratio*100:8.0f}%{star}")
        print()
