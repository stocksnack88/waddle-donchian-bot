"""
RSI(2) reversion - Larry Connors style, mean reversion.

Entry : RSI(rsi_len) below rsi_lo AND close above EMA(trend_ema) -> long
        (RSI above rsi_hi AND close below EMA(trend_ema) -> short)
Exit  : RSI crosses back past rsi_exit, OR close crosses EMA(fast_ema),
        OR fixed target tp_pct, OR stop at stop_atr * ATR.
Entry limit rests slightly below close (limit_off) for maker fills.

Classic high-win-rate profile: many small wins, a few larger losses.
"""
from __future__ import annotations

import pandas as pd

from engine import Context, Signal, Strategy
from .indicators import atr, ema, rsi


class RSI2Reversion(Strategy):
    name = "rsi2_reversion"

    def __init__(
        self,
        rsi_len: int = 2,
        rsi_lo: float = 10.0,
        rsi_hi: float = 90.0,
        rsi_exit: float = 55.0,
        trend_ema: int = 200,
        fast_ema: int = 5,
        stop_atr: float = 2.5,
        atr_period: int = 14,
        tp_pct: float = 0.0,       # 0 = no fixed target, exit by rule
        limit_off_pct: float = 0.0015,
    ):
        self.rsi_len = rsi_len
        self.rsi_lo, self.rsi_hi, self.rsi_exit = rsi_lo, rsi_hi, rsi_exit
        self.trend_ema, self.fast_ema = trend_ema, fast_ema
        self.stop_atr, self.atr_period = stop_atr, atr_period
        self.tp_pct = tp_pct
        self.limit_off_pct = limit_off_pct
        self.warmup = max(trend_ema, atr_period, 30) + 5
        self.params = dict(rsi_len=rsi_len, rsi_lo=rsi_lo, rsi_hi=rsi_hi, rsi_exit=rsi_exit,
                           trend_ema=trend_ema, fast_ema=fast_ema, stop_atr=stop_atr,
                           atr_period=atr_period, tp_pct=tp_pct, limit_off_pct=limit_off_pct)

    def prepare(self, df: pd.DataFrame, funding=None) -> pd.DataFrame:
        c = df["close"]
        f = pd.DataFrame(index=df.index)
        f["rsi"] = rsi(c, self.rsi_len)
        f["trend"] = ema(c, self.trend_ema)
        f["fast"] = ema(c, self.fast_ema)
        f["atr"] = atr(df, self.atr_period)
        return f

    def on_bar(self, ctx: Context) -> Signal:
        row = ctx.f
        price = ctx.price
        rv, tr, fa, a = row["rsi"], row["trend"], row["fast"], row["atr"]
        if any(v != v for v in (rv, tr, fa, a)) or a <= 0:
            return Signal("hold")

        if ctx.position == 1:
            if rv > self.rsi_exit or price > fa:
                return Signal("exit", tag="rsi/fast exit")
            return Signal("hold")
        if ctx.position == -1:
            if rv < (100 - self.rsi_exit) or price < fa:
                return Signal("exit", tag="rsi/fast exit")
            return Signal("hold")

        if rv < self.rsi_lo and price > tr:
            sl = price - self.stop_atr * a
            tp = price * (1 + self.tp_pct) if self.tp_pct else None
            lim = price * (1 - self.limit_off_pct)
            return Signal("enter_long", sl=sl, tp=tp, limit=lim, tag="rsi2 oversold")
        if rv > self.rsi_hi and price < tr:
            sl = price + self.stop_atr * a
            tp = price * (1 - self.tp_pct) if self.tp_pct else None
            lim = price * (1 + self.limit_off_pct)
            return Signal("enter_short", sl=sl, tp=tp, limit=lim, tag="rsi2 overbought")
        return Signal("hold")
