"""
Donchian breakout with ATR stop.

Entry : close breaks the highest high of the last `lookback` bars -> long
        (breaks the lowest low -> short, if the engine allows shorts)
Stop  : entry -/+ atr_mult * ATR(atr_period)
Target: risk * rr
Filter: optional EMA trend gate + ADX floor (trade only when there's a trend)
Exit  : opposite Donchian band (structure flip), plus stop / target
"""
from __future__ import annotations

import pandas as pd

from engine import Context, Signal, Strategy
from .indicators import adx, atr, ema


class DonchianATR(Strategy):
    name = "donchian_atr"

    def __init__(
        self,
        lookback: int = 20,
        atr_period: int = 14,
        atr_mult: float = 2.0,
        rr: float = 2.0,
        ema_filter: int = 200,      # 0 disables
        adx_min: float = 0.0,       # 0 disables
        exit_opposite: bool = True,
    ):
        self.lookback = lookback
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.rr = rr
        self.ema_filter = ema_filter
        self.adx_min = adx_min
        self.exit_opposite = exit_opposite
        self.warmup = max(lookback, atr_period, ema_filter, 30) + 5
        self.params = dict(lookback=lookback, atr_period=atr_period, atr_mult=atr_mult,
                           rr=rr, ema_filter=ema_filter, adx_min=adx_min,
                           exit_opposite=exit_opposite)

    def prepare(self, df: pd.DataFrame, funding=None) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        f["atr"] = atr(df, self.atr_period)
        # bands from bars strictly BEFORE the current one (shift(1)) - no same-bar peek
        f["don_hi"] = df["high"].rolling(self.lookback).max().shift(1)
        f["don_lo"] = df["low"].rolling(self.lookback).min().shift(1)
        f["ema"] = ema(df["close"], self.ema_filter) if self.ema_filter else float("nan")
        f["adx"] = adx(df, 14) if self.adx_min else 0.0
        return f

    def on_bar(self, ctx: Context) -> Signal:
        row = ctx.f
        a, hi, lo = row["atr"], row["don_hi"], row["don_lo"]
        close = ctx.price
        if a != a or a <= 0 or hi != hi or lo != lo:
            return Signal("hold")

        if ctx.position != 0:
            if self.exit_opposite:
                if ctx.position == 1 and close < lo:
                    return Signal("exit", tag="donchian flip")
                if ctx.position == -1 and close > hi:
                    return Signal("exit", tag="donchian flip")
            return Signal("hold")

        long_ok = short_ok = True
        if self.ema_filter:
            long_ok, short_ok = close > row["ema"], close < row["ema"]
        if self.adx_min and row["adx"] < self.adx_min:
            return Signal("hold")

        if close > hi and long_ok:
            sl = close - self.atr_mult * a
            return Signal("enter_long", sl=sl, tp=close + self.rr * (close - sl),
                          tag=f"break>{self.lookback}H")
        if close < lo and short_ok:
            sl = close + self.atr_mult * a
            return Signal("enter_short", sl=sl, tp=close - self.rr * (sl - close),
                          tag=f"break<{self.lookback}L")
        return Signal("hold")
