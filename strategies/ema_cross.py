"""
EMA cross, trend-continuation.

Entry : fast EMA crosses above slow EMA -> long (below -> short)
        optional: EMA gap must exceed `gap_pct` (filters chop)
        optional: higher-timeframe agreement via a longer EMA on the same series
Stop  : atr_mult * ATR
Target: risk * rr
Exit  : opposite cross, plus stop / target
"""
from __future__ import annotations

import pandas as pd

from engine import Context, Signal, Strategy
from .indicators import atr, ema


class EmaCross(Strategy):
    name = "ema_cross"

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        atr_period: int = 14,
        atr_mult: float = 2.0,
        rr: float = 2.0,
        gap_pct: float = 0.0,       # min |fast-slow|/price, 0 disables
        htf_ema: int = 0,          # 0 disables; e.g. 200 for a slow trend gate
    ):
        self.fast, self.slow = fast, slow
        self.atr_period, self.atr_mult, self.rr = atr_period, atr_mult, rr
        self.gap_pct = gap_pct
        self.htf_ema = htf_ema
        self.warmup = max(slow, htf_ema, atr_period, 30) + 5
        self.params = dict(fast=fast, slow=slow, atr_period=atr_period,
                           atr_mult=atr_mult, rr=rr, gap_pct=gap_pct, htf_ema=htf_ema)

    def prepare(self, df: pd.DataFrame, funding=None) -> pd.DataFrame:
        c = df["close"]
        f = pd.DataFrame(index=df.index)
        f["ef"] = ema(c, self.fast)
        f["es"] = ema(c, self.slow)
        f["atr"] = atr(df, self.atr_period)
        f["htf"] = ema(c, self.htf_ema) if self.htf_ema else float("nan")
        return f

    def on_bar(self, ctx: Context) -> Signal:
        ef, es = ctx.f["ef"], ctx.f["es"]
        ef0, es0 = ctx.fprev("ef"), ctx.fprev("es")
        a = ctx.f["atr"]
        price = ctx.price
        if a != a or a <= 0 or ef0 != ef0:
            return Signal("hold")

        cross_up = ef0 <= es0 and ef > es
        cross_dn = ef0 >= es0 and ef < es

        if ctx.position == 1 and cross_dn:
            return Signal("exit", tag="ema flip")
        if ctx.position == -1 and cross_up:
            return Signal("exit", tag="ema flip")
        if ctx.position != 0:
            return Signal("hold")

        if self.gap_pct and abs(ef - es) / price < self.gap_pct:
            return Signal("hold")
        htf_long = htf_short = True
        if self.htf_ema:
            he = ctx.f["htf"]
            htf_long, htf_short = price > he, price < he

        if cross_up and htf_long:
            sl = price - self.atr_mult * a
            return Signal("enter_long", sl=sl, tp=price + self.rr * (price - sl), tag="ema x up")
        if cross_dn and htf_short:
            sl = price + self.atr_mult * a
            return Signal("enter_short", sl=sl, tp=price - self.rr * (sl - price), tag="ema x dn")
        return Signal("hold")
