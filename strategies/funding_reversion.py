"""
Funding-rate reversion (perp-specific).

Idea: when funding sits in the top slice of its own recent range, the crowd is
aggressively long and paying to hold - crowded longs get flushed. Fade it: short
when funding is extreme-high AND price is stretched above its mean; mirror for
longs. Mean-reversion, so it wants chop, not a strong trend.

Thresholds are ROLLING PERCENTILES of the symbol's own funding history, so one
config works on BTC (funding barely moves) and on meme alts (funding wild).

Entry : funding >= rolling q-quantile   and close > EMA(mean_ema) by stretch_pct -> short
        funding <= rolling (1-q)-quantile and close < EMA(mean_ema) by stretch_pct -> long
Stop  : atr_mult * ATR
Exit  : price reverts to EMA(mean_ema), or funding falls back inside the band
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import Context, Signal, Strategy
from .indicators import atr, ema


class FundingReversion(Strategy):
    name = "funding_reversion"

    def __init__(
        self,
        q: float = 0.85,
        funding_window: int = 180,   # funding prints (~3/day) -> ~60 days
        mean_ema: int = 48,
        stretch_pct: float = 0.006,
        atr_period: int = 14,
        atr_mult: float = 2.5,
        tp_at_mean: bool = True,
    ):
        self.q = q
        self.funding_window = funding_window
        self.mean_ema = mean_ema
        self.stretch_pct = stretch_pct
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.tp_at_mean = tp_at_mean
        self.warmup = max(mean_ema, atr_period, 30) + 5
        self.params = dict(q=q, funding_window=funding_window, mean_ema=mean_ema,
                           stretch_pct=stretch_pct, atr_period=atr_period,
                           atr_mult=atr_mult, tp_at_mean=tp_at_mean)

    def prepare(self, df: pd.DataFrame, funding: pd.Series | None = None) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        f["ema"] = ema(df["close"], self.mean_ema)
        f["atr"] = atr(df, self.atr_period)
        if funding is None or len(funding) == 0:
            f["fr"] = np.nan
            f["fr_hi"] = np.nan
            f["fr_lo"] = np.nan
            return f
        fr = funding.sort_index()
        hi = fr.rolling(self.funding_window, min_periods=20).quantile(self.q)
        lo = fr.rolling(self.funding_window, min_periods=20).quantile(1 - self.q)
        # as-of align funding prints onto bar timestamps (only past prints visible)
        f["fr"] = fr.reindex(df.index, method="ffill")
        f["fr_hi"] = hi.reindex(df.index, method="ffill")
        f["fr_lo"] = lo.reindex(df.index, method="ffill")
        return f

    def on_bar(self, ctx: Context) -> Signal:
        row = ctx.f
        price = ctx.price
        m, a, fr, hi, lo = row["ema"], row["atr"], row["fr"], row["fr_hi"], row["fr_lo"]
        if any(v != v for v in (m, a, fr, hi, lo)) or a <= 0:
            return Signal("hold")

        if ctx.position != 0:
            reverted = (ctx.position == -1 and price <= m) or (ctx.position == 1 and price >= m)
            cooled = lo < fr < hi
            if reverted or cooled:
                return Signal("exit", tag="revert" if reverted else "funding cooled")
            return Signal("hold")

        stretch = (price - m) / m
        tp = m if self.tp_at_mean else None
        if fr >= hi and stretch > self.stretch_pct:
            return Signal("enter_short", sl=price + self.atr_mult * a, tp=tp, tag="fade crowded longs")
        if fr <= lo and stretch < -self.stretch_pct:
            return Signal("enter_long", sl=price - self.atr_mult * a, tp=tp, tag="fade crowded shorts")
        return Signal("hold")
