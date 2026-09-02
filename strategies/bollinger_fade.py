"""
Bollinger Band fade - mean reversion, aimed at 5-15m.

Entry : close prints below the lower band -> long (above upper -> short).
        Limit rests AT the band (natural maker entry).
Target: middle band (the SMA). In maker mode this exit fills as a limit -> low cost.
Stop  : band +/- stop_atr * ATR  (or a hard % if ATR unavailable)
Filter: optional - only fade WITH the higher trend (close vs EMA(trend_ema)),
        or only fade in low-ADX (chop) regimes.

High win rate by design, small wins, occasional bigger loss when a trend runs.
"""
from __future__ import annotations

import pandas as pd

from engine import Context, Signal, Strategy
from .indicators import adx, atr


class BollingerFade(Strategy):
    name = "bollinger_fade"

    def __init__(
        self,
        period: int = 20,
        k: float = 2.0,
        stop_atr: float = 1.5,
        atr_period: int = 14,
        trend_ema: int = 0,        # 0 disables; >0 = only fade toward this trend
        adx_max: float = 0.0,      # 0 disables; only trade when ADX below this
        min_band_pct: float = 0.0,  # require band half-width >= this frac of price
    ):
        self.period = period
        self.k = k
        self.stop_atr = stop_atr
        self.atr_period = atr_period
        self.trend_ema = trend_ema
        self.adx_max = adx_max
        self.min_band_pct = min_band_pct
        self.warmup = max(period, atr_period, trend_ema, 30) + 5
        self.params = dict(period=period, k=k, stop_atr=stop_atr, atr_period=atr_period,
                           trend_ema=trend_ema, adx_max=adx_max, min_band_pct=min_band_pct)

    def prepare(self, df: pd.DataFrame, funding=None) -> pd.DataFrame:
        c = df["close"]
        mid = c.rolling(self.period).mean()
        sd = c.rolling(self.period).std()
        f = pd.DataFrame(index=df.index)
        f["mid"] = mid
        f["lower"] = mid - self.k * sd
        f["upper"] = mid + self.k * sd
        f["atr"] = atr(df, self.atr_period)
        f["trend"] = c.ewm(span=self.trend_ema, adjust=False).mean() if self.trend_ema else float("nan")
        f["adx"] = adx(df, 14) if self.adx_max else 0.0
        return f

    def on_bar(self, ctx: Context) -> Signal:
        row = ctx.f
        price = ctx.price
        mid, lo, up, a = row["mid"], row["lower"], row["upper"], row["atr"]
        if any(v != v for v in (mid, lo, up, a)) or a <= 0:
            return Signal("hold")

        if ctx.position != 0:
            # exit handled by target(mid)/stop; also bail if price crosses mid
            if (ctx.position == 1 and price >= mid) or (ctx.position == -1 and price <= mid):
                return Signal("exit", tag="hit mid")
            return Signal("hold")

        if self.min_band_pct and (up - mid) / price < self.min_band_pct:
            return Signal("hold")
        if self.adx_max and row["adx"] > self.adx_max:
            return Signal("hold")

        long_ok = short_ok = True
        if self.trend_ema:
            long_ok, short_ok = price > row["trend"], price < row["trend"]

        if price < lo and long_ok:
            sl = lo - self.stop_atr * a
            return Signal("enter_long", sl=sl, tp=mid, limit=lo, tag="fade lower band")
        if price > up and short_ok:
            sl = up + self.stop_atr * a
            return Signal("enter_short", sl=sl, tp=mid, limit=up, tag="fade upper band")
        return Signal("hold")
