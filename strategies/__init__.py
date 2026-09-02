"""Strategy registry. Each module exposes a Strategy subclass; register it here."""
from __future__ import annotations

from .donchian_atr import DonchianATR
from .ema_cross import EmaCross
from .funding_reversion import FundingReversion
from .bollinger_fade import BollingerFade
from .rsi2_reversion import RSI2Reversion

REGISTRY = {
    "Donchian breakout + ATR stop": DonchianATR,
    "EMA cross": EmaCross,
    "Funding-rate reversion": FundingReversion,
    "Bollinger fade (mean reversion)": BollingerFade,
    "RSI(2) reversion (mean reversion)": RSI2Reversion,
}
