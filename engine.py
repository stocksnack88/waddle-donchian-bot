"""
Event-driven backtest engine.

Design goals
------------
* No lookahead. On bar i the strategy sees data up to and including bar i (closed).
  Any order it returns is filled at bar i+1's OPEN (plus slippage) - the same
  contract a live bot has: decide on the closed candle, act on the next tick.
* One engine, reused live. `Strategy.on_bar(ctx)` is the only call the live bot
  needs - swap the Backtester loop for a websocket loop.
* Fast. Indicators are computed once, vectorised, in `Strategy.prepare()`. They
  must be causal (ewm / rolling / shift) so precomputing introduces no lookahead.
  `on_bar` then only reads scalars from the current row -> O(n) backtest.
* Risk-based sizing. Every trade risks `risk_pct` of current equity to its stop.
* Costs always on: taker fee + slippage in bps on entry and exit notional, plus
  funding paid/received over the holding window.

Intrabar rule: if a bar's range touches both stop and target, assume STOP first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

from sizing import Instrument, size_position

Action = Literal["enter_long", "enter_short", "exit", "hold"]


@dataclass
class Signal:
    action: Action = "hold"
    sl: Optional[float] = None          # absolute price of stop-loss
    tp: Optional[float] = None          # absolute price of take-profit
    tag: str = ""                       # free-text reason, shows up in the trade log
    limit: Optional[float] = None       # if set + entry_mode='maker': rest a limit here


@dataclass
class Context:
    """
    What the strategy may read. `bars`/`feats` are the FULL frames; `i` is the
    index of the current CLOSED bar. Never read past `i`.
    """
    bars: pd.DataFrame
    feats: pd.DataFrame
    i: int
    position: int                      # -1 short, 0 flat, +1 long
    entry_price: float
    equity: float
    funding: Optional[pd.Series] = None

    @property
    def now(self) -> pd.Timestamp:
        return self.bars.index[self.i]

    @property
    def price(self) -> float:
        return float(self.bars["close"].iat[self.i])

    @property
    def f(self) -> pd.Series:
        """Current feature row."""
        return self.feats.iloc[self.i]

    def fprev(self, col: str, back: int = 1):
        return self.feats[col].iat[self.i - back]

    def window(self, col: str, n: int) -> pd.Series:
        return self.bars[col].iloc[max(0, self.i - n + 1): self.i + 1]


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int
    entry_price: float
    exit_price: float
    qty: float
    notional: float
    pnl: float
    pnl_pct: float
    fees: float
    bars_held: int
    mae_pct: float
    mfe_pct: float
    exit_reason: str
    entry_tag: str
    risk_pct_actual: float = 0.0


class Strategy:
    """Subclass this. Implement `prepare` (vectorised indicators) and `on_bar`."""

    name: str = "unnamed"
    warmup: int = 50

    def prepare(self, df: pd.DataFrame, funding: pd.Series | None = None) -> pd.DataFrame:
        """Return a features DataFrame aligned to df.index. Causal columns only."""
        return pd.DataFrame(index=df.index)

    def on_bar(self, ctx: Context) -> Signal:
        raise NotImplementedError


@dataclass
class Result:
    equity_curve: pd.Series
    trades: pd.DataFrame
    bars: pd.DataFrame
    params: dict = field(default_factory=dict)
    initial_equity: float = 10_000.0
    skipped_trades: int = 0
    skip_reasons: dict = field(default_factory=dict)


class Backtester:
    def __init__(
        self,
        bars: pd.DataFrame,
        strategy: Strategy,
        *,
        initial_equity: float = 10_000.0,
        fee_bps: float = 5.5,               # taker fee
        slippage_bps: float = 2.0,
        risk_pct: float = 0.01,
        max_leverage: float = 5.0,
        funding: pd.DataFrame | None = None,
        allow_short: bool = True,
        instrument: Instrument | None = None,
        max_risk_inflation: float = 3.0,
        entry_mode: str = "taker",         # 'taker' = market at next open; 'maker' = rest a limit
        maker_fee_bps: float = 2.0,         # Bybit VIP0 linear maker
        limit_wait_bars: int = 1,          # bars a resting limit stays live before cancel
        maker_exit_on_target: bool = True,  # in maker mode, target exits fill as maker (no slip)
    ):
        self.bars = bars
        self.strategy = strategy
        self.initial_equity = float(initial_equity)
        self.fee = fee_bps / 10_000.0
        self.maker_fee = maker_fee_bps / 10_000.0
        self.slip = slippage_bps / 10_000.0
        self.risk_pct = risk_pct
        self.max_leverage = max_leverage
        self.funding = (funding["funding_rate"] if funding is not None and len(funding) else None)
        self.allow_short = allow_short
        self.inst = instrument or Instrument.unconstrained()
        self.max_risk_inflation = max_risk_inflation
        self.entry_mode = entry_mode
        self.limit_wait_bars = max(1, limit_wait_bars)
        self.maker_exit_on_target = maker_exit_on_target

    def run(self) -> Result:
        b = self.bars
        feats = self.strategy.prepare(b, self.funding)
        feats = feats.reindex(b.index)

        o = b["open"].to_numpy(float)
        h = b["high"].to_numpy(float)
        l = b["low"].to_numpy(float)
        c = b["close"].to_numpy(float)
        idx = b.index
        n = len(b)
        warmup = max(self.strategy.warmup, 2)

        equity = self.initial_equity
        eq_curve = np.full(n, np.nan)
        eq_prev_close = self.initial_equity

        pos = 0
        qty = 0.0
        entry_px = 0.0
        entry_i = 0
        entry_eq = self.initial_equity
        sl = tp = np.nan
        entry_tag = ""
        entry_fee = 0.0
        run_mae = run_mfe = 0.0
        pending: Signal | None = None
        pending_i = 0
        entry_risk_actual = 0.0
        maker = self.entry_mode == "maker"

        trades: list[Trade] = []
        skipped = 0
        skip_reasons: dict = {}

        for i in range(warmup, n):
            # 1. fill order queued earlier
            if pending is not None:
                act = pending.action
                if act in ("enter_long", "enter_short") and pos == 0:
                    side = 1 if act == "enter_long" else -1
                    use_limit = maker and pending.limit is not None and pending.limit == pending.limit
                    fill = None
                    is_maker_fill = False
                    if side == -1 and not self.allow_short:
                        pending = None
                    elif use_limit:
                        lim = pending.limit
                        touched = (l[i] <= lim) if side == 1 else (h[i] >= lim)
                        if touched:
                            fill, is_maker_fill = lim, True
                        elif i - pending_i >= self.limit_wait_bars:
                            pending = None            # limit expired unfilled
                    else:
                        fill = o[i] * (1 + side * self.slip)   # taker market at open

                    if fill is not None:
                        stop = pending.sl
                        if (stop is None or stop != stop
                                or (side == 1 and stop >= fill) or (side == -1 and stop <= fill)):
                            stop = fill * (1 - side * 0.01)     # fallback 1% stop
                        sized = size_position(
                            equity=equity, entry_price=fill, stop_price=stop,
                            risk_pct=self.risk_pct, inst=self.inst,
                            max_leverage=self.max_leverage,
                            max_risk_inflation=self.max_risk_inflation,
                        )
                        if sized.qty <= 0:
                            skipped += 1
                            skip_reasons[sized.reason] = skip_reasons.get(sized.reason, 0) + 1
                        else:
                            fee_rate = self.maker_fee if is_maker_fill else self.fee
                            entry_fee = sized.qty * fill * fee_rate
                            equity -= entry_fee
                            pos, qty, entry_px, entry_i = side, sized.qty, fill, i
                            entry_eq = eq_prev_close
                            entry_risk_actual = sized.risk_pct_actual
                            sl = stop
                            tp = pending.tp if (pending.tp and pending.tp == pending.tp) else np.nan
                            entry_tag = pending.tag
                            run_mae = run_mfe = 0.0
                        pending = None
                else:
                    pending = None

            # 2. manage open position against THIS bar's range
            if pos != 0:
                adverse = (l[i] - entry_px) / entry_px if pos == 1 else (entry_px - h[i]) / entry_px
                favour = (h[i] - entry_px) / entry_px if pos == 1 else (entry_px - l[i]) / entry_px
                run_mae = min(run_mae, adverse)
                run_mfe = max(run_mfe, favour)

                exit_px = None
                reason = ""
                exit_fee_rate = self.fee
                hit_sl = (l[i] <= sl) if pos == 1 else (h[i] >= sl)
                hit_tp = (tp == tp) and ((h[i] >= tp) if pos == 1 else (l[i] <= tp))
                if hit_sl:
                    exit_px = sl * (1 - pos * self.slip)      # stop = market = taker + slip
                    reason = "stop"
                elif hit_tp:
                    if maker and self.maker_exit_on_target:
                        exit_px = tp                          # resting limit at target: no slip
                        exit_fee_rate = self.maker_fee
                    else:
                        exit_px = tp * (1 - pos * self.slip)
                    reason = "target"

                if exit_px is None:
                    sig = self.strategy.on_bar(self._ctx(b, feats, i, pos, entry_px, equity))
                    if sig.action == "exit":
                        exit_px = c[i] * (1 - pos * self.slip)
                        reason = sig.tag or "signal"

                if exit_px is not None:
                    gross = pos * (exit_px - entry_px) * qty
                    fees_exit = exit_px * qty * exit_fee_rate
                    fees_entry = entry_fee
                    fund_cost = self._funding_cost(idx, entry_i, i, pos, entry_px, qty)
                    equity += gross - fees_exit - fund_cost
                    net = gross - fees_exit - fees_entry - fund_cost
                    trades.append(Trade(
                        entry_time=idx[entry_i], exit_time=idx[i], side=pos,
                        entry_price=entry_px, exit_price=exit_px, qty=qty,
                        notional=entry_px * qty, pnl=net,
                        pnl_pct=net / max(entry_eq, 1e-9),
                        fees=fees_exit + fees_entry + fund_cost,
                        bars_held=i - entry_i, mae_pct=run_mae, mfe_pct=run_mfe,
                        exit_reason=reason, entry_tag=entry_tag,
                        risk_pct_actual=entry_risk_actual,
                    ))
                    pos, qty, entry_px = 0, 0.0, 0.0
                    sl = tp = np.nan

            # 3. if flat, ask for a new entry (decided on this closed bar)
            if pos == 0 and pending is None:
                sig = self.strategy.on_bar(self._ctx(b, feats, i, 0, 0.0, equity))
                if sig.action in ("enter_long", "enter_short"):
                    pending = sig
                    pending_i = i

            # 4. mark to market on the close
            eq_curve[i] = equity + (pos * (c[i] - entry_px) * qty if pos != 0 else 0.0)
            eq_prev_close = eq_curve[i]

        eq = pd.Series(eq_curve, index=idx).iloc[warmup:].ffill().fillna(self.initial_equity)
        return Result(
            equity_curve=eq,
            trades=_trades_to_df(trades),
            bars=b.iloc[warmup:],
            params={**getattr(self.strategy, "params", {}),
                    "fee_bps": self.fee * 10_000, "slippage_bps": self.slip * 10_000,
                    "risk_pct": self.risk_pct, "strategy": self.strategy.name,
                    "entry_mode": self.entry_mode},
            initial_equity=self.initial_equity,
            skipped_trades=skipped,
            skip_reasons=skip_reasons,
        )

    def _ctx(self, b, feats, i, pos, entry_px, equity) -> Context:
        return Context(bars=b, feats=feats, i=i, position=pos,
                       entry_price=entry_px, equity=equity, funding=self.funding)

    def _funding_cost(self, idx, i0, i1, side, entry_px, qty) -> float:
        if self.funding is None:
            return 0.0
        window = self.funding[(self.funding.index > idx[i0]) & (self.funding.index <= idx[i1])]
        if window.empty:
            return 0.0
        return float(side * window.sum() * entry_px * qty)


def _trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    cols = ["entry_time", "exit_time", "side", "entry_price", "exit_price", "qty",
            "notional", "pnl", "pnl_pct", "fees", "bars_held", "mae_pct", "mfe_pct",
            "exit_reason", "entry_tag", "risk_pct_actual"]
    if not trades:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([t.__dict__ for t in trades])[cols]
