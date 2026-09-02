"""
Multi-instrument portfolio backtester.

One shared account. Each leg (symbol + strategy) decides independently via the
same `Strategy.on_bar` contract, but position sizing risks a % of TOTAL equity
and an optional cap limits how many legs can be in the market at once.

This is what the live bot runs: same legs, same sizing, loop driven by streamed
candles instead of a historical frame.

Assumes all legs share one bar timeline (e.g. all 4h). Legs are aligned to the
intersection of their indices.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine import Signal, Strategy, Context, _trades_to_df, Trade
from sizing import Instrument, size_position


@dataclass
class Leg:
    symbol: str
    bars: pd.DataFrame
    strategy: Strategy
    funding: pd.DataFrame | None = None
    instrument: Instrument | None = None


@dataclass
class PortfolioResult:
    equity_curve: pd.Series
    trades: pd.DataFrame                 # all legs, with a 'symbol' column
    per_symbol: dict                     # symbol -> trades DataFrame
    initial_equity: float
    params: dict = field(default_factory=dict)
    skipped_trades: int = 0
    skip_reasons: dict = field(default_factory=dict)


class _LegState:
    __slots__ = ("leg", "feats", "fund", "inst", "pos", "qty", "entry_px", "entry_i",
                 "entry_eq", "sl", "tp", "tag", "mae", "mfe", "pending", "risk_actual")

    def __init__(self, leg: Leg, index: pd.DatetimeIndex):
        self.leg = leg
        self.feats = leg.strategy.prepare(leg.bars, _fund_series(leg.funding)).reindex(index)
        self.fund = _fund_series(leg.funding)
        self.inst = leg.instrument or Instrument.unconstrained()
        self.pos = 0
        self.qty = 0.0
        self.entry_px = 0.0
        self.entry_i = 0
        self.entry_eq = 0.0
        self.sl = self.tp = np.nan
        self.tag = ""
        self.mae = self.mfe = 0.0
        self.pending: Signal | None = None
        self.risk_actual = 0.0


def _fund_series(funding):
    if funding is None or len(funding) == 0:
        return None
    return funding["funding_rate"] if "funding_rate" in getattr(funding, "columns", []) else funding


class PortfolioBacktester:
    def __init__(
        self,
        legs: list[Leg],
        *,
        initial_equity: float = 50.0,
        fee_bps: float = 5.5,
        slippage_bps: float = 5.0,
        risk_pct: float = 0.01,
        max_leverage: float = 5.0,
        max_concurrent: int | None = None,
        max_risk_inflation: float = 3.0,
    ):
        self.legs = legs
        self.initial_equity = float(initial_equity)
        self.fee = fee_bps / 10_000.0
        self.slip = slippage_bps / 10_000.0
        self.risk_pct = risk_pct
        self.max_leverage = max_leverage
        self.max_concurrent = max_concurrent or len(legs)
        self.max_risk_inflation = max_risk_inflation

    def run(self) -> PortfolioResult:
        # shared timeline = intersection of all leg indices
        idx = None
        for leg in self.legs:
            idx = leg.bars.index if idx is None else idx.intersection(leg.bars.index)
        idx = idx.sort_values()

        states = [_LegState(leg, idx) for leg in self.legs]
        arrs = []
        for leg in self.legs:
            b = leg.bars.reindex(idx)
            arrs.append(dict(o=b["open"].to_numpy(float), h=b["high"].to_numpy(float),
                             l=b["low"].to_numpy(float), c=b["close"].to_numpy(float), b=b))

        warmup = max(max(s.leg.strategy.warmup for s in states), 2)
        n = len(idx)
        cash = self.initial_equity
        eq_curve = np.full(n, np.nan)
        eq_prev = self.initial_equity
        trades: list[tuple[str, Trade]] = []
        skipped = 0
        skip_reasons: dict = {}

        def n_open():
            return sum(1 for s in states if s.pos != 0)

        for i in range(warmup, n):
            # ---- fills queued last bar
            for k, s in enumerate(states):
                a = arrs[k]
                if s.pending is None:
                    continue
                act = s.pending.action
                if act in ("enter_long", "enter_short") and s.pos == 0 and n_open() < self.max_concurrent:
                    side = 1 if act == "enter_long" else -1
                    fill = a["o"][i] * (1 + side * self.slip)
                    stop = s.pending.sl
                    if (stop is None or stop != stop
                            or (side == 1 and stop >= fill) or (side == -1 and stop <= fill)):
                        stop = fill * (1 - side * 0.01)
                    sized = size_position(equity=eq_prev, entry_price=fill, stop_price=stop,
                                          risk_pct=self.risk_pct, inst=s.inst,
                                          max_leverage=self.max_leverage,
                                          max_risk_inflation=self.max_risk_inflation)
                    if sized.qty <= 0:
                        skipped += 1
                        skip_reasons[sized.reason] = skip_reasons.get(sized.reason, 0) + 1
                    else:
                        cash -= sized.qty * fill * self.fee
                        s.pos, s.qty, s.entry_px, s.entry_i = side, sized.qty, fill, i
                        s.entry_eq = eq_prev
                        s.risk_actual = sized.risk_pct_actual
                        s.sl = stop
                        s.tp = s.pending.tp if (s.pending.tp and s.pending.tp == s.pending.tp) else np.nan
                        s.tag = s.pending.tag
                        s.mae = s.mfe = 0.0
                s.pending = None

            # ---- manage open positions
            for k, s in enumerate(states):
                if s.pos == 0:
                    continue
                a = arrs[k]
                hi, lo, cl = a["h"][i], a["l"][i], a["c"][i]
                adverse = (lo - s.entry_px) / s.entry_px if s.pos == 1 else (s.entry_px - hi) / s.entry_px
                favour = (hi - s.entry_px) / s.entry_px if s.pos == 1 else (s.entry_px - lo) / s.entry_px
                s.mae = min(s.mae, adverse)
                s.mfe = max(s.mfe, favour)

                exit_px = None
                reason = ""
                hit_sl = (lo <= s.sl) if s.pos == 1 else (hi >= s.sl)
                hit_tp = (s.tp == s.tp) and ((hi >= s.tp) if s.pos == 1 else (lo <= s.tp))
                if hit_sl:
                    exit_px, reason = s.sl * (1 - s.pos * self.slip), "stop"
                elif hit_tp:
                    exit_px, reason = s.tp * (1 - s.pos * self.slip), "target"

                if exit_px is None:
                    ctx = Context(bars=a["b"], feats=s.feats, i=i, position=s.pos,
                                  entry_price=s.entry_px, equity=eq_prev, funding=s.fund)
                    sig = s.leg.strategy.on_bar(ctx)
                    if sig.action == "exit":
                        exit_px, reason = cl * (1 - s.pos * self.slip), (sig.tag or "signal")

                if exit_px is not None:
                    gross = s.pos * (exit_px - s.entry_px) * s.qty
                    fees_exit = exit_px * s.qty * self.fee
                    fees_entry = s.entry_px * s.qty * self.fee
                    fund_cost = _funding_cost(s.fund, idx, s.entry_i, i, s.pos, s.entry_px, s.qty)
                    cash += gross - fees_exit - fund_cost
                    net = gross - fees_exit - fees_entry - fund_cost
                    trades.append((s.leg.symbol, Trade(
                        entry_time=idx[s.entry_i], exit_time=idx[i], side=s.pos,
                        entry_price=s.entry_px, exit_price=exit_px, qty=s.qty,
                        notional=s.entry_px * s.qty, pnl=net,
                        pnl_pct=net / max(s.entry_eq, 1e-9),
                        fees=fees_exit + fees_entry + fund_cost,
                        bars_held=i - s.entry_i, mae_pct=s.mae, mfe_pct=s.mfe,
                        exit_reason=reason, entry_tag=s.tag, risk_pct_actual=s.risk_actual,
                    )))
                    s.pos, s.qty, s.entry_px = 0, 0.0, 0.0
                    s.sl = s.tp = np.nan

            # ---- new entries (decided on this closed bar)
            for k, s in enumerate(states):
                if s.pos != 0 or s.pending is not None:
                    continue
                a = arrs[k]
                ctx = Context(bars=a["b"], feats=s.feats, i=i, position=0,
                              entry_price=0.0, equity=eq_prev, funding=s.fund)
                sig = s.leg.strategy.on_bar(ctx)
                if sig.action in ("enter_long", "enter_short"):
                    s.pending = sig

            # ---- mark to market
            unreal = 0.0
            for k, s in enumerate(states):
                if s.pos != 0:
                    unreal += s.pos * (arrs[k]["c"][i] - s.entry_px) * s.qty
            eq_curve[i] = cash + unreal
            eq_prev = eq_curve[i]

        eq = pd.Series(eq_curve, index=idx).iloc[warmup:].ffill().fillna(self.initial_equity)
        all_tr = _trades_to_df([t for _, t in trades])
        if len(all_tr):
            all_tr.insert(0, "symbol", [sym for sym, _ in trades])
        per_symbol = {sym: all_tr[all_tr["symbol"] == sym].reset_index(drop=True)
                      for sym in {s for s, _ in trades}} if len(all_tr) else {}
        return PortfolioResult(
            equity_curve=eq, trades=all_tr, per_symbol=per_symbol,
            initial_equity=self.initial_equity,
            params=dict(risk_pct=self.risk_pct, fee_bps=self.fee * 1e4,
                        slippage_bps=self.slip * 1e4, max_concurrent=self.max_concurrent,
                        legs=[l.symbol for l in self.legs]),
            skipped_trades=skipped, skip_reasons=skip_reasons,
        )


def _funding_cost(fund, idx, i0, i1, side, entry_px, qty) -> float:
    if fund is None:
        return 0.0
    w = fund[(fund.index > idx[i0]) & (fund.index <= idx[i1])]
    if w.empty:
        return 0.0
    return float(side * w.sum() * entry_px * qty)
