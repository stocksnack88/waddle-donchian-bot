"""
Position sizing with real exchange constraints.

Shared by the single-instrument engine and the portfolio engine so both size
trades the same way. The point of this module: on a $50 account the exchange
minimums (min order qty, min $5 notional) can force a position bigger than your
intended risk. This makes that visible instead of silently pretending you can
trade 0.037 of a contract.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Instrument:
    symbol: str = "GENERIC"
    min_qty: float = 0.0
    qty_step: float = 0.0
    min_notional: float = 0.0
    tick_size: float = 0.0
    max_leverage: float = 100.0

    @classmethod
    def unconstrained(cls) -> "Instrument":
        return cls()


@dataclass
class Sized:
    qty: float                 # 0.0 => do not take the trade
    risk_pct_intended: float
    risk_pct_actual: float
    notional: float
    reason: str = ""           # why skipped, if qty == 0


def round_step(x: float, step: float, mode: str = "down") -> float:
    if step <= 0:
        return x
    n = x / step
    n = math.floor(n) if mode == "down" else math.ceil(n)
    return round(n * step, 12)


def size_position(
    *,
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float,
    inst: Instrument,
    max_leverage: float,
    max_risk_inflation: float = 3.0,
) -> Sized:
    """
    Risk `risk_pct` of equity to the stop, then snap to the instrument's grid.
    If the exchange minimum inflates real risk past `max_risk_inflation`x the
    intended amount, skip the trade (the account is too small for this stop).
    """
    risk_per_unit = abs(entry_price - stop_price)
    if risk_per_unit <= 0 or equity <= 0:
        return Sized(0.0, risk_pct, 0.0, 0.0, "bad stop / equity")

    dollar_risk = equity * risk_pct
    qty = dollar_risk / risk_per_unit

    lev_cap_qty = equity * max_leverage / entry_price
    qty = min(qty, lev_cap_qty)

    qty = round_step(qty, inst.qty_step, "down")

    # exchange minimums can push size UP
    if inst.min_qty and qty < inst.min_qty:
        qty = inst.min_qty
    if inst.min_notional and qty * entry_price < inst.min_notional:
        need = inst.min_notional / entry_price
        qty = max(qty, round_step(need, inst.qty_step, "up") or need)

    if qty <= 0:
        return Sized(0.0, risk_pct, 0.0, 0.0, "rounds to zero")

    notional = qty * entry_price
    if notional > lev_cap_qty * entry_price + 1e-9:
        return Sized(0.0, risk_pct, 0.0, notional, "min size exceeds leverage cap")

    risk_actual = qty * risk_per_unit / equity
    if risk_actual > risk_pct * max_risk_inflation:
        return Sized(0.0, risk_pct, risk_actual, notional,
                     f"min size forces {risk_actual*100:.1f}% risk "
                     f"(> {max_risk_inflation:.0f}x intended)")

    return Sized(qty, risk_pct, risk_actual, notional)
