"""
Walk-forward validation.

Roll a train/test window across history. On each fold: pick the best parameter
set using ONLY the train slice, then trade it on the untouched test slice.
Stitch the test-slice returns into one out-of-sample curve.

If that stitched curve is still healthy - and the chosen parameters don't jump
around wildly fold to fold - the edge is probably real and not hindsight.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine import Backtester
from sizing import Instrument
import metrics as M


@dataclass
class WFFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_end: pd.Timestamp
    chosen: dict
    train_sharpe: float
    test_return_pct: float
    test_sharpe: float
    test_trades: int


@dataclass
class WFResult:
    equity_curve: pd.Series          # stitched out-of-sample
    folds: list[WFFold]
    initial_equity: float
    summary: dict = field(default_factory=dict)


def _grid(param_grid: dict) -> list[dict]:
    keys = list(param_grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*param_grid.values())]


def _score(m: dict, min_trades: int, max_dd: float) -> float:
    """Objective for picking a config on the train slice. Higher = better."""
    if m["trades"] < min_trades or abs(m["max_dd_pct"]) > max_dd * 100:
        return -1e9
    s = m["sharpe"] if m["sharpe"] == m["sharpe"] else -1e9
    return s + 0.01 * m["cagr_pct"]        # Sharpe first, CAGR as tiebreak


def walk_forward(
    *,
    strategy_cls,
    param_grid: dict,
    bars: pd.DataFrame,
    timeframe: str,
    funding: pd.DataFrame | None = None,
    instrument: Instrument | None = None,
    initial_equity: float = 50.0,
    train_days: int = 300,
    test_days: int = 90,
    fee_bps: float = 5.5,
    slippage_bps: float = 5.0,
    risk_pct: float = 0.01,
    max_leverage: float = 5.0,
    min_trades: int = 20,
    max_dd: float = 0.35,
    fixed: dict | None = None,
) -> WFResult:
    fixed = fixed or {}
    combos = _grid(param_grid)
    idx = bars.index
    t0 = idx[0]
    end = idx[-1]
    train_td = pd.Timedelta(days=train_days)
    test_td = pd.Timedelta(days=test_days)
    buf = pd.Timedelta(days=40)          # indicator warm-up lead-in for test slices

    folds: list[WFFold] = []
    stitched = [pd.Series([initial_equity], index=[t0])]
    equity = initial_equity

    cur = t0
    while cur + train_td + test_td <= end:
        tr_start, tr_end = cur, cur + train_td
        te_end = tr_end + test_td
        train = bars.loc[tr_start:tr_end]

        best, best_score, best_m = None, -np.inf, None
        for combo in combos:
            strat = strategy_cls(**{**fixed, **combo})
            res = Backtester(train, strat, funding=funding, instrument=instrument,
                             initial_equity=initial_equity, fee_bps=fee_bps,
                             slippage_bps=slippage_bps, risk_pct=risk_pct,
                             max_leverage=max_leverage).run()
            m = M.compute(res, timeframe)
            sc = _score(m, min_trades, max_dd)
            if sc > best_score:
                best, best_score, best_m = combo, sc, m

        # apply best config to the untouched test slice (with a warm-up buffer)
        test = bars.loc[tr_end - buf:te_end]
        strat = strategy_cls(**{**fixed, **best})
        res = Backtester(test, strat, funding=funding, instrument=instrument,
                         initial_equity=equity, fee_bps=fee_bps,
                         slippage_bps=slippage_bps, risk_pct=risk_pct,
                         max_leverage=max_leverage).run()
        tm = M.compute(res, timeframe)
        te_curve = res.equity_curve[res.equity_curve.index >= tr_end]
        if len(te_curve):
            stitched.append(te_curve)
            equity = float(te_curve.iloc[-1])

        folds.append(WFFold(
            train_start=tr_start, train_end=tr_end, test_end=te_end, chosen=best,
            train_sharpe=best_m["sharpe"], test_return_pct=tm["return_pct"],
            test_sharpe=tm["sharpe"], test_trades=tm["trades"],
        ))
        cur = cur + test_td

    eq = pd.concat(stitched)
    eq = eq[~eq.index.duplicated(keep="last")].sort_index()

    rets = eq.pct_change().dropna()
    per_year = M._TF_PER_YEAR.get(timeframe, 8760)
    yrs = max(len(eq) / per_year, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1
    sharpe = rets.mean() / rets.std() * np.sqrt(per_year) if rets.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    prof_folds = np.mean([f.test_return_pct > 0 for f in folds]) * 100 if folds else 0

    # parameter stability: how often each param's chosen value changes fold to fold
    stability = {}
    for key in param_grid:
        vals = [f.chosen[key] for f in folds]
        changes = sum(1 for a, b in zip(vals, vals[1:]) if a != b)
        stability[key] = {"values": vals, "changes": changes,
                          "n_distinct": len(set(vals))}

    return WFResult(
        equity_curve=eq, folds=folds, initial_equity=initial_equity,
        summary=dict(
            folds=len(folds),
            oos_return_pct=(eq.iloc[-1] / eq.iloc[0] - 1) * 100,
            oos_cagr_pct=cagr * 100, oos_sharpe=sharpe, oos_max_dd_pct=dd * 100,
            profitable_folds_pct=prof_folds, final_equity=float(eq.iloc[-1]),
            param_stability=stability,
        ),
    )
