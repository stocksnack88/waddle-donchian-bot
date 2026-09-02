"""
Monte Carlo on trade order.

Take the realised per-trade returns, resample them (bootstrap with replacement)
many times, and rebuild equity curves. Tells you whether the backtest's drawdown
and final return are typical or a lucky ordering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def resample_trades(result, n_runs: int = 1000, seed: int = 7) -> dict:
    tr = result.trades
    e0 = result.initial_equity
    if len(tr) < 5:
        return {"ok": False, "reason": "need >=5 trades"}

    # per-trade return on equity, in order
    r = (tr["pnl"] / e0).to_numpy()
    rng = np.random.default_rng(seed)
    m = len(r)

    finals = np.empty(n_runs)
    max_dds = np.empty(n_runs)
    for k in range(n_runs):
        draw = rng.choice(r, size=m, replace=True)
        eq = e0 * np.cumprod(1 + draw)
        peak = np.maximum.accumulate(eq)
        dd = (eq / peak - 1.0).min()
        finals[k] = eq[-1] / e0 - 1.0
        max_dds[k] = dd

    return {
        "ok": True,
        "n_runs": n_runs,
        "return_p05": float(np.percentile(finals, 5) * 100),
        "return_p50": float(np.percentile(finals, 50) * 100),
        "return_p95": float(np.percentile(finals, 95) * 100),
        "maxdd_p05": float(np.percentile(max_dds, 5) * 100),   # worst
        "maxdd_p50": float(np.percentile(max_dds, 50) * 100),
        "maxdd_p95": float(np.percentile(max_dds, 95) * 100),   # mildest
        "prob_profit": float((finals > 0).mean() * 100),
        "prob_dd_gt_35": float((max_dds < -0.35).mean() * 100),
        "finals": finals,
        "max_dds": max_dds,
    }
