"""Performance metrics computed from an engine.Result."""
from __future__ import annotations

import numpy as np
import pandas as pd

_TF_PER_YEAR = {
    "1m": 525_600, "3m": 175_200, "5m": 105_120, "15m": 35_040, "30m": 17_520,
    "1h": 8_760, "2h": 4_380, "4h": 2_190, "6h": 1_460, "12h": 730, "1d": 365,
}

MIN_TRADES = 30            # Algovibes-style gate
SHARPE_LO, SHARPE_HI = 0.5, 2.5
MAX_DD_LIMIT = 0.35


def compute(result, timeframe: str) -> dict:
    eq = result.equity_curve
    tr = result.trades
    e0 = result.initial_equity
    bars_per_year = _TF_PER_YEAR.get(timeframe, 8_760)

    ret_total = eq.iloc[-1] / e0 - 1.0 if len(eq) else 0.0
    years = max(len(eq) / bars_per_year, 1e-9)
    cagr = (eq.iloc[-1] / e0) ** (1 / years) - 1.0 if len(eq) and eq.iloc[-1] > 0 else -1.0

    bar_ret = eq.pct_change().dropna()
    if len(bar_ret) > 2 and bar_ret.std() > 0:
        sharpe = bar_ret.mean() / bar_ret.std() * np.sqrt(bars_per_year)
        downside = bar_ret[bar_ret < 0].std()
        sortino = bar_ret.mean() / downside * np.sqrt(bars_per_year) if downside and downside > 0 else np.nan
    else:
        sharpe = sortino = np.nan

    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0
    # longest time under water, in bars
    underwater = (dd < 0).astype(int)
    uw_len = _max_run(underwater.to_numpy())

    n = len(tr)
    wins = tr[tr["pnl"] > 0] if n else tr
    losses = tr[tr["pnl"] <= 0] if n else tr
    win_rate = len(wins) / n if n else 0.0
    gross_win = wins["pnl"].sum() if n else 0.0
    gross_loss = -losses["pnl"].sum() if n else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf if gross_win > 0 else 0.0
    avg_win = wins["pnl"].mean() if len(wins) else 0.0
    avg_loss = losses["pnl"].mean() if len(losses) else 0.0
    expectancy = tr["pnl"].mean() if n else 0.0
    expectancy_r = (tr["pnl"] / (result.initial_equity * result.params.get("risk_pct", 0.01))).mean() if n else 0.0
    avg_bars = tr["bars_held"].mean() if n else 0.0
    total_fees = tr["fees"].sum() if n else 0.0

    exposure = float((tr["bars_held"].sum() / len(eq))) if n and len(eq) else 0.0

    passed = bool(
        n >= MIN_TRADES
        and (not np.isnan(sharpe)) and SHARPE_LO <= sharpe <= SHARPE_HI
        and abs(max_dd) <= MAX_DD_LIMIT
    )

    return {
        "trades": n,
        "return_pct": ret_total * 100,
        "cagr_pct": cagr * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd_pct": max_dd * 100,
        "underwater_bars": int(uw_len),
        "win_rate_pct": win_rate * 100,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "expectancy_r": expectancy_r,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_bars_held": avg_bars,
        "exposure_pct": exposure * 100,
        "total_fees": total_fees,
        "final_equity": float(eq.iloc[-1]) if len(eq) else e0,
        "passed_filter": passed,
    }


def monthly_table(result) -> pd.DataFrame:
    """Return per-calendar-month P&L and trade count."""
    tr = result.trades
    if not len(tr):
        return pd.DataFrame(columns=["month", "pnl", "trades", "win_rate"])
    g = tr.assign(month=tr["exit_time"].dt.to_period("M").astype(str)).groupby("month")
    out = g.agg(pnl=("pnl", "sum"), trades=("pnl", "size"),
               win_rate=("pnl", lambda s: (s > 0).mean() * 100)).reset_index()
    return out


def split_in_out(result, timeframe: str, oos_frac: float = 0.3) -> tuple[dict, dict]:
    """Recompute metrics on the first (1-oos_frac) vs last oos_frac of the equity curve."""
    eq = result.equity_curve
    if len(eq) < 20:
        return {}, {}
    cut = eq.index[int(len(eq) * (1 - oos_frac))]

    def _slice(lo, hi):
        sub_eq = eq[(eq.index >= lo) & (eq.index < hi)] if hi else eq[eq.index >= lo]
        sub_tr = result.trades[(result.trades["entry_time"] >= lo) &
                               ((result.trades["entry_time"] < hi) if hi else True)]
        r = type(result)(equity_curve=sub_eq / sub_eq.iloc[0] * result.initial_equity if len(sub_eq) else sub_eq,
                         trades=sub_tr.reset_index(drop=True),
                         bars=result.bars, params=result.params,
                         initial_equity=result.initial_equity)
        return compute(r, timeframe)

    return _slice(eq.index[0], cut), _slice(cut, None)


def _max_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best
