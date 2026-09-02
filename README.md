# Perp backtest lab

A small, honest, visual research tool for Bybit perpetual strategies. Built to
answer one question fast: *does this idea have an edge after costs, or not?*

## Run

```bash
cd ~/crypto-backtest/research
~/Library/Python/3.9/bin/streamlit run app.py
```

Then open http://localhost:8501. Data is fetched from Bybit's public API on first
use and cached to `.cache/*.parquet` (incremental refresh after that).

## What's here

| File | Job |
|---|---|
| `data.py` | Fetch + cache OHLCV and funding-rate history from Bybit (no API key) |
| `engine.py` | Event-driven backtester. `Strategy.on_bar(ctx)` is the same call the live bot will make |
| `metrics.py` | CAGR, Sharpe, Sortino, maxDD, profit factor, expectancy (R), exposure, in/out-of-sample split, monthly table |
| `montecarlo.py` | Bootstrap trade order → confidence bands on return and drawdown |
| `strategies/` | One file per strategy. `prepare()` = vectorised indicators, `on_bar()` = decision |
| `app.py` | Streamlit dashboard |

## Design rules that keep it honest

1. **No lookahead.** On bar `i` the strategy sees data up to and including bar `i`
   (closed). Orders fill at bar `i+1`'s **open**, plus slippage. Indicators are
   precomputed but only with causal ops (`ewm`, `rolling`, `shift`).
2. **Costs always on.** Taker fee (default 5.5 bps) + slippage (2 bps) on entry
   and exit notional, plus funding paid/received over the holding window.
3. **Risk-based sizing.** Every trade risks a fixed % of equity to its stop, so
   strategies with different stop distances stay comparable.
4. **Intrabar tie-break:** if a bar touches both stop and target, assume the stop.
5. **Validation gate** (shown pass/fail): ≥30 trades, Sharpe 0.5–2.5, maxDD ≤35%.
6. **Monte Carlo** tells you if an equity curve is skill or a lucky ordering.

## Adding a strategy

```python
# strategies/my_strat.py
from engine import Context, Signal, Strategy
import pandas as pd

class MyStrat(Strategy):
    name = "my_strat"
    def __init__(self, foo: int = 10, bar: float = 1.5):
        self.foo, self.bar = foo, bar
        self.warmup = foo + 5
        self.params = dict(foo=foo, bar=bar)          # shows up in results

    def prepare(self, df, funding=None) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        f["sma"] = df["close"].rolling(self.foo).mean()   # causal only
        return f

    def on_bar(self, ctx: Context) -> Signal:
        if ctx.position == 0 and ctx.price > ctx.f["sma"]:
            sl = ctx.price * 0.98
            return Signal("enter_long", sl=sl, tp=ctx.price * 1.04, tag="above sma")
        return Signal("hold")
```

Then register it in `strategies/__init__.py`. The `__init__` signature drives the
sidebar widgets automatically (int → stepper, float → number, bool → checkbox).

## The $50 bot

- `pages/1_50_dollar_bot.py` - Streamlit page: portfolio backtest + walk-forward
  for the XLM+XRP 4h Donchian basket.
- `portfolio.py` - `PortfolioBacktester`: one shared account, N legs, risk % of
  total equity per trade, `max_concurrent` cap.
- `walkforward.py` - roll a 300d-train / 90d-test window, pick params on train,
  trade them on the untouched test slice, stitch the out-of-sample curve.
- `sizing.py` - `size_position()` applies real Bybit constraints (min qty, qty
  step, $5 min notional). On a $50 account it flags when the exchange minimum
  forces more risk than intended; skips the trade past 3x.
- `bot.py` - **live paper-trading bot**. Same `Strategy.on_bar` code as the
  backtest. Polls Bybit REST, keeps a rolling frame, routes Signals through a
  `PaperBroker` (no real money). State + trades persisted to
  `bot_state.json` / `paper_trades.jsonl`. Run: `python3 bot.py` (loop) or
  `python3 bot.py --once`. Going live = swap `PaperBroker.market_order` for a
  real Bybit order call.

## Findings

**Costs are the constraint.** Donchian 1h BTC: +18.9% zero-fee → −20.4% with
real taker fees (296 trades, $2,709 fees / $10k). 15m: −96% (1,234 trades). Fix
= fewer, higher-timeframe trades.

**The edge: Donchian breakout, 4h, cheap alts.** Config `lookback=20,
atr_period=14, atr_mult=2.0, rr=2.0, ema_filter=200`, ~2.7yr (Dec 2023-Aug 2026),
5 bps slippage, $50 account, real instrument constraints:

| | single-config backtest | walk-forward (out-of-sample) |
|---|---|---|
| XLM | 12.8% CAGR, Sharpe 1.15, DD −8% | 6% CAGR, Sharpe 0.58, 71% folds + |
| XRP | 10.7% CAGR, Sharpe 0.94, DD −14% | 10.6% CAGR, Sharpe 0.88 |
| XLM+XRP basket, 0.5% risk/leg | 14.3% CAGR, Sharpe 1.25, DD −9% | ~8.5% CAGR, Sharpe 1.07, DD −6% |

- $50 mechanics are clean: **0 skipped trades**, actual risk holds ~1%/trade.
  Account floor ~$20-30; below $10 the exchange minimums break sizing.
- 62% of months positive, but **~85% of the profit is 2 months** (Nov 2024,
  Jul 2025). It's trend-following: bleed small, a few big moves make the year.
- TRX/ADA marginal, DOGE weak (OOS breaks down), SOL fails.
- Funding-rate reversion: no edge found yet; needs wilder-funding alts than these.

**Higher frequency (5m / 15m) does not work at retail fees.** Swept Bollinger
fade + RSI(2) reversion + Donchian on 5m/15m across XLM/XRP/DOGE/ADA, taker AND
maker (limit entries + maker-fee target exits, modeled in `engine.py`). Every
combination lost, −45% to −98%. Maker mean-reversion reaches a 55-61% win rate
(the "consistent win rate" goal) but per-trade expectancy stays negative: at
1,000-2,000 trades, fees are 15-56% of gross P&L. Even hyper-selective 15m RSI(2)
(~320 trades/400d, maker) = 56% wins but −0.07 R, −22% return. Trade count is the
killer variable — the 4h basket works because it trades ~130x/year, not 1,500x.
To make HF viable you'd need a maker-rebate fee tier or a genuinely better signal
than band-fade / RSI(2). `sweep_hf.py` reproduces this.
