"""
Streamlit research app for crypto perp strategies.

Run:  cd ~/crypto-backtest/research && streamlit run app.py
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data as datamod
import metrics as metricsmod
import montecarlo as mc
from engine import Backtester
from strategies import REGISTRY

st.set_page_config(page_title="Perp backtest lab", layout="wide")

TF_CHOICES = ["5m", "15m", "30m", "1h", "2h", "4h", "1d"]
COMMON_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT",
                  "ADAUSDT", "BNBUSDT", "LINKUSDT", "ARBUSDT", "XLMUSDT"]


@st.cache_data(show_spinner="Fetching OHLCV from Bybit…")
def get_ohlcv(symbol: str, tf: str, days: int) -> pd.DataFrame:
    return datamod.fetch_ohlcv(symbol, tf, days=days)


@st.cache_data(show_spinner="Fetching funding history…")
def get_funding(symbol: str, days: int) -> pd.DataFrame:
    try:
        return datamod.fetch_funding(symbol, days=days)
    except Exception as e:  # noqa: BLE001
        st.warning(f"funding fetch failed: {e}")
        return pd.DataFrame(columns=["funding_rate"]).rename_axis("ts")


def param_widgets(cls) -> dict:
    """Build sidebar widgets from a strategy __init__ signature."""
    sig = inspect.signature(cls.__init__)
    out: dict = {}
    st.sidebar.markdown("**Strategy parameters**")
    for pname, p in sig.parameters.items():
        if pname == "self":
            continue
        default = p.default
        key = f"p_{cls.__name__}_{pname}"
        if isinstance(default, bool):
            out[pname] = st.sidebar.checkbox(pname, value=default, key=key)
        elif isinstance(default, int):
            out[pname] = st.sidebar.number_input(pname, value=default, step=1, key=key)
        elif isinstance(default, float):
            step = 10 ** (np.floor(np.log10(abs(default))) - 1) if default else 0.01
            out[pname] = st.sidebar.number_input(pname, value=float(default),
                                                 step=float(step), format="%.6g", key=key)
        else:
            out[pname] = st.sidebar.text_input(pname, value=str(default), key=key)
    return out


# ----------------------------------------------------------------- sidebar
st.sidebar.title("Perp backtest lab")
symbol = st.sidebar.selectbox("Symbol", COMMON_SYMBOLS, index=0,
                              accept_new_options=True)
tf = st.sidebar.selectbox("Timeframe", TF_CHOICES, index=3)
days = st.sidebar.slider("History (days)", 60, 720, 400, step=20)

strat_name = st.sidebar.selectbox("Strategy", list(REGISTRY))
strat_cls = REGISTRY[strat_name]
sparams = param_widgets(strat_cls)

st.sidebar.markdown("**Execution model**")
fee_bps = st.sidebar.number_input("Taker fee (bps)", value=5.5, step=0.5)
slip_bps = st.sidebar.number_input("Slippage (bps)", value=2.0, step=0.5)
risk_pct = st.sidebar.number_input("Risk per trade (%)", value=1.0, step=0.25) / 100
max_lev = st.sidebar.number_input("Max leverage", value=5.0, step=1.0)
allow_short = st.sidebar.checkbox("Allow shorts", value=True)
use_funding = st.sidebar.checkbox("Apply funding costs", value=True)
oos_frac = st.sidebar.slider("Out-of-sample tail (%)", 10, 50, 30, step=5) / 100
mc_runs = st.sidebar.select_slider("Monte Carlo runs", [200, 500, 1000, 2000, 5000], value=1000)

run = st.sidebar.button("Run backtest", type="primary", use_container_width=True)

# ----------------------------------------------------------------- main
st.title(f"{symbol} · {tf} · {strat_name}")

if not run:
    st.info("Set parameters in the sidebar, then **Run backtest**. "
            "Data is cached locally after the first fetch.")
    st.stop()

bars = get_ohlcv(symbol, tf, days)
if len(bars) < 200:
    st.error(f"only {len(bars)} bars returned — try a longer history or different symbol")
    st.stop()
funding = get_funding(symbol, days) if use_funding else None

strategy = strat_cls(**sparams)
bt = Backtester(bars, strategy, fee_bps=fee_bps, slippage_bps=slip_bps,
                risk_pct=risk_pct, max_leverage=max_lev, funding=funding,
                allow_short=allow_short)
res = bt.run()
m = metricsmod.compute(res, tf)
ins, oos = metricsmod.split_in_out(res, tf, oos_frac)

# ---- KPI row
ok = m["passed_filter"]
st.subheader(("✅ passes" if ok else "❌ fails") + " the validation gate "
             f"(≥{metricsmod.MIN_TRADES} trades · Sharpe {metricsmod.SHARPE_LO}–{metricsmod.SHARPE_HI} · maxDD ≤{int(metricsmod.MAX_DD_LIMIT*100)}%)")
c = st.columns(8)
c[0].metric("Return", f"{m['return_pct']:.1f}%")
c[1].metric("CAGR", f"{m['cagr_pct']:.1f}%")
c[2].metric("Sharpe", f"{m['sharpe']:.2f}" if m['sharpe'] == m['sharpe'] else "—")
c[3].metric("Max DD", f"{m['max_dd_pct']:.1f}%")
c[4].metric("Profit factor", f"{m['profit_factor']:.2f}")
c[5].metric("Win rate", f"{m['win_rate_pct']:.0f}%")
c[6].metric("Trades", m["trades"])
c[7].metric("Expectancy", f"{m['expectancy_r']:.2f} R")

tab_eq, tab_mc, tab_split, tab_month, tab_price, tab_trades = st.tabs(
    ["Equity & drawdown", "Monte Carlo", "In/out of sample", "Monthly", "Price & trades", "Trades table"])

# ---- Equity & drawdown
with tab_eq:
    eq = res.equity_curve
    bh = bars["close"].reindex(eq.index).ffill()
    bh = bh / bh.iloc[0] * res.initial_equity
    dd = (eq / eq.cummax() - 1.0) * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Strategy equity",
                             line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bh.index, y=bh.values, name="Buy & hold",
                             line=dict(width=1, dash="dot"), opacity=0.6), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown %",
                             fill="tozeroy", line=dict(width=1, color="#c0392b")), row=2, col=1)
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="DD %", row=2, col=1)
    fig.update_layout(height=560, legend=dict(orientation="h", y=1.08), margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

    cc = st.columns(4)
    cc[0].metric("Final equity", f"${m['final_equity']:,.0f}")
    cc[1].metric("Sortino", f"{m['sortino']:.2f}" if m['sortino'] == m['sortino'] else "—")
    cc[2].metric("Exposure", f"{m['exposure_pct']:.0f}%")
    cc[3].metric("Fees paid", f"${m['total_fees']:,.0f}")

# ---- Monte Carlo
with tab_mc:
    r = mc.resample_trades(res, n_runs=int(mc_runs))
    if not r["ok"]:
        st.warning(r["reason"])
    else:
        cc = st.columns(4)
        cc[0].metric("Return p05 / p50 / p95",
                     f"{r['return_p05']:.0f} / {r['return_p50']:.0f} / {r['return_p95']:.0f} %")
        cc[1].metric("Max DD p50 (typical)", f"{r['maxdd_p50']:.1f}%")
        cc[2].metric("Max DD p05 (bad run)", f"{r['maxdd_p05']:.1f}%")
        cc[3].metric("P(profit) / P(DD>35%)", f"{r['prob_profit']:.0f}% / {r['prob_dd_gt_35']:.0f}%")
        f1 = go.Figure()
        f1.add_trace(go.Histogram(x=r["finals"] * 100, nbinsx=60, name="final return %"))
        f1.update_layout(height=260, title="Bootstrapped final return distribution (%)",
                         margin=dict(t=40, b=10))
        st.plotly_chart(f1, use_container_width=True)
        f2 = go.Figure()
        f2.add_trace(go.Histogram(x=r["max_dds"] * 100, nbinsx=60, name="max DD %",
                                  marker_color="#c0392b"))
        f2.update_layout(height=260, title="Bootstrapped max-drawdown distribution (%)",
                         margin=dict(t=40, b=10))
        st.plotly_chart(f2, use_container_width=True)
        st.caption("Resamples the realised per-trade P&L with replacement. If the "
                   "backtest's drawdown sits far from the p50 here, the curve was "
                   "ordering-lucky.")

# ---- In / out of sample
with tab_split:
    if ins and oos:
        rows = ["trades", "return_pct", "cagr_pct", "sharpe", "max_dd_pct",
                "win_rate_pct", "profit_factor", "expectancy_r"]
        tbl = pd.DataFrame({
            "in-sample": {k: ins[k] for k in rows},
            "out-of-sample": {k: oos[k] for k in rows},
        }).round(2)
        st.dataframe(tbl, use_container_width=True)
        deg = (ins["sharpe"] - oos["sharpe"]) if (ins["sharpe"] == ins["sharpe"] and oos["sharpe"] == oos["sharpe"]) else np.nan
        st.caption(f"Sharpe drop in-→out-of sample: {deg:.2f}. "
                   "A large positive drop = the config was fit to the early period.")
    else:
        st.warning("not enough data/trades to split")

# ---- Monthly
with tab_month:
    mt = metricsmod.monthly_table(res)
    if len(mt):
        colr = ["#2ecc71" if v > 0 else "#c0392b" for v in mt["pnl"]]
        f = go.Figure(go.Bar(x=mt["month"], y=mt["pnl"], marker_color=colr))
        f.update_layout(height=320, title="P&L by calendar month ($)", margin=dict(t=40))
        st.plotly_chart(f, use_container_width=True)
        pos_months = (mt["pnl"] > 0).mean() * 100
        st.metric("Profitable months", f"{pos_months:.0f}%")
        st.dataframe(mt.round(2), use_container_width=True)
    else:
        st.warning("no trades")

# ---- Price & trades
with tab_price:
    show_bars = bars.iloc[-min(len(bars), 1500):]
    f = go.Figure(go.Candlestick(x=show_bars.index, open=show_bars["open"],
                                 high=show_bars["high"], low=show_bars["low"],
                                 close=show_bars["close"], name=symbol))
    tr = res.trades
    if len(tr):
        longs = tr[tr["side"] == 1]
        shorts = tr[tr["side"] == -1]
        f.add_trace(go.Scatter(x=longs["entry_time"], y=longs["entry_price"], mode="markers",
                               marker=dict(symbol="triangle-up", size=9, color="#2ecc71"),
                               name="long entry"))
        f.add_trace(go.Scatter(x=shorts["entry_time"], y=shorts["entry_price"], mode="markers",
                               marker=dict(symbol="triangle-down", size=9, color="#c0392b"),
                               name="short entry"))
        f.add_trace(go.Scatter(x=tr["exit_time"], y=tr["exit_price"], mode="markers",
                               marker=dict(symbol="x", size=7, color="#7f8c8d"), name="exit"))
    f.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(t=20),
                    legend=dict(orientation="h", y=1.06))
    st.plotly_chart(f, use_container_width=True)
    st.caption("Last 1500 bars shown.")

# ---- Trades table
with tab_trades:
    tr = res.trades.copy()
    if len(tr):
        for col in ["entry_price", "exit_price", "pnl", "notional", "fees"]:
            tr[col] = tr[col].round(4)
        tr["pnl_pct"] = (tr["pnl_pct"] * 100).round(2)
        tr["mae_pct"] = (tr["mae_pct"] * 100).round(2)
        tr["mfe_pct"] = (tr["mfe_pct"] * 100).round(2)
        st.dataframe(tr, use_container_width=True, height=460)
        st.download_button("Download trades CSV", tr.to_csv(index=False),
                           file_name=f"{symbol}_{tf}_{strategy.name}_trades.csv")
    else:
        st.warning("no trades")
