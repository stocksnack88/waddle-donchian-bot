"""
The $50 bot: portfolio backtest + walk-forward validation for the
XLM + XRP  4h Donchian-breakout basket.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data as datamod
import metrics as metricsmod
from portfolio import PortfolioBacktester, Leg
from sizing import Instrument
from strategies.donchian_atr import DonchianATR
from walkforward import walk_forward

st.set_page_config(page_title="$50 bot", layout="wide")
st.title("The $50 bot — XLM + XRP · 4h Donchian breakout")

DEFAULT_SYMBOLS = ["XLMUSDT", "XRPUSDT"]


@st.cache_data(show_spinner="Fetching data…")
def load_leg(sym: str, days: int):
    return (datamod.fetch_ohlcv(sym, "4h", days=days),
            datamod.fetch_funding(sym, days=days),
            datamod.fetch_instrument(sym))


# ---------------- sidebar
st.sidebar.header("Basket")
syms = st.sidebar.multiselect("Symbols (4h)",
    ["XLMUSDT", "XRPUSDT", "TRXUSDT", "ADAUSDT", "DOGEUSDT", "SOLUSDT"],
    default=DEFAULT_SYMBOLS)
days = st.sidebar.slider("History (days)", 200, 1000, 1000, step=50)

st.sidebar.header("Strategy (fixed Donchian)")
lookback = st.sidebar.number_input("lookback", value=20, step=1)
atr_mult = st.sidebar.number_input("atr_mult", value=2.0, step=0.25)
rr = st.sidebar.number_input("rr", value=2.0, step=0.25)
ema_filter = st.sidebar.number_input("ema_filter", value=200, step=10)

st.sidebar.header("Account")
equity0 = st.sidebar.number_input("Start capital ($)", value=50.0, step=10.0)
risk_pct = st.sidebar.number_input("Risk per leg (%)", value=0.5, step=0.1) / 100
max_conc = st.sidebar.slider("Max positions open at once", 1, 4, min(2, len(syms) or 1))
slip = st.sidebar.number_input("Slippage (bps)", value=5.0, step=1.0)

run = st.sidebar.button("Run", type="primary", use_container_width=True)

if not run:
    st.info("Pick the basket and account settings, then **Run**. "
            "Tab 1 = portfolio backtest on all history. Tab 2 = walk-forward "
            "(re-optimise every 90 days, fully out-of-sample).")
    st.stop()
if not syms:
    st.error("pick at least one symbol")
    st.stop()

cfg = dict(lookback=int(lookback), atr_period=14, atr_mult=atr_mult,
           rr=rr, ema_filter=int(ema_filter))

legs = []
for s in syms:
    bars, fund, spec = load_leg(s, days)
    legs.append(Leg(symbol=s, bars=bars, strategy=DonchianATR(**cfg),
                    funding=fund, instrument=Instrument(**spec)))

tab_bt, tab_wf = st.tabs(["Portfolio backtest", "Walk-forward validation"])

# ================================================================ backtest
with tab_bt:
    res = PortfolioBacktester(legs, initial_equity=equity0, risk_pct=risk_pct,
                              slippage_bps=slip, max_concurrent=max_conc).run()
    m = metricsmod.compute(res, "4h")
    eq = res.equity_curve

    c = st.columns(7)
    c[0].metric("Final equity", f"${m['final_equity']:,.2f}")
    c[1].metric("Return", f"{m['return_pct']:.1f}%")
    c[2].metric("CAGR", f"{m['cagr_pct']:.1f}%")
    c[3].metric("Sharpe", f"{m['sharpe']:.2f}")
    c[4].metric("Max DD", f"{m['max_dd_pct']:.1f}%")
    c[5].metric("Trades", m["trades"])
    c[6].metric("Skipped (too small)", res.skipped_trades)

    dd = (eq / eq.cummax() - 1.0) * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Portfolio equity",
                             line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown %", fill="tozeroy",
                             line=dict(width=1, color="#c0392b")), row=2, col=1)
    fig.update_layout(height=520, legend=dict(orientation="h", y=1.1), margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

    mt = metricsmod.monthly_table(res)
    if len(mt):
        colr = ["#2ecc71" if v > 0 else "#c0392b" for v in mt["pnl"]]
        fm = go.Figure(go.Bar(x=mt["month"], y=mt["pnl"], marker_color=colr))
        fm.update_layout(height=280, title=f"Monthly P&L ($) — "
                         f"{(mt['pnl']>0).mean()*100:.0f}% of months positive",
                         margin=dict(t=40))
        st.plotly_chart(fm, use_container_width=True)

    st.subheader("Per symbol")
    rows = []
    for sym, tr in res.per_symbol.items():
        if not len(tr):
            continue
        rows.append(dict(symbol=sym, trades=len(tr), pnl=round(tr["pnl"].sum(), 2),
                         win_rate=f"{(tr['pnl']>0).mean()*100:.0f}%",
                         avg_risk=f"{tr['risk_pct_actual'].mean()*100:.2f}%"))
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    if res.skip_reasons:
        st.caption(f"skip reasons: {res.skip_reasons}")

# ================================================================ walk-forward
with tab_wf:
    st.caption("Each 90-day test slice is traded with parameters chosen only from "
               "the prior 300 days. Curves are stitched into one out-of-sample line.")
    grid = dict(lookback=[14, 20, 30], atr_mult=[1.5, 2.0, 2.5], rr=[1.5, 2.0, 3.0])
    curves = {}
    fold_tbl = []
    prog = st.progress(0.0)
    for j, leg in enumerate(legs):
        wf = walk_forward(strategy_cls=DonchianATR, param_grid=grid, bars=leg.bars,
                          timeframe="4h", funding=leg.funding, instrument=leg.instrument,
                          initial_equity=equity0 / len(legs), train_days=300, test_days=90,
                          slippage_bps=slip, risk_pct=risk_pct,
                          fixed=dict(atr_period=14, ema_filter=int(ema_filter)))
        curves[leg.symbol] = wf.equity_curve
        s = wf.summary
        fold_tbl.append(dict(symbol=leg.symbol, folds=s["folds"],
                             oos_cagr=f"{s['oos_cagr_pct']:.1f}%",
                             oos_sharpe=f"{s['oos_sharpe']:.2f}",
                             oos_maxdd=f"{s['oos_max_dd_pct']:.1f}%",
                             profitable_folds=f"{s['profitable_folds_pct']:.0f}%"))
        prog.progress((j + 1) / len(legs))
    prog.empty()

    port = pd.concat(curves.values(), axis=1).ffill().dropna().sum(axis=1)
    r = port.pct_change().dropna()
    yrs = len(port) / 2190
    cagr = (port.iloc[-1] / port.iloc[0]) ** (1 / yrs) - 1 if port.iloc[-1] > 0 else -1
    sharpe = r.mean() / r.std() * np.sqrt(2190) if r.std() > 0 else np.nan
    ddp = (port / port.cummax() - 1)

    c = st.columns(5)
    c[0].metric("OOS final", f"${port.iloc[-1]:,.2f}")
    c[1].metric("OOS return", f"{(port.iloc[-1]/port.iloc[0]-1)*100:.1f}%")
    c[2].metric("OOS CAGR", f"{cagr*100:.1f}%")
    c[3].metric("OOS Sharpe", f"{sharpe:.2f}")
    c[4].metric("OOS max DD", f"{ddp.min()*100:.1f}%")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=port.index, y=port.values, name="Walk-forward equity",
                             line=dict(width=2, color="#8e44ad")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ddp.index, y=ddp.values * 100, name="DD %", fill="tozeroy",
                             line=dict(width=1, color="#c0392b")), row=2, col=1)
    fig.update_layout(height=480, legend=dict(orientation="h", y=1.1), margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame(fold_tbl), use_container_width=True)
    st.caption("If OOS CAGR/Sharpe here are much weaker than the backtest tab, the "
               "backtest was partly curve-fit. Some gap is normal and expected.")
