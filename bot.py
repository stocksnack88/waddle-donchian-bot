"""
Live / paper trading bot for the XLM+XRP 4h Donchian basket.

Same strategy code as the backtest (`DonchianATR.prepare` + `on_bar`). Decisions
are made ONLY on a closed 4h candle (never the forming one); stop / target are
watched continuously.

    python3 bot.py                     # paper, loop  (default, safe)
    python3 bot.py --once              # paper, single pass
    python3 bot.py --live --testnet    # real orders on Bybit TESTNET
    python3 bot.py --live              # real orders on MAINNET  (needs BYBIT_LIVE=1)

Live mode needs env BYBIT_API_KEY / BYBIT_API_SECRET, and for mainnet also
BYBIT_LIVE=1. Entries carry exchange-native stopLoss/takeProfit, so positions
stay protected if this process dies.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import pandas as pd

import data as datamod
from engine import Context
from sizing import Instrument, size_position
from strategies.donchian_atr import DonchianATR

HERE = Path(__file__).parent
STATE_PATH = HERE / "bot_state.json"
TRADES_PATH = HERE / "paper_trades.jsonl"

RUN_MODE = "paper"          # set in main(): paper | testnet | live

CFG = dict(lookback=20, atr_period=14, atr_mult=2.0, rr=2.0, ema_filter=200)
LEGS = [dict(symbol="XLMUSDT", timeframe="4h"), dict(symbol="XRPUSDT", timeframe="4h")]
ACCOUNT_START = 50.0
RISK_PCT = 0.005
SLIPPAGE_BPS = 5.0
FEE_BPS = 5.5
MAX_LEVERAGE = 5.0
MAX_CONCURRENT = 2         # never hold more than this many positions at once
HIST_DAYS = 120


# ============================================================ paper broker
@dataclass
class PaperBroker:
    cash: float = ACCOUNT_START
    positions: dict = field(default_factory=dict)
    seen_closed_ts: dict = field(default_factory=dict)
    realized_pnl: float = 0.0
    n_trades: int = 0
    wins: int = 0
    exchange_exits: bool = False          # paper bot watches stop/target itself

    # -- reads
    def get_equity(self, marks: dict) -> float:
        eq = self.cash
        for s, p in self.positions.items():
            if s in marks:
                eq += p["side"] * (marks[s] - p["entry"]) * p["qty"]
        return eq

    def get_positions(self) -> dict:
        return self.positions

    # -- writes
    def market_order(self, symbol, side, qty, price, *, sl, tp, tag, ts):
        fill = price * (1 + side * SLIPPAGE_BPS / 1e4)
        self.cash -= qty * fill * FEE_BPS / 1e4
        self.positions[symbol] = dict(side=side, qty=qty, entry=fill, sl=sl, tp=tp,
                                      entry_ts=ts, tag=tag)
        _log(dict(event="open", symbol=symbol, side=side, qty=qty, price=round(fill, 6),
                  sl=round(sl, 6), tp=(round(tp, 6) if tp == tp else None), tag=tag, ts=ts,
                  cash=round(self.cash, 4)))

    def close_position(self, symbol, price, reason, ts):
        p = self.positions.pop(symbol)
        fill = price * (1 - p["side"] * SLIPPAGE_BPS / 1e4)
        gross = p["side"] * (fill - p["entry"]) * p["qty"]
        fees = fill * p["qty"] * FEE_BPS / 1e4
        net = gross - fees
        self.cash += gross - fees
        self.realized_pnl += net
        self.n_trades += 1
        self.wins += int(net > 0)
        _log(dict(event="close", symbol=symbol, side=p["side"], qty=p["qty"],
                  entry=round(p["entry"], 6), exit=round(fill, 6), pnl=round(net, 4),
                  reason=reason, ts=ts, cash=round(self.cash, 4)))
        return net


def _log(rec: dict):
    rec["mode"] = RUN_MODE
    rec["logged_at"] = pd.Timestamp.utcnow().isoformat()
    with open(TRADES_PATH, "a") as fh:                       # always-on local log
        fh.write(json.dumps(rec) + "\n")
    tail = f"pnl={rec['pnl']}" if rec["event"] == "close" else f"{rec.get('tag')}"
    print(f"  [{rec['event'].upper()}] {rec['symbol']} {tail}")
    _supa_log(rec)                                           # best-effort remote log


def _supa_log(rec: dict):
    """POST one trade event to Supabase for the waddle-ops dashboard. Never raises."""
    url = os.getenv("WADDLE_SUPABASE_URL")
    key = os.getenv("WADDLE_SUPABASE_KEY")
    if not (url and key):
        return
    row = {
        "event": rec["event"], "mode": rec.get("mode", RUN_MODE),
        "symbol": rec["symbol"], "side": rec.get("side"),
        "qty": rec.get("qty"), "price": rec.get("price") or rec.get("exit"),
        "entry_price": rec.get("entry"), "sl": rec.get("sl"), "tp": rec.get("tp"),
        "pnl": rec.get("pnl"), "reason": rec.get("reason"), "tag": rec.get("tag"),
        "candle_ts": rec.get("ts"), "equity_after": rec.get("cash"),
    }
    try:
        httpx.post(
            f"{url.rstrip('/')}/rest/v1/waddle_paper_trades",
            headers={"apikey": key, "Authorization": f"Bearer {key}",
                     "Content-Type": "application/json", "Prefer": "return=minimal"},
            json=row, timeout=8,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  (supabase log skipped: {e})")


def _supa_creds():
    return os.getenv("WADDLE_SUPABASE_URL"), os.getenv("WADDLE_SUPABASE_KEY")


def load_paper() -> PaperBroker:
    if STATE_PATH.exists():
        return PaperBroker(**json.loads(STATE_PATH.read_text()))
    # Railway / fresh host: local file gone -> pull last state from Supabase
    url, key = _supa_creds()
    if url and key:
        try:
            r = httpx.get(
                f"{url.rstrip('/')}/rest/v1/waddle_bot_state?id=eq.1&select=state",
                headers={"apikey": key, "Authorization": f"Bearer {key}",
                         "Accept": "application/json"}, timeout=8,
            )
            if r.status_code == 404:
                print("  [state] !! Supabase 404 — table 'waddle_bot_state' not found. "
                      "WADDLE_SUPABASE_URL must point to the project where you ran "
                      "supabase_paper_trades.sql. Nothing will be logged until fixed.")
                return PaperBroker()
            r.raise_for_status()
            rows = r.json() if r.text.strip() else []
            if rows and rows[0].get("state"):
                print("  [state] restored from Supabase")
                return PaperBroker(**rows[0]["state"])
            print("  [state] no prior state in Supabase — starting fresh")
        except Exception as e:  # noqa: BLE001
            print(f"  (state restore skipped: {e})")
    return PaperBroker()


def save_paper(b: PaperBroker):
    if not isinstance(b, PaperBroker):
        return
    blob = asdict(b)
    STATE_PATH.write_text(json.dumps(blob, indent=2, default=str))
    url, key = _supa_creds()
    if not (url and key):
        return
    try:
        httpx.post(
            f"{url.rstrip('/')}/rest/v1/waddle_bot_state?on_conflict=id",
            headers={"apikey": key, "Authorization": f"Bearer {key}",
                     "Content-Type": "application/json",
                     "Prefer": "resolution=merge-duplicates,return=minimal"},
            json={"id": 1, "state": blob, "updated_at": pd.Timestamp.utcnow().isoformat()},
            timeout=8,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  (state sync skipped: {e})")


# ============================================================ core step
def step(broker, insts: dict, verbose=True):
    marks: dict = {}
    positions = broker.get_positions()

    for leg in LEGS:
        sym, tf = leg["symbol"], leg["timeframe"]
        fresh = datamod.fetch_ohlcv(sym, tf, days=HIST_DAYS)
        if len(fresh) < CFG["ema_filter"] + 30:
            continue
        last_price = float(fresh["close"].iloc[-1])
        marks[sym] = last_price

        # the last row is the FORMING candle -> decide on the one before it
        closed = fresh.iloc[:-1]
        closed_ts = str(closed.index[-1])
        dec_i = len(closed) - 1
        strat = DonchianATR(**CFG)
        feats = strat.prepare(closed, None)
        eq = broker.get_equity(marks)

        # --- continuous stop/target watch (paper only; live uses exchange orders)
        if not broker.exchange_exits and sym in positions:
            p = positions[sym]
            hi = float(fresh["high"].iloc[-1]); lo = float(fresh["low"].iloc[-1])
            if (p["side"] == 1 and lo <= p["sl"]) or (p["side"] == -1 and hi >= p["sl"]):
                broker.close_position(sym, p["sl"], "stop", closed_ts); continue
            if p["tp"] == p["tp"] and ((p["side"] == 1 and hi >= p["tp"]) or (p["side"] == -1 and lo <= p["tp"])):
                broker.close_position(sym, p["tp"], "target", closed_ts); continue

        # --- decisions only once per freshly-closed candle
        if broker.seen_closed_ts.get(sym) == closed_ts:
            continue
        broker.seen_closed_ts[sym] = closed_ts

        if sym in positions:
            ctx = Context(bars=closed, feats=feats, i=dec_i, position=positions[sym]["side"],
                          entry_price=positions[sym]["entry"], equity=eq, funding=None)
            sig = strat.on_bar(ctx)
            if sig.action == "exit":
                broker.close_position(sym, last_price, sig.tag or "signal", closed_ts)
        else:
            if len(broker.get_positions()) >= MAX_CONCURRENT:
                continue
            ctx = Context(bars=closed, feats=feats, i=dec_i, position=0,
                          entry_price=0.0, equity=eq, funding=None)
            sig = strat.on_bar(ctx)
            if sig.action in ("enter_long", "enter_short"):
                side = 1 if sig.action == "enter_long" else -1
                stop = sig.sl if sig.sl else last_price * (1 - side * 0.01)
                sized = size_position(equity=eq, entry_price=last_price, stop_price=stop,
                                      risk_pct=RISK_PCT, inst=insts[sym], max_leverage=MAX_LEVERAGE)
                if sized.qty > 0:
                    broker.market_order(sym, side, sized.qty, last_price, sl=stop,
                                        tp=sig.tp if (sig.tp and sig.tp == sig.tp) else float("nan"),
                                        tag=sig.tag, ts=closed_ts)
                elif verbose:
                    print(f"  {sym}: {sig.tag} signal, size skipped ({sized.reason})")

    eq = broker.get_equity(marks)
    if verbose:
        n = getattr(broker, "n_trades", 0)
        wr = (broker.wins / n * 100) if getattr(broker, "wins", 0) and n else 0
        print(f"  equity ${eq:.2f} | open {list(broker.get_positions())} | "
              f"closed {n} (win {wr:.0f}%)")
    save_paper(broker)
    return eq


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--poll", type=int, default=45)
    ap.add_argument("--live", action="store_true", help="send real orders to Bybit")
    ap.add_argument("--testnet", action="store_true", help="use Bybit testnet")
    args = ap.parse_args()

    global RUN_MODE
    RUN_MODE = "testnet" if (args.live and args.testnet) else "live" if args.live else "paper"
    if not args.live and os.getenv("DATA_SOURCE", "bybit").lower() == "okx":
        RUN_MODE = "paper-okx"       # paper record priced off OKX, not Bybit

    if args.live:
        from broker_bybit import LiveBroker
        broker = LiveBroker(testnet=args.testnet, dry_run=False)
        broker.exchange_exits = True
        for leg in LEGS:
            broker.set_leverage(leg["symbol"], MAX_LEVERAGE)
        mode = f"LIVE ({'testnet' if args.testnet else 'MAINNET'}, "
        mode += "ARMED" if getattr(broker, "armed", False) else "dry-run"
        mode += ")"
    else:
        broker = load_paper()
        mode = "PAPER"

    insts = {l["symbol"]: Instrument(**datamod.fetch_instrument(l["symbol"])) for l in LEGS}
    print(f"bot [{mode}] | legs {[l['symbol'] for l in LEGS]} | cfg {CFG} | risk {RISK_PCT*100}%/leg")

    if args.once:
        step(broker, insts); return
    while True:
        try:
            print(pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
            step(broker, insts)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {type(e).__name__}: {e}")
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
