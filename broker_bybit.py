"""
Bybit v5 live broker - same interface as bot.PaperBroker.

SAFETY
------
* dry_run=True by default. Nothing is sent to the exchange unless you pass
  dry_run=False AND set env BYBIT_LIVE=1.
* API keys are read from env only (BYBIT_API_KEY / BYBIT_API_SECRET). Never
  hard-code them, never commit them.
* Entries are sent as Market orders with exchange-native stopLoss / takeProfit
  attached, so the position is protected even if this process dies.
* Use a Bybit API key restricted to "Contract - Orders & Positions". Do NOT
  enable withdrawals on the key.
* Test on testnet first: dry_run=False, testnet=True (env BYBIT_LIVE=1).

This module is provided so the strategy can run live. Arming it with real funds
is the operator's decision and responsibility.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time

import httpx

MAINNET = "https://api.bybit.com"
TESTNET = "https://api-testnet.bybit.com"
RECV_WINDOW = "5000"


class BybitError(RuntimeError):
    pass


class LiveBroker:
    def __init__(self, *, testnet: bool = True, dry_run: bool = True):
        self.base = TESTNET if testnet else MAINNET
        self.key = os.getenv("BYBIT_API_KEY", "")
        self.secret = os.getenv("BYBIT_API_SECRET", "")
        self.armed = (not dry_run) and os.getenv("BYBIT_LIVE") == "1"
        self.dry_run = not self.armed
        if not self.armed:
            print("  [LiveBroker] DRY-RUN — no orders will be sent "
                  "(set dry_run=False and env BYBIT_LIVE=1 to arm).")
        elif not (self.key and self.secret):
            raise BybitError("armed but BYBIT_API_KEY / BYBIT_API_SECRET not set")
        self._client = httpx.Client(timeout=15)
        # interface parity with bot.PaperBroker
        self.exchange_exits = True
        self.seen_closed_ts: dict = {}
        self.n_trades = 0
        self.wins = 0

    # ---------------------------------------------------------------- signing
    def _sign(self, ts: str, payload: str) -> str:
        raw = ts + self.key + RECV_WINDOW + payload
        return hmac.new(self.secret.encode(), raw.encode(), hashlib.sha256).hexdigest()

    def _headers(self, ts: str, payload: str) -> dict:
        return {
            "X-BAPI-API-KEY": self.key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": RECV_WINDOW,
            "X-BAPI-SIGN": self._sign(ts, payload),
            "Content-Type": "application/json",
        }

    def _get(self, path: str, params: dict) -> dict:
        ts = str(int(time.time() * 1000))
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        r = self._client.get(self.base + path, params=params,
                             headers=self._headers(ts, query))
        return self._unwrap(r)

    def _post(self, path: str, body: dict) -> dict:
        ts = str(int(time.time() * 1000))
        payload = json.dumps(body, separators=(",", ":"))
        r = self._client.post(self.base + path, content=payload,
                              headers=self._headers(ts, payload))
        return self._unwrap(r)

    @staticmethod
    def _unwrap(r: httpx.Response) -> dict:
        r.raise_for_status()
        d = r.json()
        if d.get("retCode") != 0:
            raise BybitError(f"{d.get('retCode')}: {d.get('retMsg')}")
        return d.get("result", {})

    # ---------------------------------------------------------------- reads
    def get_equity(self, *_a, **_k) -> float:
        if not (self.key and self.secret):
            return float(os.getenv("PAPER_EQUITY", "50"))
        res = self._get("/v5/account/wallet-balance", {"accountType": "UNIFIED"})
        for coin in res["list"][0]["coin"]:
            if coin["coin"] == "USDT":
                return float(coin["equity"])
        return 0.0

    def get_positions(self) -> dict:
        if not (self.key and self.secret):
            return {}
        res = self._get("/v5/position/list", {"category": "linear", "settleCoin": "USDT"})
        out = {}
        for p in res["list"]:
            if float(p["size"]) > 0:
                out[p["symbol"]] = dict(
                    side=1 if p["side"] == "Buy" else -1,
                    qty=float(p["size"]), entry=float(p["avgPrice"]),
                    sl=float(p["stopLoss"] or 0), tp=float(p["takeProfit"] or 0),
                )
        return out

    # ---------------------------------------------------------------- writes
    def set_leverage(self, symbol: str, lev: float = 5.0):
        if self.dry_run:
            print(f"  [dry] set_leverage {symbol} {lev}x"); return
        try:
            self._post("/v5/position/set-leverage", {
                "category": "linear", "symbol": symbol,
                "buyLeverage": str(lev), "sellLeverage": str(lev)})
        except BybitError as e:
            if "110043" not in str(e):     # 110043 = leverage not modified
                raise

    def market_order(self, symbol, side, qty, price, *, sl, tp, tag, ts):
        s = "Buy" if side == 1 else "Sell"
        body = {
            "category": "linear", "symbol": symbol, "side": s,
            "orderType": "Market", "qty": _fmt(qty), "timeInForce": "IOC",
            "positionIdx": 0,
            "stopLoss": _fmt(sl), "slTriggerBy": "LastPrice",
        }
        if tp and tp == tp:
            body["takeProfit"] = _fmt(tp)
            body["tpTriggerBy"] = "LastPrice"
        if self.dry_run:
            print(f"  [dry] ENTRY {s} {symbol} qty={_fmt(qty)} sl={_fmt(sl)} tp={_fmt(tp)}  ({tag})")
            return {"dry_run": True}
        res = self._post("/v5/order/create", body)
        print(f"  [LIVE] ENTRY {s} {symbol} qty={_fmt(qty)} orderId={res.get('orderId')}")
        return res

    def close_position(self, symbol, price, reason, ts):
        pos = self.get_positions().get(symbol)
        if not pos:
            print(f"  [LiveBroker] close({symbol}): no open position"); return 0.0
        s = "Sell" if pos["side"] == 1 else "Buy"
        body = {"category": "linear", "symbol": symbol, "side": s, "orderType": "Market",
                "qty": _fmt(pos["qty"]), "reduceOnly": True, "timeInForce": "IOC",
                "positionIdx": 0}
        if self.dry_run:
            print(f"  [dry] CLOSE {symbol} ({reason})"); return 0.0
        self._post("/v5/order/create", body)
        print(f"  [LIVE] CLOSE {symbol} ({reason})")
        return 0.0


def _fmt(x) -> str:
    if x is None or x != x:
        return "0"
    return f"{float(x):.8f}".rstrip("0").rstrip(".")
