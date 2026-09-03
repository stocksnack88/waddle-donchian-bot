"""
OKX data backend — used when DATA_SOURCE=okx (e.g. from GitHub Actions, whose
IPs Bybit blocks). OKX serves the same XLM/XRP perps from any IP.

Everything is normalised to the SAME shape data.py's Bybit functions return:
  * OHLCV: DataFrame[open,high,low,close,volume] indexed by naive UTC timestamp
  * funding: DataFrame[funding_rate] indexed by naive UTC timestamp
  * instrument: {symbol,min_qty,qty_step,min_notional,tick_size,max_leverage}
    with sizes converted from OKX *contracts* to *base-coin units* (1 contract
    = ctVal coins), so the engine can stay contract-agnostic.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pandas as pd

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

OKX = "https://www.okx.com"
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"

_TF_MAP = {"1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
           "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H", "1d": "1D"}
_TF_MIN = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120,
           "4h": 240, "6h": 360, "12h": 720, "1d": 1440}


def _inst_id(symbol: str) -> str:
    s = symbol.replace("/", "").replace(":USDT", "").upper()
    base = s[:-4] if s.endswith("USDT") else s
    return f"{base}-USDT-SWAP"


def _get(client: httpx.Client, path: str, params: dict) -> list:
    r = client.get(f"{OKX}{path}", params=params, headers={"User-Agent": _UA})
    r.raise_for_status()
    d = r.json()
    if d.get("code") not in ("0", 0):
        raise RuntimeError(f"OKX error {d.get('code')}: {d.get('msg')}")
    return d.get("data", [])


# --------------------------------------------------------------------------- instrument
def okx_instrument(symbol: str, *, force: bool = False) -> dict:
    iid = _inst_id(symbol)
    path = CACHE_DIR / f"okx_instrument_{iid}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    with httpx.Client(timeout=20) as c:
        d = _get(c, "/api/v5/public/instruments", {"instType": "SWAP", "instId": iid})[0]
    ct = float(d["ctVal"])                       # base coins per contract
    out = {
        "symbol": symbol.replace("/", "").replace(":USDT", "").upper(),
        "min_qty": float(d["minSz"]) * ct,       # -> base-coin units
        "qty_step": float(d["lotSz"]) * ct,
        "min_notional": 1.0,                      # OKX has no explicit $ min; min_qty governs
        "tick_size": float(d["tickSz"]),
        "max_leverage": float(d.get("lever", 50)),
    }
    path.write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- OHLCV
def okx_ohlcv(symbol: str, timeframe: str = "4h", days: int = 120, *, force: bool = False) -> pd.DataFrame:
    if timeframe not in _TF_MAP:
        raise ValueError(f"unsupported timeframe {timeframe!r}")
    iid = _inst_id(symbol)
    path = CACHE_DIR / f"okx_ohlcv_{iid}_{timeframe}.parquet"
    cached = pd.read_parquet(path) if path.exists() and not force else None

    now_ms = int(time.time() * 1000)
    want_start = now_ms - days * 86_400_000
    have_until = int(cached.index[-1].timestamp() * 1000) if cached is not None and len(cached) else 0
    stop_at = max(want_start, have_until)

    bar = _TF_MAP[timeframe]
    rows: list[list] = []
    after = None
    with httpx.Client(timeout=20) as c:
        while True:
            params = {"instId": iid, "bar": bar, "limit": "100"}
            if after:
                params["after"] = str(after)
            batch = _get(c, "/api/v5/market/history-candles", params)   # newest-first
            if not batch:
                break
            rows.extend(batch)
            oldest = int(batch[-1][0])
            if oldest <= stop_at or len(batch) < 100:
                break
            after = oldest
            time.sleep(0.12)

    fresh = _rows_to_df(rows)
    out = fresh if cached is None else (
        pd.concat([cached, fresh]).pipe(lambda d: d[~d.index.duplicated(keep="last")]).sort_index()
    )
    if len(out):
        out.to_parquet(path)
    cut = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
    return out[out.index >= cut].copy()


def _rows_to_df(rows: list[list]) -> pd.DataFrame:
    # Keep ALL candles including the still-forming one as the last row, to match
    # Bybit's behaviour — bot.py drops the last row to get closed candles. The
    # cache dedup (keep="last") replaces the forming candle once it confirms.
    cols = ["open", "high", "low", "close", "volume"]
    if not rows:
        return pd.DataFrame(columns=cols).rename_axis("ts")
    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "vol", "volCcy", "volQuote", "confirm"])
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
    for src, dst in zip(["o", "h", "l", "c", "vol"], cols):
        df[dst] = df[src].astype("float64")
    return df.set_index("ts")[cols].sort_index()


# --------------------------------------------------------------------------- funding
def okx_funding(symbol: str, days: int = 120, *, force: bool = False) -> pd.DataFrame:
    iid = _inst_id(symbol)
    path = CACHE_DIR / f"okx_funding_{iid}.parquet"
    if path.exists() and not force:
        cached = pd.read_parquet(path)
        if len(cached) and cached.index[-1] > pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=8):
            return cached

    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    out: list[dict] = []
    after = None
    with httpx.Client(timeout=20) as c:
        while True:
            params = {"instId": iid, "limit": "100"}
            if after:
                params["after"] = str(after)
            batch = _get(c, "/api/v5/public/funding-rate-history", params)
            if not batch:
                break
            out.extend(batch)
            oldest = int(batch[-1]["fundingTime"])
            if oldest <= start or len(batch) < 100:
                break
            after = oldest
            time.sleep(0.12)

    if not out:
        df = pd.DataFrame(columns=["funding_rate"]).rename_axis("ts")
    else:
        df = pd.DataFrame(out)
        df["ts"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms")
        df["funding_rate"] = df["fundingRate"].astype("float64")
        df = df.set_index("ts")[["funding_rate"]].sort_index()
        df = df[~df.index.duplicated(keep="last")]
    if len(df):
        df.to_parquet(path)
    return df


if __name__ == "__main__":
    for s in ["XLMUSDT", "XRPUSDT"]:
        i = okx_instrument(s)
        b = okx_ohlcv(s, "4h", days=120)
        f = okx_funding(s, days=120)
        print(f"{s}: {len(b)} bars {b.index[0]}..{b.index[-1]} | funding {len(f)} | "
              f"min_qty {i['min_qty']} step {i['qty_step']}")
