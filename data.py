"""
Data layer: fetch Bybit perpetual OHLCV + funding rate, cache to local parquet.

No API key needed - all endpoints used here are public market data.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
import pandas as pd

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# Bybit geo-/bot-blocks some datacenter IPs with a Cloudflare 403. Mitigations:
#  1) send a browser-ish User-Agent (fixes most CF blocks)
#  2) try the alternate domain api.bytick.com
#  3) honour BYBIT_BASE if you point it at a proxy
BYBIT = os.getenv("BYBIT_BASE", "https://api.bybit.com")
_BYBIT_HOSTS = [BYBIT, "https://api.bytick.com", "https://api.bybit.com"]
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _bybit_get(client: httpx.Client, path: str, params: dict) -> dict:
    """GET {host}{path} trying each Bybit host; return parsed JSON of the first that works."""
    seen: set[str] = set()
    last_err: Exception | None = None
    for host in _BYBIT_HOSTS:
        if host in seen:
            continue
        seen.add(host)
        try:
            r = client.get(f"{host}{path}", params=params, headers={"User-Agent": _UA})
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"all Bybit hosts failed for {path}: {last_err}")

# our timeframe string -> Bybit v5 "interval" value
_TF_MAP = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
    "1d": "D",
}
# minutes per bar, for pagination math
_TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
    "1d": 1440,
}


def _norm_symbol(symbol: str) -> str:
    """'BTC/USDT' or 'BTCUSDT' or 'BTC/USDT:USDT' -> 'BTCUSDT'."""
    return symbol.replace("/", "").replace(":USDT", "").upper()


def _cache_path(kind: str, symbol: str, timeframe: str) -> Path:
    return CACHE_DIR / f"{kind}_{_norm_symbol(symbol)}_{timeframe}.parquet"


# --------------------------------------------------------------------------- OHLCV


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "15m",
    days: int = 540,
    *,
    force: bool = False,
    category: str = "linear",
) -> pd.DataFrame:
    """
    Return a DataFrame indexed by UTC timestamp with columns
    [open, high, low, close, volume]. Cached to parquet; incremental refresh
    on subsequent calls (only fetches bars newer than the cache).
    """
    if timeframe not in _TF_MAP:
        raise ValueError(f"unsupported timeframe {timeframe!r}; pick from {list(_TF_MAP)}")

    path = _cache_path("ohlcv", symbol, timeframe)
    cached: pd.DataFrame | None = None
    if path.exists() and not force:
        cached = pd.read_parquet(path)

    now_ms = int(time.time() * 1000)
    want_start_ms = now_ms - days * 86_400_000
    sym = _norm_symbol(symbol)
    interval = _TF_MAP[timeframe]
    parts = [cached] if cached is not None and len(cached) else []

    if cached is not None and len(cached):
        cache_start_ms = int(cached.index[0].timestamp() * 1000)
        cache_end_ms = int(cached.index[-1].timestamp() * 1000)
        # forward fill: newer bars since the cache ends
        parts.append(_klines_to_df(_paginate_klines(sym, interval, cache_end_ms + 1, now_ms, category, timeframe)))
        # backward fill: older bars if a larger `days` is now requested
        if want_start_ms < cache_start_ms:
            parts.append(_klines_to_df(_paginate_klines(sym, interval, want_start_ms, cache_start_ms - 1, category, timeframe)))
    else:
        parts.append(_klines_to_df(_paginate_klines(sym, interval, want_start_ms, now_ms, category, timeframe)))

    parts = [p for p in parts if p is not None and len(p)]
    out = pd.concat(parts) if parts else _klines_to_df([])
    out = out[~out.index.duplicated(keep="last")].sort_index()

    # persist the FULL history (never shrink the cache) ...
    out.to_parquet(path)
    # ... but only return the requested window
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
    return out[out.index >= cutoff].copy()


def _paginate_klines(symbol, interval, start_ms, end_ms, category, timeframe) -> list[list]:
    """
    Bybit returns <=1000 bars/call, newest first, anchored to `end`. Walk
    BACKWARD from end_ms toward start_ms, one 1000-bar window at a time.
    """
    step = _TF_MINUTES[timeframe] * 60_000
    all_rows: list[list] = []
    cursor_end = end_ms
    with httpx.Client(timeout=20) as client:
        while cursor_end > start_ms:
            payload = _bybit_get(client, "/v5/market/kline", {
                "category": category,
                "symbol": symbol,
                "interval": interval,
                "start": start_ms,
                "end": cursor_end,
                "limit": 1000,
            })
            if payload.get("retCode") != 0:
                raise RuntimeError(f"Bybit error: {payload.get('retMsg')}")
            batch = payload["result"]["list"]  # newest first
            if not batch:
                break
            all_rows.extend(batch)
            oldest_ts = int(batch[-1][0])
            if oldest_ts <= start_ms or len(batch) < 1000:
                break
            cursor_end = oldest_ts - step
            time.sleep(0.12)  # stay well under rate limits
    return all_rows


def _klines_to_df(rows: list[list]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).rename_axis("ts")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype("float64")
    return df.set_index("ts")[["open", "high", "low", "close", "volume"]].sort_index()


# --------------------------------------------------------------------------- funding


def fetch_funding(symbol: str, days: int = 540, *, force: bool = False) -> pd.DataFrame:
    """Return DataFrame indexed by UTC timestamp with a single 'funding_rate' column."""
    path = _cache_path("funding", symbol, "8h")
    if path.exists() and not force:
        return pd.read_parquet(path)

    sym = _norm_symbol(symbol)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    out: list[list] = []
    cursor_end = end_ms
    with httpx.Client(timeout=20) as client:
        while cursor_end > start_ms:
            payload = _bybit_get(client, "/v5/market/funding/history",
                                 {"category": "linear", "symbol": sym,
                                  "endTime": cursor_end, "limit": 200})
            lst = payload["result"]["list"]  # newest first
            if not lst:
                break
            out.extend(lst)
            cursor_end = int(lst[-1]["fundingRateTimestamp"]) - 1
            time.sleep(0.12)

    if not out:
        df = pd.DataFrame(columns=["funding_rate"]).rename_axis("ts")
    else:
        df = pd.DataFrame(out)
        df["ts"] = pd.to_datetime(df["fundingRateTimestamp"].astype("int64"), unit="ms")
        df["funding_rate"] = df["fundingRate"].astype("float64")
        df = df.set_index("ts")[["funding_rate"]].sort_index()
        df = df[~df.index.duplicated(keep="last")]
    df.to_parquet(path)
    return df


# --------------------------------------------------------------------------- freqtrade feathers


def fetch_instrument(symbol: str, *, force: bool = False, category: str = "linear") -> dict:
    """
    Return the trading constraints that matter for a small account:
    {min_qty, qty_step, min_notional, tick_size, max_leverage}.
    Cached to .cache/instrument_<SYM>.json.
    """
    import json

    sym = _norm_symbol(symbol)
    path = CACHE_DIR / f"instrument_{sym}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())

    with httpx.Client(timeout=20) as client:
        payload = _bybit_get(client, "/v5/market/instruments-info",
                             {"category": category, "symbol": sym})
        d = payload["result"]["list"][0]
    lot, price, lev = d["lotSizeFilter"], d["priceFilter"], d["leverageFilter"]
    out = {
        "symbol": sym,
        "min_qty": float(lot["minOrderQty"]),
        "qty_step": float(lot["qtyStep"]),
        "min_notional": float(lot.get("minNotionalValue", 5)),
        "tick_size": float(price["tickSize"]),
        "max_leverage": float(lev["maxLeverage"]),
    }
    path.write_text(json.dumps(out, indent=2))
    return out


def load_freqtrade_feather(symbol: str, timeframe: str, base: str | Path | None = None) -> pd.DataFrame:
    """Load one of the already-downloaded freqtrade .feather files as a fallback."""
    base = Path(base or (Path(__file__).parent.parent / "user_data" / "data" / "bybit"))
    sym = _norm_symbol(symbol).replace("USDT", "_USDT")
    for cand in [base / f"{sym}-{timeframe}.feather", base / "futures" / f"{sym}_USDT-{timeframe}-futures.feather"]:
        if cand.exists():
            df = pd.read_feather(cand)
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            return df.set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()
    raise FileNotFoundError(f"no freqtrade feather for {symbol} {timeframe} under {base}")


if __name__ == "__main__":
    d = fetch_ohlcv("BTCUSDT", "1h", days=30)
    print(d.tail())
    print(f"{len(d)} bars, {d.index[0]} -> {d.index[-1]}")
