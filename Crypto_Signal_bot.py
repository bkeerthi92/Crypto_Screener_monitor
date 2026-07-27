#!/usr/bin/env python3
"""
CoinDCX Multi-Timeframe Crypto Signal Bot
==========================================

Pulls LIVE candle data directly from CoinDCX's public API for a coin of your
choice, computes RSI / MACD / Bollinger Bands / PVT / Volume Oscillator / ATR /
RVI across 1m, 15m, 1h, 4h, and 1d timeframes, and prints a Bull / Bear /
Neutral verdict table - plus an overall composite call and a mechanical,
ATR-based Entry / Stop-Loss / Target suggestion.

It also pulls the latest crypto-related headlines (including any mentioning
Trump / Musk / major figures) from free Google News RSS - no API key needed.

USAGE
-----
    pip install requests pandas numpy --break-system-packages
    python crypto_signal_bot.py HBAR
    python crypto_signal_bot.py ADA
    python crypto_signal_bot.py BTC

NOTE ON LIMITATIONS (please read)
----------------------------------
- This is a technical + headline summary tool, NOT a trading signal
  guarantee. RSI/MACD/Bollinger/PVT are lagging/momentum indicators and
  can and do give false signals, especially in choppy or thin markets.
- Real-time Twitter/X monitoring of specific individuals (Trump, Musk, etc.)
  requires a paid X API developer account. This script instead pulls public
  news headlines via Google News RSS, which covers most viral tweets/news
  anyway (journalists report on major tweets within minutes).
- This tool does not place trades. It only reads public market data.
- Nothing here is financial advice.
"""

import sys
import time
import math
import argparse
import concurrent.futures
from datetime import datetime, timezone, timedelta
from urllib.parse import quote

import requests
import pandas as pd
import numpy as np

PUBLIC_BASE = "https://public.coindcx.com"
API_BASE = "https://api.coindcx.com"

TIMEFRAMES = [
    ("1m", "1m", "1 Minute"),
    ("15m", "15m", "15 Minute"),
    ("1h", "1h", "1 Hour"),
    ("4h", "4h", "4 Hour"),
    ("1d", "1d", "1 Day"),
]

# Higher timeframes get more weight in the composite verdict (balanced default)
TIMEFRAME_WEIGHTS = {"1m": 0.5, "15m": 1.0, "1h": 1.5, "4h": 2.0, "1d": 2.5}

# --------------------------------------------------------------------------
# Trading-style profiles. A scalper and a trend trader genuinely need
# different machines: a scalper's edge lives on 1m-1h (the 1d chart is
# near-irrelevant to a 30-minute hold), while a trend trader's edge lives
# on 4h-1d (1m noise is actively harmful). One set of weights can't serve
# both honestly, so the style adjusts ALL of these consistently:
#   - composite timeframe weights
#   - which chart sizes the trade levels (ATR basis)
#   - which timeframes the fast pre-scan reads
#   - default monitoring check interval
# --------------------------------------------------------------------------
STYLE_PROFILES = {
    "scalp": {
        "weights": {"1m": 1.5, "15m": 2.5, "1h": 2.0, "4h": 1.0, "1d": 0.5},
        "basis_order": ["15m", "1h", "1m", "4h", "1d"],
        "prescan_tfs": ("15m", "1h"),
        "poll_minutes": 1,
        "note": "LTF-weighted: 15m/1h dominate; trade levels sized off the 15m chart",
    },
    "trend": {
        "weights": {"1m": 0.25, "15m": 0.75, "1h": 1.5, "4h": 2.5, "1d": 3.0},
        "basis_order": ["4h", "1d", "1h", "15m", "1m"],
        "prescan_tfs": ("4h", "1d"),
        "poll_minutes": 15,
        "note": "HTF-weighted: 4h/1d dominate; trade levels sized off the 4h chart",
    },
    "balanced": {
        "weights": {"1m": 0.5, "15m": 1.0, "1h": 1.5, "4h": 2.0, "1d": 2.5},
        "basis_order": ["4h", "1h", "1d", "15m", "1m"],
        "prescan_tfs": ("1h", "1d"),
        "poll_minutes": 5,
        "note": "original mixed weighting",
    },
}
ACTIVE_STYLE = {"name": "balanced", **STYLE_PROFILES["balanced"]}


def apply_style(style_name: str):
    """Switches the active trading style, mutating shared config in place so every module sees it."""
    profile = STYLE_PROFILES.get(style_name, STYLE_PROFILES["balanced"])
    TIMEFRAME_WEIGHTS.clear()
    TIMEFRAME_WEIGHTS.update(profile["weights"])
    TRADE_BASIS_FALLBACK_ORDER[:] = profile["basis_order"]
    ACTIVE_STYLE.clear()
    ACTIVE_STYLE.update({"name": style_name, **profile})
    print(f"[style] {style_name.upper()} mode - {profile['note']}; "
          f"monitoring default every {profile['poll_minutes']} min.")

# CoinDCX's own confirmed commodity perpetual futures pairs (as of their
# commodity fee announcement). These trade USDT-margined exactly like
# crypto futures, so the rest of the script needs no special handling for
# them beyond friendly-name aliases and better news query wording.
COMMODITY_ALIASES = {
    "GOLD": "XAU", "GOLDUSDT": "XAU",
    "SILVER": "XAG",
    "OIL": "CL", "CRUDE": "CL", "CRUDEOIL": "CL", "WTI": "CL",
    "BRENT": "BZ", "BRENTOIL": "BZ", "BRENTCRUDE": "BZ",
    "GAS": "NATGAS", "NATURALGAS": "NATGAS",
}
COMMODITY_FUTURES_SYMBOLS = {"XAU", "PAXG", "XAG", "CL", "BZ", "NATGAS"}


# --------------------------------------------------------------------------
# Pair resolution
# --------------------------------------------------------------------------

FUTURES_API_BASE = "https://api.coindcx.com/exchange/v1/derivatives/futures"

# Cache for the active-instruments list. Without this, an all-pairs screener
# scan would re-download the SAME list once per symbol (hundreds of identical
# requests); with it, the whole scan uses one fetch.
_INSTRUMENTS_CACHE = {"data": None, "ts": 0.0}


def _get_active_instruments(cache_seconds: int = 600):
    now = time.time()
    if _INSTRUMENTS_CACHE["data"] is not None and now - _INSTRUMENTS_CACHE["ts"] < cache_seconds:
        return _INSTRUMENTS_CACHE["data"]
    resp = requests.get(f"{FUTURES_API_BASE}/data/active_instruments", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    _INSTRUMENTS_CACHE["data"] = data
    _INSTRUMENTS_CACHE["ts"] = now
    return data


def fetch_all_futures_symbols(quote_ccy: str = "USDT") -> list:
    """Returns every symbol with an active CoinDCX futures pair against the quote currency."""
    instruments = _get_active_instruments()
    suffix = f"_{quote_ccy}"
    symbols, seen = [], set()
    for inst in instruments:
        if "-" in inst and inst.endswith(suffix):
            sym = inst.split("-", 1)[1][: -len(suffix)]
            if sym and sym not in seen:
                seen.add(sym)
                symbols.append(sym)
    return symbols


def resolve_pair(symbol: str, quote_ccy: str = "USDT") -> str:
    """
    Looks up the correct CoinDCX FUTURES instrument pair (e.g. 'B-HBAR_USDT')
    via the official active_instruments endpoint (cached). This script
    analyzes FUTURES data specifically (not spot) since that's what actually
    determines margin/leverage/liquidation for a real futures position, and
    it's the only way commodity pairs (XAU, XAG, CL, BZ, NATGAS) work at
    all - they have no spot market, only futures.
    """
    symbol = symbol.upper().replace("-", "_")
    if "_" in symbol:
        symbol = symbol.split("_")[0]

    try:
        instruments = _get_active_instruments()
        target_suffix = f"{symbol}_{quote_ccy}"
        for inst in instruments:
            if "-" in inst and inst.split("-", 1)[1] == target_suffix:
                return inst
    except Exception as e:
        print(f"[warn] Could not fetch active futures instruments ({e}); guessing pair format.")

    return f"B-{symbol}_{quote_ccy}"


# --------------------------------------------------------------------------
# Candle fetching
# --------------------------------------------------------------------------

# Confirmed directly from CoinDCX's official Futures API PDF: the REST
# candlesticks endpoint only supports these four resolutions. There is NO
# native 15-minute or 4-hour resolution - both are synthesized locally by
# resampling finer data (5m x3 -> 15m, 1h x4 -> 4h).
FUTURES_RESOLUTION_SECONDS = {"1": 60, "5": 300, "60": 3600, "1D": 86400}


def fetch_orderbook(pair: str, depth: int = 20) -> dict:
    """
    Fetches the FUTURES orderbook. Confirmed URL pattern from CoinDCX's
    official docs: path-based depth (10/20/50) and a '-futures' suffix on
    the pair - both different from a naive guess at the spot orderbook URL.
    """
    url = f"{PUBLIC_BASE}/market_data/v3/orderbook/{pair}-futures/{depth}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and ("bids" in data or "asks" in data):
        return data
    raise ValueError(f"Unexpected orderbook response shape for {pair}")


def compute_liquidity_metrics(ob: dict, depth: int = 20) -> dict:
    """
    Detects thin/illiquid market conditions from the live orderbook.
    Bid-ask spread % is the classic, scale-invariant measure of liquidity -
    it means the same thing whether the asset trades at $50,000 or $0.01,
    unlike an absolute depth-in-dollars threshold which would need
    per-coin calibration. Also reports total resting depth (notional
    value) near the top of book as supporting context.
    """
    bids = ob.get("bids", {}) or {}
    asks = ob.get("asks", {}) or {}
    if not bids or not asks:
        return {"available": False}

    try:
        bid_prices = sorted((float(p) for p in bids.keys()), reverse=True)
        ask_prices = sorted((float(p) for p in asks.keys()))
        best_bid, best_ask = bid_prices[0], ask_prices[0]
        bid_items = sorted(((float(p), float(q)) for p, q in bids.items()), key=lambda x: -x[0])[:depth]
        ask_items = sorted(((float(p), float(q)) for p, q in asks.items()), key=lambda x: x[0])[:depth]
    except (ValueError, TypeError, IndexError):
        return {"available": False}

    if best_ask <= 0 or best_bid <= 0 or best_ask < best_bid:
        return {"available": False}

    mid = (best_bid + best_ask) / 2
    spread_pct = (best_ask - best_bid) / mid * 100
    depth_notional = sum(p * q for p, q in bid_items) + sum(p * q for p, q in ask_items)

    if spread_pct >= 0.5:
        verdict = "THIN"
    elif spread_pct >= 0.15:
        verdict = "CAUTION"
    else:
        verdict = "NORMAL"

    return {
        "available": True, "verdict": verdict, "spread_pct": spread_pct,
        "best_bid": best_bid, "best_ask": best_ask, "depth_notional": depth_notional,
    }


def compute_orderbook_imbalance(ob: dict, depth: int = 20) -> float:
    """
    CoinDCX's orderbook returns bids/asks as DICTS of {price_str: qty_str},
    not lists of [price, qty] pairs. Sorts by price properly (best bids =
    highest price, best asks = lowest price) and sums the top `depth`
    levels on each side. Returns a value in [-1, 1]; positive = more resting
    buy interest near the top of book, negative = more resting sell interest.
    """
    bids = ob.get("bids", {}) or {}
    asks = ob.get("asks", {}) or {}
    if not bids or not asks:
        return 0.0
    try:
        bid_items = sorted(((float(p), float(q)) for p, q in bids.items()), key=lambda x: -x[0])[:depth]
        ask_items = sorted(((float(p), float(q)) for p, q in asks.items()), key=lambda x: x[0])[:depth]
    except (ValueError, TypeError):
        return 0.0

    bid_qty = sum(q for _, q in bid_items)
    ask_qty = sum(q for _, q in ask_items)
    total = bid_qty + ask_qty
    return (bid_qty - ask_qty) / total if total else 0.0


def assess_liquidity(ob: dict, df_1h: pd.DataFrame, volume_lookback: int = 20) -> dict:
    """
    Flags thin/illiquid market conditions using two independent, self-
    contained signals (no external liquidity benchmark needed):

    1. Bid-ask spread (from the live orderbook) - a wide spread relative to
       price means it's costly and risky to enter/exit a position right now.
    2. Relative volume - current volume vs that SAME asset's own recent
       rolling average, so it's self-calibrated rather than an arbitrary
       universal threshold (a "normal" volume for NATGAS is very different
       from BTC, but comparing each asset to its own recent history sidesteps
       that problem entirely).

    This does NOT feed into the Bull/Bear direction - it's a separate
    execution-risk flag, since a thin market makes ANY signal (technical,
    strategy, or news) less trustworthy, not more bullish or bearish.
    """
    bids = ob.get("bids", {}) or {}
    asks = ob.get("asks", {}) or {}
    spread_pct = None
    if bids and asks:
        try:
            best_bid = max(float(p) for p in bids.keys())
            best_ask = min(float(p) for p in asks.keys())
            mid = (best_bid + best_ask) / 2
            spread_pct = ((best_ask - best_bid) / mid) * 100 if mid else None
        except (ValueError, TypeError):
            spread_pct = None

    relative_volume = None
    if df_1h is not None and len(df_1h) > volume_lookback:
        recent_avg = df_1h["volume"].iloc[-(volume_lookback + 1):-1].mean()
        current_vol = df_1h["volume"].iloc[-1]
        if recent_avg and recent_avg > 0:
            relative_volume = current_vol / recent_avg

    reasons = []
    thin_flags = 0
    total_checks = 0

    if spread_pct is not None:
        total_checks += 1
        if spread_pct > 0.20:
            thin_flags += 1
            reasons.append(f"Wide bid-ask spread ({spread_pct:.3f}%) - costly to enter/exit right now")
        elif spread_pct > 0.08:
            reasons.append(f"Moderate bid-ask spread ({spread_pct:.3f}%)")
        else:
            reasons.append(f"Tight bid-ask spread ({spread_pct:.3f}%) - healthy liquidity")

    if relative_volume is not None:
        total_checks += 1
        if relative_volume < 0.4:
            thin_flags += 1
            reasons.append(f"Volume is only {relative_volume*100:.0f}% of its recent "
                            f"{volume_lookback}-bar average - unusually quiet")
        elif relative_volume < 0.7:
            reasons.append(f"Volume is {relative_volume*100:.0f}% of its recent average "
                            f"- somewhat below normal")
        else:
            reasons.append(f"Volume is {relative_volume*100:.0f}% of its recent average - normal or active")

    if total_checks == 0:
        verdict = "UNKNOWN"
    elif thin_flags == total_checks:
        verdict = "THIN"
    elif thin_flags > 0:
        verdict = "CAUTION"
    else:
        verdict = "NORMAL"

    return {"verdict": verdict, "spread_pct": spread_pct,
            "relative_volume": relative_volume, "reasons": reasons}


def print_liquidity_assessment(liquidity: dict):
    print(f"\n{'-'*78}\n  MARKET LIQUIDITY CHECK\n{'-'*78}")
    for r in liquidity["reasons"]:
        print(f"  - {r}")

    verdict = liquidity["verdict"]
    if verdict == "THIN":
        print("\n  ⚠ THIN MARKET - both liquidity checks came back weak.")
        print("  Every indicator in this report (RSI/MACD/Bollinger/PVT/etc.) is MORE likely")
        print("  to give a false signal right now. Consider skipping new entries, or using a")
        print("  much smaller position size than usual, until liquidity normalizes.")
    elif verdict == "CAUTION":
        print("\n  CAUTION - one liquidity check came back weak. Trade a bit more carefully than usual.")
    elif verdict == "NORMAL":
        print("\n  Liquidity looks normal - no extra caution needed on this front specifically.")
    else:
        print("\n  Could not assess liquidity this cycle (missing orderbook or volume data).")


def fetch_futures_candles(pair: str, resolution: str, limit: int = 300) -> pd.DataFrame:
    """
    Fetches candles from CoinDCX's futures-specific endpoint. Confirmed
    directly from CoinDCX's official Futures API PDF:
      GET https://public.coindcx.com/market_data/candlesticks
      params: pair, from (epoch SECONDS), to (epoch SECONDS),
              resolution ('1'/'5'/'60'/'1D' only), pcode='f'
      response: {"s": "ok", "data": [{"open":..,"high":..,"low":..,
                 "close":..,"volume":..,"time": epoch_ms}, ...]}
    """
    if resolution not in FUTURES_RESOLUTION_SECONDS:
        raise ValueError(f"Unsupported futures resolution: {resolution}")

    interval_seconds = FUTURES_RESOLUTION_SECONDS[resolution]
    to_time = int(time.time())
    from_time = to_time - interval_seconds * (limit + 5)

    url = f"{PUBLIC_BASE}/market_data/candlesticks"
    params = {"pair": pair, "from": from_time, "to": to_time, "resolution": resolution, "pcode": "f"}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    if not isinstance(payload, dict) or payload.get("s") != "ok" or not payload.get("data"):
        raise ValueError(f"No futures candle data for pair={pair} resolution={resolution} "
                          f"(raw response: {str(payload)[:200]})")

    df = pd.DataFrame(payload["data"])
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Combines finer candles into coarser ones (e.g. 5m->15m, 1h->4h) via standard OHLCV aggregation."""
    d = df.set_index("timestamp")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return d.resample(rule).agg(agg).dropna(how="any").reset_index()


def fetch_timeframe_candles(pair: str, tf_key: str, limit: int = 300) -> pd.DataFrame:
    """
    Returns candles for one of our logical timeframes (1m/15m/1h/4h/1d),
    using CoinDCX's futures API natively for 1m/1h/1d, and synthesizing
    15m (from 5m x3) and 4h (from 1h x4) since neither resolution exists
    natively on the futures candlestick endpoint.
    """
    if tf_key == "1m":
        return fetch_futures_candles(pair, "1", limit)
    elif tf_key == "15m":
        base = fetch_futures_candles(pair, "5", limit * 3)
        return resample_ohlcv(base, "15min")
    elif tf_key == "1h":
        return fetch_futures_candles(pair, "60", limit)
    elif tf_key == "4h":
        base = fetch_futures_candles(pair, "60", limit * 4)
        return resample_ohlcv(base, "4h")
    elif tf_key == "1d":
        return fetch_futures_candles(pair, "1D", limit)
    raise ValueError(f"Unknown timeframe key: {tf_key}")


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def pvt(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    volume = df["volume"]
    pct_change = close.pct_change().fillna(0)
    pvt_series = (pct_change * volume).cumsum()
    return pvt_series


def volume_oscillator(volume: pd.Series, fast: int = 5, slow: int = 10) -> pd.Series:
    """Percentage difference between a fast and slow volume moving average."""
    fast_ma = volume.rolling(fast).mean()
    slow_ma = volume.rolling(slow).mean()
    return ((fast_ma - slow_ma) / slow_ma.replace(0, np.nan)) * 100


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range - measures volatility, used for stop/target sizing."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def rvi(df: pd.DataFrame, period: int = 10) -> tuple:
    """
    Relative Vigor Index - measures conviction of a move by comparing
    close-open range to high-low range, smoothed with a symmetrical weighting.
    Returns (rvi_line, signal_line).
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    num = (c - o) + 2 * (c.shift(1) - o.shift(1)) + 2 * (c.shift(2) - o.shift(2)) + (c.shift(3) - o.shift(3))
    den = (h - l) + 2 * (h.shift(1) - l.shift(1)) + 2 * (h.shift(2) - l.shift(2)) + (h.shift(3) - l.shift(3))
    num = num / 6.0
    den = den / 6.0

    num_sma = num.rolling(period).mean()
    den_sma = den.rolling(period).mean()
    rvi_line = num_sma / den_sma.replace(0, np.nan)

    signal_line = (rvi_line + 2 * rvi_line.shift(1) + 2 * rvi_line.shift(2) + rvi_line.shift(3)) / 6.0
    return rvi_line, signal_line


def adx(df: pd.DataFrame, period: int = 14):
    """
    Average Directional Index - measures TREND STRENGTH, not direction.
    Returns (adx_line, plus_di, minus_di). A common mistake (seen in another
    script) is treating high ADX as bullish - it isn't; ADX is high during
    strong trends in EITHER direction and low during ranging/choppy markets.
    This is used as a confidence multiplier on the score, not a vote.
    """
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    smoothed_tr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False).mean() / smoothed_tr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False).mean() / smoothed_tr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx_line, plus_di, minus_di


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume Weighted Average Price, anchored to each calendar day (resets
    daily) rather than accumulated across the whole fetched history. A
    VWAP that never resets drifts into a meaningless long-run average and
    loses the "fair value for today" meaning VWAP is supposed to have.
    """
    date_key = df["timestamp"].dt.date
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    tpv = typical_price * df["volume"]
    cum_tpv = tpv.groupby(date_key).cumsum()
    cum_vol = df["volume"].groupby(date_key).cumsum()
    return cum_tpv / cum_vol.replace(0, np.nan)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: adds full volume on up-closes, subtracts on down-closes."""
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RSI"] = rsi(df["close"], 14)
    macd_line, signal_line, hist = macd(df["close"])
    df["MACD"] = macd_line
    df["MACD_signal"] = signal_line
    df["MACD_hist"] = hist
    bb_up, bb_mid, bb_low = bollinger_bands(df["close"], 20, 2.0)
    df["BB_upper"] = bb_up
    df["BB_mid"] = bb_mid
    df["BB_lower"] = bb_low
    df["PVT"] = pvt(df)
    df["VolOsc"] = volume_oscillator(df["volume"], 5, 10)
    df["ATR"] = atr(df, 14)
    rvi_line, rvi_signal = rvi(df, 10)
    df["RVI"] = rvi_line
    df["RVI_signal"] = rvi_signal
    adx_line, plus_di, minus_di = adx(df, 14)
    df["ADX"] = adx_line
    df["PlusDI"] = plus_di
    df["MinusDI"] = minus_di
    df["VWAP"] = session_vwap(df)
    df["OBV"] = obv(df["close"], df["volume"])
    return df


# --------------------------------------------------------------------------
# Signal classification
# --------------------------------------------------------------------------

def classify_timeframe(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    score = 0
    reasons = []

    # RSI
    if pd.notna(latest["RSI"]):
        if latest["RSI"] >= 60:
            score += 1
            reasons.append(f"RSI {latest['RSI']:.1f} (bullish momentum)")
        elif latest["RSI"] <= 40:
            score -= 1
            reasons.append(f"RSI {latest['RSI']:.1f} (bearish momentum)")
        else:
            reasons.append(f"RSI {latest['RSI']:.1f} (neutral)")

    # MACD histogram + direction
    if pd.notna(latest["MACD_hist"]):
        if latest["MACD_hist"] > 0:
            score += 1
            if prev["MACD_hist"] <= 0:
                reasons.append("MACD just crossed bullish")
            else:
                reasons.append("MACD histogram positive")
        else:
            score -= 1
            if prev["MACD_hist"] >= 0:
                reasons.append("MACD just crossed bearish")
            else:
                reasons.append("MACD histogram negative")

    # Bollinger position
    if pd.notna(latest["BB_mid"]):
        _adx_now = latest.get("ADX")
        _strong_trend = pd.notna(_adx_now) and _adx_now >= 25
        if latest["close"] > latest["BB_upper"]:
            # Context-dependent: an upper-band break in a STRONG trend tends to
            # continue (breakout); in a weak/ranging market it tends to snap
            # back (mean reversion). Previously this always scored +0.5 while
            # the label said "overextended" - contradictory messaging, now fixed.
            if _strong_trend:
                score += 0.5
                reasons.append(f"Price above upper BB with strong ADX (breakout continuation)")
            else:
                score -= 0.5
                reasons.append(f"Price above upper BB with weak ADX (overextended - mean-reversion risk)")
        elif latest["close"] < latest["BB_lower"]:
            if _strong_trend:
                score -= 1.5
                reasons.append("Price below lower BB with strong ADX (breakdown)")
            else:
                score -= 0.5
                reasons.append("Price below lower BB with weak ADX (oversold - bounce risk)")
        elif latest["close"] > latest["BB_mid"]:
            score += 1
            reasons.append("Price above BB midline")
        else:
            score -= 1
            reasons.append("Price below BB midline")

    # PVT trend (rough slope over last 10 bars)
    if len(df) > 10 and pd.notna(latest["PVT"]):
        pvt_slope = df["PVT"].iloc[-1] - df["PVT"].iloc[-10]
        if pvt_slope > 0:
            score += 0.5
            reasons.append("PVT rising (buying volume pressure)")
        else:
            score -= 0.5
            reasons.append("PVT falling (selling volume pressure)")

    # RVI (Relative Vigor Index) - conviction behind the move
    if pd.notna(latest.get("RVI")):
        if latest["RVI"] > latest.get("RVI_signal", 0):
            score += 0.5
            reasons.append(f"RVI {latest['RVI']:.2f} above signal (bullish conviction)")
        else:
            score -= 0.5
            reasons.append(f"RVI {latest['RVI']:.2f} below signal (bearish conviction)")

    # VWAP (session-anchored) - price above/below today's volume-weighted fair value
    if pd.notna(latest.get("VWAP")):
        if latest["close"] > latest["VWAP"]:
            score += 0.75
            reasons.append(f"Price above session VWAP ({latest['VWAP']:.6f})")
        else:
            score -= 0.75
            reasons.append(f"Price below session VWAP ({latest['VWAP']:.6f})")

    # OBV trend - weighted lower than PVT since testing showed they're ~0.8
    # correlated (both are volume/price-direction indicators); OBV still adds
    # some independent information (it weights by raw volume, not % price
    # move), just not a full extra vote's worth.
    if len(df) > 10 and pd.notna(latest.get("OBV")):
        obv_slope = df["OBV"].iloc[-1] - df["OBV"].iloc[-10]
        if obv_slope > 0:
            score += 0.25
            reasons.append("OBV rising (accumulation)")
        else:
            score -= 0.25
            reasons.append("OBV falling (distribution)")

    # Volume Oscillator - a CONVICTION MULTIPLIER, like ADX (never a
    # directional vote, since volume expands in rallies AND panic dumps).
    # Strong short-term volume expansion means the current move has real
    # participation -> amplify whatever the score already says; a move on
    # sharply dying volume is less trustworthy -> dampen it. Multipliers are
    # deliberately gentler than ADX's (1.15/0.75) because VolOsc is a fast
    # 5/10-bar measure and much noisier than a 14-period ADX.
    if pd.notna(latest.get("VolOsc")):
        if latest["VolOsc"] >= 20:
            score *= 1.10
            reasons.append(f"Volume Osc {latest['VolOsc']:.1f}% (volume expanding - conviction amplified)")
        elif latest["VolOsc"] <= -30:
            score *= 0.85
            reasons.append(f"Volume Osc {latest['VolOsc']:.1f}% (volume drying up - conviction dampened)")
        elif latest["VolOsc"] > 0:
            reasons.append(f"Volume Osc {latest['VolOsc']:.1f}% (short-term volume expanding)")
        else:
            reasons.append(f"Volume Osc {latest['VolOsc']:.1f}% (short-term volume contracting)")

    # ADX - a CONFIDENCE MULTIPLIER on the score, not a directional vote.
    # ADX only measures trend strength: high during strong trends in EITHER
    # direction, low when ranging. Scaling the score by it means the same
    # raw signal produces a more decisive verdict when the trend is
    # confirmed strong, and a more cautious one in a choppy/ranging market.
    adx_note = None
    if pd.notna(latest.get("ADX")):
        if latest["ADX"] >= 25:
            score *= 1.15
            adx_note = f"ADX {latest['ADX']:.1f} (strong trend - conviction boosted)"
        elif latest["ADX"] < 20:
            score *= 0.75
            adx_note = f"ADX {latest['ADX']:.1f} (weak/ranging - conviction dampened)"
        else:
            adx_note = f"ADX {latest['ADX']:.1f} (moderate trend strength)"
        reasons.append(adx_note)

    if score >= 1.5:
        verdict = "BULL"
    elif score <= -1.5:
        verdict = "BEAR"
    else:
        verdict = "NEUTRAL"

    return {
        "verdict": verdict,
        "score": score,
        "price": latest["close"],
        "rsi": latest["RSI"],
        "macd_hist": latest["MACD_hist"],
        "bb_mid": latest["BB_mid"],
        "bb_upper": latest["BB_upper"],
        "bb_lower": latest["BB_lower"],
        "atr": latest.get("ATR"),
        "vol_osc": latest.get("VolOsc"),
        "rvi": latest.get("RVI"),
        "adx": latest.get("ADX"),
        "vwap": latest.get("VWAP"),
        "reasons": reasons,
    }


def compute_timeframe_alignment(results: dict) -> dict:
    """
    Explicit measure of how many timeframes actually agree on direction.
    A weighted composite can land on BULL even if only 2 of 5 timeframes
    are individually bullish, if the higher-weighted ones dominate - this
    makes that visible and usable as its own confidence signal, separate
    from ADX (which measures trend strength WITHIN one timeframe, not
    agreement ACROSS timeframes).
    """
    verdicts = [r["verdict"] for r in results.values() if "verdict" in r]
    if not verdicts:
        return {"available": False}
    bulls = verdicts.count("BULL")
    bears = verdicts.count("BEAR")
    total = len(verdicts)
    dominant = max(bulls, bears)
    alignment_pct = dominant / total * 100
    dominant_direction = "BULL" if bulls >= bears else "BEAR"
    return {"available": True, "alignment_pct": alignment_pct, "bulls": bulls,
            "bears": bears, "neutral": total - bulls - bears, "total": total,
            "dominant_direction": dominant_direction}


def composite_verdict(results: dict) -> dict:
    total = 0.0
    max_possible = 0.0
    for tf_key, res in results.items():
        w = TIMEFRAME_WEIGHTS.get(tf_key, 1.0)
        total += res["score"] * w
        max_possible += 3.5 * w  # rough max score per timeframe

    raw_normalized = total / max_possible if max_possible else 0

    alignment = compute_timeframe_alignment(results)
    if alignment.get("available"):
        if alignment["alignment_pct"] >= 80:
            alignment_mult = 1.15
        elif alignment["alignment_pct"] <= 40:
            alignment_mult = 0.75
        else:
            alignment_mult = 1.0
    else:
        alignment_mult = 1.0

    # 3.5 is a "typical strong timeframe" scale, NOT the true maximum (a
    # maxed-out timeframe can hit ~5.5, x1.15 with the ADX boost). On extreme
    # days where several timeframes are near-max in the same direction (seen
    # live on BZ: printed -1.74 on a scale claiming +/-1), the ratio overflows
    # - so clamp AFTER the alignment multiplier. Chosen over rescaling the
    # denominator to keep all existing verdict-threshold calibration intact;
    # anything beyond +/-1 carries no extra meaning ("maximally bearish" is
    # maximally bearish).
    normalized = max(min(raw_normalized * alignment_mult, 1.0), -1.0)
    if normalized > 0.15:
        verdict = "BULL"
    elif normalized < -0.15:
        verdict = "BEAR"
    else:
        verdict = "NEUTRAL"
    return {"verdict": verdict, "normalized_score": normalized,
            "raw_score": raw_normalized, "alignment": alignment}


# --------------------------------------------------------------------------
# Strategy signals: Market Structure, FVG/Imbalance, Liquidity Trap,
# Fibonacci Pullback, HTF Bias, Volume Profile (HVN), ICT Silver Bullet
#
# HONESTY NOTE: Market Structure Shift, FVG, Liquidity Trap, Fib Pullback,
# HTF Bias, and Volume Profile/HVN are objectively-defined technical
# concepts - implemented and tested below with synthetic data.
# The ICT "Silver Bullet" is a specific, discretionary trading-course
# concept with community variations on its exact rules. What's below is
# ONE reasonable systemization of the most widely-cited version (a
# liquidity sweep + FVG reversal during the 10-11 AM New York session
# window) - not a validated high-win-rate system, despite how it's often
# marketed. Treat it the same as the others: a mechanical reading of a
# concept, not a guarantee.
# --------------------------------------------------------------------------

def find_swing_points(df: pd.DataFrame, left: int = 2, right: int = 2):
    """
    Fractal-style swing high/low detection: a bar is a swing high if its
    high is the strict max within [left, right] bars around it (same for lows).
    Returns (swing_highs, swing_lows) as lists of (index, price).
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    swing_highs, swing_lows = [], []
    for i in range(left, n - right):
        h_window = highs[i - left:i + right + 1]
        if highs[i] == h_window.max() and (h_window == highs[i]).sum() == 1:
            swing_highs.append((i, highs[i]))
        l_window = lows[i - left:i + right + 1]
        if lows[i] == l_window.min() and (l_window == lows[i]).sum() == 1:
            swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows



def detect_market_structure(df: pd.DataFrame, left: int = 2, right: int = 2) -> dict:
    """
    Uses the last two swing highs/lows to classify trend (higher-highs +
    higher-lows = BULLISH, lower-highs + lower-lows = BEARISH), and flags a
    Market Structure Shift (MSS) if the current close breaks the most
    recent opposite-direction swing point.
    """
    swing_highs, swing_lows = find_swing_points(df, left, right)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"trend": "UNKNOWN", "mss": None}

    _, last_high = swing_highs[-1]
    _, prev_high = swing_highs[-2]
    _, last_low = swing_lows[-1]
    _, prev_low = swing_lows[-2]

    if last_high > prev_high and last_low > prev_low:
        trend = "BULLISH"
    elif last_high < prev_high and last_low < prev_low:
        trend = "BEARISH"
    else:
        trend = "RANGING"

    current_close = df["close"].iloc[-1]
    mss = None
    if trend == "BULLISH" and current_close < last_low:
        mss = "BEARISH_MSS"
    elif trend == "BEARISH" and current_close > last_high:
        mss = "BULLISH_MSS"

    return {"trend": trend, "mss": mss, "last_swing_high": last_high, "last_swing_low": last_low}


def detect_fvg(df: pd.DataFrame, lookback: int = 50):
    """
    Standard 3-candle Fair Value Gap / imbalance: a bullish FVG forms when
    candle[i-2].high < candle[i].low (a gap the price hasn't traded through);
    a bearish FVG forms when candle[i-2].low > candle[i].high.
    """
    gaps = []
    n = len(df)
    start = max(2, n - lookback)
    for i in range(start, n):
        c1_high, c1_low = df["high"].iloc[i - 2], df["low"].iloc[i - 2]
        c3_high, c3_low = df["high"].iloc[i], df["low"].iloc[i]
        if c1_high < c3_low:
            gaps.append({"index": i, "type": "BULLISH", "top": c3_low, "bottom": c1_high})
        elif c1_low > c3_high:
            gaps.append({"index": i, "type": "BEARISH", "top": c1_low, "bottom": c3_high})
    return gaps


def latest_unfilled_fvg_signal(df: pd.DataFrame, gaps: list):
    """If price is currently trading inside the most recent FVG, that's a common entry trigger."""
    if not gaps:
        return None
    current_price = df["close"].iloc[-1]
    recent = gaps[-1]
    if recent["bottom"] <= current_price <= recent["top"]:
        return recent["type"]
    return None


def detect_liquidity_trap(df: pd.DataFrame, swing_lookback: int = 20):
    """
    A 'liquidity trap' (a.k.a. stop hunt): price wicks beyond a recent swing
    high/low (sweeping resting stop orders) then closes back inside the
    prior range - a common precursor to a reversal against the breakout.
    """
    sub = df.iloc[-swing_lookback:].reset_index(drop=True)
    swing_highs, swing_lows = find_swing_points(sub, left=1, right=1)
    if not swing_highs and not swing_lows:
        return None

    last = df.iloc[-1]
    recent_low = min((p for _, p in swing_lows), default=None)
    recent_high = max((p for _, p in swing_highs), default=None)

    if recent_low is not None and last["low"] < recent_low and last["close"] > recent_low:
        return "BULLISH_TRAP"
    if recent_high is not None and last["high"] > recent_high and last["close"] < recent_high:
        return "BEARISH_TRAP"
    return None


def fibonacci_pullback_signal(df: pd.DataFrame, bias_direction: str, left: int = 3, right: int = 3):
    """
    Finds the most recent swing high/low, and checks whether price has
    pulled back into the 61.8%-78.6% 'golden zone' of that leg, in the
    direction of the given higher-timeframe bias.
    """
    swing_highs, swing_lows = find_swing_points(df, left, right)
    if not swing_highs or not swing_lows:
        return None

    last_high_idx, last_high = swing_highs[-1]
    last_low_idx, last_low = swing_lows[-1]
    current_price = df["close"].iloc[-1]
    leg = last_high - last_low
    if leg <= 0:
        return None

    if bias_direction == "BULLISH" and last_low_idx < last_high_idx:
        level_618 = last_high - 0.618 * leg
        level_786 = last_high - 0.786 * leg
        if level_786 <= current_price <= level_618:
            return {"direction": "BULLISH", "level_618": level_618, "level_786": level_786}
    elif bias_direction == "BEARISH" and last_high_idx < last_low_idx:
        level_618 = last_low + 0.618 * leg
        level_786 = last_low + 0.786 * leg
        if level_618 <= current_price <= level_786:
            return {"direction": "BEARISH", "level_618": level_618, "level_786": level_786}
    return None


def compute_htf_bias(results: dict) -> str:
    """
    Combines the 1D and 4h verdicts into a higher-timeframe bias label.
    Degrades gracefully if one timeframe's data is missing (e.g. CoinDCX
    doesn't have 4h candles cached for a thin/newer pair) by basing the
    bias on whichever one is actually available, rather than giving up.
    """
    d, h4 = results.get("1d"), results.get("4h")

    if not d and not h4:
        return "UNKNOWN"

    if d and h4:
        bulls = sum(1 for r in (d, h4) if r["verdict"] == "BULL")
        bears = sum(1 for r in (d, h4) if r["verdict"] == "BEAR")
        if bulls == 2:
            return "BULLISH"
        if bears == 2:
            return "BEARISH"
        if bulls == 1 and bears == 0:
            return "LEAN_BULLISH"
        if bears == 1 and bulls == 0:
            return "LEAN_BEARISH"
        return "MIXED"

    # Only one of the two is available - use it alone, one notch weaker
    only = d or h4
    if only["verdict"] == "BULL":
        return "LEAN_BULLISH"
    if only["verdict"] == "BEAR":
        return "LEAN_BEARISH"
    return "MIXED"


def detect_divergence(df: pd.DataFrame, indicator_col: str = "RSI", left: int = 2,
                       right: int = 2, lookback: int = 60):
    """
    Classic bullish/bearish divergence between price and an oscillator
    (default RSI): price makes a LOWER low while the oscillator makes a
    HIGHER low (bullish - downward momentum is fading despite the new
    low), or price makes a HIGHER high while the oscillator makes a LOWER
    high (bearish - upward momentum is fading). This is one of the more
    respected classic reversal signals and is distinct from the level-based
    RSI/MACD checks already in the scoring (those look at where the
    indicator IS; this looks at whether its trajectory agrees with price's).
    """
    sub = df.iloc[-lookback:].reset_index(drop=True) if len(df) > lookback else df.reset_index(drop=True)
    swing_highs, swing_lows = find_swing_points(sub, left, right)

    signal = None
    if len(swing_lows) >= 2:
        (idx1, low1), (idx2, low2) = swing_lows[-2], swing_lows[-1]
        if low2 < low1:
            ind1, ind2 = sub[indicator_col].iloc[idx1], sub[indicator_col].iloc[idx2]
            if pd.notna(ind1) and pd.notna(ind2) and ind2 > ind1:
                signal = "BULLISH_DIVERGENCE"

    if len(swing_highs) >= 2:
        (idx1, high1), (idx2, high2) = swing_highs[-2], swing_highs[-1]
        if high2 > high1:
            ind1, ind2 = sub[indicator_col].iloc[idx1], sub[indicator_col].iloc[idx2]
            if pd.notna(ind1) and pd.notna(ind2) and ind2 < ind1:
                signal = "BEARISH_DIVERGENCE" if signal is None else signal

    return signal


def detect_bollinger_squeeze(df: pd.DataFrame, lookback: int = 120, percentile_threshold: float = 10.0):
    """
    Flags a Bollinger Band squeeze: current band width is unusually narrow
    relative to its own recent history - a classic precursor to a
    volatility breakout. Does NOT predict direction, only that a bigger
    move than recent conditions may be coming soon.
    """
    if len(df) < lookback or "BB_upper" not in df.columns:
        return None
    bb_width = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"].replace(0, np.nan)
    recent = bb_width.iloc[-lookback:]
    current = bb_width.iloc[-1]
    if pd.isna(current) or recent.isna().all():
        return None
    percentile_rank = (recent <= current).mean() * 100
    return {"is_squeeze": percentile_rank <= percentile_threshold,
            "width_percentile": percentile_rank, "current_width": current}


def compute_volume_profile_hvn(df: pd.DataFrame, bins: int = 20, lookback: int = 100, top_n: int = 2):
    """
    Approximate volume profile: bins the traded price range into levels and
    assigns each candle's volume to the bin containing its close. Returns
    the top_n highest-volume bins (High Volume Nodes) as (low, high, volume).
    This is a bar-close approximation, not true tick-by-tick volume profile.
    """
    sub = df.iloc[-lookback:]
    price_min, price_max = sub["low"].min(), sub["high"].max()
    if price_max <= price_min:
        return []
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    vol_per_bin = np.zeros(bins)
    for close, vol in zip(sub["close"], sub["volume"]):
        idx = min(max(np.searchsorted(bin_edges, close, side="right") - 1, 0), bins - 1)
        vol_per_bin[idx] += vol
    top_idx = np.argsort(vol_per_bin)[-top_n:]
    return [(bin_edges[i], bin_edges[i + 1], vol_per_bin[i]) for i in sorted(top_idx)]


def hvn_reaction_signal(df: pd.DataFrame, hvns: list, proximity_pct: float = 0.5):
    """Checks if the latest close is near a High Volume Node and reacting off it."""
    if not hvns:
        return None
    price = df["close"].iloc[-1]
    for lo, hi, _ in hvns:
        mid = (lo + hi) / 2
        if mid > 0 and abs(price - mid) / mid * 100 <= proximity_pct:
            last = df.iloc[-1]
            if last["close"] > last["open"]:
                return {"level": mid, "reaction": "BULLISH_BOUNCE"}
            elif last["close"] < last["open"]:
                return {"level": mid, "reaction": "BEARISH_REJECTION"}
    return None


def ict_silver_bullet_signal(df_1m: pd.DataFrame, now_utc=None) -> dict:
    """
    One systemization of the widely-cited ICT 'Silver Bullet': the 10:00-11:00
    AM New York time window, looking for a liquidity sweep of a recent 1m
    swing point followed by an FVG in the reversal direction. See the module
    docstring above for the honesty caveat on this specific strategy.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        ny_time = now_utc.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        from datetime import timedelta as _td
        ny_time = now_utc - _td(hours=4)  # rough EDT fallback if tzdata is unavailable

    if ny_time.hour != 10:
        return {"active": False, "ny_time": ny_time.strftime("%H:%M"), "signal": None}

    swing_highs, swing_lows = find_swing_points(df_1m, left=2, right=2)
    gaps = detect_fvg(df_1m, lookback=20)
    if not swing_highs or not swing_lows or not gaps:
        return {"active": True, "ny_time": ny_time.strftime("%H:%M"), "signal": None}

    last = df_1m.iloc[-1]
    recent_low = swing_lows[-1][1]
    recent_high = swing_highs[-1][1]
    recent_gap = gaps[-1]

    signal = None
    if last["low"] < recent_low and recent_gap["type"] == "BULLISH":
        signal = "BULLISH_SILVER_BULLET"
    elif last["high"] > recent_high and recent_gap["type"] == "BEARISH":
        signal = "BEARISH_SILVER_BULLET"

    return {"active": True, "ny_time": ny_time.strftime("%H:%M"), "signal": signal}


def compute_strategy_signals(dfs_by_tf: dict, results: dict, orderbook_imbalance: float = None,
                              btc_regime: dict = None) -> dict:
    """Runs all strategy modules and aggregates them into one strategy_score in [-1, 1]."""
    signals = {}
    htf_bias = compute_htf_bias(results)
    signals["htf_bias"] = htf_bias

    df_1h = dfs_by_tf.get("1h")
    signals["market_structure"] = (
        detect_market_structure(df_1h) if df_1h is not None and len(df_1h) > 20
        else {"trend": "UNKNOWN", "mss": None}
    )

    df_15m = dfs_by_tf.get("15m")
    fvg_signal, hvn_signal = None, None
    if df_15m is not None and len(df_15m) > 10:
        fvg_signal = latest_unfilled_fvg_signal(df_15m, detect_fvg(df_15m, lookback=50))
    if df_15m is not None and len(df_15m) > 30:
        hvns = compute_volume_profile_hvn(df_15m, bins=20, lookback=min(150, len(df_15m)))
        hvn_signal = hvn_reaction_signal(df_15m, hvns)
    signals["fvg_15m"] = fvg_signal
    signals["hvn_15m"] = hvn_signal

    signals["liquidity_trap_1h"] = (
        detect_liquidity_trap(df_1h) if df_1h is not None and len(df_1h) > 20 else None
    )

    df_4h = dfs_by_tf.get("4h")
    fib_signal = None
    if df_4h is not None and len(df_4h) > 20 and htf_bias in ("BULLISH", "BEARISH", "LEAN_BULLISH", "LEAN_BEARISH"):
        direction = "BULLISH" if "BULLISH" in htf_bias else "BEARISH"
        fib_signal = fibonacci_pullback_signal(df_4h, direction)
    signals["fib_pullback"] = fib_signal

    df_1m = dfs_by_tf.get("1m")
    signals["silver_bullet"] = (
        ict_silver_bullet_signal(df_1m) if df_1m is not None and len(df_1m) > 10 else {"active": False, "signal": None}
    )

    signals["orderbook_imbalance"] = orderbook_imbalance

    # Divergence is a stronger, more respected signal than most others here -
    # checked on 1h for a reasonable balance of noise vs timeliness.
    divergence_signal = detect_divergence(df_1h, "RSI") if df_1h is not None and len(df_1h) > 30 else None
    # Confirmation filter (incorporated from external review): a divergence
    # alone is an early-warning, not a turn. It gets FULL vote weight only
    # once RSI has actually crossed back through 50 in the divergence's
    # direction; before that it counts at reduced weight as "unconfirmed".
    divergence_confirmed = False
    if divergence_signal and df_1h is not None:
        last_rsi = df_1h["RSI"].iloc[-1]
        if pd.notna(last_rsi):
            if divergence_signal == "BULLISH_DIVERGENCE" and last_rsi > 50:
                divergence_confirmed = True
            elif divergence_signal == "BEARISH_DIVERGENCE" and last_rsi < 50:
                divergence_confirmed = True
    signals["divergence_confirmed"] = divergence_confirmed
    signals["divergence_1h"] = divergence_signal

    squeeze = detect_bollinger_squeeze(df_1h) if df_1h is not None else None
    signals["bollinger_squeeze"] = squeeze

    signals["btc_regime"] = btc_regime

    votes = []
    if signals["market_structure"].get("mss") == "BULLISH_MSS":
        votes.append(1)
    elif signals["market_structure"].get("mss") == "BEARISH_MSS":
        votes.append(-1)
    if fvg_signal == "BULLISH":
        votes.append(1)
    elif fvg_signal == "BEARISH":
        votes.append(-1)
    if signals["liquidity_trap_1h"] == "BULLISH_TRAP":
        votes.append(1)
    elif signals["liquidity_trap_1h"] == "BEARISH_TRAP":
        votes.append(-1)
    if fib_signal:
        votes.append(1 if fib_signal["direction"] == "BULLISH" else -1)
    if hvn_signal:
        votes.append(0.5 if hvn_signal["reaction"] == "BULLISH_BOUNCE" else -0.5)
    if signals["silver_bullet"].get("signal"):
        votes.append(1 if signals["silver_bullet"]["signal"] == "BULLISH_SILVER_BULLET" else -1)
    if orderbook_imbalance is not None:
        if orderbook_imbalance >= 0.10:
            votes.append(0.5)
        elif orderbook_imbalance <= -0.10:
            votes.append(-0.5)
    if divergence_signal == "BULLISH_DIVERGENCE":
        votes.append(1.5 if divergence_confirmed else 0.5)
    elif divergence_signal == "BEARISH_DIVERGENCE":
        votes.append(-1.5 if divergence_confirmed else -0.5)
    if btc_regime and btc_regime.get("available"):
        regime = btc_regime["regime"]
        if regime == "BULLISH":
            votes.append(0.75)
        elif regime == "BEARISH":
            votes.append(-0.75)
        elif regime == "LEAN_BULLISH":
            votes.append(0.4)
        elif regime == "LEAN_BEARISH":
            votes.append(-0.4)
    # Squeeze deliberately casts NO directional vote - it flags "big move
    # coming" with no direction, so it can't honestly push the score either way.

    # FIXED denominator = the max weight possible if MSS(1) + FVG(1) + Trap(1)
    # + Fib(1) + HVN(0.5) + SilverBullet(1) + Orderbook(0.5) + Divergence(1.5)
    # + BTCRegime(0.75) ALL fired unanimously in the same direction = 8.25.
    #
    # BUG FIX (found via external review + verified against actual output):
    # the old code divided by len(votes) - the number of signals that
    # actually fired - not by this fixed total. That meant just 2 of 9
    # possible signals firing (e.g. only Divergence + BTC Regime, both
    # bullish) would average to 1.125, clamp to a maxed-out +1.00, and get
    # treated identically to full 9/9 unanimous agreement. That's how a
    # -0.94 technical read (strong, broad-based bearish evidence across
    # all 5 timeframes) got dragged all the way to NEUTRAL by just 2 sparse
    # bullish signals. Dividing by the fixed max instead means conviction
    # now scales with how MUCH of the total possible evidence actually
    # showed up, not just the average of whatever happened to fire.
    STRATEGY_MAX_WEIGHT = 8.25
    signals["strategy_score"] = max(min((sum(votes) / STRATEGY_MAX_WEIGHT) if votes else 0.0, 1.0), -1.0)

    if signals["strategy_score"] > 0.15:
        signals["verdict"] = "BULL"
    elif signals["strategy_score"] < -0.15:
        signals["verdict"] = "BEAR"
    else:
        signals["verdict"] = "NEUTRAL"

    # Market-state label (incorporated from external review): a bullish
    # strategy score while HTF bias / market structure are still bearish is
    # NOT a confirmed bull trend - it's an unconfirmed reversal attempt, and
    # labeling it plain "BULL" overstates it. Internal BULL/BEAR/NEUTRAL is
    # kept unchanged for all downstream machinery; this is the honest
    # human-facing description.
    ms_trend = signals.get("market_structure", {}).get("trend", "")
    ctx_bearish = ("BEARISH" in str(signals.get("htf_bias", ""))) or (ms_trend == "BEARISH")
    ctx_bullish = ("BULLISH" in str(signals.get("htf_bias", ""))) or (ms_trend == "BULLISH")
    if signals["verdict"] == "BULL":
        if ctx_bearish and not ctx_bullish:
            signals["market_state"] = "POTENTIAL BULLISH REVERSAL (unconfirmed - trend still bearish)"
        elif signals["strategy_score"] >= 0.45:
            signals["market_state"] = "STRONG BULL"
        else:
            signals["market_state"] = "BULL"
    elif signals["verdict"] == "BEAR":
        if ctx_bullish and not ctx_bearish:
            signals["market_state"] = "POTENTIAL BEARISH REVERSAL (unconfirmed - trend still bullish)"
        elif signals["strategy_score"] <= -0.45:
            signals["market_state"] = "STRONG BEAR"
        else:
            signals["market_state"] = "BEAR"
    else:
        signals["market_state"] = "NEUTRAL"

    return signals


def print_strategy_signals(signals: dict):
    print(f"\n{'-'*78}\n  STRATEGY SIGNALS (SMC / ICT-style readings)\n{'-'*78}")
    print(f"  HTF Bias (1D+4H)      : {signals['htf_bias']}")
    ms = signals["market_structure"]
    print(f"  Market Structure (1H) : trend={ms['trend']}, MSS={ms.get('mss') or 'none'}")
    print(f"  15m Imbalance/FVG     : {signals['fvg_15m'] or 'price not inside a recent FVG'}")
    hvn = signals["hvn_15m"]
    print(f"  15m Volume Node       : {hvn['reaction'] if hvn else 'no reaction at a High Volume Node right now'}")
    print(f"  1H Liquidity Trap     : {signals['liquidity_trap_1h'] or 'none detected'}")
    ob_imb = signals.get("orderbook_imbalance")
    if ob_imb is not None:
        tilt = "bid-heavy (buy pressure)" if ob_imb >= 0.10 else \
               "ask-heavy (sell pressure)" if ob_imb <= -0.10 else "roughly balanced"
        print(f"  Orderbook Imbalance   : {ob_imb:+.2f}  ({tilt})")
    else:
        print("  Orderbook Imbalance   : unavailable this cycle")

    liq = signals.get("liquidity", {})
    if liq.get("available"):
        print(f"  Market Liquidity      : {liq['verdict']}  "
              f"(spread {liq['spread_pct']:.3f}%, top-of-book depth ~{liq['depth_notional']:.2f} USDT)")
        if liq["verdict"] == "THIN":
            print("    >>> Spread is unusually wide right now - this often means low liquidity."
                  " Expect slippage, and consider NOT trading until it normalizes. <<<")
        elif liq["verdict"] == "CAUTION":
            print("    Spread is a bit wider than a deeply liquid market - trade smaller size"
                  " or use limit orders if you do trade.")
    else:
        print("  Market Liquidity      : unavailable this cycle")

    chop = signals.get("choppiness", {})
    if chop.get("available"):
        print(f"  Market Choppiness     : {chop['verdict']}  (ADX {chop['adx']:.1f} on {chop['timeframe']} chart)")
        if chop["verdict"] == "CHOPPY":
            print("    >>> Market is ranging, not trending (low ADX). Trend-following signals are"
                  " much less reliable here - expect more false signals and whipsaws. Consider"
                  " NOT trading until a clearer trend forms. <<<")
        elif chop["verdict"] == "MODERATE":
            print("    Trend strength is moderate - not a clean trend, not fully ranging either.")
    else:
        print("  Market Choppiness     : unavailable this cycle")
    fib = signals["fib_pullback"]
    if fib:
        print(f"  Fibonacci Pullback    : {fib['direction']} golden zone "
              f"({fib['level_786']:.6f} - {fib['level_618']:.6f})")
    else:
        print("  Fibonacci Pullback    : not currently in a golden-zone pullback")
    sb = signals["silver_bullet"]
    if sb.get("active"):
        print(f"  ICT Silver Bullet     : window ACTIVE (NY time {sb['ny_time']}), signal={sb.get('signal') or 'none'}")
    else:
        print(f"  ICT Silver Bullet     : window not active (NY time {sb.get('ny_time','?')}, needs 10:00-11:00 NY)")

    div = signals.get("divergence_1h")
    print(f"  RSI Divergence (1H)   : {div or 'none detected'}")

    btc = signals.get("btc_regime")
    if btc and btc.get("available"):
        print(f"  BTC Market Regime     : {btc['regime']}")
    elif btc is not None:
        print("  BTC Market Regime     : unavailable this cycle")

    sq = signals.get("bollinger_squeeze")
    if sq:
        squeeze_note = "SQUEEZE - low volatility, breakout may be near (direction unknown)" if sq["is_squeeze"] \
            else "no squeeze"
        print(f"  Bollinger Squeeze     : {squeeze_note} (width percentile: {sq['width_percentile']:.1f}%)")
    else:
        print("  Bollinger Squeeze     : unavailable this cycle")

    print(f"\n  STRATEGY VERDICT: {signals['verdict']}  (composite score {signals['strategy_score']:+.2f}, range -1 to +1)")
    if signals.get("market_state") and signals["market_state"] != signals["verdict"]:
        print(f"  MARKET STATE    : {signals['market_state']}")
    print("  [Note: Silver Bullet is a discretionary ICT concept - see script docstring for caveat]")


def assess_choppiness(results: dict) -> dict:
    """
    Flags choppy/ranging conditions using ADX (already computed per
    timeframe - no extra fetch needed). Reuses the same basis-timeframe
    fallback order as the trade-level suggestion, so the choppiness read
    always matches whichever chart's ATR is sizing the trade.
    """
    for tf in TRADE_BASIS_FALLBACK_ORDER:
        r = results.get(tf)
        if r is not None and pd.notna(r.get("adx")):
            adx_val = r["adx"]
            if adx_val < 20:
                verdict = "CHOPPY"
            elif adx_val >= 25:
                verdict = "TRENDING"
            else:
                verdict = "MODERATE"
            return {"available": True, "verdict": verdict, "adx": adx_val, "timeframe": tf}
    return {"available": False}


# --------------------------------------------------------------------------
# Trade level suggestions (entry / stop-loss / target)
# --------------------------------------------------------------------------

# Preferred order for the trade-level basis timeframe: 4h is the ideal
# middle ground for swing-style entries, but if a thin/newer pair doesn't
# have 4h candles cached on CoinDCX, fall back to the next-closest thing
# rather than silently landing on noisy 1m data.
TRADE_BASIS_FALLBACK_ORDER = ["4h", "1h", "1d", "15m", "1m"]

# Risk:Reward multiples applied to ATR
STOP_ATR_MULT = 1.5
TARGET1_ATR_MULT = 1.5   # ~1:1 R:R
TARGET2_ATR_MULT = 3.0   # ~1:2 R:R


def project_price_range(current_price: float, atr: float, periods_ahead: int, z: float = 1.0):
    """
    Projects a statistical price range using ATR scaled by sqrt(time) - the
    standard random-walk volatility scaling law (same principle behind an
    options "expected move" calculation, just using historical ATR instead
    of implied volatility).

    HONESTY NOTE: this is NOT a directional prediction. It says "if recent
    volatility continues, price will PROBABLY stay within this envelope" -
    z=1 is roughly a 68% confidence band, z=2 roughly 95%, under a
    normal-distribution assumption. Real markets have fat tails, so big
    moves beyond this range happen more often than these odds suggest.
    """
    if pd.isna(atr) or atr <= 0 or periods_ahead <= 0:
        return None
    move = z * atr * math.sqrt(periods_ahead)
    return {"lower": current_price - move, "upper": current_price + move, "move": move}


def compute_price_projections(results: dict) -> dict:
    """Builds near-term (1h-based) and medium-term (1d-based) price range projections."""
    projections = {}
    h1 = results.get("1h")
    if h1 is not None and pd.notna(h1.get("atr")):
        price, atr = h1["price"], h1["atr"]
        projections["next_4h_68"] = project_price_range(price, atr, 4, z=1.0)
        projections["next_4h_95"] = project_price_range(price, atr, 4, z=2.0)
        projections["next_24h_68"] = project_price_range(price, atr, 24, z=1.0)
        projections["next_24h_95"] = project_price_range(price, atr, 24, z=2.0)
    d1 = results.get("1d")
    if d1 is not None and pd.notna(d1.get("atr")):
        price, atr = d1["price"], d1["atr"]
        projections["next_3d_68"] = project_price_range(price, atr, 3, z=1.0)
        projections["next_3d_95"] = project_price_range(price, atr, 3, z=2.0)
        projections["next_7d_68"] = project_price_range(price, atr, 7, z=1.0)
        projections["next_7d_95"] = project_price_range(price, atr, 7, z=2.0)
    return projections


def compute_pivot_points(prior_high: float, prior_low: float, prior_close: float) -> dict:
    """Classic floor-trader pivot points, computed from the prior completed period's H/L/C."""
    pp = (prior_high + prior_low + prior_close) / 3.0
    r1 = 2 * pp - prior_low
    s1 = 2 * pp - prior_high
    r2 = pp + (prior_high - prior_low)
    s2 = pp - (prior_high - prior_low)
    r3 = prior_high + 2 * (pp - prior_low)
    s3 = prior_low - 2 * (prior_high - pp)
    return {"pp": pp, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3}


def daily_pivot_points(df_1d: pd.DataFrame):
    """Uses the PRIOR completed daily bar (not the still-forming current one) - the standard approach."""
    if df_1d is None or len(df_1d) < 2:
        return None
    prior = df_1d.iloc[-2]
    return compute_pivot_points(prior["high"], prior["low"], prior["close"])


def classify_pivot_zone(price: float, pivots: dict) -> str:
    if price > pivots["r3"]:
        return "Above R3 (strongly overextended up)"
    elif price > pivots["r2"]:
        return "Between R2 and R3"
    elif price > pivots["r1"]:
        return "Between R1 and R2"
    elif price > pivots["pp"]:
        return "Between Pivot and R1"
    elif price > pivots["s1"]:
        return "Between S1 and Pivot"
    elif price > pivots["s2"]:
        return "Between S1 and S2"
    elif price > pivots["s3"]:
        return "Between S2 and S3"
    else:
        return "Below S3 (strongly overextended down)"


def print_price_outlook(results: dict, dfs: dict):
    print(f"\n{'-'*78}\n  PRICE OUTLOOK (statistical projection + pivot levels)\n{'-'*78}")
    print("  NOT a directional prediction - just how far price could reasonably move")
    print("  if recent volatility continues, and the classic reaction levels traders")
    print("  watch. Real markets have 'fat tails': big moves happen more often than")
    print("  a normal-distribution model like this suggests.")

    def fmt(v):
        return f"{v:.6f}" if v < 1 else f"{v:.4f}"

    projections = compute_price_projections(results)
    labels = [
        ("next_4h", "Next 4 hours  (1h volatility)"),
        ("next_24h", "Next 24 hours (1h volatility)"),
        ("next_3d", "Next 3 days   (daily volatility)"),
        ("next_7d", "Next 7 days   (daily volatility)"),
    ]
    any_shown = False
    for key, label in labels:
        p68, p95 = projections.get(f"{key}_68"), projections.get(f"{key}_95")
        if p68 and p95:
            any_shown = True
            print(f"\n  {label}:")
            print(f"    ~68% likely range: {fmt(p68['lower'])}  to  {fmt(p68['upper'])}")
            print(f"    ~95% likely range: {fmt(p95['lower'])}  to  {fmt(p95['upper'])}")
    if not any_shown:
        print("\n  Not enough data to project a price range this cycle.")

    df_1d = dfs.get("1d")
    pivots = daily_pivot_points(df_1d)
    if pivots and "1d" in results:
        price = results["1d"]["price"]
        zone = classify_pivot_zone(price, pivots)
        print(f"\n  Daily Pivot Levels (from prior day's H/L/C):")
        print(f"    R3: {fmt(pivots['r3'])}   R2: {fmt(pivots['r2'])}   R1: {fmt(pivots['r1'])}")
        print(f"    Pivot: {fmt(pivots['pp'])}")
        print(f"    S1: {fmt(pivots['s1'])}   S2: {fmt(pivots['s2'])}   S3: {fmt(pivots['s3'])}")
        print(f"    Current price zone: {zone}")
    else:
        print("\n  Not enough daily history to compute pivot levels this cycle.")


def suggest_trade_levels(results: dict, composite: dict) -> dict:
    """
    Produces a mechanical, ATR-based entry / stop-loss / target suggestion.
    This is a heuristic, not a backtested strategy - treat it as a
    starting point for your own risk management, not a guarantee.
    """
    basis, basis_tf = None, None
    for tf in TRADE_BASIS_FALLBACK_ORDER:
        candidate = results.get(tf)
        if candidate is not None and pd.notna(candidate.get("atr")):
            basis, basis_tf = candidate, tf
            break

    if basis is None:
        return {"available": False}

    entry = basis["price"]
    atr_val = basis["atr"]
    direction = composite["verdict"]

    if direction == "BULL":
        stop_loss = entry - STOP_ATR_MULT * atr_val
        if pd.notna(basis.get("bb_lower")) and basis["bb_lower"] < entry:
            stop_loss = min(stop_loss, basis["bb_lower"])
        side = "LONG"
    elif direction == "BEAR":
        stop_loss = entry + STOP_ATR_MULT * atr_val
        if pd.notna(basis.get("bb_upper")) and basis["bb_upper"] > entry:
            stop_loss = max(stop_loss, basis["bb_upper"])
        side = "SHORT"
    else:
        return {"available": False, "reason": "Composite verdict is NEUTRAL - no clean directional trade suggested."}

    risk = abs(entry - stop_loss)
    risk_atr_multiple = risk / atr_val if atr_val else float("nan")

    # Trade-quality gate (incorporated from external review, adapted): if the
    # band-adjusted stop lands absurdly far from entry relative to current
    # volatility, the setup is structurally poor - refuse it rather than
    # print levels with a collapsed reward:risk (the exact failure the
    # 0.31:1 BZ example exposed).
    if pd.notna(risk_atr_multiple) and risk_atr_multiple > 3.5:
        return {"available": False,
                "reason": (f"NO TRADE - the nearest safe stop is {risk_atr_multiple:.1f}x ATR away "
                            f"({risk:.4f} vs ATR {atr_val:.4f}). Risk is too wide relative to "
                            f"volatility for a sensible reward:risk; wait for a better structure.")}

    # Targets scale from the ACTUAL risk distance (1x and 2x), so reward:risk
    # stays ~1:1 / ~2:1 by construction even when the stop was widened to the
    # band edge. (Bug fix: the printed explanation always claimed this, but
    # the code previously used fixed ATR multiples regardless of the stop.)
    if side == "LONG":
        target1 = entry + risk
        target2 = entry + 2 * risk
    else:
        target1 = entry - risk
        target2 = entry - 2 * risk

    reward1 = abs(target1 - entry)
    rr1 = (reward1 / risk) if risk else float("nan")

    return {
        "available": True,
        "side": side,
        "basis_timeframe": basis_tf,
        "basis_synthetic": bool(basis.get("synthetic")),
        "entry": entry,
        "stop_loss": stop_loss,
        "target1": target1,
        "target2": target2,
        "risk_reward_t1": rr1,
        "risk_atr_multiple": risk_atr_multiple,
        "atr_used": atr_val,
    }


def print_trade_levels(trade: dict, symbol: str, liquidity: dict = None, choppiness: dict = None):
    print(f"\n{'-'*78}\n  SUGGESTED TRADE LEVELS (mechanical ATR-based heuristic - not advice)\n{'-'*78}")
    if liquidity and liquidity.get("available") and liquidity["verdict"] == "THIN":
        print(f"  ⚠ MARKET IS THIN right now (spread {liquidity['spread_pct']:.3f}%). Signals below"
              f" are especially unreliable in thin conditions - slippage could easily eat past"
              f" these levels. Consider waiting for liquidity to normalize before entering.")
    if choppiness and choppiness.get("available") and choppiness["verdict"] == "CHOPPY":
        print(f"  ⚠ MARKET IS CHOPPY right now (ADX {choppiness['adx']:.1f} on {choppiness['timeframe']}"
              f" chart, below 20 = ranging). These levels assume a directional move - in a ranging"
              f" market, expect more false signals and whipsaws. Consider waiting for a clearer trend.")
    if not trade.get("available"):
        print(f"  {trade.get('reason', 'Not enough data to suggest levels.')}")
        return

    def fmt(v):
        return f"{v:.6f}" if v < 1 else f"{v:.4f}"

    print(f"  Direction     : {trade['side']}  (based on {trade['basis_timeframe']} chart)")
    if trade.get("basis_synthetic"):
        print(f"  NOTE: CoinDCX's futures API has no native 4h resolution, so these 4h candles"
              f" were synthesized locally from 1h data instead. Should be close to the real"
              f" thing, but isn't exchange-native 4h data.")
    elif trade["basis_timeframe"] != "4h":
        print(f"  NOTE: 4h candle data wasn't available for this pair, so this fell back to"
              f" the {trade['basis_timeframe']} chart instead. Sizing may be tighter/looser than usual.")
    print(f"  Entry (~mkt)  : {fmt(trade['entry'])}")
    print(f"  Stop Loss     : {fmt(trade['stop_loss'])}")
    print(f"  Target 1      : {fmt(trade['target1'])}  (~{trade['risk_reward_t1']:.2f}:1 reward:risk)")
    print(f"  Target 2      : {fmt(trade['target2'])}  (~{(trade['risk_reward_t1']*2):.2f}:1 reward:risk)")
    print(f"  ATR used      : {trade['atr_used']:.6f}")
    print("\n  How these were built: Stop = entry +/- 1.5x ATR (or the nearest Bollinger")
    print("  Band edge if that's tighter). Target 1 = 1x that risk distance, Target 2 = 2x.")
    print("  This is a volatility-based heuristic, NOT a backtested strategy. Always size")
    print(f"  your position so the Stop Loss distance is an amount you can afford to lose.")
    print(f"  {symbol} can gap through any of these levels, especially in fast markets.")


# --------------------------------------------------------------------------
# News headlines (free, no API key - Google News RSS)
# --------------------------------------------------------------------------

def fetch_news_headlines(query: str, max_items: int = 5):
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    IST_OFFSET = timedelta(hours=5, minutes=30)  # India Standard Time, no DST

    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")[:max_items]
        headlines = []
        for item in items:
            title = item.findtext("title", default="").strip()
            pub_date_raw = item.findtext("pubDate", default="").strip()
            link = item.findtext("link", default="").strip()

            pub_date_fmt = pub_date_raw
            if pub_date_raw:
                try:
                    dt = parsedate_to_datetime(pub_date_raw)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    dt_ist = dt.astimezone(timezone.utc) + IST_OFFSET
                    pub_date_fmt = dt_ist.strftime("%d %b %Y, %H:%M IST")
                except Exception:
                    pass  # fall back to the raw RSS string if parsing fails

            if title:
                headlines.append({"title": title, "date": pub_date_fmt, "link": link})
        return headlines
    except Exception as e:
        return [{"title": f"[news fetch failed: {e}]", "date": "", "link": ""}]


# --------------------------------------------------------------------------
# Output formatting
# --------------------------------------------------------------------------

def print_table(symbol: str, results: dict, composite: dict):
    ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)

    print("\n" + "=" * 78)
    print(f"  TECHNICAL SIGNAL TABLE — {symbol}/USDT (live from CoinDCX)")
    print(f"  Generated: {ist_now.strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("=" * 78)

    header = (f"{'Timeframe':<10}{'Price':<14}{'RSI':<7}{'MACD H.':<10}"
               f"{'ATR':<10}{'ADX':<7}{'RVI':<7}{'Verdict':<9}")
    print(header)
    print("-" * 78)
    any_synthetic = False
    for tf_key, _, tf_label in TIMEFRAMES:
        if tf_key not in results:
            continue
        r = results[tf_key]
        label = tf_label + ("*" if r.get("synthetic") else "")
        any_synthetic = any_synthetic or r.get("synthetic", False)
        price_str = f"{r['price']:.6f}" if r["price"] < 1 else f"{r['price']:.4f}"
        rsi_str = f"{r['rsi']:.1f}" if pd.notna(r["rsi"]) else "N/A"
        macd_str = f"{r['macd_hist']:.5f}" if pd.notna(r["macd_hist"]) else "N/A"
        atr_str = f"{r['atr']:.5f}" if pd.notna(r.get("atr")) else "N/A"
        adx_str = f"{r['adx']:.1f}" if pd.notna(r.get("adx")) else "N/A"
        rvi_str = f"{r['rvi']:.2f}" if pd.notna(r.get("rvi")) else "N/A"
        print(f"{label:<10}{price_str:<14}{rsi_str:<7}{macd_str:<10}"
              f"{atr_str:<10}{adx_str:<7}{rvi_str:<7}{r['verdict']:<9}")
    if any_synthetic:
        print("* = synthesized from finer candles (CoinDCX's futures API has no native "
              "15m or 4h resolution - only 1m/5m/1h/1D)")

    print("-" * 78)
    print("\nWhy each timeframe scored the way it did:")
    for tf_key, _, tf_label in TIMEFRAMES:
        if tf_key not in results:
            continue
        r = results[tf_key]
        print(f"  [{tf_label}] {r['verdict']}: " + "; ".join(r["reasons"]))

    print("\n" + "=" * 78)
    print(f"  COMPOSITE VERDICT: {composite['verdict']}  "
          f"(weighted score: {composite['normalized_score']:+.2f}, range -1 to +1)")
    align = composite.get("alignment", {})
    if align.get("available"):
        print(f"  Timeframe Alignment: {align['bulls']} bull / {align['bears']} bear / "
              f"{align['neutral']} neutral out of {align['total']} "
              f"({align['alignment_pct']:.0f}% agree on {align['dominant_direction']})")
    print("=" * 78)


# --------------------------------------------------------------------------
# News sentiment scoring (simple keyword-based - no paid NLP/API needed)
# --------------------------------------------------------------------------

BULLISH_WORDS = [
    "rall",  # rally, rallies, rallying
    "surg",  # surge, surges, surging
    "soar",  # soar, soars, soaring
    "breakout", "bullish", "adopt", "partnership",
    "upgrad",  # upgrade, upgrades, upgraded
    "record high", "all-time high", "ath", "buy signal",
    "accumulat",  # accumulate, accumulation
    "institutional", "inflow", "listing", "integrat",  # integrate, integration
    "approv",  # approval, approved
    "launch", "gain", "jump",
    "recover",  # recover, recovery, recovered
    "rebound", "green", "outperform", "upbeat",
    "optimist",  # optimistic, optimism
    "boost", "mileston",  # milestone, milestones
]

BEARISH_WORDS = [
    "crash", "plung", "dump", "sell-off", "selloff", "bearish",
    "hack", "exploit", "lawsuit", "sues", "sued", "ban", "crackdown",
    "delist", "fraud", "scam",
    "collaps",  # collapse, collapses, collapsing
    "declin",  # decline, declines, declining
    "liquidat",  # liquidation, liquidated, liquidating
    "warn",  # warn, warns, warning, warned
    "risk", "outflow",
    "downgrad",  # downgrade, downgraded
    "underperform", "fear", "sell signal", "correction", "slump",
    "tumbl",  # tumble, tumbles, tumbling
    "drop",
]


def score_headline(title: str) -> int:
    """Simple net keyword sentiment: bullish hits minus bearish hits, capped."""
    t = title.lower()
    bull_hits = sum(1 for w in BULLISH_WORDS if w in t)
    bear_hits = sum(1 for w in BEARISH_WORDS if w in t)
    net = bull_hits - bear_hits
    return max(min(net, 3), -3)  # cap so one spammy headline can't dominate


# NOTE: I tested swapping this for a general-purpose NLP sentiment library
# (VADER) and it misread finance jargon - e.g. it rated "Bitcoin crashes
# below key support amid sell-off" as BULLISH, because it reads "support"
# as a warm/supportive word rather than a technical price level. A generic
# sentiment model isn't safe for this domain, so instead the improvement
# here is negation-awareness layered on top of the finance-tuned lexicon:
# "no crash expected" or "not bullish" now flip the polarity correctly,
# which plain keyword-counting could not do before.

NEGATORS = ["not ", "no ", "n't", "never ", "without ", "isn't", "wasn't",
            "doesn't", "won't", "don't", "didn't", "hasn't", "unlikely to"]


def _is_negated(text: str, match_index: int, window: int = 18) -> bool:
    """Checks a short window of text just before a keyword match for a negator."""
    preceding = text[max(0, match_index - window):match_index]
    return any(neg in preceding for neg in NEGATORS)


import re as _re

# Special-case patterns for short/generic stems that would otherwise collide
# with unrelated whole words as a prefix. "ban" needs to still match
# "bans"/"banned"/"banning" but NOT "bank"/"banking" - a negative lookahead
# handles that precisely, better than a blunt full-word-only match would.
SPECIAL_PATTERNS = {
    "ban": r"\bban(?!k)",
    "fear": r"\bfear(?!less)",
}


def _keyword_pattern(w: str) -> str:
    if w in SPECIAL_PATTERNS:
        return SPECIAL_PATTERNS[w]
    return r"\b" + _re.escape(w)


def get_headline_score(title: str) -> float:
    """
    Negation-aware keyword sentiment score in [-1, 1].
    Finds each bullish/bearish keyword hit, checks whether it's negated by a
    nearby word like "not"/"no"/"never", and flips its polarity if so.

    Matching requires a word boundary BEFORE each keyword (so "gain" matches
    "gains"/"gained" but not the middle of "again") while still allowing the
    stem to match any suffix (so "gain" alone still catches "gaining").
    A small set of short/generic stems (e.g. "ban") require a FULL word
    match instead, since as a prefix they collide with unrelated words like
    "banking".
    """
    t = title.lower()
    bull_score = 0
    bear_score = 0

    for w in BULLISH_WORDS:
        m = _re.search(_keyword_pattern(w), t)
        if m:
            if _is_negated(t, m.start()):
                bear_score += 1
            else:
                bull_score += 1

    for w in BEARISH_WORDS:
        m = _re.search(_keyword_pattern(w), t)
        if m:
            if _is_negated(t, m.start()):
                bull_score += 1
            else:
                bear_score += 1

    net = max(min(bull_score - bear_score, 3), -3)
    return net / 3.0


def sentiment_label(score) -> str:
    if score > 0.05:
        return "BULLISH"
    elif score < -0.05:
        return "BEARISH"
    return "NEUTRAL"


# Category weights for the combined news score (sum to 1.0).
# Coin-specific news matters most; Trump/Musk/macro news matter, but less
# directly, since they move the whole market rather than this coin alone.
NEWS_CATEGORY_WEIGHTS = {
    "coin": 0.50,
    "trump": 0.20,
    "musk": 0.20,
    "market": 0.10,
}


def analyze_news_sentiment(symbol: str, max_items: int = 4) -> dict:
    """
    Fetches headlines per category, scores each with the keyword lexicon,
    and produces a weighted overall news sentiment score in [-1, 1].
    """
    if symbol in COMMODITY_FUTURES_SYMBOLS:
        coin_query = f"{symbol} price"
        market_query = "commodity market today"
    else:
        coin_query = f"{symbol} crypto"
        market_query = "crypto market today"

    queries = {
        "coin": coin_query,
        "trump": "Trump crypto",
        "musk": "Elon Musk crypto",
        "market": market_query,
    }

    category_data = {}
    weighted_total = 0.0

    for cat, q in queries.items():
        headlines = fetch_news_headlines(q, max_items=max_items)
        scored = []
        for h in headlines:
            s = get_headline_score(h["title"])
            scored.append({**h, "score": s, "label": sentiment_label(s)})

        valid_scores = [h["score"] for h in scored if not h["title"].startswith("[news fetch failed")]
        avg_score = (sum(valid_scores) / len(valid_scores)) if valid_scores else 0.0
        normalized = avg_score  # already in [-1, 1] from get_headline_score

        category_data[cat] = {"headlines": scored, "avg_score": avg_score, "normalized": normalized}
        weighted_total += normalized * NEWS_CATEGORY_WEIGHTS.get(cat, 0)

    if weighted_total > 0.1:
        overall_verdict = "BULLISH"
    elif weighted_total < -0.1:
        overall_verdict = "BEARISH"
    else:
        overall_verdict = "NEUTRAL"

    return {
        "categories": category_data,
        "overall_score": weighted_total,
        "overall_verdict": overall_verdict,
    }


def combined_verdict(technical_composite: dict, news: dict = None, strategy: dict = None) -> dict:
    """
    Blends technical, strategy-signal, and news sentiment scores.
    - If both news and strategy are available: Technical 40% / Strategy 30% / News 30%
    - If only news: Technical 65% / News 35% (original behavior)
    - If only strategy: Technical 55% / Strategy 45%
    - If neither: Technical 100%
    """
    tech_score = technical_composite["normalized_score"]
    news_score = news["overall_score"] if news else 0.0
    strat_score = strategy["strategy_score"] if strategy else 0.0

    if news and strategy:
        blended = 0.40 * tech_score + 0.30 * strat_score + 0.30 * news_score
    elif news:
        blended = 0.65 * tech_score + 0.35 * news_score
    elif strategy:
        blended = 0.55 * tech_score + 0.45 * strat_score
    else:
        blended = tech_score

    if blended > 0.15:
        verdict = "BULL"
    elif blended < -0.15:
        verdict = "BEAR"
    else:
        verdict = "NEUTRAL"

    return {"verdict": verdict, "blended_score": blended,
            "tech_score": tech_score, "news_score": news_score, "strat_score": strat_score}



def print_news(symbol: str, news: dict):
    print(f"\n{'-'*78}\n  RECENT HEADLINES + SENTIMENT (free Google News RSS, no key needed)\n{'-'*78}")

    if symbol in COMMODITY_FUTURES_SYMBOLS:
        labels = {
            "coin": f"{symbol} price",
            "trump": "Trump crypto",
            "musk": "Elon Musk crypto",
            "market": "commodity market today",
        }
    else:
        labels = {
            "coin": f"{symbol} crypto",
            "trump": "Trump crypto",
            "musk": "Elon Musk crypto",
            "market": "crypto market today",
        }

    for cat, cat_data in news["categories"].items():
        weight_pct = int(NEWS_CATEGORY_WEIGHTS.get(cat, 0) * 100)
        print(f"\n  >> {labels.get(cat, cat)}  (weight in news score: {weight_pct}%)")
        for h in cat_data["headlines"]:
            date_str = f" [{h['date']}]" if h.get("date") else ""
            print(f"     - ({h['label']:<8}) {h['title']}{date_str}")
        print(f"     Category sentiment: {sentiment_label(cat_data['avg_score'])} "
              f"(avg score {cat_data['avg_score']:+.2f})")

    print(f"\n  Overall NEWS sentiment: {news['overall_verdict']}  "
          f"(weighted score: {news['overall_score']:+.2f}, range -1 to +1)")
    print("  [keyword-based lexicon - a rough read of public mood, not true NLP sentiment analysis]")


def compute_trade_confidence(snap: dict, trade: dict) -> dict:
    """
    A 0-100 CONFIDENCE rating for the current combined verdict, built from
    independent conditions. Incorporates the reviewed presentation idea,
    with one honest correction: this measures how strongly the evidence
    AGREES right now - it is NOT a probability of profit, and it is never
    labeled "accuracy". Even maximum-confidence setups lose regularly.
    """
    combo = snap["combo"]
    if combo["verdict"] == "NEUTRAL":
        return {"score": 0, "tier": "NO-TRADE (verdict is NEUTRAL)", "direction": "NEUTRAL",
                "components": [("Verdict is NEUTRAL", 0, 100)],
                "neutral_note": ("Confidence is 0 BY DEFINITION when the verdict is NEUTRAL - "
                                  "there is no directional call to be confident in. This means "
                                  "'stand aside', NOT 'nothing will happen'.")}

    direction = combo["verdict"]
    components = []

    # 1. Blended conviction strength (0-30)
    conviction = min(abs(combo["blended_score"]) / 0.6, 1.0) * 30
    components.append(("Blended conviction", round(conviction), 30))

    # 2. Timeframe alignment (0-25)
    align = snap.get("composite", {}).get("alignment", {})
    align_pts = 0
    if align.get("available") and align.get("dominant_direction") == direction:
        align_pts = (align["alignment_pct"] / 100) * 25
    components.append(("Timeframe alignment", round(align_pts), 25))

    # 3. Technical & strategy agreement (0-20)
    strat = snap.get("strategy_signals")
    agree_pts = 0
    if strat:
        if strat.get("verdict") == direction:
            agree_pts = 20
        elif strat.get("verdict") == "NEUTRAL":
            agree_pts = 8
    components.append(("Tech/strategy agreement", round(agree_pts), 20))

    # 4. Market conditions (0-15): full marks only if liquid AND trending
    cond_pts = 15
    liq = snap.get("liquidity", {})
    chop = snap.get("choppiness", {})
    if liq.get("available") and liq["verdict"] == "THIN":
        cond_pts -= 8
    elif liq.get("available") and liq["verdict"] == "CAUTION":
        cond_pts -= 4
    if chop.get("available") and chop["verdict"] == "CHOPPY":
        cond_pts -= 7
    elif chop.get("available") and chop["verdict"] == "MODERATE":
        cond_pts -= 3
    cond_pts = max(cond_pts, 0)
    components.append(("Market conditions", cond_pts, 15))

    # 5. Reward:risk quality (0-10)
    rr_pts = 0
    if trade.get("available"):
        ram = trade.get("risk_atr_multiple")
        if pd.notna(ram):
            # Full marks for a clean ~1.5x-ATR stop, scaling to 0 as the stop
            # widens toward the 3.5x-ATR rejection threshold. (Targets are now
            # risk-scaled, so R:R itself is constant by construction - stop
            # tightness is the honest quality measure left.)
            rr_pts = max(0.0, min(1.0, (3.5 - ram) / (3.5 - 1.5))) * 10
    components.append(("Stop quality", round(rr_pts), 10))

    score = round(sum(p for _, p, _ in components))
    if score >= 70:
        tier = "HIGH CONFIDENCE"
    elif score >= 50:
        tier = "MODERATE"
    elif score >= 30:
        tier = "LOW - consider skipping"
    else:
        tier = "NO-TRADE - conditions too weak"

    # A strong directional read with NO valid entry structure is "right about
    # direction, wrong time to enter" - say so, or "HIGH CONFIDENCE" printed
    # under a NO-TRADE gate reads like a contradiction (seen live on BZ).
    if not trade.get("available") and score >= 50:
        tier += " in DIRECTION - but NO TRADE (no sensible entry structure right now)"

    return {"score": score, "tier": tier, "direction": direction, "components": components}


def print_trade_confidence(conf: dict):
    print(f"\n{'-'*78}\n  TRADE CONFIDENCE: {conf['score']}/100 - {conf['tier']}"
          f"{'' if conf['direction']=='NEUTRAL' else ' (' + conf['direction'] + ')'}\n{'-'*78}")
    for name, pts, max_pts in conf["components"]:
        bar = "#" * int(pts / max_pts * 10) if max_pts else ""
        print(f"    {name:<26} {pts:>3}/{max_pts:<4} {bar}")
    if conf.get("neutral_note"):
        print(f"  {conf['neutral_note']}")
    print("  [Confidence = how strongly the evidence currently agrees. It is NOT a")
    print("   probability of profit - even maximum-confidence setups lose regularly.]")


def print_combined_verdict(combined: dict):
    print("\n" + "=" * 78)
    print(f"  FINAL COMBINED VERDICT (Technical + Strategy Signals + News Sentiment)")
    print("=" * 78)
    print(f"  Technical score : {combined['tech_score']:+.2f}")
    print(f"  Strategy score  : {combined['strat_score']:+.2f}")
    print(f"  News score      : {combined['news_score']:+.2f}")
    print(f"  Blended score   : {combined['blended_score']:+.2f}")
    print(f"  >>> {combined['verdict']} <<<")
    print("=" * 78)


# --------------------------------------------------------------------------
# Alarm system - tries an Android notification, then an audible beep, then
# always falls back to a big printed banner + terminal bell (which always
# works, even with no extra packages installed).
# --------------------------------------------------------------------------

def _generate_beep_wav(path: str, freq: int = 1000, duration_ms: int = 700, volume: float = 0.6):
    """Synthesizes a plain sine-wave beep tone as a WAV file - no external audio file needed."""
    import wave
    import struct
    import math

    framerate = 44100
    n_frames = int(framerate * duration_ms / 1000)
    with wave.open(path, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(framerate)
        frames = bytearray()
        for i in range(n_frames):
            value = int(volume * 32767 * math.sin(2 * math.pi * freq * i / framerate))
            frames += struct.pack("<h", value)
        wav_file.writeframesraw(bytes(frames))


def _play_beep():
    import tempfile
    import os
    import subprocess

    tmp_path = os.path.join(tempfile.gettempdir(), "crypto_bot_alarm.wav")
    if not os.path.exists(tmp_path):
        _generate_beep_wav(tmp_path)

    try:
        import playsound
        playsound.playsound(tmp_path)
        return
    except Exception:
        pass

    # Termux fallback (requires the separate Termux:API app + termux-api package)
    try:
        subprocess.run(["termux-media-player", "play", tmp_path], timeout=5,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def trigger_alarm(message: str, severity: str = "WARNING"):
    """
    Tries, in order: an Android system notification (via plyer, works well in
    Pydroid 3), an audible beep (via playsound, if installed), and always
    prints a big banner + terminal bell as a guaranteed fallback.
    """
    icon = "[!!!]" if severity == "DANGER" else "[! ]"
    print("\n" + "!" * 78)
    print(f"{icon} {severity} ALERT {icon}")
    print(message)
    print("!" * 78 + "\n")

    try:
        from plyer import notification as plyer_notification
        plyer_notification.notify(title=f"Crypto Alert: {severity}", message=message, timeout=15)
    except Exception:
        pass

    try:
        _play_beep()
    except Exception:
        pass

    for _ in range(3):
        print("\a", end="", flush=True)
        time.sleep(0.3)


# --------------------------------------------------------------------------
# Position-aware continuous monitoring
# --------------------------------------------------------------------------

_BTC_REGIME_CACHE = {"data": None, "ts": 0.0}


def get_btc_market_regime(args, cache_seconds: int = 600) -> dict:
    """
    Fetches a LIGHTWEIGHT BTC market regime (just 4h+1D HTF bias, not the
    full 5-timeframe analysis) to use as market-wide context. Altcoins tend
    to underperform in a BTC bear regime and outperform in a BTC bull
    regime, so checking BTC's own trend can improve precision on every
    other symbol's call, not just add a feature for its own sake.

    Cached for a few minutes: BTC's 4h/1D bias doesn't change second to
    second, and during an all-pairs screener scan this would otherwise be
    re-fetched once per symbol (hundreds of identical requests).
    """
    now = time.time()
    if _BTC_REGIME_CACHE["data"] is not None and now - _BTC_REGIME_CACHE["ts"] < cache_seconds:
        return _BTC_REGIME_CACHE["data"]
    try:
        btc_pair = resolve_pair("BTC", args.quote)
        results = {}
        for tf_key in ("4h", "1d"):
            df = fetch_timeframe_candles(btc_pair, tf_key, limit=args.limit)
            df = compute_indicators(df)
            results[tf_key] = classify_timeframe(df)
        regime = compute_htf_bias(results)
        out = {"available": True, "regime": regime}
        _BTC_REGIME_CACHE["data"] = out
        _BTC_REGIME_CACHE["ts"] = now
        return out
    except Exception as e:
        return {"available": False, "error": str(e)}


def run_analysis_cycle(symbol: str, pair: str, args, include_news: bool = True):
    """
    Runs one full fetch -> indicators -> strategy signals -> news -> combined
    verdict cycle. Shared by both the initial one-shot report and every loop
    of the continuous monitor, so the two can never drift out of sync.

    Uses CoinDCX's FUTURES candle data for every symbol (not spot) since
    that's what actually determines margin/leverage/liquidation for a real
    futures position - and it's the only data source that exists at all for
    commodity pairs (XAU, XAG, CL, BZ, NATGAS), which have no spot market.
    """
    def _fetch_and_process(tf_key, tf_label):
        """Runs in a worker thread: fetch + compute indicators + classify for one timeframe."""
        df = fetch_timeframe_candles(pair, tf_key, limit=args.limit)
        df = compute_indicators(df)
        result = classify_timeframe(df)
        if tf_key in ("15m", "4h"):
            result["synthetic"] = True
        return tf_key, df, result

    # Kick off the BTC market-regime check concurrently with the main fetch
    # loop below, so it adds minimal extra latency (skipped entirely when
    # the symbol being analyzed IS BTC - checking BTC against itself is
    # redundant).
    btc_future = None
    btc_executor = None
    if symbol.upper() not in ("BTC", "BTC_USDT"):
        btc_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        btc_future = btc_executor.submit(get_btc_market_regime, args)

    results, dfs = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(TIMEFRAMES)) as executor:
        future_to_tf = {
            executor.submit(_fetch_and_process, tf_key, tf_label): (tf_key, tf_label)
            for tf_key, _, tf_label in TIMEFRAMES
        }
        for future in concurrent.futures.as_completed(future_to_tf):
            tf_key, tf_label = future_to_tf[future]
            try:
                _, df, result = future.result()
                results[tf_key] = result
                dfs[tf_key] = df
            except Exception as e:
                print(f"[warn] Failed to fetch/compute {tf_label}: {e}")

    btc_regime = {"available": False}
    if btc_future is not None:
        try:
            btc_regime = btc_future.result()
        except Exception as e:
            btc_regime = {"available": False, "error": str(e)}
        btc_executor.shutdown(wait=False)

    if not results:
        return None

    orderbook_imbalance = None
    liquidity = {"available": False}
    if not args.no_strategy:
        try:
            ob = fetch_orderbook(pair)
            orderbook_imbalance = compute_orderbook_imbalance(ob)
            liquidity = compute_liquidity_metrics(ob)
        except Exception as e:
            print(f"[warn] Orderbook fetch failed: {e}")

    composite = composite_verdict(results)
    choppiness = assess_choppiness(results)
    strategy_signals = None if args.no_strategy else compute_strategy_signals(
        dfs, results, orderbook_imbalance, btc_regime=btc_regime)
    if strategy_signals is not None:
        strategy_signals["liquidity"] = liquidity
        strategy_signals["choppiness"] = choppiness
    news = analyze_news_sentiment(symbol) if (include_news and not args.no_news) else None
    combo = combined_verdict(composite, news=news, strategy=strategy_signals)

    return {"results": results, "dfs": dfs, "composite": composite,
            "strategy_signals": strategy_signals, "news": news, "combo": combo,
            "liquidity": liquidity, "choppiness": choppiness}


def fetch_usdt_inr_rate():
    """
    Fetches CoinDCX's own live USDT/INR rate (their most-traded pair) so
    margin can be entered in INR and converted automatically. Using
    CoinDCX's own rate (rather than an external forex source) keeps it
    consistent with what the exchange itself is pricing USDT at. Returns
    None if the ticker can't be fetched, so callers can fall back gracefully.
    """
    try:
        resp = requests.get(f"{API_BASE}/exchange/ticker", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for item in data:
            if item.get("market", "").upper() == "USDTINR":
                return float(item["last_price"])
    except Exception:
        pass
    return None


def _ask_float(prompt: str, allow_blank: bool = False):
    """Repeatedly prompts until a valid float is entered (or blank, if allowed)."""
    while True:
        raw = input(prompt).strip()
        if allow_blank and raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a plain number (e.g. 0.0768).")


def ask_position():
    """
    Asks whether the user already holds a futures position, and if so,
    collects everything needed to track real P&L and protect against loss:
    side, entry price, margin, leverage, and optional target/stop-loss.
    Margin can be entered in USDT or INR (auto-converted via CoinDCX's own
    live USDT/INR rate).
    """
    ans = input("\nHave you already entered a crypto futures position on this coin? (yes/no): ").strip().lower()
    if ans not in ("y", "yes"):
        return None

    side = input("Are you LONG or SHORT? ").strip().upper()
    while side not in ("LONG", "SHORT"):
        side = input("Please type LONG or SHORT: ").strip().upper()

    entry_price = _ask_float("Entry price: ")

    currency = input("Enter margin in USDT or INR? (usdt/inr): ").strip().lower()
    while currency not in ("usdt", "inr"):
        currency = input("Please type 'usdt' or 'inr': ").strip().lower()

    inr_rate = None
    if currency == "inr":
        margin_inr = _ask_float("Margin used (in INR): ")
        print("  Fetching CoinDCX's live USDT/INR rate...")
        inr_rate = fetch_usdt_inr_rate()
        if inr_rate:
            margin = margin_inr / inr_rate
            print(f"  Live rate: 1 USDT = ~{inr_rate:.2f} INR -> margin = {margin:.4f} USDT")
            print("  [Note: this is CoinDCX's live spot USDT/INR rate right now - it may differ"
                  " slightly from whatever rate your futures wallet used at the time, so treat"
                  " the converted figure as very close, not necessarily bit-for-bit identical.]")
        else:
            print("  [warn] Couldn't fetch a live rate. Please enter margin in USDT instead.")
            margin = _ask_float("Margin used (in USDT): ")
    else:
        margin = _ask_float("Margin used (in USDT): ")

    leverage = _ask_float("Leverage (e.g. 10 for 10x): ")
    target_price = _ask_float("Target price (press Enter to skip): ", allow_blank=True)
    stop_loss = _ask_float("Stop-loss price (press Enter to skip): ", allow_blank=True)

    position = {
        "side": side, "entry_price": entry_price, "margin": margin,
        "leverage": leverage, "target_price": target_price, "stop_loss": stop_loss,
        "inr_rate": inr_rate,
    }

    notional = margin * leverage
    quantity = notional / entry_price if entry_price else 0
    print(f"\n  Position summary: {side} | Notional {notional:.2f} USDT | Quantity ~{quantity:.4f}")
    print("  [Note: liquidation price shown during monitoring is an ESTIMATE only -"
          " it ignores maintenance margin rate and fees, which vary by exchange."
          " Check CoinDCX's actual liquidation price for your position.]")

    return position


def compute_position_metrics(position: dict, current_price: float) -> dict:
    """
    Computes live unrealized P&L and a rough liquidation price estimate.
    The liquidation estimate is SIMPLIFIED (ignores maintenance margin rate
    and fees) - treat it as a rough guide, not the exact exchange figure.
    """
    entry = position["entry_price"]
    margin = position["margin"]
    leverage = position["leverage"]
    side = position["side"]

    notional = margin * leverage
    quantity = notional / entry if entry else 0

    if side == "LONG":
        pnl = (current_price - entry) * quantity
        liq_estimate = entry * (1 - 1 / leverage) if leverage else None
    else:
        pnl = (entry - current_price) * quantity
        liq_estimate = entry * (1 + 1 / leverage) if leverage else None

    pnl_pct_margin = (pnl / margin * 100) if margin else None

    return {"notional": notional, "quantity": quantity, "pnl": pnl,
            "pnl_pct_margin": pnl_pct_margin, "liq_estimate": liq_estimate}


def check_exit_conditions(position: dict, current_price: float, metrics: dict,
                           combo_verdict: str, soft_stop: float = None, liquidity: dict = None) -> list:
    """
    Checks every reason you might want to exit right now, independent of
    each other, so a manual target/stop, a technical verdict flip, and a
    raw P&L danger zone are ALL covered - not just whichever one you set.
    Returns a list of (severity, message) tuples, most important first.
    """
    side = position["side"]
    reasons = []

    if position.get("stop_loss") is not None:
        sl = position["stop_loss"]
        if (side == "LONG" and current_price <= sl) or (side == "SHORT" and current_price >= sl):
            reasons.append(("DANGER", f"Price ({current_price:.6f}) hit your stop-loss ({sl:.6f}). Exit now."))

    if position.get("target_price") is not None:
        tp = position["target_price"]
        if (side == "LONG" and current_price >= tp) or (side == "SHORT" and current_price <= tp):
            reasons.append(("TARGET", f"Price ({current_price:.6f}) reached your target ({tp:.6f})."))

    if position.get("stop_loss") is None and soft_stop is not None:
        if (side == "LONG" and current_price <= soft_stop) or (side == "SHORT" and current_price >= soft_stop):
            reasons.append(("DANGER", f"Price broke a computed safety level ({soft_stop:.6f}) - "
                                       f"you didn't set a manual stop-loss, so this is an ATR-based estimate."))

    risk = determine_position_risk(side, combo_verdict)
    if risk == "DANGER":
        reasons.append(("DANGER", f"Combined verdict flipped to {combo_verdict}, against your {side} position."))
    elif risk == "CAUTION":
        reasons.append(("CAUTION", "Momentum weakening (verdict now NEUTRAL) - not a full reversal yet."))

    if metrics.get("pnl_pct_margin") is not None:
        pct = metrics["pnl_pct_margin"]
        if pct <= -50:
            reasons.append(("DANGER", f"Unrealized loss is {pct:.1f}% of your margin - approaching liquidation risk."))
        elif pct <= -25:
            reasons.append(("CAUTION", f"Unrealized loss is {pct:.1f}% of your margin."))

    if liquidity and liquidity.get("available"):
        if liquidity["verdict"] == "THIN":
            reasons.append(("CAUTION", f"Market just went THIN (spread {liquidity['spread_pct']:.3f}%) - "
                                        f"if you need to exit, expect slippage beyond your stop/target."))

    return reasons


def determine_position_risk(position_side: str, combo_verdict: str) -> str:
    """
    Maps the combined verdict onto a risk level for the held position:
    - Verdict flips fully against you      -> DANGER
    - Verdict weakens to NEUTRAL           -> CAUTION (early warning)
    - Verdict still agrees with your side  -> SAFE
    """
    opposite = "BEAR" if position_side == "LONG" else "BULL"
    if combo_verdict == opposite:
        return "DANGER"
    elif combo_verdict == "NEUTRAL":
        return "CAUTION"
    return "SAFE"


def get_current_price(results: dict):
    """Returns the most granular available price as a live-price proxy."""
    for tf in ("1m", "15m", "1h", "4h", "1d"):
        if tf in results:
            return results[tf]["price"]
    return None


def format_timeframe_verdicts(results: dict) -> str:
    """One line showing each individual timeframe's own verdict - not a rolled-up average."""
    parts = []
    for tf_key, _, tf_label in TIMEFRAMES:
        if tf_key in results:
            parts.append(f"{tf_label}={results[tf_key]['verdict']}")
    return "  ".join(parts) if parts else "no data"


def format_strategy_verdicts(signals: dict) -> str:
    """One line showing each individual strategy signal's own reading - not a rolled-up score."""
    if not signals:
        return "N/A (strategy signals disabled)"

    parts = [f"HTF={signals.get('htf_bias', 'N/A')}"]

    ms = signals.get("market_structure", {})
    parts.append(f"MSS={ms.get('mss') or 'none'}({ms.get('trend', 'N/A')})")

    parts.append(f"FVG={signals.get('fvg_15m') or 'none'}")
    parts.append(f"Trap={signals.get('liquidity_trap_1h') or 'none'}")

    fib = signals.get("fib_pullback")
    parts.append(f"Fib={fib['direction'] if fib else 'none'}")

    div = signals.get("divergence_1h")
    parts.append(f"Div={div or 'none'}")

    sq = signals.get("bollinger_squeeze")
    parts.append(f"Squeeze={'YES' if sq and sq.get('is_squeeze') else 'no'}")

    ob = signals.get("orderbook_imbalance")
    parts.append(f"OrderBook={ob:+.2f}" if ob is not None else "OrderBook=N/A")

    btc = signals.get("btc_regime")
    parts.append(f"BTCRegime={btc.get('regime') if btc and btc.get('available') else 'N/A'}")

    sb = signals.get("silver_bullet", {})
    sb_val = sb.get("signal") or ("active-no-signal" if sb.get("active") else "off")
    parts.append(f"SilverBullet={sb_val}")

    _ms_label = signals.get("market_state") or signals.get("verdict", "N/A")
    parts.append(f"=> STRATEGY VERDICT={_ms_label}")
    return "  ".join(parts)


def monitor_position(symbol: str, pair: str, args, position: dict,
                      poll_seconds: int = 300, news_every_n_cycles: int = 6):
    """
    Repeats the full analysis on a timer and alarms the user on ANY exit
    condition: stop-loss hit, target hit, verdict flip against the position,
    or unrealized loss crossing a danger threshold - even if no stop-loss
    was set. Runs until Ctrl+C.
    """
    side = position["side"]
    print(f"\n{'='*78}")
    print(f"  MONITORING STARTED - {symbol} - Position: {side}")
    print(f"  Entry: {position['entry_price']}  Margin: {position['margin']} USDT  "
          f"Leverage: {position['leverage']}x")
    print(f"  Target: {position['target_price'] or 'not set'}   "
          f"Stop-loss: {position['stop_loss'] or 'not set (using computed ATR safety level)'}")
    print(f"  Checking every {poll_seconds} seconds. Press Ctrl+C to stop.")
    print(f"{'='*78}")

    cycle = 0
    last_alert_key = None
    try:
        while True:
            cycle += 1
            include_news = (cycle % news_every_n_cycles == 1)
            snap = run_analysis_cycle(symbol, pair, args, include_news=include_news)
            ist_now = (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%H:%M:%S IST")

            if snap is None:
                print(f"[{ist_now}] Cycle {cycle}: data fetch failed, will retry next interval.")
                time.sleep(poll_seconds)
                continue

            combo = snap["combo"]
            current_price = get_current_price(snap["results"])
            metrics = compute_position_metrics(position, current_price)

            soft_stop = None
            if position.get("stop_loss") is None:
                soft_trade = suggest_trade_levels(snap["results"], {"verdict": "BULL" if side == "LONG" else "BEAR"})
                if soft_trade.get("available"):
                    soft_stop = soft_trade["stop_loss"]

            pnl_str = f"{metrics['pnl']:+.2f} USDT ({metrics['pnl_pct_margin']:+.1f}% of margin)"
            if position.get("inr_rate"):
                pnl_str += f" | ~₹{metrics['pnl'] * position['inr_rate']:+.2f}"

            print(f"\n[{ist_now}] Cycle {cycle} | price={current_price:.6f} | P&L: {pnl_str} | "
                  f"liq est: {metrics['liq_estimate']:.6f}")
            print(f"  Timeframes : {format_timeframe_verdicts(snap['results'])}")
            print(f"  Strategy   : {format_strategy_verdicts(snap.get('strategy_signals'))}")
            print(f"  Combined   : {combo['verdict']} (score {combo['blended_score']:+.2f})")
            _conf_trade = suggest_trade_levels(snap["results"], {"verdict": combo["verdict"]})
            _conf = compute_trade_confidence(snap, _conf_trade)
            print(f"  Confidence : {_conf['score']}/100 - {_conf['tier']}")

            reasons = check_exit_conditions(position, current_price, metrics, combo["verdict"],
                                             soft_stop, liquidity=snap.get("liquidity"))
            if reasons:
                worst_severity = "DANGER" if any(r[0] == "DANGER" for r in reasons) else \
                                  "TARGET" if any(r[0] == "TARGET" for r in reasons) else "CAUTION"
                alert_key = (worst_severity, tuple(r[1] for r in reasons))

                should_alert = (alert_key != last_alert_key) or \
                               (worst_severity == "DANGER" and cycle % 3 == 0)
                if should_alert:
                    combined_msg = " | ".join(r[1] for r in reasons)
                    trigger_alarm(f"{symbol} {side}: {combined_msg}", severity=worst_severity)
                last_alert_key = alert_key
            else:
                last_alert_key = None

            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\nMonitoring stopped. Stay safe out there.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def recommend_style(results: dict) -> dict:
    """
    Suggests a trading style from CURRENT market conditions for this symbol:
    - Strong, aligned 4h/1d trend           -> TREND (ride it on higher timeframes)
    - Quiet higher TFs but active 15m/1h    -> SCALP (the movement is short-term)
    - Nothing clearly moving                -> BALANCED (and trade cautiously)
    A suggestion, not an order - the trader's own plan always wins.
    """
    def _adx(tf):
        r = results.get(tf)
        return r["adx"] if r is not None and pd.notna(r.get("adx")) else None
    def _verdict(tf):
        r = results.get(tf)
        return r["verdict"] if r is not None else None

    htf_adx = [a for a in (_adx("4h"), _adx("1d")) if a is not None]
    ltf_adx = [a for a in (_adx("15m"), _adx("1h")) if a is not None]
    htf_avg = sum(htf_adx) / len(htf_adx) if htf_adx else 0
    ltf_avg = sum(ltf_adx) / len(ltf_adx) if ltf_adx else 0
    htf_aligned = (_verdict("4h") == _verdict("1d")) and _verdict("4h") in ("BULL", "BEAR")
    ltf_active = any(_verdict(tf) in ("BULL", "BEAR") for tf in ("15m", "1h"))

    if htf_avg >= 25 and htf_aligned:
        return {"style": "trend",
                "reason": f"4h & 1d agree ({_verdict('4h')}) with strong trend (avg ADX {htf_avg:.0f})"}
    if ltf_avg >= 25 and ltf_active:
        return {"style": "scalp",
                "reason": f"higher TFs unclear, but 15m/1h are moving (avg ADX {ltf_avg:.0f})"}
    return {"style": "balanced",
            "reason": f"no clearly dominant trend on any horizon (HTF ADX {htf_avg:.0f}, LTF ADX {ltf_avg:.0f})"}


def recompute_snapshot_for_style(snap: dict, args) -> dict:
    """
    Re-scores an already-fetched snapshot under the newly active style.
    Weights, trade basis, and choppiness basis are style-dependent, but the
    underlying candles/indicators/strategy detections are not - so this
    reuses everything already fetched with zero extra network requests.
    """
    snap["composite"] = composite_verdict(snap["results"])
    snap["choppiness"] = assess_choppiness(snap["results"])
    if snap.get("strategy_signals") is not None:
        snap["strategy_signals"]["choppiness"] = snap["choppiness"]
    snap["combo"] = combined_verdict(snap["composite"], news=snap.get("news"),
                                      strategy=snap.get("strategy_signals"))
    return snap


def run_single_symbol_flow(symbol: str, pair: str, args, snap: dict = None):
    """
    Prints the full report for one symbol (table, strategy signals, price
    outlook, news, combined verdict, trade levels) and then offers position
    tracking + continuous monitoring. Shared by normal single-symbol mode
    and the screener's drill-down, so a symbol already analyzed by the
    screener doesn't need to be re-fetched.
    """
    if snap is None:
        snap = run_analysis_cycle(symbol, pair, args, include_news=True)
    if snap is None:
        print("No data could be fetched at all. Check the symbol and your internet connection.")
        return

    # Style suggestion based on this symbol's CURRENT conditions (skipped if
    # the user pinned a style with --style). Accepting reuses the data
    # already fetched - only the scoring is redone, nothing is re-downloaded.
    if getattr(args, "style", None) is None:
        rec = recommend_style(snap["results"])
        choice = input(f"\nSuggested trading style for {symbol} right now: {rec['style'].upper()} "
                        f"- {rec['reason']}.\nPress Enter to accept, or type scalp/trend/balanced: ").strip().lower()
        chosen = choice if choice in STYLE_PROFILES else rec["style"]
        apply_style(chosen)
        snap = recompute_snapshot_for_style(snap, args)

    print_table(symbol, snap["results"], snap["composite"])
    if snap["strategy_signals"] is not None:
        print_strategy_signals(snap["strategy_signals"])
    print_price_outlook(snap["results"], snap["dfs"])
    if snap["news"] is not None:
        print_news(symbol, snap["news"])

    combo = snap["combo"]
    print_combined_verdict(combo)

    trade = suggest_trade_levels(snap["results"], {"verdict": combo["verdict"]})
    print_trade_levels(trade, symbol, liquidity=snap.get("liquidity"), choppiness=snap.get("choppiness"))
    print_trade_confidence(compute_trade_confidence(snap, trade))

    print("\nReminder: technical indicators, strategy signals, and news sentiment are a starting"
          " point for your own research, not financial advice. Markets can and do move against"
          " every signal here - and Silver Bullet in particular is a discretionary concept, not"
          " a validated system.")

    position = ask_position()
    if position:
        if args.poll_minutes:
            poll_minutes = args.poll_minutes
        else:
            _default_poll = ACTIVE_STYLE.get("poll_minutes", 5)
            raw = input(f"Check interval in minutes (default {_default_poll}): ").strip()
            poll_minutes = int(raw) if raw.isdigit() else _default_poll
        monitor_position(symbol, pair, args, position, poll_seconds=poll_minutes * 60)
    else:
        print("\nNo position given - starting live WATCH mode instead (updates in real time).")
        monitor_watchlist(symbol, pair, args)


def monitor_watchlist(symbol: str, pair: str, args):
    """
    Live WATCH mode for when you have NO position yet: re-runs the full
    analysis on a timer and prints, every cycle, the individual timeframe
    verdicts, individual strategy signals, the trade levels (or the NO-TRADE
    reason), and the full trade-confidence breakdown - so you can watch a
    setup form in real time. Rings the alarm ONCE when conditions first
    become a HIGH-confidence setup with a valid entry structure (re-arms if
    conditions degrade and then recover).
    """
    if args.poll_minutes:
        poll_minutes = args.poll_minutes
    else:
        _default_poll = ACTIVE_STYLE.get("poll_minutes", 5)
        raw = input(f"Check interval in minutes (default {_default_poll}): ").strip()
        poll_minutes = int(raw) if raw.isdigit() else _default_poll

    print(f"\n{'='*78}\n  WATCH MODE - {symbol} - no position, hunting for a setup\n"
          f"  Checking every {poll_minutes} min. Alarm fires when confidence >= 70 AND a valid\n"
          f"  entry structure exists. Press Ctrl+C to stop.\n{'='*78}")

    def fmt(v):
        return f"{v:.6f}" if v < 1 else f"{v:.4f}"

    ist = timezone(timedelta(hours=5, minutes=30))
    last_ready_key = None
    cycle = 0
    try:
        while True:
            cycle += 1
            include_news = (not args.no_news) and (cycle % 6 == 1)
            snap = run_analysis_cycle(symbol, pair, args, include_news=include_news)
            ist_now = datetime.now(ist).strftime("%H:%M:%S IST")
            if snap is None:
                print(f"[{ist_now}] Cycle {cycle}: data fetch failed, will retry next interval.")
                time.sleep(poll_minutes * 60)
                continue

            combo = snap["combo"]
            trade = suggest_trade_levels(snap["results"], {"verdict": combo["verdict"]})
            conf = compute_trade_confidence(snap, trade)
            basis = next((snap["results"][tf] for tf in TRADE_BASIS_FALLBACK_ORDER
                          if tf in snap["results"]), None)
            price = basis["price"] if basis else float("nan")

            print(f"\n[{ist_now}] Cycle {cycle} | price={fmt(price)}")
            print(f"  Timeframes : {format_timeframe_verdicts(snap['results'])}")
            print(f"  Strategy   : {format_strategy_verdicts(snap.get('strategy_signals'))}")
            print(f"  Combined   : {combo['verdict']} (score {combo['blended_score']:+.2f})")
            if trade.get("available"):
                print(f"  Trade idea : {trade['side']} | entry ~{fmt(trade['entry'])} | "
                      f"stop {fmt(trade['stop_loss'])} | T1 {fmt(trade['target1'])} | "
                      f"T2 {fmt(trade['target2'])}  (basis {trade['basis_timeframe']})")
            else:
                print(f"  {trade.get('reason', 'No trade available this cycle.')}")
            print_trade_confidence(conf)

            ready = conf["score"] >= 70 and trade.get("available")
            ready_key = f"{combo['verdict']}|{trade.get('side')}" if ready else None
            if ready and ready_key != last_ready_key:
                trigger_alarm(f"{symbol}: HIGH-confidence {trade['side']} setup forming "
                               f"({conf['score']}/100). Entry ~{fmt(trade['entry'])}, "
                               f"stop {fmt(trade['stop_loss'])}. Verify before acting.",
                               severity="TARGET")
            last_ready_key = ready_key

            time.sleep(poll_minutes * 60)
    except KeyboardInterrupt:
        print("\nWatch stopped. Stay safe out there.")


# --------------------------------------------------------------------------
# Screener - scans multiple pairs and ranks them by signal conviction
# --------------------------------------------------------------------------

# A practical default list: liquid majors + popular alts + all 6 confirmed
# commodity pairs. Scanning literally every active CoinDCX futures
# instrument (100+) would be slow and hit the public API hard for little
# extra value - this covers what most people actually want to compare.
DEFAULT_SCREENER_SYMBOLS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "HBAR", "INJ",
    "AVAX", "LTC", "POL",
    "XAU", "PAXG", "XAG", "CL", "BZ", "NATGAS",
]


def compute_screener_score(snap: dict, trade: dict) -> float:
    """
    Ranks a pair by how strong AND trustworthy its current signal is:
    conviction (blended score magnitude) scaled down if the market is
    currently thin (slippage risk) or choppy (whipsaw risk) - so a huge
    score in an untradeable market doesn't rank above a moderate score in
    clean conditions. NEUTRAL verdicts always sink to the bottom.
    """
    combo = snap["combo"]
    score = abs(combo["blended_score"])
    if combo["verdict"] == "NEUTRAL":
        return -1.0  # always ranks below any directional call

    liquidity = snap.get("liquidity", {})
    if liquidity.get("available"):
        if liquidity["verdict"] == "THIN":
            score *= 0.5
        elif liquidity["verdict"] == "CAUTION":
            score *= 0.8

    choppiness = snap.get("choppiness", {})
    if choppiness.get("available"):
        if choppiness["verdict"] == "CHOPPY":
            score *= 0.5
        elif choppiness["verdict"] == "MODERATE":
            score *= 0.85

    if trade.get("available") and pd.notna(trade.get("risk_reward_t1")):
        rr = trade["risk_reward_t1"]
        if rr >= 1.5:
            score *= 1.1
        elif rr < 0.5:
            score *= 0.85

    return score


def quick_scan_symbol(symbol: str, args) -> dict:
    """
    STAGE-1 fast pre-scan: fetches only 1h + 1d candles (2 requests instead
    of the ~7 a full analysis costs) and computes a lightweight two-timeframe
    conviction score. Used to shortlist candidates from a full-exchange scan
    before spending the expensive full analysis on them. Deliberately skips
    news, orderbook, strategy signals, and the faster timeframes - this is a
    coarse filter, not a final verdict.
    """
    try:
        pair = resolve_pair(symbol, args.quote)
        prescan_tfs = ACTIVE_STYLE.get("prescan_tfs", ("1h", "1d"))
        results = {}
        for tf_key in prescan_tfs:
            df = fetch_timeframe_candles(pair, tf_key, limit=min(args.limit, 150))
            df = compute_indicators(df)
            results[tf_key] = classify_timeframe(df)
        total_w = sum(TIMEFRAME_WEIGHTS.get(tf, 1.0) for tf in prescan_tfs)
        score = sum(results[tf]["score"] * TIMEFRAME_WEIGHTS.get(tf, 1.0) for tf in prescan_tfs) / (3.5 * total_w)
        return {"symbol": symbol, "ok": True, "pair": pair, "quick_score": score}
    except Exception as e:
        return {"symbol": symbol, "ok": False, "error": str(e)}


def run_fast_prescan(symbols: list, args, workers: int = 8) -> list:
    """Runs the stage-1 quick scan across all symbols in parallel with a progress counter."""
    out = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(quick_scan_symbol, s, args): s for s in symbols}
        for future in concurrent.futures.as_completed(futures):
            out.append(future.result())
            done += 1
            if done % 25 == 0 or done == len(symbols):
                print(f"  ...pre-scanned {done}/{len(symbols)} pairs")
    return out


def run_screener(symbols: list, args) -> list:
    """Runs the full analysis on each symbol in parallel and returns a list of result summaries."""
    def _screen_one(symbol):
        try:
            pair = resolve_pair(symbol, args.quote)
            snap = run_analysis_cycle(symbol, pair, args, include_news=not args.no_news)
            if snap is None:
                return {"symbol": symbol, "ok": False, "error": "no data returned"}
            combo = snap["combo"]
            trade = suggest_trade_levels(snap["results"], {"verdict": combo["verdict"]})
            score = compute_screener_score(snap, trade)
            return {"symbol": symbol, "ok": True, "pair": pair, "snap": snap,
                     "trade": trade, "combo": combo, "score": score}
        except Exception as e:
            return {"symbol": symbol, "ok": False, "error": str(e)}

    screener_results = []
    # Bounded to a modest number of concurrent PAIRS (each pair itself already
    # fires 5 concurrent timeframe requests) - keeps total concurrent load
    # on CoinDCX's public API reasonable rather than firing everything at once.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_screen_one, s): s for s in symbols}
        for future in concurrent.futures.as_completed(futures):
            screener_results.append(future.result())

    return screener_results


def print_screener_table(screener_results: list):
    ok_results = [r for r in screener_results if r["ok"]]
    failed = [r for r in screener_results if not r["ok"]]
    ok_results.sort(key=lambda r: r["score"], reverse=True)

    print("\n" + "=" * 90)
    print("  SCREENER RESULTS - ranked by signal conviction (adjusted for liquidity/choppiness)")
    print("=" * 90)
    header = f"{'#':<3}{'Symbol':<8}{'Verdict':<9}{'Score':<8}{'Liquidity':<11}{'Choppiness':<12}{'R:R':<7}{'Price':<12}"
    print(header)
    print("-" * 90)
    for i, r in enumerate(ok_results, 1):
        combo = r["combo"]
        liq = r["snap"].get("liquidity", {})
        chop = r["snap"].get("choppiness", {})
        trade = r["trade"]
        price = r["snap"]["results"].get("1h", next(iter(r["snap"]["results"].values())))["price"]
        price_str = f"{price:.6f}" if price < 1 else f"{price:.4f}"
        liq_str = liq.get("verdict", "N/A") if liq.get("available") else "N/A"
        chop_str = chop.get("verdict", "N/A") if chop.get("available") else "N/A"
        rr_str = f"{trade['risk_reward_t1']:.2f}" if trade.get("available") and pd.notna(trade.get("risk_reward_t1")) else "N/A"
        print(f"{i:<3}{r['symbol']:<8}{combo['verdict']:<9}{r['score']:<8.2f}"
              f"{liq_str:<11}{chop_str:<12}{rr_str:<7}{price_str:<12}")

    if failed:
        print("\n  Skipped (couldn't fetch):")
        for r in failed:
            print(f"    {r['symbol']}: {r['error']}")

    print("\n  IMPORTANT: this ranks by conviction + tradeable conditions, NOT by expected profit."
          " It is not a prediction of which pair will make money - a high score just means many"
          " signals currently agree in clean (liquid, trending) conditions. Verify independently.")


def backtest_symbol(symbol: str, pair: str, args, timeframe: str = "1h",
                     forward_periods: int = 24, history_limit: int = 1000) -> dict:
    """
    Walk-forward backtest of classify_timeframe()'s own scoring against a
    symbol's own history. At each historical point, only data up to and
    including that bar is used - verified to introduce ZERO lookahead bias,
    since every indicator here (RSI, MACD, ATR, ADX, VWAP, OBV, PVT, RVI,
    Bollinger Bands) is backward-looking only. Indicators are computed once
    on the whole series (fast, vectorized) rather than recomputed per step,
    since slicing a causal indicator produces an identical value to
    computing it fresh on just that truncated history - verified directly
    before writing this function.

    HONESTY NOTE: this is a SINGLE historical path for one symbol and one
    timeframe, not a statistically powered sample across many market
    regimes. It does not model transaction costs, slippage, or funding
    rates. Treat results as informative about how this scoring has behaved
    on THIS pair's recent history, not a guarantee of future performance.
    """
    try:
        df = fetch_timeframe_candles(pair, timeframe, limit=history_limit)
    except Exception as e:
        return {"available": False, "reason": f"couldn't fetch history: {e}"}

    if len(df) < 100:
        return {"available": False, "reason": "not enough historical data"}

    df = compute_indicators(df)
    min_lookback = 40  # need enough bars for ADX/BB/MACD/OBV-slope to be valid

    records = []
    for i in range(min_lookback, len(df) - forward_periods):
        window = df.iloc[:i + 1]
        result = classify_timeframe(window)
        entry_price = df["close"].iloc[i]
        future_price = df["close"].iloc[i + forward_periods]
        fwd_return_pct = (future_price - entry_price) / entry_price * 100
        records.append({"verdict": result["verdict"], "fwd_return_pct": fwd_return_pct})

    if not records:
        return {"available": False, "reason": "no valid backtest points in this history window"}

    records_df = pd.DataFrame(records)
    summary = {"available": True, "symbol": symbol, "timeframe": timeframe,
               "forward_periods": forward_periods, "total_samples": len(records_df),
               "by_verdict": {}}

    for verdict in ("BULL", "BEAR", "NEUTRAL"):
        sub = records_df[records_df["verdict"] == verdict]
        if len(sub) == 0:
            continue
        if verdict == "BULL":
            win_rate = (sub["fwd_return_pct"] > 0).mean() * 100
        elif verdict == "BEAR":
            win_rate = (sub["fwd_return_pct"] < 0).mean() * 100
        else:
            win_rate = None
        summary["by_verdict"][verdict] = {
            "count": len(sub),
            "pct_of_samples": len(sub) / len(records_df) * 100,
            "avg_fwd_return_pct": sub["fwd_return_pct"].mean(),
            "median_fwd_return_pct": sub["fwd_return_pct"].median(),
            "win_rate_pct": win_rate,
        }
    return summary


def print_backtest_report(summary: dict):
    print(f"\n{'='*78}\n  BACKTEST REPORT (walk-forward, zero lookahead bias)\n{'='*78}")
    if not summary.get("available"):
        print(f"  Could not run backtest: {summary.get('reason', 'unknown error')}")
        return

    print(f"  Symbol: {summary['symbol']}   Timeframe: {summary['timeframe']}   "
          f"Forward window: {summary['forward_periods']} bars")
    print(f"  Total historical points tested: {summary['total_samples']}")
    print("\n  HONESTY NOTE: single historical path for one symbol/timeframe, not a")
    print("  statistically powered sample. No transaction costs, slippage, or funding")
    print("  modeled. This describes past behavior, not a promise about the future.")

    print(f"\n  {'Verdict':<10}{'Count':<8}{'% of time':<12}{'Avg fwd move':<15}{'Median fwd move':<17}{'Win rate':<10}")
    print("  " + "-" * 72)
    for verdict in ("BULL", "BEAR", "NEUTRAL"):
        stat = summary["by_verdict"].get(verdict)
        if not stat:
            continue
        win_str = f"{stat['win_rate_pct']:.1f}%" if stat["win_rate_pct"] is not None else "N/A"
        print(f"  {verdict:<10}{stat['count']:<8}{stat['pct_of_samples']:<12.1f}"
              f"{stat['avg_fwd_return_pct']:+.2f}%{'':<8}{stat['median_fwd_return_pct']:+.2f}%{'':<10}{win_str:<10}")

    bull = summary["by_verdict"].get("BULL")
    bear = summary["by_verdict"].get("BEAR")
    print()
    if bull and bull["win_rate_pct"] is not None:
        edge = bull["win_rate_pct"] - 50
        print(f"  BULL calls were right {bull['win_rate_pct']:.1f}% of the time ({edge:+.1f} points vs. a coin flip).")
    if bear and bear["win_rate_pct"] is not None:
        edge = bear["win_rate_pct"] - 50
        print(f"  BEAR calls were right {bear['win_rate_pct']:.1f}% of the time ({edge:+.1f} points vs. a coin flip).")
    print("\n  If win rates are close to 50%, the scoring isn't adding real edge on this")
    print("  pair/timeframe/horizon - that's a genuinely useful, if humbling, result to know.")


import os as _os
import json as _json


def send_telegram_message(text: str) -> bool:
    """
    Sends a message via a Telegram bot. Credentials come from environment
    variables (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) so they can live in CI
    secrets (e.g. GitHub Actions) instead of in code. Returns True on success.
    """
    token = _os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = _os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[telegram] Not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars missing) - skipping send.")
        return False
    try:
        resp = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              json={"chat_id": chat_id, "text": text}, timeout=15)
        resp.raise_for_status()
        print("[telegram] Alert sent.")
        return True
    except Exception as e:
        print(f"[telegram] Send failed: {e}")
        return False


def run_ci_monitor(args, max_duration_seconds: float = 5.5 * 3600):
    """
    Runs the REAL continuous monitor loop non-interactively - built for a
    single long-lived job (GitHub Actions or any scheduler that allows a
    long-running process), NOT periodic short checks. Loops until
    max_duration_seconds elapses (default 5.5h, safely under GitHub
    Actions' 6h hard kill of any job), printing full per-cycle status to
    stdout (visible live in the Actions log) and pushing a Telegram alert -
    deduped across cycles so an unchanged condition doesn't spam - only on
    a genuine exit condition (position mode) or a HIGH-confidence
    actionable setup (watch mode, no position).
    """
    position = None
    if args.position_file and _os.path.exists(args.position_file):
        try:
            with open(args.position_file) as f:
                position = _json.load(f)
        except Exception as e:
            print(f"[ci-monitor] Could not read position file: {e}")

    symbol = (args.symbol or (position or {}).get("symbol") or "").upper()
    if not symbol:
        print("[ci-monitor] No symbol given (pass one on the command line or in the position file). Exiting.")
        return
    symbol = COMMODITY_ALIASES.get(symbol, symbol)
    apply_style(args.style or (position or {}).get("style") or "balanced")
    pair = resolve_pair(symbol, args.quote)

    poll_seconds = (args.poll_minutes or ACTIVE_STYLE.get("poll_minutes", 5)) * 60
    ist = timezone(timedelta(hours=5, minutes=30))
    start = time.time()
    cycle = 0
    last_alert_key = None

    mode_desc = f"position: {position['side']} @ {position['entry_price']}" if position else "watch mode (no position)"
    send_telegram_message(f"▶ Monitoring session started - {symbol} ({mode_desc}). "
                           f"Checking every {poll_seconds // 60} min for up to "
                           f"{max_duration_seconds / 3600:.1f}h this session.")
    print(f"[ci-monitor] {symbol} ({pair}) - {mode_desc} - checking every {poll_seconds // 60} min "
          f"for up to {max_duration_seconds / 3600:.1f}h")

    while time.time() - start < max_duration_seconds:
        cycle += 1
        include_news = (not args.no_news) and (cycle % 6 == 1)
        snap = run_analysis_cycle(symbol, pair, args, include_news=include_news)
        ist_now = datetime.now(ist).strftime("%H:%M:%S IST")

        if snap is None:
            print(f"[{ist_now}] Cycle {cycle}: data fetch failed, will retry next interval.")
            time.sleep(poll_seconds)
            continue

        combo = snap["combo"]
        trade = suggest_trade_levels(snap["results"], {"verdict": combo["verdict"]})
        conf = compute_trade_confidence(snap, trade)
        print(f"\n[{ist_now}] Cycle {cycle}")
        print(f"  Timeframes : {format_timeframe_verdicts(snap['results'])}")
        print(f"  Strategy   : {format_strategy_verdicts(snap.get('strategy_signals'))}")
        print(f"  Combined   : {combo['verdict']} (score {combo['blended_score']:+.2f})")
        print(f"  Confidence : {conf['score']}/100 - {conf['tier']}")

        alerts = []
        if position:
            basis = next((snap["results"][tf] for tf in TRADE_BASIS_FALLBACK_ORDER
                          if tf in snap["results"]), None)
            current_price = basis["price"] if basis else None
            if current_price is not None:
                metrics = compute_position_metrics(position, current_price)
                soft_stop = None
                if position.get("stop_loss") is None and basis and pd.notna(basis.get("atr")):
                    soft_stop = (position["entry_price"] - 2 * basis["atr"]) if position["side"] == "LONG" \
                        else (position["entry_price"] + 2 * basis["atr"])
                alerts = check_exit_conditions(position, current_price, metrics, combo["verdict"],
                                                soft_stop, liquidity=snap.get("liquidity"))
                print(f"  Position   : {position['side']} @ {position['entry_price']} | "
                      f"P&L {metrics['pnl']:+.2f} USDT ({metrics['pnl_pct_margin']:+.1f}% of margin)")
        elif conf["score"] >= 70 and trade.get("available"):
            alerts = [("TARGET", f"HIGH-confidence {trade['side']} setup - entry ~{trade['entry']:.6f}, "
                                  f"stop {trade['stop_loss']:.6f}, T1 {trade['target1']:.6f}")]

        if alerts:
            for sev, msg in alerts:
                print(f"  [{sev}] {msg}")
            alert_key = ";".join(f"{sev}:{msg.split('(')[0].strip()}" for sev, msg in alerts)
            if alert_key != last_alert_key:
                lines = [f"🚨 {symbol} {position['side'] if position else ''}".strip()]
                lines += [f"[{sev}] {msg}" for sev, msg in alerts]
                lines.append(f"Verdict: {combo['verdict']} ({combo['blended_score']:+.2f}), "
                             f"confidence {conf['score']}/100")
                send_telegram_message("\n".join(lines))
            else:
                print("  (same alert as last cycle - not re-sending)")
            last_alert_key = alert_key
        else:
            last_alert_key = None

        remaining = max_duration_seconds - (time.time() - start)
        if remaining <= poll_seconds:
            break
        time.sleep(poll_seconds)

    send_telegram_message(f"⏹ Monitoring session for {symbol} ended (time limit reached). "
                           f"Re-run the workflow to start another session.")
    print(f"\n[ci-monitor] Session duration limit reached after {cycle} cycles. Exiting cleanly.")


def run_headless_check(args):
    """
    NON-INTERACTIVE single check, built for schedulers (GitHub Actions cron,
    Termux cron, any CI): runs ONE full analysis cycle, evaluates exit
    conditions if a position file is present, pushes Telegram alerts (with a
    state file deduping repeats across runs so the same alert isn't re-sent
    every 15 minutes), prints everything to stdout for the run logs, exits.
    Deliberately NOT an infinite loop - schedulers provide the repetition,
    and CI runners kill long-lived processes anyway.
    """
    position = None
    if args.position_file and _os.path.exists(args.position_file):
        try:
            with open(args.position_file) as f:
                position = _json.load(f)
        except Exception as e:
            print(f"[headless] Could not read position file: {e}")

    symbol = (args.symbol or (position or {}).get("symbol") or "").upper()
    if not symbol:
        print("[headless] No symbol given (pass one on the command line or in the position file). Exiting.")
        return
    symbol = COMMODITY_ALIASES.get(symbol, symbol)
    apply_style(args.style or (position or {}).get("style") or "balanced")

    pair = resolve_pair(symbol, args.quote)
    ist_now = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%d %b %H:%M IST")
    print(f"[headless] {ist_now} - checking {symbol} ({pair})")
    snap = run_analysis_cycle(symbol, pair, args, include_news=not args.no_news)
    if snap is None:
        send_telegram_message(f"⚠ {symbol} monitor: data fetch FAILED at {ist_now} - will retry next scheduled run.")
        return

    combo = snap["combo"]
    trade = suggest_trade_levels(snap["results"], {"verdict": combo["verdict"]})
    conf = compute_trade_confidence(snap, trade)
    tf_line = format_timeframe_verdicts(snap["results"])
    strat_line = format_strategy_verdicts(snap.get("strategy_signals"))
    print(f"  Timeframes : {tf_line}")
    print(f"  Strategy   : {strat_line}")
    print(f"  Combined   : {combo['verdict']} ({combo['blended_score']:+.2f}) | Confidence {conf['score']}/100")

    alerts = []
    summary_price = None
    if position:
        basis = next((snap["results"][tf] for tf in TRADE_BASIS_FALLBACK_ORDER if tf in snap["results"]), None)
        current_price = basis["price"] if basis else None
        summary_price = current_price
        if current_price is not None:
            metrics = compute_position_metrics(position, current_price)
            soft_stop = None
            if position.get("stop_loss") is None and basis and pd.notna(basis.get("atr")):
                soft_stop = (position["entry_price"] - 2 * basis["atr"]) if position["side"] == "LONG" \
                    else (position["entry_price"] + 2 * basis["atr"])
            alerts = check_exit_conditions(position, current_price, metrics, combo["verdict"],
                                            soft_stop, liquidity=snap.get("liquidity"))
            print(f"  Position   : {position['side']} @ {position['entry_price']} | "
                  f"P&L {metrics['pnl']:+.2f} USDT ({metrics['pnl_pct_margin']:+.1f}% of margin)")

    # Dedupe across scheduled runs via the state file
    alert_key = ";".join(f"{sev}:{msg.split('(')[0].strip()}" for sev, msg in alerts)
    prev_key = ""
    if args.state_file and _os.path.exists(args.state_file):
        try:
            with open(args.state_file) as f:
                prev_key = _json.load(f).get("last_alert_key", "")
        except Exception:
            pass

    if alerts:
        for sev, msg in alerts:
            print(f"  [{sev}] {msg}")
        if alert_key != prev_key:
            lines = [f"🚨 {symbol} {position['side'] if position else ''} @ {summary_price}"]
            lines += [f"[{sev}] {msg}" for sev, msg in alerts]
            lines.append(f"Verdict: {combo['verdict']} ({combo['blended_score']:+.2f}), "
                          f"confidence {conf['score']}/100")
            send_telegram_message("\n".join(lines))
        else:
            print("  (same alerts as last run - Telegram send skipped to avoid spam)")
    elif args.heartbeat:
        send_telegram_message(f"{symbol}: {combo['verdict']} ({combo['blended_score']:+.2f}), "
                               f"confidence {conf['score']}/100, no alerts. {ist_now}")

    if args.state_file:
        try:
            with open(args.state_file, "w") as f:
                _json.dump({"last_alert_key": alert_key, "checked_at": ist_now}, f)
        except Exception as e:
            print(f"[headless] Could not write state file: {e}")


def run_backtest_flow(args):
    symbol = input("\nSymbol to backtest (e.g. BTC, DOGE, XAU): ").strip().upper()
    symbol = COMMODITY_ALIASES.get(symbol, symbol)
    tf_raw = input("Timeframe to test - 1h or 1d (default 1h): ").strip().lower()
    timeframe = tf_raw if tf_raw in ("1h", "1d") else "1h"
    fp_raw = input(f"How many bars ahead to check the outcome (default 24 {timeframe} bars): ").strip()
    forward_periods = int(fp_raw) if fp_raw.isdigit() else 24

    pair = resolve_pair(symbol, args.quote)
    print(f"\nRunning walk-forward backtest for {symbol} ({pair}) on {timeframe}, "
          f"checking outcomes {forward_periods} bars ahead. This may take a little while...")
    summary = backtest_symbol(symbol, pair, args, timeframe=timeframe, forward_periods=forward_periods)
    print_backtest_report(summary)


def run_screener_flow(args):
    raw = input("\nEnter comma-separated symbols to scan, type 'all' to scan EVERY active "
                "CoinDCX futures pair, or press Enter for a default list of majors + commodities: ").strip()
    if raw.upper() == "ALL":
        try:
            all_symbols = fetch_all_futures_symbols(args.quote)
        except Exception as e:
            print(f"Couldn't fetch the full instrument list ({e}). Falling back to the default list.")
            symbols = DEFAULT_SCREENER_SYMBOLS
        else:
            print(f"\nFound {len(all_symbols)} active CoinDCX futures pairs.")
            top_raw = input("How many top candidates should get the FULL deep analysis after the "
                             "fast pre-scan? (default 15): ").strip()
            top_n = int(top_raw) if top_raw.isdigit() and int(top_raw) > 0 else 15
            _ptfs = "+".join(ACTIVE_STYLE.get("prescan_tfs", ("1h", "1d")))
            print(f"\nStage 1: fast pre-scan of all {len(all_symbols)} pairs "
                   f"({_ptfs} charts, 2 quick requests each, ~1-4 min total)...")
            prescan = run_fast_prescan(all_symbols, args)
            ok = [r for r in prescan if r["ok"]]
            failed = [r for r in prescan if not r["ok"]]
            ok.sort(key=lambda r: abs(r["quick_score"]), reverse=True)
            symbols = [r["symbol"] for r in ok[:top_n]]
            if failed:
                print(f"  ({len(failed)} pairs skipped - no data returned)")
            print(f"\nStage 2: full deep analysis (technical + strategy"
                   f"{' + news' if not args.no_news else ''}) on the top {len(symbols)}: "
                   f"{', '.join(symbols)}")
    elif raw:
        symbols = [COMMODITY_ALIASES.get(s.strip().upper(), s.strip().upper()) for s in raw.split(",") if s.strip()]
    else:
        symbols = DEFAULT_SCREENER_SYMBOLS

    print(f"\nScanning {len(symbols)} pairs - this may take a little while "
          f"(each pair runs the full technical + strategy{' + news' if not args.no_news else ''} analysis)...")
    screener_results = run_screener(symbols, args)
    print_screener_table(screener_results)

    ok_by_symbol = {r["symbol"]: r for r in screener_results if r["ok"]}
    choice = input("\nEnter a symbol from above for the full detailed report (or press Enter to exit): ").strip().upper()
    choice = COMMODITY_ALIASES.get(choice, choice)
    if choice in ok_by_symbol:
        r = ok_by_symbol[choice]
        run_single_symbol_flow(choice, r["pair"], args, snap=r["snap"])
    elif choice:
        print(f"'{choice}' wasn't in the scanned list. Run the script again and enter it directly instead.")


def main():
    parser = argparse.ArgumentParser(description="CoinDCX multi-timeframe crypto signal bot")
    parser.add_argument("symbol", nargs="?", default=None,
                         help="Coin symbol, e.g. HBAR, ADA, BTC (paired against USDT)")
    parser.add_argument("--quote", default="USDT", help="Quote currency (default USDT)")
    parser.add_argument("--no-news", action="store_true", help="Skip the news headline section")
    parser.add_argument("--no-strategy", action="store_true", help="Skip the SMC/ICT strategy signals section")
    parser.add_argument("--limit", type=int, default=300, help="Candles to fetch per timeframe")
    parser.add_argument("--poll-minutes", type=int, default=None,
                         help="Monitoring check interval in minutes (skips the interactive prompt if set)")
    parser.add_argument("--screener", action="store_true",
                         help="Scan multiple pairs and rank them, instead of analyzing one symbol")
    parser.add_argument("--backtest", action="store_true",
                         help="Walk-forward backtest the scoring system against a symbol's own history")
    parser.add_argument("--style", choices=["scalp", "trend", "balanced"], default=None,
                         help="Trading style: adjusts timeframe weights, trade-level sizing, and monitoring cadence")
    parser.add_argument("--headless-check", action="store_true",
                         help="Non-interactive single check for schedulers (GitHub Actions/cron): one cycle, Telegram alerts, exit")
    parser.add_argument("--position-file", default=None, help="JSON file with position details for headless mode")
    parser.add_argument("--state-file", default=None, help="JSON file persisting alert-dedupe state between headless runs")
    parser.add_argument("--heartbeat", action="store_true", help="Headless mode: send a Telegram summary even when there are no alerts")
    parser.add_argument("--ci-monitor", action="store_true",
                         help="Run the REAL continuous monitor loop for one long session (e.g. a single GitHub "
                              "Actions job), capped safely under its 6h hard job-kill limit, instead of a single check")
    parser.add_argument("--max-hours", type=float, default=5.5,
                         help="Max session duration in hours for --ci-monitor (default 5.5, safely under GitHub's 6h limit)")
    args = parser.parse_args()

    if args.ci_monitor:
        run_ci_monitor(args, max_duration_seconds=args.max_hours * 3600)
        return

    if args.headless_check:
        run_headless_check(args)
        return

    if args.style:
        apply_style(args.style)
    # (No upfront style question - for single-symbol analysis the style is
    # SUGGESTED from that symbol's live conditions after fetching. Screener
    # and backtest run in balanced mode unless --style pins one.)

    symbol = args.symbol
    if args.screener:
        run_screener_flow(args)
        return
    if args.backtest:
        run_backtest_flow(args)
        return
    if not symbol:
        # No command-line argument given (common on mobile Python apps like
        # Pydroid 3 that just hit "Run") - ask interactively instead.
        symbol = input("Enter symbol - crypto (HBAR, ADA, BTC), commodity (GOLD, SILVER, OIL, "
                        "BRENT, GAS), 'screen' to scan multiple pairs, or 'backtest' to validate "
                        "the scoring against history: ").strip()
    if symbol.upper() in ("SCREEN", "SCREENER"):
        run_screener_flow(args)
        return
    if symbol.upper() == "BACKTEST":
        run_backtest_flow(args)
        return

    symbol = symbol.upper()
    symbol = COMMODITY_ALIASES.get(symbol, symbol)
    is_commodity = symbol in COMMODITY_FUTURES_SYMBOLS
    if is_commodity:
        print(f"[info] {symbol} recognized as a CoinDCX commodity futures pair "
              f"({symbol}-USDT), not a cryptocurrency.")
    print(f"Resolving CoinDCX pair for {symbol}/{args.quote} ...")
    pair = resolve_pair(symbol, args.quote)
    print(f"Using pair code: {pair}")

    run_single_symbol_flow(symbol, pair, args)


if __name__ == "__main__":
    main()
