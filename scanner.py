"""
Midcap Reversal Desk -- daily scanner.

Reads daily OHLC for every stock in the chosen index universes, computes
Wilder RSI(14) and ATR(14), and looks for three setups:

  * bullish RSI divergence -- price prints a LOWER bottom while RSI prints a
    HIGHER one; the peak between those bottoms is the resistance to break
  * bullish engulfing      -- a green candle swallowing the previous red one
                              after a downtrend; buy above its high, stop at
                              its low
  * tweezer bottom         -- two sessions bottoming at the same level after a
                              downtrend; buy above the second high, stop on
                              the shared low

Horizontal levels are found by clustering pivots, so a level is a price MANY
touches agree on rather than a single swing high. Each setup is sized off its
own entry and stop against the risk-per-trade rule (RPT = total risk / 50).

Writes data.json for the dashboard and alerts.md for the e-mail step.
Nothing here needs an API key.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from universes import (UNIVERSES as UNIVERSE_CATALOGUE, GROUPS as UNIVERSE_GROUPS,
                       build_watchlist, yahoo_symbol)

# ----------------------------------------------------------------------------
# WHICH UNIVERSE(S) TO SCAN -- step 1 of the system.
#
# Everything listed here is scanned in ONE run, and every stock keeps the tag
# of each index it belongs to. That is what lets the dashboard switch between
# indexes instantly: the work is already done, the page just filters.
#
# Overlap is free. Nifty 500 already contains Nifty 50, Next 50, Nifty 100,
# Nifty 200, all the midcap and smallcap indexes and Nifty Bank, so a stock in
# eight of these lists is still downloaded exactly once. The only list below
# that adds genuinely new stocks is Microcap 250 (ranks 501-750), so the real
# cost is about 750 stocks, not the 2,000-odd you get by adding the counts up.
#
# To add sector indexes -- IT, Pharma, Auto, FMCG, Metal, Energy, Realty,
# Infrastructure -- just append their keys. They cost close to nothing in time,
# because their members are already inside Nifty 500 and so already fetched.
# Full list of keys: universes.py.
# ----------------------------------------------------------------------------
UNIVERSES    = [
    # broad market
    "NIFTY50", "NIFTYNEXT50", "NIFTY100", "NIFTY200", "NIFTY500",
    # by size
    "NIFTYMIDCAP50", "NIFTYMIDCAP100", "NIFTYMIDCAP150", "NIFTYMIDCAPSELECT",
    "NIFTYSMLCAP50", "NIFTYSMLCAP100", "NIFTYSMLCAP250", "NIFTYMICROCAP250",
    # banking
    "NIFTYBANK",
]
BENCHMARK    = "^NSEI"  # Nifty 50 -- the trend line for comparative strength

# ----------------------------------------------------------------------------
# Strategy settings -- change these and the dashboard follows.
# ----------------------------------------------------------------------------
RSI_PERIOD   = 14
ATR_PERIOD   = 14
ATR_MULT     = 1.5      # stop sits ATR_MULT x ATR below the breakout level
TOTAL_RISK   = 100000   # default risk capital; RPT = TOTAL_RISK / divisor
RPT_DIVISOR  = 50       # the notes use 50 for swing, 65 for the S1 system
RR_TARGETS   = [2, 3]   # book targets at these multiples of the risk
CRS_PERIOD   = 100      # comparative relative strength average, in sessions
ATRPCT_FLOOR = 3.0      # "momentum" floor from the notes: ATR% above 3%
TURNOVER_DAYS = 20      # sessions averaged for the liquidity proxy

# --- setups and levels ------------------------------------------------------
SETUPS       = ["divergence", "engulfing", "tweezer"]   # what the scan looks for
FRESH_BARS   = 5        # a candlestick setup goes stale after this many sessions
TREND_BARS   = 10       # sessions of decline that count as "a downtrend before it"
TWEEZER_TOL_ATR = 0.15  # how equal two lows must be, as a fraction of ATR
LEVEL_WINDOW = 250      # sessions scanned for horizontal levels
LEVEL_TOL_ATR = 0.6     # pivots within this x ATR collapse into one level
LEVEL_MIN_TOUCHES = 2   # "a major level, not a single point"
LEVEL_SNAP_ATR = 1.5    # how near a cluster must be to snap to the bounce peak
PRIMARY_RULE = "widest"   # which setup leads a row: "widest" or "tightest" stop.
                          # Widest is the default because a fixed RPT turns a
                          # tight stop into a large share count -- routinely
                          # more capital than one position should hold.
SWING_BARS   = 5        # bars either side that define a pivot low/high
DIV_WINDOW   = 120      # how far back (trading days) to hunt for the two bottoms
NEAR_PCT     = 20.0     # an armed setup further than this below its resistance
                        # is stale -- kept on the dashboard, left out of the e-mail
ALERT_MAX    = 30       # most lines per section in the e-mail. Across ~750
                        # stocks a full list would be unreadable; the dashboard
                        # is where you go for everything.
# --- timeframes -------------------------------------------------------------
# Which candle sizes to analyse. Weekly costs no extra downloads: the daily
# frame is resampled, so it is one fetch and two passes of the same maths.
#
# Every window below (DIV_WINDOW, LEVEL_WINDOW, SWING_BARS, TREND_BARS,
# FRESH_BARS) is counted in BARS, not days, so they carry across untouched --
# FRESH_BARS = 5 means five sessions on daily and five weeks on weekly, which
# is exactly what the exit-on-the-fifth-candle rule means on each chart.
TIMEFRAMES   = ["daily", "weekly"]
TF_LABEL     = {"daily": "Daily", "weekly": "Weekly"}
TF_RULE      = {"weekly": "W-FRI"}      # NSE weeks end Friday
BASE_TF      = "daily"  # the timeframe the row sort and the 52-week range use

HISTORY      = "5y"     # history pulled per stock. Daily analysis only ever
                        # looks at the tail of this, but weekly needs the depth:
                        # 250 weekly bars of levels IS five years of chart.
CHUNK        = 25       # stocks per yfinance request. Bigger means fewer
                        # round-trips over ~750 stocks; too big and one refused
                        # request loses a lot of names at once, so 25 is the
                        # compromise. Failed chunks are retried per stock below.
CHUNK_PAUSE  = 1.2      # seconds between chunks -- politeness, and it keeps
                        # Yahoo from rate-limiting a long run
IST          = timezone(timedelta(hours=5, minutes=30))


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------
def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing: seed with the simple average of the first `period`
    readings, then carry it forward as (prev * (n-1) + new) / n.

    The seeding is what makes this match the RSI and ATR that TradingView,
    Zerodha and Wilder's own book print -- a plain EMA differs for the first
    few dozen bars.
    """
    vals = series.to_numpy(dtype=float)
    out = np.full(vals.shape, np.nan)
    valid = ~np.isnan(vals)
    if not valid.any():
        return pd.Series(out, index=series.index)
    first = int(np.argmax(valid))
    seed_end = first + period - 1
    if seed_end >= len(vals) or np.isnan(vals[first : seed_end + 1]).any():
        return pd.Series(out, index=series.index)
    out[seed_end] = vals[first : seed_end + 1].mean()
    for i in range(seed_end + 1, len(vals)):
        prev, cur = out[i - 1], vals[i]
        out[i] = prev if np.isnan(cur) else (prev * (period - 1) + cur) / period
    return pd.Series(out, index=series.index)


def wilder_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI -- the same definition TradingView and Zerodha plot."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi[(avg_loss == 0) & avg_gain.notna()] = 100.0
    return rsi


def wilder_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Average True Range, Wilder smoothing."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return wilder_smooth(tr, period)


def pivot_lows(series: pd.Series, k: int = SWING_BARS) -> list:
    """Indices where the value is the lowest in the k bars either side."""
    vals = series.values
    out = []
    for i in range(k, len(vals) - k):
        window = vals[i - k : i + k + 1]
        if np.isnan(window).any():
            continue
        if vals[i] == window.min() and (window.min() < window.max()):
            out.append(i)
    return out


def pivot_highs(series: pd.Series, k: int = SWING_BARS) -> list:
    vals = series.values
    out = []
    for i in range(k, len(vals) - k):
        window = vals[i - k : i + k + 1]
        if np.isnan(window).any():
            continue
        if vals[i] == window.max() and (window.max() > window.min()):
            out.append(i)
    return out


# ----------------------------------------------------------------------------
# The setup, exactly as written in the strategy notes
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Horizontal levels, the way the notes define them: a level must join MANY
# points and sit on a major level, never a single point. So every pivot high
# and low is clustered into bands, and a band's strength is its touch count.
# ----------------------------------------------------------------------------
def touch_levels(df: pd.DataFrame, atr: float, window: int = LEVEL_WINDOW,
                 tol_mult: float = LEVEL_TOL_ATR) -> list:
    sub = df.iloc[-window:]
    if len(sub) < SWING_BARS * 2 + 5:
        return []
    marks = []
    for i in pivot_highs(sub["High"]):
        marks.append((float(sub["High"].iloc[i]), sub.index[i]))
    for i in pivot_lows(sub["Low"]):
        marks.append((float(sub["Low"].iloc[i]), sub.index[i]))
    if not marks:
        return []

    last = float(sub["Close"].iloc[-1])
    tol = (atr if atr and np.isfinite(atr) else last * 0.01) * tol_mult
    marks.sort(key=lambda m: m[0])

    clusters, cur = [], [marks[0]]
    for m in marks[1:]:
        if m[0] - cur[-1][0] <= tol:       # close enough to be the same level
            cur.append(m)
        else:
            clusters.append(cur)
            cur = [m]
    clusters.append(cur)

    levels = []
    for c in clusters:
        prices = [x[0] for x in c]
        levels.append({
            "price": round(float(np.mean(prices)), 2),
            "touches": len(c),
            "lastTouch": max(x[1] for x in c).strftime("%Y-%m-%d"),
        })
    return sorted(levels, key=lambda l: (-l["touches"], l["price"]))


def pick_resistance(levels: list, price: float, floor: float = None) -> dict:
    """The nearest level above price that price has actually respected."""
    above = [l for l in levels if l["price"] > price and
             (floor is None or l["price"] > floor)]
    if not above:
        return None
    strong = [l for l in above if l["touches"] >= LEVEL_MIN_TOUCHES]
    pool = strong or above
    return min(pool, key=lambda l: l["price"])


# ----------------------------------------------------------------------------
# Candlestick setups. Both need a downtrend in front of them, both are graded
# stale after FRESH_BARS sessions -- the notes exit on the 5th candle, so a
# pattern older than that has already had its run.
# ----------------------------------------------------------------------------
def in_downtrend(close: pd.Series, i: int, lookback: int = TREND_BARS) -> bool:
    if i < lookback:
        return False
    return float(close.iloc[i]) < float(close.iloc[i - lookback])


def find_engulfing(df: pd.DataFrame, fresh: int = FRESH_BARS) -> list:
    """Red candle, then a green one that swallows its body whole.

    Buy above the green candle's high; its low is the stop. Straight from the
    notes -- "it's high is the position to buy & low must be set as stop loss".
    """
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    n, found = len(df), []
    for i in range(max(1, n - fresh), n):
        prev_red = c.iloc[i - 1] < o.iloc[i - 1]
        green = c.iloc[i] > o.iloc[i]
        swallows = (o.iloc[i] <= c.iloc[i - 1]) and (c.iloc[i] >= o.iloc[i - 1])
        bigger = (c.iloc[i] - o.iloc[i]) > (o.iloc[i - 1] - c.iloc[i - 1])
        if prev_red and green and swallows and bigger and in_downtrend(c, i - 1):
            found.append({
                "type": "engulfing", "label": "Bullish engulfing",
                "date": df.index[i].strftime("%Y-%m-%d"), "ageBars": n - 1 - i,
                "entry": round(float(h.iloc[i]), 2),
                "stop": round(float(l.iloc[i]), 2),
                "detail": (f"Green candle engulfed the red one "
                           f"({round(float(o.iloc[i-1]),2)}-{round(float(c.iloc[i-1]),2)}) "
                           f"after a downtrend."),
            })
    return found


def find_tweezer(df: pd.DataFrame, atr: float, fresh: int = FRESH_BARS) -> list:
    """Two candles bottoming at the same level after a downtrend."""
    l, h, c = df["Low"], df["High"], df["Close"]
    n, found = len(df), []
    last = float(c.iloc[-1])
    tol = (atr if atr and np.isfinite(atr) else last * 0.002) * TWEEZER_TOL_ATR
    for i in range(max(1, n - fresh), n):
        matched = abs(float(l.iloc[i]) - float(l.iloc[i - 1])) <= tol
        if matched and in_downtrend(c, i - 1):
            shared = round(min(float(l.iloc[i]), float(l.iloc[i - 1])), 2)
            found.append({
                "type": "tweezer", "label": "Tweezer bottom",
                "date": df.index[i].strftime("%Y-%m-%d"), "ageBars": n - 1 - i,
                "entry": round(float(h.iloc[i]), 2),
                "stop": shared,
                "detail": (f"Two sessions bottomed together at {shared} "
                           f"after a downtrend."),
            })
    return found


def size_setup(setup: dict, price: float) -> dict:
    """Attach the risk maths to one setup, using ITS own entry and stop."""
    entry, stop = setup.get("entry"), setup.get("stop")
    rps = (entry - stop) if (entry is not None and stop is not None) else None
    if rps is not None and rps <= 0:
        rps = None
    rpt = TOTAL_RISK / RPT_DIVISOR
    qty = int(rpt // rps) if rps else None
    setup["riskPerShare"] = round(rps, 2) if rps else None
    setup["qty"] = qty
    setup["deploy"] = round(qty * entry, 0) if (qty and entry) else None
    setup["atRisk"] = round(qty * rps, 0) if (qty and rps) else None
    setup["targets"] = ([round(entry + rps * m, 2) for m in RR_TARGETS]
                        if (entry and rps) else [])
    # A single share can risk more than the whole risk-per-trade budget --
    # common on a 1.5-lakh-rupee share. That is not a buy at this risk level,
    # so say so instead of quietly reporting a quantity of zero.
    setup["tooSmall"] = bool(rps and qty == 0)
    setup["capitalForOne"] = round(rps * RPT_DIVISOR, 0) if setup["tooSmall"] else None
    setup["state"] = ("triggered" if (price is not None and entry is not None
                                      and price >= entry) else "armed")
    setup["distancePct"] = (round((entry / price - 1) * 100, 2)
                            if (entry and price) else None)
    return setup


def comparative_strength(close: pd.Series, bench: pd.Series, period: int = CRS_PERIOD):
    """Comparative Relative Strength against the Nifty 50.

    The ratio of the stock to the index, measured against its own moving
    average: above the line the stock is outperforming the index, below it is
    lagging. This is the 'is it worth buying at all' filter from the notes.
    """
    if bench is None or bench.empty:
        return None, None
    joined = pd.concat([close.rename("s"), bench.rename("b")], axis=1).dropna()
    if len(joined) < period + 5:
        return None, None
    ratio = joined["s"] / joined["b"]
    avg = ratio.rolling(period).mean()
    last, last_avg = float(ratio.iloc[-1]), float(avg.iloc[-1])
    if not np.isfinite(last_avg) or last_avg == 0:
        return None, None
    return (last >= last_avg), round((last / last_avg - 1) * 100, 2)


def resample_tf(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Daily bars -> the chosen candle size. Open is the week's first trade,
    Close its last, High/Low the extremes, Volume the sum."""
    rule = TF_RULE.get(tf)
    if not rule:
        return df
    out = df.resample(rule).agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    })
    return out.dropna(subset=["Close"])


def bar_is_complete(daily: pd.DataFrame, resampled: pd.DataFrame, tf: str) -> bool:
    """Is the newest bar finished, or still forming?

    Mid-week the last weekly bar holds Monday to today and is labelled with the
    coming Friday. A tweezer on a bar that has three days left to run can still
    disappear, so the page has to be able to say so."""
    if tf == BASE_TF or resampled.empty or daily.empty:
        return True
    return resampled.index[-1].date() <= daily.index[-1].date()


def analyse(symbol: str, name: str, meta: dict, df: pd.DataFrame,
            bench: pd.Series = None) -> dict:
    df = df.dropna(subset=["Close"]).copy()
    industry = (meta or {}).get("industry", "Unclassified")
    tags = (meta or {}).get("universes", [])
    if len(df) < max(RSI_PERIOD, ATR_PERIOD) + SWING_BARS * 2 + 10:
        return {"symbol": symbol, "name": name, "industry": industry,
                "universes": tags, "status": "nodata",
                "reason": "not enough history"}

    df["RSI"] = wilder_rsi(df["Close"])
    df["ATR"] = wilder_atr(df)

    close = df["Close"]
    low = df["Low"]
    rsi = df["RSI"]

    last_price = float(close.iloc[-1])
    prev_price = float(close.iloc[-2]) if len(close) > 1 else last_price
    last_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else None
    last_atr = float(df["ATR"].iloc[-1]) if not np.isnan(df["ATR"].iloc[-1]) else None
    as_of = df.index[-1].strftime("%Y-%m-%d")

    # --- step 1: the divergence -------------------------------------------
    window_start = max(0, len(df) - DIV_WINDOW)
    lows_idx = [i for i in pivot_lows(low) if i >= window_start and not np.isnan(rsi.iloc[i])]

    divergence = False
    marks = None
    mark_idx = None
    if len(lows_idx) >= 2:
        # walk the most recent bottom back against earlier ones; the first
        # pair that satisfies "lower bottom on price, higher bottom on RSI"
        # is the live divergence.
        b = lows_idx[-1]
        for a in reversed(lows_idx[:-1]):
            price_lower = float(low.iloc[b]) < float(low.iloc[a])
            rsi_higher = float(rsi.iloc[b]) > float(rsi.iloc[a])
            if price_lower and rsi_higher:
                divergence = True
                marks = {
                    "dateA": df.index[a].strftime("%Y-%m-%d"),
                    "lowA": round(float(low.iloc[a]), 2),
                    "rsiA": round(float(rsi.iloc[a]), 1),
                    "dateB": df.index[b].strftime("%Y-%m-%d"),
                    "lowB": round(float(low.iloc[b]), 2),
                    "rsiB": round(float(rsi.iloc[b]), 1),
                }
                mark_idx = (a, b)
                break
            if price_lower and not rsi_higher:
                break  # a lower bottom with a lower RSI kills the divergence

    # --- step 2: the resistance that has to break --------------------------
    # Levels come from clustered pivots, so a level is one price MANY touches
    # agree on rather than a single swing high.
    levels = touch_levels(df, last_atr)

    # The divergence's resistance is anchored to the SECOND BOTTOM, not to
    # today's price: it is the level that formed above the bottoms and has to
    # break. Anchoring it to today's price would silently move the goalposts
    # up every time price cleared a level, hiding the breakout that just
    # happened.
    res_level = None
    if divergence and mark_idx:
        # "the previous major resistance" is the peak the price made between
        # the two bottoms. Snap that peak to a clustered level when one sits
        # near it, so the trigger is a price many touches agree on.
        a_i, b_i = mark_idx
        span = df["High"].iloc[a_i : b_i + 1]
        if len(span):
            peak = float(span.max())
            tol = (last_atr if last_atr else peak * 0.01) * LEVEL_SNAP_ATR
            near = [l for l in levels if abs(l["price"] - peak) <= tol]
            res_level = (max(near, key=lambda l: l["touches"]) if near
                         else {"price": round(peak, 2), "touches": 1})
    if res_level is None:
        res_level = pick_resistance(levels, last_price)
    resistance = res_level["price"] if res_level else None
    res_touches = res_level["touches"] if res_level else None

    if divergence and mark_idx and resistance is None:
        a_i, b_i = mark_idx
        between = df["High"].iloc[a_i : b_i + 1]
        if len(between):
            resistance = round(float(between.max()), 2)
    if resistance is None:
        tail = df["High"].iloc[-DIV_WINDOW:]
        resistance = round(float(tail.max()), 2) if len(tail) else None

    broke = bool(resistance is not None and last_price >= resistance)

    # --- steps 3 and 4: ATR stop and RPT sizing ----------------------------
    entry = resistance if resistance is not None else last_price
    risk_per_share = last_atr * ATR_MULT if last_atr else None
    stop = entry - risk_per_share if (entry and risk_per_share) else None
    rpt = TOTAL_RISK / RPT_DIVISOR
    qty = int(rpt // risk_per_share) if risk_per_share and risk_per_share > 0 else None

    # Targets at the risk/reward multiples the notes use (1:2, 1:3).
    targets = ([round(entry + risk_per_share * m, 2) for m in RR_TARGETS]
               if (entry and risk_per_share) else [])

    # --- every setup this stock is showing right now -----------------------
    setups = []
    if divergence and "divergence" in SETUPS:
        d = {"type": "divergence", "label": "RSI divergence",
             "date": marks["dateB"] if marks else as_of, "ageBars": None,
             "entry": round(entry, 2) if entry else None,
             "stop": round(stop, 2) if stop else None,
             "detail": (f"Price {marks['lowA']} → {marks['lowB']} (lower bottom), "
                        f"RSI {marks['rsiA']} → {marks['rsiB']} (higher bottom)."
                        if marks else "Bullish RSI divergence."),
             "marks": marks}
        setups.append(size_setup(d, last_price))
    if "engulfing" in SETUPS:
        for s in find_engulfing(df):
            setups.append(size_setup(s, last_price))
    if "tweezer" in SETUPS:
        for s in find_tweezer(df, last_atr):
            setups.append(size_setup(s, last_price))

    # one setup leads the row; the rest stay visible in the drawer
    primary = None
    sized = [s for s in setups if s.get("riskPerShare")]
    if sized:
        # prefer setups that size to at least one share, then triggered ones,
        # then apply the stop-width rule
        buyable = [s for s in sized if s.get("qty")] or sized
        fired = [s for s in buyable if s["state"] == "triggered"] or buyable
        primary = (min(fired, key=lambda s: s["riskPerShare"])
                   if PRIMARY_RULE == "tightest"
                   else max(fired, key=lambda s: s["riskPerShare"]))
    elif setups:
        primary = setups[0]

    if primary:
        entry = primary.get("entry", entry)
        stop = primary.get("stop", stop)
        risk_per_share = primary.get("riskPerShare", risk_per_share)
        qty = primary.get("qty", qty)
        targets = primary.get("targets", targets)

    # Selection measures: velocity, liquidity, relative strength.
    atr_pct = round(last_atr / last_price * 100, 2) if (last_atr and last_price) else None
    turnover = None
    if "Volume" in df.columns:
        tv = (df["Close"] * df["Volume"]).tail(TURNOVER_DAYS).mean()
        if np.isfinite(tv):
            turnover = round(float(tv), 0)
    outperforming, rs_gap = comparative_strength(close, bench)

    states = {s["state"] for s in setups}
    if "triggered" in states:
        status = "triggered"
    elif "armed" in states:
        status = "armed"
    elif last_rsi is not None and last_rsi < 30:
        status = "oversold"
    elif last_rsi is not None and last_rsi > 70:
        status = "overbought"
    else:
        status = "watching"

    return {
        "symbol": symbol,
        "name": name,
        "industry": industry,
        "universes": tags,
        "status": status,
        "atrPct": atr_pct,
        "turnover": turnover,
        "outperforming": outperforming,
        "rsGap": rs_gap,
        "targets": targets,
        "setups": setups,
        "setupTypes": sorted({s["type"] for s in setups}),
        "primary": primary["type"] if primary else None,
        "resTouches": res_touches,
        "levels": levels[:6],
        "asOf": as_of,
        "price": round(last_price, 2),
        "changePct": round((last_price / prev_price - 1) * 100, 2) if prev_price else None,
        "rsi": round(last_rsi, 1) if last_rsi is not None else None,
        "atr": round(last_atr, 2) if last_atr is not None else None,
        "resistance": round(resistance, 2) if resistance is not None else None,
        "distancePct": round((resistance / last_price - 1) * 100, 2)
                        if resistance and last_price else None,
        "divergence": divergence,
        "marks": marks,
        "broke": broke,
        "entry": round(entry, 2) if entry else None,
        "stop": round(stop, 2) if stop else None,
        "riskPerShare": round(risk_per_share, 2) if risk_per_share else None,
        "qty": qty,
        "deploy": round(qty * entry, 0) if (qty and entry) else None,
        "low52": round(float(df["Low"].iloc[-250:].min()), 2),
        "high52": round(float(df["High"].iloc[-250:].max()), 2),
    }


# ----------------------------------------------------------------------------
# Fetch
# ----------------------------------------------------------------------------
# Fields that describe the STOCK rather than the chart you are looking at, so
# they are stored once instead of once per timeframe.
SHARED_FIELDS = ("symbol", "name", "industry", "universes")
STOCK_FIELDS  = ("turnover", "low52", "high52")


def build_record(symbol: str, name: str, meta: dict, daily: pd.DataFrame,
                 benches: dict) -> dict:
    """One stock across every timeframe, as a single record."""
    results, complete = {}, {}
    for tf in TIMEFRAMES:
        frame = resample_tf(daily, tf)
        complete[tf] = bar_is_complete(daily, frame, tf)
        try:
            results[tf] = analyse(symbol, name, meta, frame, benches.get(tf))
        except Exception as exc:  # noqa: BLE001
            results[tf] = {
                "symbol": symbol, "name": name,
                "industry": (meta or {}).get("industry", "Unclassified"),
                "universes": (meta or {}).get("universes", []),
                "status": "nodata", "reason": str(exc),
            }

    base = results.get(BASE_TF) or next(iter(results.values()))
    rec = {k: base.get(k) for k in SHARED_FIELDS}
    # Liquidity and the 52-week range come from the daily frame whichever
    # timeframe you are viewing: weekly volume sums read about five times
    # daily, and a "52-week high" off 250 weekly bars would quietly mean five
    # years. Neither is a property of the candle size.
    for k in STOCK_FIELDS:
        rec[k] = base.get(k)

    rec["tf"] = {}
    last_daily = daily.index[-1].strftime("%Y-%m-%d") if len(daily) else None
    for tf, r in results.items():
        slim = {k: v for k, v in r.items()
                if k not in SHARED_FIELDS and k not in STOCK_FIELDS}
        slim["barComplete"] = complete[tf]
        if not complete[tf]:
            # Resampling labels a week by its Friday, so mid-week that label is
            # a date that has not happened yet. Report the real last trading
            # day as "as of" and keep the Friday separately as the week's end.
            slim["barEnds"] = slim.get("asOf")
            slim["asOf"] = last_daily
        rec["tf"][tf] = slim

    # A setup that prints on more than one candle size is the stronger read.
    seen = {}
    for tf, r in results.items():
        for t in (r.get("setupTypes") or []):
            seen.setdefault(t, []).append(tf)
    rec["agree"] = {t: tfs for t, tfs in seen.items() if len(tfs) > 1}

    # Top-level status drives the row order only; the page overlays the status
    # of whichever timeframe you are actually looking at.
    rec["status"] = base.get("status", "nodata")
    rec["asOf"] = base.get("asOf")
    return rec


def fetch_benchmark() -> pd.Series:
    """Daily closes for the Nifty 50, the comparative-strength trend line."""
    try:
        data = yf.download(BENCHMARK, period=HISTORY, interval="1d",
                           auto_adjust=False, progress=False, timeout=30)
        if data is None or data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data = data[BENCHMARK] if BENCHMARK in data.columns.levels[0] else data.droplevel(1, axis=1)
        return data["Close"].dropna()
    except Exception as exc:  # noqa: BLE001
        print(f"benchmark {BENCHMARK} unavailable ({exc}); "
              f"relative strength will be blank", flush=True)
        return None


def _download_batch(pairs: list, frames: dict) -> list:
    """Fetch one batch. Returns the symbols it could not get."""
    missing = []
    ytickers = [y for _, y in pairs]
    data = None
    for attempt in range(3):
        try:
            data = yf.download(
                ytickers, period=HISTORY, interval="1d", group_by="ticker",
                auto_adjust=False, threads=True, progress=False, timeout=30,
            )
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  retry {attempt + 1}: {exc}", flush=True)
            time.sleep(5 * (attempt + 1))

    if data is None or data.empty:
        return [sym for sym, _ in pairs]

    for sym, yt in pairs:
        try:
            # With a single ticker yfinance returns plain columns, not a
            # MultiIndex -- which is exactly the shape the rescue pass below
            # produces, so handle both.
            if isinstance(data.columns, pd.MultiIndex):
                sub = data[yt] if yt in data.columns.get_level_values(0) else None
            else:
                sub = data
            if sub is None:
                missing.append(sym)
                continue
            sub = sub.dropna(how="all")
            if sub.empty or sub["Close"].dropna().empty:
                missing.append(sym)
                continue
            frames[sym] = sub
        except Exception:  # noqa: BLE001
            missing.append(sym)
    return missing


def fetch_frames(symbols: list) -> tuple:
    """Two passes. The first goes in chunks, which is fast. The second retries
    whatever the first pass lost, a few at a time -- because one refused
    request should not cost you 25 stocks out of 750."""
    frames, missing = {}, []
    pairs = [(sym, yahoo_symbol(sym)) for sym in symbols]
    lookup = dict(pairs)

    for start in range(0, len(pairs), CHUNK):
        batch = pairs[start : start + CHUNK]
        print(f"fetching {start + 1}-{start + len(batch)} of {len(pairs)}", flush=True)
        missing.extend(_download_batch(batch, frames))
        time.sleep(CHUNK_PAUSE)

    if missing:
        print(f"rescue pass: retrying {len(missing)} stock(s) in small batches",
              flush=True)
        retry, missing = sorted(set(missing)), []
        for start in range(0, len(retry), 5):
            small = [(s, lookup[s]) for s in retry[start : start + 5]]
            missing.extend(_download_batch(small, frames))
            time.sleep(CHUNK_PAUSE)
        if missing:
            print(f"  still no data for {len(missing)}: "
                  f"{', '.join(sorted(missing)[:12])}"
                  f"{' ...' if len(missing) > 12 else ''}", flush=True)

    return frames, missing


def main() -> int:
    print(f"building watchlist from {len(UNIVERSES)} universes", flush=True)
    watchlist, sources = build_watchlist(UNIVERSES)
    if not watchlist:
        print("No universe could be loaded -- leaving data.json untouched.", file=sys.stderr)
        return 1
    meta = {row["symbol"]: row for row in watchlist}

    for key in UNIVERSES:
        s = sources.get(key, {})
        print(f"  {s.get('label', key):<20} {s.get('count', 0):>4} stocks"
              f"  ({s.get('source', '?')})", flush=True)
    empty = [k for k in UNIVERSES if not sources.get(k, {}).get("count")]
    if empty:
        print(f"  !! no constituents for: {', '.join(empty)}", flush=True)
    print(f"{len(watchlist)} unique stocks to fetch "
          f"(overlap between indexes is downloaded once)", flush=True)

    bench = fetch_benchmark()
    # The index has to be measured on the same candle size as the stock, or
    # comparative strength compares five months against two years.
    benches = {BASE_TF: bench}
    for tf in TIMEFRAMES:
        if tf == BASE_TF:
            continue
        rule = TF_RULE.get(tf)
        benches[tf] = (bench.resample(rule).last().dropna()
                       if (bench is not None and not bench.empty and rule) else None)

    frames, missing = fetch_frames([row["symbol"] for row in watchlist])
    if not frames:
        print("No price data came back at all -- leaving data.json untouched.", file=sys.stderr)
        return 1

    rows = []
    for row in watchlist:
        sym = row["symbol"]
        if sym not in frames:
            continue
        try:
            rows.append(build_record(sym, row["name"], row, frames[sym], benches))
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym}: {exc}", flush=True)
            missing.append(sym)

    rank = {"triggered": 0, "armed": 1, "oversold": 2, "overbought": 3,
            "watching": 4, "nodata": 5}
    rows.sort(key=lambda r: (rank.get(r["status"], 9), -(r.get("turnover") or 0)))

    as_of = max((r.get("asOf") for r in rows if r.get("asOf")), default=None)
    payload = {
        "generatedAt": datetime.now(IST).isoformat(timespec="seconds"),
        "asOf": as_of,
        "settings": {
            "rsiPeriod": RSI_PERIOD, "atrPeriod": ATR_PERIOD,
            "atrMult": ATR_MULT, "totalRisk": TOTAL_RISK,
            "rptDivisor": RPT_DIVISOR, "swingBars": SWING_BARS,
            "divWindow": DIV_WINDOW, "rrTargets": RR_TARGETS,
            "setups": SETUPS, "freshBars": FRESH_BARS,
            "levelMinTouches": LEVEL_MIN_TOUCHES, "primaryRule": PRIMARY_RULE,
            "crsPeriod": CRS_PERIOD, "atrPctFloor": ATRPCT_FLOOR,
            "benchmark": "Nifty 50", "hasBenchmark": bench is not None,
        },
        "baseTimeframe": BASE_TF,
        "timeframes": [
            {"key": tf, "label": TF_LABEL.get(tf, tf),
             # Every stock shares the same week boundary, so the first row that
             # has an opinion answers for all of them.
             "barComplete": next((r["tf"][tf].get("barComplete", True)
                                  for r in rows if tf in r.get("tf", {})), True),
             "lastBar": next((r["tf"][tf].get("asOf")
                              for r in rows if r.get("tf", {}).get(tf, {}).get("asOf")), None),
             "barEnds": next((r["tf"][tf].get("barEnds")
                              for r in rows if r.get("tf", {}).get(tf, {}).get("barEnds")), None)}
            for tf in TIMEFRAMES
        ],
        "universeGroups": [{"key": g, "label": lbl} for g, lbl in UNIVERSE_GROUPS],
        "universes": [
            {"key": k, "label": UNIVERSE_CATALOGUE.get(k, {}).get("label", k),
             "group": UNIVERSE_CATALOGUE.get(k, {}).get("group", "other"),
             "count": sources.get(k, {}).get("count", 0),
             "source": sources.get(k, {}).get("source", "")}
            for k in UNIVERSES
        ],
        "sectors": sorted({r.get("industry") for r in rows
                           if r.get("industry") and r["industry"] != "Unclassified"}),
        "missing": sorted(set(missing)),
        "rows": rows,
    }
    # Compact separators, not indent=1. At ~750 stocks the pretty version is
    # about 1.2 MB and this one about 800 KB, for identical content -- and the
    # file is re-committed every trading day, so the saving compounds.
    with open("data.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))

    # Which index to name for a stock that sits in eight of them. The smallest
    # one it belongs to is the most informative: "Nifty 50" tells you more than
    # "Nifty 500", and "Nifty Bank" more than either.
    sizes = {k: (sources.get(k, {}).get("count") or 10 ** 6) for k in UNIVERSES}

    def tag_of(r) -> str:
        tags = [t for t in (r.get("universes") or []) if t in sizes]
        if not tags:
            return ""
        best = min(tags, key=lambda t: sizes[t])
        return UNIVERSE_CATALOGUE.get(best, {}).get("label", best)

    def view(r, tf) -> dict:
        """The stock as it looks on one candle size, shared fields included."""
        merged = {k: r.get(k) for k in SHARED_FIELDS}
        merged.update({k: r.get(k) for k in STOCK_FIELDS})
        merged.update(r.get("tf", {}).get(tf, {}))
        return merged

    def agree_note(r, tf) -> str:
        """'also weekly' -- the same pattern on another candle size."""
        others = sorted({o for tfs in (r.get("agree") or {}).values()
                         for o in tfs if o != tf})
        return f", also {'/'.join(TF_LABEL.get(o, o).lower() for o in others)}" if others else ""

    lines = [f"# Midcap Reversal Desk -- {as_of}", ""]
    counts = {"triggered": 0, "armed": 0}

    # One section per candle size. A line that does not say which chart it came
    # from is useless: a daily tweezer and a weekly tweezer are different trades.
    for tf in TIMEFRAMES:
        label = TF_LABEL.get(tf, tf)
        views = [view(r, tf) for r in rows]
        fired = [(r, v) for r, v in zip(rows, views) if v.get("status") == "triggered"]
        arm   = [(r, v) for r, v in zip(rows, views) if v.get("status") == "armed"]
        counts["triggered"] += len(fired)
        counts["armed"] += len(arm)

        meta_tf = next((t for t in payload["timeframes"] if t["key"] == tf), {})
        forming = "" if meta_tf.get("barComplete", True) else \
            f" — the current {label.lower()} candle is still forming, so these can change"

        if not fired and not arm:
            continue
        lines.append(f"# {label} candles{forming}")
        lines.append("")

        if fired:
            lines.append(f"## Triggered on {label.lower()} ({len(fired)})")
            lines.append("Divergence confirmed and price has cleared the resistance.")
            lines.append("")
            for r, v in fired[:ALERT_MAX]:
                lines.append(
                    f"- **{v['symbol']}** ({v['name']}, {tag_of(v)}{agree_note(r, tf)}) "
                    f"at Rs {v['price']:,} -- broke {v['resistance']:,}, "
                    f"RSI {v['rsi']}, stop {v['stop']:,}, qty {v['qty']}"
                )
            if len(fired) > ALERT_MAX:
                lines.append(f"- _...and {len(fired) - ALERT_MAX} more "
                             f"-- see the dashboard._")
            lines.append("")

        near = sorted(
            ((r, v) for r, v in arm
             if v.get("distancePct") is not None and v["distancePct"] <= NEAR_PCT),
            key=lambda rv: rv[1]["distancePct"],
        )
        if near:
            lines.append(f"## Armed on {label.lower()}, within {NEAR_PCT:.0f}% "
                         f"of the trigger ({len(near)})")
            lines.append("Closest to the resistance break first.")
            lines.append("")
            for r, v in near[:ALERT_MAX]:
                lines.append(
                    f"- **{v['symbol']}** ({tag_of(v)}{agree_note(r, tf)}) "
                    f"at Rs {v['price']:,} -- needs {v['distancePct']}% to clear "
                    f"{v['resistance']:,} (RSI {v['rsi']})"
                )
            if len(near) > ALERT_MAX:
                lines.append(f"- _...and {len(near) - ALERT_MAX} more within "
                             f"{NEAR_PCT:.0f}%._")
            lines.append("")

        far = len(arm) - len(near)
        if far > 0:
            lines.append(f"_{far} more armed on {label.lower()} sit further than "
                         f"{NEAR_PCT:.0f}% below their resistance._")
            lines.append("")

    triggered = [r for r in rows if r["status"] == "triggered"]
    armed = [r for r in rows if r["status"] == "armed"]
    if not counts["triggered"] and not counts["armed"]:
        lines.append("No setups on any timeframe today.")
    if missing:
        miss = sorted(set(missing))
        shown = ", ".join(miss[:20])
        more = f" and {len(miss) - 20} more" if len(miss) > 20 else ""
        lines.append("")
        lines.append(f"_No data for: {shown}{more}_")

    with open("alerts.md", "w", encoding="utf-8") as fh:
        # Trailing newline matters: the workflow feeds this file into a
        # GITHUB_OUTPUT heredoc, and without it the closing delimiter lands on
        # the same line as the last sentence and is never recognised.
        fh.write("\n".join(lines) + "\n")

    print(f"scanned {len(rows)} stocks | missing {len(set(missing))}")
    for tf in TIMEFRAMES:
        t = sum(1 for r in rows if r.get("tf", {}).get(tf, {}).get("status") == "triggered")
        a = sum(1 for r in rows if r.get("tf", {}).get(tf, {}).get("status") == "armed")
        meta_tf = next((x for x in payload["timeframes"] if x["key"] == tf), {})
        forming = "" if meta_tf.get("barComplete", True) else "  (candle still forming)"
        print(f"  {TF_LABEL.get(tf, tf):<8} triggered {t:>3} | armed {a:>3}{forming}")
    both = sum(1 for r in rows if r.get("agree"))
    print(f"  {both} stock(s) show the same setup on more than one timeframe")

    # tell the workflow whether an e-mail is worth sending
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as fh:
            fh.write(f"has_alerts={'true' if (counts['triggered'] or counts['armed']) else 'false'}\n")
            fh.write(f"subject=Reversal desk: {counts['triggered']} triggered, "
                     f"{counts['armed']} armed across "
                     f"{len(TIMEFRAMES)} timeframes ({as_of})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
