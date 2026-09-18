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

# --- risk : reward ----------------------------------------------------------
# Your rule: never take a trade whose reward is not worth the risk. 1:1 is the
# floor nobody should trade below -- you would need to be right more than half
# the time just to break even, before costs. 2:1 is the working minimum.
#
# Reward is NOT assumed. It is the distance to somewhere price can actually
# reach: the pattern's own measured move, or the next clustered resistance,
# whichever is nearer. A setup that cannot clear MIN_RR is shown and flagged,
# not hidden -- you may still want it, and hiding it would hide the near misses.
MIN_RR       = 2.0

# --- what makes a stop a real stop ------------------------------------------
# A pattern's own stop is the candle's low, and when that candle is tiny the
# stop lands a few paise under the entry. That is not a stop: ordinary noise
# takes it out on the first tick. Worse, risk-per-share near zero makes the
# quantity explode and the risk:reward ratio go to infinity, so screening for
# a HIGH ratio would surface exactly the worst trades. These put a floor on it.
MIN_STOP_ATR = 0.5      # a stop closer to entry than this is not believable
MAX_STOP_ATR = 4.0      # a support level further than this is not worth using
SUPPORT_BUFFER_ATR = 0.25   # the stop sits this far BELOW the support level

# --- setups and levels ------------------------------------------------------
# Bullish ones are entries. "doubletop" and "hs" are TOPPING patterns: they are
# carried as warnings on the row and never given an entry, stop or quantity.
SETUPS       = ["divergence", "engulfing", "tweezer",
                "doublebottom", "invhs"]
WARNINGS     = ["doubletop", "hs"]

# double bottom / double top
DBL_WINDOW   = 180      # bars searched for the pattern
DBL_TOL_ATR  = 0.6      # how equal the two feet must be, in ATR
DBL_MIN_GAP  = 6        # bars between the feet: closer than this is one dip
DBL_MAX_GAP  = 90       # further than this and they are unrelated lows
DBL_MIN_DEPTH_ATR = 2.0 # the peak between must stand this far clear, in ATR
DBL_MAX_AGE  = 40       # the second foot must be this recent. Without it a W
                        # from nine months ago is still reported as live, and
                        # nearly every stock ends up carrying one.
DBL_LATE_FRAC = 0.5     # if price is already this far through the measured
                        # move, the trade is gone -- do not offer it

# head and shoulders, both directions
HS_WINDOW    = 200
HS_MIN_GAP   = 5        # bars between shoulder and head
HS_MAX_GAP   = 60
HS_HEAD_ATR  = 1.0      # how far the head must stand clear of the shoulders
HS_SHOULDER_ATR = 2.0   # how unequal the two shoulders may be
HS_MAX_AGE   = 40       # right shoulder must be this recent

# volume confirmation
VOL_LOOKBACK = 20       # bars averaged for "normal" volume
VOL_CONFIRM_MULT = 1.2  # signal bar must beat the average by this much

# --- backtest ---------------------------------------------------------------
BACKTEST      = True    # set False to skip it and shorten the run
BT_SAMPLE     = 200     # stocks sampled. Five years x 200 names already gives
                        # thousands of occurrences per pattern; scanning all
                        # 750 would cost minutes to change a number in the
                        # third decimal place.
BT_TARGET_R   = 2.0     # "success" = reached this multiple of risk
BT_TRIGGER_BARS = 20    # bars allowed for the entry to trigger at all
BT_HOLD_BARS  = 60      # bars allowed to reach the target before giving up

# --- recent bars, for the trade journal -------------------------------------
# The Dashboard needs to know whether a stop or a target was actually TOUCHED
# on some day since you entered, not merely whether today's close is past it --
# a stop hit on Tuesday and recovered by Friday is still a stop hit. That needs
# the daily highs and lows, so the scan publishes the last few weeks of them.
# The dates are shared across every stock, which is what keeps this affordable:
# per stock it is three numbers a day, not a date string as well.
BAR_HISTORY  = 25      # five trading weeks: enough to review recent trades
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
WATCHLIST_FILE = "watchlist.txt"   # one NSE symbol per line, '#' for comments
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


# ----------------------------------------------------------------------------
# Structural patterns: double bottom / top, head and shoulders both ways.
#
# These differ from the candlestick setups in one important way -- they carry a
# MEASURED TARGET. The distance from the pattern's extreme to its neckline,
# projected from the neckline, is where the move is conventionally expected to
# reach. That is a real target derived from the chart, not a multiple of risk.
# ----------------------------------------------------------------------------
def _similar(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def find_double_bottom(df: pd.DataFrame, atr: float) -> list:
    """Two lows at the same level with a peak between: a W.

    The peak is the neckline. Nothing is a buy until price clears it, the stop
    goes below the lower foot, and the target is the neckline plus the depth of
    the pattern.
    """
    low, high, close = df["Low"], df["High"], df["Close"]
    n = len(df)
    # Enough bars for the pattern itself -- two pivots plus the gap between
    # them -- rather than an arbitrary round number.
    if n < DBL_MIN_GAP + SWING_BARS * 2 + 4:
        return []
    start = max(0, n - DBL_WINDOW)
    lows = [i for i in pivot_lows(low) if i >= start]
    if len(lows) < 2:
        return []

    last_price = float(close.iloc[-1])
    unit = atr if (atr and np.isfinite(atr)) else last_price * 0.01
    tol = unit * DBL_TOL_ATR
    found = []

    # newest pair first: a recent W matters more than one from two years ago
    for x in range(len(lows) - 1, 0, -1):
        for y in range(x - 1, -1, -1):
            a, b = lows[y], lows[x]
            gap = b - a
            if gap < DBL_MIN_GAP or gap > DBL_MAX_GAP:
                continue
            la, lb = float(low.iloc[a]), float(low.iloc[b])
            if not _similar(la, lb, tol):
                continue
            seg = high.iloc[a:b + 1]
            neck = float(seg.max())
            foot = min(la, lb)
            depth = neck - foot
            # A W that is barely a W is noise, not a pattern.
            if depth < unit * DBL_MIN_DEPTH_ATR:
                continue
            if n - 1 - b > DBL_MAX_AGE:
                continue
            # Already halfway to the measured move? The trade has left.
            if last_price > neck + depth * DBL_LATE_FRAC:
                continue
            found.append({
                "type": "doublebottom", "label": "Double bottom",
                "direction": "long",
                "date": df.index[b].strftime("%Y-%m-%d"),
                "ageBars": n - 1 - b,
                "entry": round(neck, 2),
                "stop": round(foot - unit * 0.25, 2),
                "measured": round(neck + depth, 2),
                "detail": (f"Two feet at {round(la, 2)} and {round(lb, 2)}, "
                           f"neckline {round(neck, 2)}. Measured move "
                           f"{round(neck + depth, 2)}."),
            })
            break                      # one pairing per right-hand foot
        if found:
            break                      # only the most recent W
    return found


def find_double_top(df: pd.DataFrame, atr: float) -> list:
    """The bearish mirror: two highs at one level with a trough between.

    This is a WARNING, never a buy. It says the level above has been rejected
    twice, so it is marked on the row and left out of the sizing entirely.
    """
    low, high, close = df["Low"], df["High"], df["Close"]
    n = len(df)
    if n < DBL_MIN_GAP + SWING_BARS * 2 + 4:
        return []
    start = max(0, n - DBL_WINDOW)
    highs = [i for i in pivot_highs(high) if i >= start]
    if len(highs) < 2:
        return []

    last_price = float(close.iloc[-1])
    unit = atr if (atr and np.isfinite(atr)) else last_price * 0.01
    tol = unit * DBL_TOL_ATR

    for x in range(len(highs) - 1, 0, -1):
        for y in range(x - 1, -1, -1):
            a, b = highs[y], highs[x]
            gap = b - a
            if gap < DBL_MIN_GAP or gap > DBL_MAX_GAP:
                continue
            ha, hb = float(high.iloc[a]), float(high.iloc[b])
            if not _similar(ha, hb, tol):
                continue
            neck = float(low.iloc[a:b + 1].min())
            crest = max(ha, hb)
            height = crest - neck
            if height < unit * DBL_MIN_DEPTH_ATR:
                continue
            if n - 1 - b > DBL_MAX_AGE:
                continue
            return [{
                "type": "doubletop", "label": "Double top",
                "direction": "warn",
                "date": df.index[b].strftime("%Y-%m-%d"),
                "ageBars": n - 1 - b,
                "level": round(neck, 2),
                "broken": bool(last_price < neck),
                "detail": (f"Rejected twice near {round(crest, 2)}. "
                           f"Support to lose is {round(neck, 2)}."),
            }]
    return []


def _shoulders(pivots: list, values, unit: float, invert: bool):
    """Three pivots forming a head with a shoulder either side."""
    for k in range(len(pivots) - 1, 1, -1):
        r = pivots[k]
        for j in range(k - 1, 0, -1):
            h = pivots[j]
            for i in range(j - 1, -1, -1):
                ls = pivots[i]
                if not (HS_MIN_GAP <= h - ls <= HS_MAX_GAP):
                    continue
                if not (HS_MIN_GAP <= r - h <= HS_MAX_GAP):
                    continue
                vl, vh, vr = (float(values.iloc[ls]), float(values.iloc[h]),
                              float(values.iloc[r]))
                head_ok = (vh < vl and vh < vr) if invert else (vh > vl and vh > vr)
                if not head_ok:
                    continue
                # the head has to stand clear of both shoulders
                if min(abs(vh - vl), abs(vh - vr)) < unit * HS_HEAD_ATR:
                    continue
                # and the shoulders should roughly match each other
                if abs(vl - vr) > unit * HS_SHOULDER_ATR:
                    continue
                return ls, h, r
    return None


def find_inverse_hs(df: pd.DataFrame, atr: float) -> list:
    """Inverse head and shoulders -- three lows, the middle one deepest.

    The bullish one. Neckline is the highest point between the shoulders; the
    target is the neckline plus the drop from neckline to head.
    """
    low, high, close = df["Low"], df["High"], df["Close"]
    n = len(df)
    start = max(0, n - HS_WINDOW)
    lows = [i for i in pivot_lows(low) if i >= start]
    if len(lows) < 3:
        return []
    last_price = float(close.iloc[-1])
    unit = atr if (atr and np.isfinite(atr)) else last_price * 0.01

    hit = _shoulders(lows, low, unit, invert=True)
    if not hit:
        return []
    ls, head, rs = hit
    if n - 1 - rs > HS_MAX_AGE:
        return []
    neck = float(high.iloc[ls:rs + 1].max())
    depth = neck - float(low.iloc[head])
    if depth <= 0 or last_price > neck + depth * DBL_LATE_FRAC:
        return []
    return [{
        "type": "invhs", "label": "Inverse head & shoulders",
        "direction": "long",
        "date": df.index[rs].strftime("%Y-%m-%d"),
        "ageBars": n - 1 - rs,
        "entry": round(neck, 2),
        "stop": round(float(low.iloc[rs]) - unit * 0.25, 2),
        "measured": round(neck + depth, 2),
        "detail": (f"Head {round(float(low.iloc[head]), 2)} between shoulders "
                   f"{round(float(low.iloc[ls]), 2)} and "
                   f"{round(float(low.iloc[rs]), 2)}; neckline {round(neck, 2)}."),
    }]


def find_head_shoulders(df: pd.DataFrame, atr: float) -> list:
    """Standard head and shoulders -- three highs, middle one tallest.

    A topping pattern, so a WARNING on the row rather than an entry.
    """
    low, high, close = df["Low"], df["High"], df["Close"]
    n = len(df)
    start = max(0, n - HS_WINDOW)
    highs = [i for i in pivot_highs(high) if i >= start]
    if len(highs) < 3:
        return []
    last_price = float(close.iloc[-1])
    unit = atr if (atr and np.isfinite(atr)) else last_price * 0.01

    hit = _shoulders(highs, high, unit, invert=False)
    if not hit:
        return []
    ls, head, rs = hit
    if n - 1 - rs > HS_MAX_AGE:
        return []
    neck = float(low.iloc[ls:rs + 1].min())
    return [{
        "type": "hs", "label": "Head & shoulders",
        "direction": "warn",
        "date": df.index[rs].strftime("%Y-%m-%d"),
        "ageBars": n - 1 - rs,
        "level": round(neck, 2),
        "broken": bool(last_price < neck),
        "detail": (f"Head {round(float(high.iloc[head]), 2)} between shoulders; "
                   f"neckline {round(neck, 2)} is the line to hold."),
    }]


def volume_state(df: pd.DataFrame, idx: int) -> dict:
    """Did anyone actually show up for this bar?

    A reversal on thin volume is a reversal nobody voted for. This compares the
    signal bar against its own recent average rather than any absolute number,
    so it works the same on a giant and on a microcap.
    """
    if "Volume" not in df.columns or idx is None or idx < 0 or idx >= len(df):
        return {"volume": None, "volumeAvg": None, "volumeConfirmed": None}
    vol = df["Volume"]
    lo = max(0, idx - VOL_LOOKBACK)
    window = vol.iloc[lo:idx]
    if window.empty:
        return {"volume": None, "volumeAvg": None, "volumeConfirmed": None}
    avg = float(window.mean())
    here = float(vol.iloc[idx])
    if not np.isfinite(avg) or avg <= 0 or not np.isfinite(here):
        return {"volume": None, "volumeAvg": None, "volumeConfirmed": None}
    return {
        "volume": round(here, 0),
        "volumeAvg": round(avg, 0),
        "volumeRatio": round(here / avg, 2),
        "volumeConfirmed": bool(here >= avg * VOL_CONFIRM_MULT),
    }


def ground_stop(setup: dict, atr: float, levels: list) -> dict:
    """Put the stop somewhere defensible, and record why it is there.

    Order of preference:
      1. the pattern's own stop -- but only if it is a believable distance away
      2. just below the nearest real SUPPORT level under the entry, because a
         level many touches agree on is where price actually tends to hold
      3. the ATR rule from the notes, when the chart offers no support nearby

    Without step 1's sanity check, a doji-shaped tweezer produces a stop 0.04%
    under the entry, a quantity in the thousands, and a deployment several times
    the account. That is the single worst failure this scan can produce, because
    every downstream number looks superficially fine.
    """
    entry = setup.get("entry")
    if entry is None or entry <= 0:
        return setup
    unit = atr if (atr and np.isfinite(atr) and atr > 0) else entry * 0.01

    stop = setup.get("stop")
    if stop is not None and (entry - stop) >= unit * MIN_STOP_ATR:
        setup["stopSource"] = "pattern"
        return setup

    tight = stop                                   # remember what we rejected
    below = [l for l in (levels or []) if l["price"] < entry * 0.999]
    support = max(below, key=lambda l: l["price"]) if below else None

    if support is not None and (entry - support["price"]) <= unit * MAX_STOP_ATR:
        setup["stop"] = round(support["price"] - unit * SUPPORT_BUFFER_ATR, 2)
        setup["stopSource"] = "support"
        setup["stopTouches"] = support.get("touches")
    else:
        setup["stop"] = round(entry - unit * ATR_MULT, 2)
        setup["stopSource"] = "ATR"

    if tight is not None:
        setup["stopWidened"] = True
        setup["patternStop"] = round(tight, 2)
    return setup


def attach_reward(setup: dict, levels: list, price: float,
                  high52: float = None) -> dict:
    """The reward half of risk:reward, and where it comes from.

    Preference order, because they are not equally trustworthy:
      1. the pattern's own measured move, when it has one
      2. the next clustered resistance above entry -- where price is likely to
         stall whether you like it or not
      3. a plain multiple of risk, when the chart offers nothing
    """
    entry, stop = setup.get("entry"), setup.get("stop")
    rps = setup.get("riskPerShare")
    if not entry or not rps or rps <= 0:
        setup["rr"] = None
        setup["target"] = None
        setup["targetSource"] = None
        setup["poorRR"] = False
        return setup

    target, source = None, None
    measured = setup.get("measured")
    if measured and measured > entry:
        target, source = measured, "measured move"

    above = sorted((l for l in (levels or []) if l["price"] > entry * 1.002),
                   key=lambda l: l["price"])
    # A level price has touched once is barely a level. Prefer the nearest one
    # with real agreement behind it, and only fall back to a single touch when
    # the chart offers nothing better.
    strong = [l for l in above if (l.get("touches") or 1) >= LEVEL_MIN_TOUCHES]
    above = strong or above
    if above:
        wall = above[0]
        # A wall below the measured move caps what is realistically reachable:
        # price has to get through it before the measured move can happen.
        if target is None or wall["price"] < target:
            target = wall["price"]
            n = wall.get("touches") or 1
            source = f"resistance, {n} touch" + ("es" if n != 1 else "")
            setup["targetTouches"] = wall.get("touches")

    if target is None and high52 and high52 > entry * 1.002:
        # No clustered level above, but the 52-week high is a real price that
        # real sellers remember.
        target, source = high52, "52-week high"

    if target is None:
        # Nothing above the entry on the chart. Rather than invent a target out
        # of a risk multiple and print a confident ratio from it, say so.
        setup["target"] = None
        setup["targetSource"] = "no resistance above"
        setup["rr"] = None
        setup["poorRR"] = False
        return setup

    rr = (target - entry) / rps
    setup["target"] = round(target, 2)
    setup["targetSource"] = source
    setup["rr"] = round(rr, 2)
    setup["poorRR"] = bool(rr < MIN_RR)
    return setup


# ----------------------------------------------------------------------------
# Backtest: does each pattern actually earn its place?
#
# For every historical occurrence, wait for the entry to trigger, then see
# whether price reached 2R before it reached the stop. That is the only
# question worth asking of a setup, and the answer is often humbling.
#
# It runs on a SAMPLE of the universe, not all of it. A few hundred stocks over
# five years already gives thousands of occurrences per pattern, and scanning
# every one of 750 would push the daily run past its timeout for a number that
# would not move in the third decimal place.
# ----------------------------------------------------------------------------
def historical_signals(df: pd.DataFrame) -> list:
    """Every occurrence of every bullish setup across the whole series.

    One pass, reusing the pivot lists, rather than re-running the detectors at
    every bar -- which would be a thousand times more work for the same answer.
    """
    n = len(df)
    if n < 60:
        return []
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    atr = wilder_atr(df).to_numpy(dtype=float)
    rsi = wilder_rsi(c).to_numpy(dtype=float)
    lows_arr, highs_arr = l.to_numpy(float), h.to_numpy(float)
    out = []

    def unit(i):
        a = atr[i]
        return a if (a and np.isfinite(a) and a > 0) else float(c.iloc[i]) * 0.01

    # --- candlestick setups: local, so just walk the bars -------------------
    for i in range(TREND_BARS + 1, n):
        if not in_downtrend(c, i - 1):
            continue
        po, pc = float(o.iloc[i - 1]), float(c.iloc[i - 1])
        co, cc = float(o.iloc[i]), float(c.iloc[i])
        if pc < po and cc > co and cc >= po and co <= pc:
            out.append(("engulfing", i, float(h.iloc[i]), float(l.iloc[i])))
        tol = unit(i) * TWEEZER_TOL_ATR
        if abs(lows_arr[i] - lows_arr[i - 1]) <= tol:
            out.append(("tweezer", i, float(h.iloc[i]),
                        min(lows_arr[i], lows_arr[i - 1])))

    plows = pivot_lows(l)
    phighs = pivot_highs(h)

    # --- RSI divergence: lower low on price, higher low on RSI --------------
    for x in range(1, len(plows)):
        b = plows[x]
        for y in range(x - 1, -1, -1):
            a = plows[y]
            if b - a > DIV_WINDOW:
                break
            if lows_arr[b] >= lows_arr[a]:
                continue
            if not (np.isfinite(rsi[a]) and np.isfinite(rsi[b])):
                continue
            if rsi[b] > rsi[a]:
                neck = float(h.iloc[a:b + 1].max())
                out.append(("divergence", b, neck, neck - unit(b) * ATR_MULT))
            break

    # --- double bottom ------------------------------------------------------
    for x in range(1, len(plows)):
        b = plows[x]
        for y in range(x - 1, -1, -1):
            a = plows[y]
            gap = b - a
            if gap > DBL_MAX_GAP:
                break
            if gap < DBL_MIN_GAP:
                continue
            if abs(lows_arr[a] - lows_arr[b]) > unit(b) * DBL_TOL_ATR:
                continue
            neck = float(h.iloc[a:b + 1].max())
            foot = min(lows_arr[a], lows_arr[b])
            if neck - foot < unit(b) * DBL_MIN_DEPTH_ATR:
                continue
            out.append(("doublebottom", b, neck, foot - unit(b) * 0.25))
            break

    # --- inverse head and shoulders ----------------------------------------
    for k in range(2, len(plows)):
        r = plows[k]
        for j in range(k - 1, 0, -1):
            head = plows[j]
            if not (HS_MIN_GAP <= r - head <= HS_MAX_GAP):
                continue
            for i in range(j - 1, -1, -1):
                ls = plows[i]
                if not (HS_MIN_GAP <= head - ls <= HS_MAX_GAP):
                    continue
                u = unit(r)
                vl, vh, vr = lows_arr[ls], lows_arr[head], lows_arr[r]
                if not (vh < vl and vh < vr):
                    continue
                if min(abs(vh - vl), abs(vh - vr)) < u * HS_HEAD_ATR:
                    continue
                if abs(vl - vr) > u * HS_SHOULDER_ATR:
                    continue
                neck = float(h.iloc[ls:r + 1].max())
                out.append(("invhs", r, neck, vr - u * 0.25))
                break
            else:
                continue
            break
    return out


def evaluate_signal(df: pd.DataFrame, i: int, entry: float, stop: float) -> str:
    """Wait for the trigger, then race the stop against the 2R target.

    Returns 'win', 'loss', 'open' (neither inside the horizon) or '' when the
    entry never triggered at all.
    """
    if not (entry and stop) or entry <= stop:
        return ""
    high, low = df["High"].to_numpy(float), df["Low"].to_numpy(float)
    n = len(df)
    fire = None
    for j in range(i + 1, min(n, i + 1 + BT_TRIGGER_BARS)):
        if high[j] >= entry:
            fire = j
            break
    if fire is None:
        return ""
    risk = entry - stop
    target = entry + risk * BT_TARGET_R
    for j in range(fire, min(n, fire + BT_HOLD_BARS)):
        hit_stop = low[j] <= stop
        hit_target = high[j] >= target
        # Same bar touched both: assume the worse outcome rather than
        # flattering the pattern. Without intraday data there is no way to
        # know which came first, and an optimistic guess here would quietly
        # inflate every number on the page.
        if hit_stop:
            return "loss"
        if hit_target:
            return "win"
    return "open"


def backtest_patterns(frames: dict, symbols: list) -> dict:
    """Aggregate hit rates per pattern and timeframe across a sample."""
    stats = {}
    scanned = 0
    for sym in symbols:
        daily = frames.get(sym)
        if daily is None or len(daily) < 120:
            continue
        scanned += 1
        for tf in TIMEFRAMES:
            frame = resample_tf(daily, tf)
            if len(frame) < 80:
                continue
            try:
                signals = historical_signals(frame)
            except Exception:  # noqa: BLE001
                continue
            for kind, i, entry, stop in signals:
                verdict = evaluate_signal(frame, i, entry, stop)
                if not verdict:
                    continue
                key = f"{kind}|{tf}"
                rec = stats.setdefault(key, {"pattern": kind, "timeframe": tf,
                                             "win": 0, "loss": 0, "open": 0})
                rec[verdict] += 1

    out = []
    for rec in stats.values():
        decided = rec["win"] + rec["loss"]
        rec["n"] = decided + rec["open"]
        rec["hitRate"] = round(rec["win"] / decided * 100, 1) if decided else None
        out.append(rec)
    out.sort(key=lambda r: (r["timeframe"], -(r["hitRate"] or 0)))
    return {"sample": scanned, "targetR": BT_TARGET_R,
            "holdBars": BT_HOLD_BARS, "rows": out}


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
    # The 52-week extremes are needed before sizing: they are the last resort
    # for a target when the chart has no clustered level above the entry.
    low_52 = round(float(df["Low"].iloc[-250:].min()), 2)
    high_52 = round(float(df["High"].iloc[-250:].max()), 2)

    raw = []
    if divergence and "divergence" in SETUPS:
        raw.append({"type": "divergence", "label": "RSI divergence",
                    "date": marks["dateB"] if marks else as_of, "ageBars": None,
                    "entry": round(entry, 2) if entry else None,
                    "stop": round(stop, 2) if stop else None,
                    "detail": (f"Price {marks['lowA']} → {marks['lowB']} (lower bottom), "
                               f"RSI {marks['rsiA']} → {marks['rsiB']} (higher bottom)."
                               if marks else "Bullish RSI divergence."),
                    "marks": marks})
    if "engulfing" in SETUPS:
        raw.extend(find_engulfing(df))
    if "tweezer" in SETUPS:
        raw.extend(find_tweezer(df, last_atr))
    if "doublebottom" in SETUPS:
        raw.extend(find_double_bottom(df, last_atr))
    if "invhs" in SETUPS:
        raw.extend(find_inverse_hs(df, last_atr))

    # Order matters. Ground the stop on real support FIRST, because every other
    # number -- quantity, deployment, amount at risk, risk:reward -- is derived
    # from the distance between entry and stop. Size it, then measure the reward
    # against a price the chart can actually reach.
    setups = []
    for s in raw:
        s.setdefault("direction", "long")
        ground_stop(s, last_atr, levels)
        size_setup(s, last_price)
        attach_reward(s, levels, last_price, high_52)
        age = s.get("ageBars")
        sig_idx = (len(df) - 1 - age) if isinstance(age, int) else len(df) - 1
        s.update(volume_state(df, sig_idx))
        setups.append(s)

    # --- topping patterns: warnings, never entries -------------------------
    warnings = []
    if "doubletop" in WARNINGS:
        warnings.extend(find_double_top(df, last_atr))
    if "hs" in WARNINGS:
        warnings.extend(find_head_shoulders(df, last_atr))
    for w in warnings:
        w["direction"] = "warn"

    # one setup leads the row; the rest stay visible in the drawer
    primary = None
    sized = [s for s in setups if s.get("riskPerShare")]
    if sized:
        # prefer setups that size to at least one share, then ones that clear
        # the minimum risk:reward, then triggered ones, then the stop-width rule
        buyable = [s for s in sized if s.get("qty")] or sized
        worth = [s for s in buyable if not s.get("poorRR")] or buyable
        fired = [s for s in worth if s["state"] == "triggered"] or worth
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
        "warnings": warnings,
        "warningTypes": sorted({w["type"] for w in warnings}),
        "rr": (primary or {}).get("rr"),
        "target": (primary or {}).get("target"),
        "targetSource": (primary or {}).get("targetSource"),
        "poorRR": bool((primary or {}).get("poorRR")),
        "stopSource": (primary or {}).get("stopSource"),
        "stopWidened": bool((primary or {}).get("stopWidened")),
        "targetTouches": (primary or {}).get("targetTouches"),
        "volumeConfirmed": (primary or {}).get("volumeConfirmed"),
        "volumeRatio": (primary or {}).get("volumeRatio"),
        "setupTypes": sorted({s["type"] for s in setups}),
        "primary": primary["type"] if primary else None,
        "resTouches": res_touches,
        "levels": levels[:4],
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
        "low52": low_52,
        "high52": high_52,
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

    # Recent daily bars for the journal: [high, low, close] per session,
    # aligned to the payload's shared date axis.
    tail = daily.tail(BAR_HISTORY)
    rec["barDates"] = [d.strftime("%Y-%m-%d") for d in tail.index]
    rec["bars"] = [[round(float(h), 2), round(float(l), 2), round(float(c), 2)]
                   for h, l, c in zip(tail["High"], tail["Low"], tail["Close"])]

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

    # One date axis for everyone. Stocks that missed a session get a null in
    # that slot, so a row's bars always line up with the axis by index.
    axis = sorted({d for r in rows for d in (r.get("barDates") or [])})[-BAR_HISTORY:]
    slot = {d: i for i, d in enumerate(axis)}
    for r in rows:
        dates, bars = r.pop("barDates", None) or [], r.get("bars") or []
        packed = [None] * len(axis)
        for d, bar in zip(dates, bars):
            i = slot.get(d)
            if i is not None:
                packed[i] = bar
        r["bars"] = packed

    backtest = None
    if BACKTEST:
        # Evenly spaced across the watchlist so the sample spans large, mid,
        # small and micro caps rather than whichever names sort first.
        have = [r["symbol"] for r in watchlist if r["symbol"] in frames]
        step = max(1, len(have) // BT_SAMPLE)
        sample = have[::step][:BT_SAMPLE]
        print(f"backtesting {len(sample)} of {len(have)} stocks "
              f"({BT_TARGET_R:.0f}R target, {BT_HOLD_BARS}-bar horizon)", flush=True)
        t0 = time.time()
        backtest = backtest_patterns(frames, sample)
        print(f"  took {time.time() - t0:.0f}s", flush=True)
        for rec in backtest["rows"]:
            rate = f"{rec['hitRate']}%" if rec["hitRate"] is not None else "n/a"
            print(f"  {rec['pattern']:<14} {rec['timeframe']:<7} "
                  f"{rate:>6} of {rec['win'] + rec['loss']:>5} decided", flush=True)

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
            "setups": SETUPS, "warnings": WARNINGS, "freshBars": FRESH_BARS,
            "levelMinTouches": LEVEL_MIN_TOUCHES, "primaryRule": PRIMARY_RULE,
            "crsPeriod": CRS_PERIOD, "atrPctFloor": ATRPCT_FLOOR,
            "benchmark": "Nifty 50", "hasBenchmark": bench is not None,
        },
        "backtest": backtest,
        "minRR": MIN_RR,
        "barDates": axis,
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

    # --- your own watchlist leads the e-mail --------------------------------
    # watchlist.txt is one NSE symbol per line, '#' for comments. The star
    # button on the dashboard has a Copy button that produces exactly this.
    follow = []
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, encoding="utf-8") as fh:
                follow = [ln.strip().upper() for ln in fh
                          if ln.strip() and not ln.strip().startswith("#")]
        except Exception as exc:  # noqa: BLE001
            print(f"could not read {WATCHLIST_FILE} ({exc})", flush=True)
    if follow:
        print(f"watchlist: following {len(follow)} stock(s)", flush=True)

    lines = [f"# Midcap Reversal Desk -- {as_of}", ""]
    counts = {"triggered": 0, "armed": 0}

    if follow:
        want = set(follow)
        mine = []
        for tf in TIMEFRAMES:
            for r in rows:
                if r["symbol"] not in want:
                    continue
                v = view(r, tf)
                if v.get("status") in ("triggered", "armed"):
                    mine.append((tf, r, v))
        lines.append(f"# Your watchlist ({len(mine)} of {len(follow)} set up)")
        lines.append("")
        if mine:
            for tf, r, v in mine:
                rr = f", R:R {v['rr']}:1" if v.get("rr") else ""
                lines.append(
                    f"- **{v['symbol']}** {v['status']} on {TF_LABEL.get(tf, tf).lower()} "
                    f"at Rs {v['price']:,} -- entry {v.get('entry')}, "
                    f"stop {v.get('stop')}{rr}{agree_note(r, tf)}"
                )
        else:
            lines.append("_Nothing on your list is set up today._")
        lines.append("")
        missing_syms = sorted(want - {r["symbol"] for r in rows})
        if missing_syms:
            lines.append(f"_Not in the scanned universe: {', '.join(missing_syms[:15])}_")
            lines.append("")

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
            watch_hit = bool(follow) and any(
                view(r, tf).get("status") in ("triggered", "armed")
                for tf in TIMEFRAMES for r in rows if r["symbol"] in set(follow))
            fh.write(f"has_alerts={'true' if (counts['triggered'] or counts['armed'] or watch_hit) else 'false'}\n")
            fh.write(f"subject=Reversal desk: {counts['triggered']} triggered, "
                     f"{counts['armed']} armed across "
                     f"{len(TIMEFRAMES)} timeframes ({as_of})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
