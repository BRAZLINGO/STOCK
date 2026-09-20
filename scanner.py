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
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import delivery
except Exception as _exc:  # noqa: BLE001
    # delivery.py is an addition, not a dependency. If it did not get uploaded
    # alongside this file the scan should still produce a page rather than
    # dying at the import line with a traceback nobody asked for.
    delivery = None
    print(f"delivery.py not available ({_exc}) -- the scan will run without "
          f"delivery percentages", file=sys.stderr)

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
SETUPS       = ["divergence", "flowdiv", "engulfing", "tweezer",
                "doublebottom", "invhs",
                # continuation and base patterns
                "flag", "pennant", "rectangle", "asctriangle", "symtriangle",
                "rounding", "cuphandle"]
WARNINGS     = ["doubletop", "hs", "desctriangle"]

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

# --- continuation and base patterns -----------------------------------------
# All of these are CONSOLIDATIONS: a move, a pause, and a level that ends the
# pause. Every one of them needs a freshness limit for the same reason the
# double bottom does -- a pause from eight months ago is not a trade today.
CONT_MAX_AGE   = 6      # the consolidation's last bar must be this recent.
                        # These re-qualify every day while they hold, so an
                        # age above zero means the pause has already broken
                        # one way or the other -- a short window is honest.
CONT_LATE_FRAC = 0.5    # price already this far through the measured move
                        # means the move happened without you
CONT_DEDUPE    = 5      # bars between two signals of the same kind
# Patterns whose right-hand edge is a pivot, and which therefore cannot be
# fresher than the pivot confirmation lag.
PIVOT_PATTERNS = ("rectangle", "asctriangle", "symtriangle", "desctriangle")
ROUND_DEDUPE   = 20     # cups are long, so their clusters are wider

# flag and pennant
# Flags run 5 to 15 bars on a daily chart. Past about twenty the sources
# agree it is momentum exhaustion rather than a pause, so 25 came out.
# Five bars is inside the textbook range but below the resolution of the
# shape test: over five bars the intrabar noise is wider than the channel, so
# "parallel" and "converging" are decided by the wiggle rather than the trend.
# That is exactly how a pole with no flag on it got reported as a flag.
FLAG_LENS    = (8, 11, 15)          # candidate lengths of the pause
FLAG_POLES   = (5, 10, 16, 24)      # candidate lengths of the run into it
FLAG_POLE_ATR = 3.5     # how far the pole must travel, in ATR
FLAG_POLE_EFF = 0.70    # and how much of that range was spent going ONE way.
                        # Without this a random walk qualifies: over 24 bars
                        # its own range is already 4-5 ATR wide.
FLAG_MAX_ATR = 3.5      # the pause itself must be tight, in ATR
FLAG_MAX_WIDTH = 0.6    # the pause must be smaller than the run
FLAG_MAX_RETRACE = 0.50 # a flag gives back at most half the pole. Past that,
                        # published failure rates jump above 40%.
FLAG_MIN_RETRACE = 0.12 # ...and it must give back SOMETHING. Without this a
                        # tight drift at the top of the run qualifies, which
                        # is a pole with no flag on it.
FLAG_TOUCH_ATR = 0.30   # how close a bar must come to count as touching a line
FLAG_MIN_TOUCH = 2      # touches needed on each line before it is a line
FLAG_SLOPE_TOL = 0.05   # slope allowed against the rule, per bar, in ATR
FLAG_PARALLEL = 0.60    # a flag's channel stays roughly this parallel
FLAG_VOL_MAX = 1.20     # the flag must not be LOUDER than the pole. The books
                        # say volume should fall during a flag; requiring a
                        # strict fall rejects half of everything on data where
                        # volume is flat, so the test is for the informative
                        # case -- a noisy consolidation after a run, which is
                        # distribution rather than a pause.
PENNANT_NARROW = 0.70   # a pennant's far end is this much tighter than its near

# rectangle and triangles
TRI_SPANS    = (20, 35, 55, 80, 120)  # widths searched, tightest first
TRI_MIN_BARS = 15
TRI_FLAT_ATR = 1.0      # how equal a "flat" edge's pivots must be, in ATR.
                        # Measured over a long window the noise alone moves
                        # the pivots more than 0.8 ATR, so a genuinely flat
                        # floor stopped reading as flat once the window grew
                        # wide enough to hold three pivots.
TRI_SLOPE_ATR = 0.5     # how much a sloping edge must actually slope, in ATR
TRI_CONVERGE = 0.70     # a symmetrical triangle's far end, against its near end
TRI_MIN_FLAT = 2        # pivots needed on a flat edge
TRI_MIN_SLOPE = 3       # ...and on a sloping one: "a series of higher lows"
                        # means three, not two
TRI_MAX_APEX = 0.75     # break before three-quarters of the way to the apex
TRI_BREACH   = 0.35     # how far price may poke through a "flat" edge, in ATR.
                        # This is what separates an edge price respected from
                        # two pivots that happened to land on the same number.
TRI_MONO     = 0.35     # slack allowed when checking a sloping edge really
                        # slopes the whole way rather than stepping once
RECT_MIN_BARS = 15
RECT_MAX_HEIGHT_ATR = 6.0   # your note: "very narrow support & resistance"
RECT_TREND_BARS = 30    # bars of run-in examined for a prior trend
RECT_TREND_ATR = 3.0    # how far that run must have travelled, in ATR

# rounding bottom and cup with handle
ROUND_SPANS  = (60, 110, 180)   # a base is a long, slow thing
CUP_MAX_SPAN = 180      # A cup runs one to six months. The search window
                        # reaches back past the left rim to see the approach,
                        # so this caps the WINDOW, and the cup's own shape is
                        # constrained by the curve fit and the depth tests
                        # rather than by a second duration rule.
ROUND_MIN_BARS = 40
ROUND_STRIDE = 10       # bars between candidate right-hand edges
ROUND_MIN_DEPTH_ATR = 3.0
ROUND_FIT    = 0.55     # R^2 of the parabola fitted through the closes
ROUND_EDGE   = 0.15     # fraction of the base treated as its right-hand climb
ROUND_RECOVER = 0.45    # how far back up the right side must have come
ROUND_RIM_TOL = 0.35    # the two rims must be within this much of the cup's
                        # depth of each other -- a cup, not a ski slope
HANDLE_MIN_BARS = 5     # one to four weeks
HANDLE_MAX_BARS = 20
HANDLE_MAX_DEPTH = 0.33 # a handle retraces at most a THIRD of the cup. It was
                        # 0.45 here, which let a second leg down pass as a
                        # handle and put the stop far too low.

# candlestick quality. Both of these patterns were being detected on shape
# alone, which is how a two-bar coincidence ends up on the page as a trade.
TWEEZER_VOL_MIN = 0.90  # the second day must trade about as much as the first
ENGULF_LOOKBACK = 10    # bars averaged for "a normal body for this stock"
ENGULF_BODY_MULT = 1.10 # the engulfing candle must be bigger than that
ENGULF_VOL_MULT = 1.00  # ...on at least average volume

# volume confirmation
VOL_LOOKBACK = 20       # bars averaged for "normal" volume
VOL_CONFIRM_MULT = 1.2  # signal bar must beat the average by this much

# --- delivery percentage (NSE only) -----------------------------------------
DELIVERY      = True    # set False to skip the NSE delivery download entirely
DELIVERY_DAYS = 260     # weekdays of history kept. The first run backfills
                        # this; later runs fetch the one day they are missing.
DELIV_HIGH    = 65.0    # "high delivery" in absolute terms
DELIV_SPIKE   = 1.30    # ...or this much above the stock's own 20-day average,
                        # which is the more useful test: a stock that always
                        # delivers 70% has not told you anything today

# --- money flow and volume at price -----------------------------------------
FLOW_MAX_AGE = 30       # a money-flow divergence's bottom must be this recent

# Volume profile. Swing pivots say where price TURNED; this says where it
# TRADED, which is the better guide to what will stop a move.
VP_BINS      = 40       # price buckets across the range
VP_LOOKBACK  = 250      # bars of history in the profile (about a year daily)
VP_VALUE_AREA = 0.70    # the band holding this share of the volume
VP_SHELF_SHARE = 0.045  # a single bin holding this much volume is a wall
VP_THIN      = 0.12     # overhead volume below this share is a clear runway

# --- pattern families, for the cross-tab ------------------------------------
# Splitting thirteen patterns by six filters gives 156 cells, and the rarer
# patterns do not have the trades to fill their own. Families do: they answer
# "does this filter help THIS KIND of setup" while there is still too little
# history to answer it pattern by pattern.
FAMILIES = {
    "structure":    ("doublebottom", "invhs", "rounding", "cuphandle",
                     "asctriangle", "symtriangle"),
    "continuation": ("flag", "pennant", "rectangle"),
    "candle":       ("engulfing", "tweezer"),
    "divergence":   ("divergence", "flowdiv"),
}
FAMILY_LABEL = {
    "structure":    "Structures (H&S, cup, rounding, triangles, double bottom)",
    "continuation": "Continuations (flag, pennant, rectangle)",
    "candle":       "Candlesticks (engulfing, tweezer)",
    "divergence":   "Divergences (RSI, money flow)",
}
FAMILY_OF = {p: f for f, ps in FAMILIES.items() for p in ps}

# A cross-tab cell needs more evidence than a one-factor row, because there are
# far more of them and therefore far more chances for one to look good by
# accident. 156 cells at ordinary significance would hand you about eight
# convincing findings from pure noise.
BT_MIN_CELL = 40

# --- backtest ---------------------------------------------------------------
BACKTEST      = True    # set False to skip it and shorten the run
BT_SAMPLE     = 0       # 0 = every stock. It was 200, which was plenty for the
                        # one-factor table but not for the CROSS-TAB: splitting
                        # 190 inverse-head-and-shoulders trades by a filter that
                        # passes a fifth of signals leaves 39 trades, and 39
                        # trades answer nothing. The whole universe multiplies
                        # every cell by about 3.75 and costs two more minutes.
                        # Set a number here to sample instead of scanning
                        # every stock, if the runtime ever matters more than
                        # the cross-tab's sample sizes.
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
BAR_HISTORY  = 25

# --- sector strength --------------------------------------------------------
# Money moves into sectors over months, not days, which is why this is useless
# to a day trader and worth having when you hold for weeks. Both windows SKIP
# the most recent month: at roughly a one-month lookback equities show
# short-term REVERSAL, so including it points the wrong way.
SECTOR_SKIP   = 21      # trading days left out at the near end (~1 month)
SECTOR_LONG   = 252     # ~12 months
SECTOR_SHORT  = 126     # ~6 months
SECTOR_MIN_N  = 3       # a "sector" of two stocks is two stocks, not a sector
REGIME_MA     = 200     # benchmark moving average that splits risk-on from off      # five trading weeks: enough to review recent trades
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
CHUNK        = 40       # stocks per yfinance request. Bigger means fewer
                        # round-trips over ~750 stocks; too big and one refused
                        # request loses a lot of names at once, so 25 is the
                        # compromise. Failed chunks are retried per stock below.
CHUNK_PAUSE  = 1.2      # seconds between chunks -- politeness, and it keeps
                        # Yahoo from rate-limiting a long run
IST          = timezone(timedelta(hours=5, minutes=30))


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------
def json_safe(o):
    """Make a payload legal JSON.

    NaN and Infinity are legal in Python and ILLEGAL in JSON. One of them
    anywhere in data.json and the browser's JSON.parse throws, the dashboard
    shows "no scan results yet", and a perfectly good scan looks like a failed
    one -- with no error anybody can see. Numpy types need coercing for the
    same reason.

    This lives at module level rather than inside main() so the test suite can
    reach it. A guard nothing can test is not much of a guard: this exact bug
    reached the live page once already.
    """
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, np.floating):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    return o


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


def ad_line(df: pd.DataFrame) -> pd.Series:
    """The Accumulation / Distribution line -- OBV with better manners.

    Plain OBV adds a bar's whole volume to the running total if the close was
    up and subtracts all of it if the close was down, so a bar that opened at
    its low, ran all day and closed a paisa lower counts as pure distribution.
    A/D weights each bar by WHERE in its own range the close landed:

        ((close - low) - (high - close)) / (high - low)   x   volume

    +1 when it closes on the high, -1 on the low, 0 in the middle. A rising
    A/D under a falling price is the thing worth knowing: the price is making
    lower bottoms while the buying underneath it is getting stronger.
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    if "Volume" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    span = (high - low).replace(0, np.nan)
    clv = (((close - low) - (high - close)) / span).fillna(0.0)
    return (clv * df["Volume"].astype(float)).cumsum()


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
    v = df["Volume"] if "Volume" in df.columns else None
    bodies = (c - o).abs()
    n, found = len(df), []
    for i in range(max(1, n - fresh), n):
        prev_red = c.iloc[i - 1] < o.iloc[i - 1]
        green = c.iloc[i] > o.iloc[i]
        swallows = (o.iloc[i] <= c.iloc[i - 1]) and (c.iloc[i] >= o.iloc[i - 1])
        bigger = (c.iloc[i] - o.iloc[i]) > (o.iloc[i - 1] - c.iloc[i - 1])
        # Engulfing the bar before it is the minimum. What separates a signal
        # from a shrug is that the candle is big for THIS stock -- a real
        # change of hands rather than one quiet bar swallowing a quieter one.
        recent = bodies.iloc[max(0, i - ENGULF_LOOKBACK):i]
        avg_body = float(recent.mean()) if len(recent) else None
        stands_out = (avg_body is None or not np.isfinite(avg_body) or avg_body <= 0
                      or float(bodies.iloc[i]) >= avg_body * ENGULF_BODY_MULT)
        vol_ok = True
        if v is not None:
            win = v.iloc[max(0, i - VOL_LOOKBACK):i]
            avg_v = float(win.mean()) if len(win) else None
            if avg_v and np.isfinite(avg_v) and avg_v > 0:
                vol_ok = float(v.iloc[i]) >= avg_v * ENGULF_VOL_MULT
        if (prev_red and green and swallows and bigger and stands_out
                and vol_ok and in_downtrend(c, i - 1)):
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
    """Two candles bottoming at the same level, the second one turning up.

    Half this pattern's definition was missing. "Two equal lows" alone is not
    a tweezer bottom -- the FIRST candle has to be the falling one and the
    SECOND has to close up, because the whole story is "sellers hit the same
    floor twice and the second time buyers took it back". Two red candles
    with equal lows is a stock still going down, and the scan was reporting
    those as buy signals. It is the worst performer on your own data at
    29.7%, and this is a large part of why.

    Volume matters too: the second day should trade at least as much as the
    first, or nobody actually turned up to defend the level.
    """
    o, l, h, c = df["Open"], df["Low"], df["High"], df["Close"]
    v = df["Volume"] if "Volume" in df.columns else None
    n, found = len(df), []
    last = float(c.iloc[-1])
    tol = (atr if atr and np.isfinite(atr) else last * 0.002) * TWEEZER_TOL_ATR
    for i in range(max(1, n - fresh), n):
        matched = abs(float(l.iloc[i]) - float(l.iloc[i - 1])) <= tol
        first_falls = float(c.iloc[i - 1]) < float(o.iloc[i - 1])
        second_turns = float(c.iloc[i]) > float(o.iloc[i])
        vol_ok = True
        if v is not None:
            v1, v2 = float(v.iloc[i - 1]), float(v.iloc[i])
            if np.isfinite(v1) and np.isfinite(v2) and v1 > 0:
                vol_ok = v2 >= v1 * TWEEZER_VOL_MIN
        if (matched and first_falls and second_turns and vol_ok
                and in_downtrend(c, i - 1)):
            shared = round(min(float(l.iloc[i]), float(l.iloc[i - 1])), 2)
            found.append({
                "type": "tweezer", "label": "Tweezer bottom",
                "date": df.index[i].strftime("%Y-%m-%d"), "ageBars": n - 1 - i,
                "entry": round(float(h.iloc[i]), 2),
                "stop": shared,
                "detail": (f"Two sessions bottomed together at {shared} after "
                           f"a downtrend; the first closed down, the second "
                           f"closed up."),
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


def _neckline(high: pd.Series, a: int, b: int, c: int, invert: bool) -> float:
    """The neckline of a head-and-shoulders, drawn the way the books draw it.

    It joins the two REACTION points -- the peak between the left shoulder and
    the head, and the peak between the head and the right shoulder -- and the
    entry is where that line sits at the right shoulder.

    This code used to take the highest high across the whole formation
    instead. On a tidy pattern the two are the same number. On a real one,
    where some unrelated spike sits inside the window, the "neckline" came out
    far above the line anyone would draw, which pushed the entry up, shrank
    the reward and made the measured move too big. The desk's best-performing
    pattern was being measured against the wrong level.
    """
    pick = (lambda seg: float(seg.max())) if invert else (lambda seg: float(seg.min()))
    left = high.iloc[a:b + 1]
    right = high.iloc[b:c + 1]
    if not len(left) or not len(right):
        return None
    p1, v1 = int(np.argmax(left.values) if invert else np.argmin(left.values)), pick(left)
    p2, v2 = int(np.argmax(right.values) if invert else np.argmin(right.values)), pick(right)
    x1, x2 = a + p1, b + p2
    if x2 == x1:
        return v2
    # Necklines slope. Project the line joining the two reaction points
    # forward to the right shoulder, which is where the break happens.
    slope = (v2 - v1) / (x2 - x1)
    projected = v2 + slope * (c - x2)
    # A steeply sloping neckline, projected far enough, lands BELOW both of
    # the points that defined it -- and an entry under the pattern's own
    # rallies is not a breakout level, it is a price already passed. Clamp it
    # inside the two reaction points.
    lo_v, hi_v = (v1, v2) if v1 <= v2 else (v2, v1)
    if invert:
        return float(min(max(projected, lo_v), hi_v * 1.10))
    return float(max(min(projected, hi_v), lo_v * 0.90))


def find_inverse_hs(df: pd.DataFrame, atr: float) -> list:
    """Inverse head and shoulders -- three lows, the middle one deepest.

    The bullish one. The neckline joins the two rallies either side of the
    head, and the target is the neckline plus the drop from neckline to head.
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
    neck = _neckline(high, ls, head, rs, invert=True)
    if neck is None:
        return []
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
    # Same correction as the bullish one: the neckline joins the two reaction
    # LOWS either side of the head, not the lowest point in the window.
    neck = _neckline(df["Low"], ls, head, rs, invert=False)
    if neck is None:
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


# ============================================================================
# CONTINUATION AND BASE PATTERNS
#
# Flags, pennants, rectangles, triangles, rounding bottoms and cup-and-handle.
# Everything above this point is a REVERSAL pattern -- something that says the
# fall is over. These are the other half of the market: a stock already moving,
# pausing, and then carrying on.
#
# One implementation, two callers. `scan_structures()` finds every occurrence
# across the whole series; the live scan keeps the ones whose last bar is
# recent, and the backtest keeps all of them. When a detector and its backtest
# are separate pieces of code they drift apart, and you end up trading a
# pattern that was never the thing the backtest measured.
# ============================================================================
def _fit_slope(xs, ys):
    """Least-squares slope of ys against xs. None when it cannot be fitted."""
    if len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    xm, ym = x.mean(), y.mean()
    den = float(((x - xm) ** 2).sum())
    if den <= 0:
        return None
    return float(((x - xm) * (y - ym)).sum() / den)


def _unit_array(close: np.ndarray, atr: np.ndarray) -> np.ndarray:
    """ATR per bar, with a 1%-of-price fallback wherever ATR is not yet valid."""
    fallback = close * 0.01
    return np.where(np.isfinite(atr) & (atr > 0), atr, fallback)


def _space_out(hits: list, gap: int) -> list:
    """Thin a run of near-identical signals down to one.

    A 20-bar flag is still a 20-bar flag one bar later, so a naive detector
    reports the same setup twenty times and the backtest counts it twenty
    times. Keep the first of each cluster, per pattern type.
    """
    kept, last = [], {}
    for h in sorted(hits, key=lambda x: x["i"]):
        prev = last.get(h["type"])
        if prev is not None and h["i"] - prev < gap:
            continue
        last[h["type"]] = h["i"]
        kept.append(h)
    return kept


# --- flags and pennants -----------------------------------------------------
def _hull_line(ys: np.ndarray, upper: bool, tol: float,
               min_sep: int = 2) -> tuple:
    """The trendline along a run of highs (or lows). Returns (slope, b, touches).

    Two wrong answers came before this one, and both are worth naming because
    they look right until you test them.

      1. Least-squares, then slide the line up until it sits above every high.
         That envelope touches exactly ONE point by construction, so a
         "two touches" rule can never pass.
      2. A line through the two HIGHEST points. In a falling channel the two
         highest highs are both early, so the line drops too steeply and later
         highs poke out above it -- it contains nothing.

    The right construction is the one a person performs with a ruler: lay the
    edge across the top and rotate it until it cannot go lower without cutting
    through a bar. That is the upper convex hull, and every edge of it has all
    the points below it by definition. Take the edge that spans the most bars,
    because that is the line describing the whole pause rather than two
    neighbours.
    """
    n = len(ys)
    if n < 3:
        return None, None, 0
    pts = [(float(i), float(ys[i])) for i in range(n)]

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1]) -
                (a[1] - o[1]) * (b[0] - o[0]))

    hull = []
    for p in pts:
        # upper hull keeps clockwise turns, lower hull counter-clockwise
        while len(hull) >= 2 and (cross(hull[-2], hull[-1], p) >= 0 if upper
                                  else cross(hull[-2], hull[-1], p) <= 0):
            hull.pop()
        hull.append(p)
    if len(hull) < 2:
        return None, None, 0

    # Which hull edge is THE trendline? Not simply the longest one: with a
    # handful of bars the longest edge runs from the first point to the last,
    # so its slope is decided by two noisy endpoints and the shape in between
    # is ignored. That produced channels that appeared to converge when the
    # drawing was parallel. Take the edge the price RESPECTED most -- the one
    # with the most bars sitting on it -- and use span only to break ties.
    xs = np.arange(n, dtype=float)
    best = None
    for a, b in zip(hull, hull[1:]):
        span = b[0] - a[0]
        if span < min_sep:
            continue
        slope = (b[1] - a[1]) / span
        intercept = a[1] - slope * a[0]
        touches = int(np.sum(np.abs(ys - (slope * xs + intercept)) <= tol))
        key = (touches, span)
        if best is None or key > best[0]:
            best = (key, slope, intercept, touches)
    if best is None:
        return None, None, 0
    _, slope, intercept, touches = best
    return float(slope), float(intercept), int(touches)


def scan_flags(df: pd.DataFrame, atr: np.ndarray, since: int = 0) -> list:
    """A sharp run (the pole), then a real pause that leans against it.

    Your note number 3 for this pattern says: "join a line with the points of
    lows & second on the point of highs". That is what this does now. The
    first version measured the pause as a BOX -- the highest high and lowest
    low of the last few bars -- and a box has no slope, no touches and no
    shape. Two consequences, both of which you saw on the live page:

      * a stock that ran hard and then drifted UP quietly for five bars
        qualified, because the box was tight and the old code allowed the
        flag's high to sit above the pole's top. That is a pole with no flag.
      * there was no requirement that price ever pulled back at all.

    So the rules here are the ones the textbooks actually give:

      pole      a sharp directional run, most of its range spent going one way
      flag      5 to 15 bars (past ~20 it is exhaustion, not a pause)
      shape     two lines, roughly parallel, sloping DOWN or sideways, each
                touched at least twice
      depth     retraces at most half the pole, and at least a little -- a
                pause that does not pause is not a flag
      ceiling   never makes a new high above the pole
      volume    quieter in the flag than in the pole
      entry     a CLOSE above the flag, not a wick through it
      stop      the flag's low, exactly as your notes say
      target    the pole's height projected from the breakout

    Two stages for speed: a cheap vectorised pre-filter over rolling windows,
    then the line fitting only on the handful of bars that survive it.
    """
    n = len(df)
    if n < 40:
        return []
    high, low, close = df["High"], df["Low"], df["Close"]
    hv, lv, cv = (high.to_numpy(float), low.to_numpy(float), close.to_numpy(float))
    vol = df["Volume"].to_numpy(float) if "Volume" in df.columns else None
    unit = _unit_array(cv, atr)
    found = []

    for L in FLAG_LENS:
        if L + max(FLAG_POLES) + 2 >= n:
            continue
        fh = high.rolling(L).max().to_numpy(float)              # flag high
        fl = low.rolling(L).min().to_numpy(float)               # flag low

        for P in FLAG_POLES:
            pt = high.shift(L).rolling(P).max().to_numpy(float)       # pole top
            pb = low.shift(L).rolling(P).min().to_numpy(float)        # pole base
            pole_h = pt - pb
            flag_h = fh - fl
            net = (close.shift(L) - close.shift(L + P - 1)).to_numpy(float)

            ok = (np.isfinite(fh) & np.isfinite(fl) & np.isfinite(net) &
                  np.isfinite(pt) & np.isfinite(pb) & np.isfinite(pole_h))
            ok &= pole_h >= unit * FLAG_POLE_ATR   # the run has to be a real run
            ok &= net >= unit * FLAG_POLE_ATR      # and it has to be UPWARD
            ok &= net >= pole_h * FLAG_POLE_EFF    # spent going one way
            ok &= flag_h > 0
            # Only the RELATIVE width test survives here. An absolute "the
            # pause must be under N ATR wide" cap was mine, not the textbooks',
            # and it rejects every honest pennant, which starts wide and
            # narrows -- the width at the start is the whole point.
            ok &= flag_h <= pole_h * FLAG_MAX_WIDTH
            # Retraces at most half the pole...
            ok &= fl >= pt - pole_h * FLAG_MAX_RETRACE
            # ...and at least a little. THIS is the test that was missing: a
            # "flag" that never gave anything back is the top of the pole.
            ok &= fl <= pt - pole_h * FLAG_MIN_RETRACE
            # A flag does not make new highs. The old code allowed half an ATR
            # above the pole, which let a continuing run pass as a pause.
            ok &= fh <= pt

            if since:
                ok[:since] = False
            for i in np.flatnonzero(ok):
                a = i - L + 1
                if a < 1:
                    continue
                xs = np.arange(L, dtype=float)
                his, los = hv[a:i + 1], lv[a:i + 1]
                if not (np.isfinite(his).all() and np.isfinite(los).all()):
                    continue
                u = unit[i]
                tol = u * FLAG_TOUCH_ATR
                s_hi, b_hi, t_hi = _hull_line(his, True, tol)
                s_lo, b_lo, t_lo = _hull_line(los, False, tol)
                if s_hi is None or s_lo is None:
                    continue

                # Each line has to have been touched at least twice, or it is
                # not a line anybody drew -- it is a boundary round noise.
                top_line = s_hi * xs + b_hi
                bot_line = s_lo * xs + b_lo
                if t_hi < FLAG_MIN_TOUCH or t_lo < FLAG_MIN_TOUCH:
                    continue

                w_start = float(top_line[0] - bot_line[0])
                w_end = float(top_line[-1] - bot_line[-1])
                if w_start <= 0 or w_end <= 0:
                    continue
                if vol is not None:
                    fvol = float(np.nanmean(vol[a:i + 1]))
                    pvol = float(np.nanmean(vol[max(0, a - P):a]))
                    if (np.isfinite(fvol) and np.isfinite(pvol) and pvol > 0
                            and fvol > pvol * FLAG_VOL_MAX):
                        continue          # louder than the pole: not a pause

                # Classify FIRST, then apply that shape's slope rule. The two
                # shapes want opposite things from the lower line -- a flag
                # leans down against the trend, a pennant's lows RISE into the
                # apex -- so a single "no rising lows" test before the
                # classification threw every honest pennant away.
                slack = u * FLAG_SLOPE_TOL
                # Convergence is a RELATIVE property: the two lines approach
                # each other. Demanding that the upper line also fall in
                # absolute terms threw away honest pennants whose roof was
                # merely flat, which is most of them on a short window.
                converging = (w_end <= w_start * PENNANT_NARROW and
                              (s_lo - s_hi) >= slack)
                parallel = (w_end >= w_start * FLAG_PARALLEL and
                            s_hi <= slack and s_lo <= slack)
                if converging:
                    kind, label = "pennant", "Pennant"
                elif parallel:
                    kind, label = "flag", "Bullish flag"
                else:
                    continue              # neither parallel nor converging

                found.append({
                    "type": kind, "label": label, "i": int(i),
                    "entry": round(float(fh[i]), 2),
                    "stop": round(float(fl[i]), 2),
                    "measured": round(float(fh[i] + pole_h[i]), 2),
                    "quality": float(L * 100 + pole_h[i] / u),
                    "detail": (f"Pole {round(float(pb[i]), 2)} \u2192 "
                               f"{round(float(pt[i]), 2)} over {P} bars, then a "
                               f"{L}-bar {'pennant' if kind == 'pennant' else 'flag'} "
                               f"between two lines "
                               f"({round(float(bot_line[-1]), 2)}\u2013"
                               f"{round(float(top_line[-1]), 2)} today), "
                               f"giving back "
                               f"{round(float((pt[i] - fl[i]) / pole_h[i]) * 100)}% "
                               f"of the run."),
                })

    # Several (flag length, pole length) pairs describe the same pause, and a
    # pause that converges is a pennant whichever window spotted it. One
    # verdict per bar: pennant beats flag, then the biggest pole wins.
    best = {}
    for f in found:
        key = f["i"]
        cur = best.get(key)
        if cur is None:
            best[key] = f
            continue
        better = ((f["type"] == "pennant") > (cur["type"] == "pennant") or
                  (f["type"] == cur["type"] and f["quality"] > cur["quality"]))
        if better:
            best[key] = f
    return sorted(best.values(), key=lambda f: f["i"])


# --- rectangles and triangles ----------------------------------------------
def _edge(pivots: list, values: np.ndarray, lo: int, hi: int) -> list:
    """The pivots of one kind that fall inside [lo, hi], with their values."""
    return [(p, float(values[p])) for p in pivots if lo <= p <= hi]


def scan_boxes(df: pd.DataFrame, atr: np.ndarray, since: int = 0) -> list:
    """Rectangles and the three triangles, all from the same pivot walk.

    Every one of these is two edges -- one drawn through the highs, one through
    the lows -- and the pattern's name is just which edges are flat and which
    slope:

        rectangle    flat highs      flat lows
        ascending    flat highs      rising lows
        descending   falling highs   flat lows      <- bearish, a warning only
        symmetrical  falling highs   rising lows

    Your notes had the symmetrical triangle as "equal bottoms & lower highs",
    but the drawing underneath it shows a rising lower line. The drawing is the
    standard pattern and it is what is coded here; "equal bottoms and lower
    highs" is the descending triangle, which is the bearish one.
    """
    n = len(df)
    if n < TRI_MIN_BARS + SWING_BARS * 2:
        return []
    high, low = df["High"], df["Low"]
    hv, lv = high.to_numpy(float), low.to_numpy(float)
    cv = df["Close"].to_numpy(float)
    unit = _unit_array(cv, atr)
    plows, phighs = pivot_lows(low), pivot_highs(high)
    if len(plows) < 2 or len(phighs) < 2:
        return []

    found = []
    # The right edge of a pattern is always a pivot: that is what makes this
    # affordable. A hundred pivots per stock, not twelve hundred bars.
    edges = [e for e in sorted(set(plows + phighs)) if e >= since]
    for r in edges:
        u = unit[r]
        if not (np.isfinite(u) and u > 0):
            continue
        for span in TRI_SPANS:
            lo = r - span
            if lo < 0:
                continue
            hs = _edge(phighs, hv, lo, r)
            ls = _edge(plows, lv, lo, r)
            # Two pivots make a line; three make a pattern. The sources are
            # consistent that a triangle needs at least three higher lows (or
            # three lower highs) on its sloping edge -- with two, "a series of
            # higher lows" is just a swing.
            if len(hs) < TRI_MIN_FLAT or len(ls) < TRI_MIN_FLAT:
                continue

            top_vals = [v for _, v in hs]
            bot_vals = [v for _, v in ls]
            # A flat edge is defined by the pivots AT that level, not by every
            # swing in the window. Over a long window a sine-ish chart throws
            # up intermediate swing points well away from the floor, and
            # judging flatness across all of them made a perfectly flat floor
            # read as an 8-rupee spread.
            flat_tol0 = unit[r] * TRI_FLAT_ATR
            roof_hits = [v for v in top_vals if v >= max(top_vals) - flat_tol0]
            floor_hits = [v for v in bot_vals if v <= min(bot_vals) + flat_tol0]
            band = max(top_vals) - min(bot_vals)
            if band <= 0:
                continue

            flat_tol = u * TRI_FLAT_ATR
            slope_min = u * TRI_SLOPE_ATR
            top_flat = len(roof_hits) >= TRI_MIN_FLAT
            bot_flat = len(floor_hits) >= TRI_MIN_FLAT
            top_drop = top_vals[0] - top_vals[-1]      # falling highs when > 0
            bot_rise = bot_vals[-1] - bot_vals[0]      # rising lows when > 0
            top_falls = top_drop >= slope_min
            bot_rises = bot_rise >= slope_min

            roof = float(np.mean(roof_hits)) if top_flat else max(top_vals)
            floor_ = float(np.mean(floor_hits)) if bot_flat else min(bot_vals)
            height = max(top_vals) - min(bot_vals)
            last_bar = r

            # Containment. Two pivots happening to sit at the same price inside
            # a much wider range is a coincidence, not an edge. A flat edge is
            # only flat if price actually respected it: nothing of consequence
            # traded through it across the whole span.
            # Containment, measured on CLOSES. Against highs this test became
            # nearly free once the roof was defined as the average of the
            # highest pivots -- the roof is built from the high, so the high
            # respects it by construction. What actually matters is that price
            # never CLOSED beyond the level while the pattern was forming: a
            # close above the roof means it already broke out, and whatever is
            # left is not the pattern any more.
            # BOTH tests, not either. Highs alone became weak once the roof
            # was built from the highest pivots (the roof then respects itself
            # by construction); closes alone are weaker still, because a close
            # always sits inside its own bar. Together they say what is meant:
            # nothing traded meaningfully through the level, and nothing
            # closed through it either.
            roof_holds = (float(hv[lo:r + 1].max()) <= roof + u * TRI_BREACH and
                          float(cv[lo:r + 1].max()) <= roof + u * TRI_BREACH * 0.5)
            floor_holds = (float(lv[lo:r + 1].min()) >= floor_ - u * TRI_BREACH and
                           float(cv[lo:r + 1].min()) >= floor_ - u * TRI_BREACH * 0.5)
            # A sloping edge should slope the whole way, not jump once and sit.
            tops_fall = all(top_vals[k] >= top_vals[k + 1] - u * TRI_MONO
                            for k in range(len(top_vals) - 1))
            bots_rise = all(bot_vals[k] <= bot_vals[k + 1] + u * TRI_MONO
                            for k in range(len(bot_vals) - 1))

            kind = None
            if top_flat and bot_flat and roof_holds and floor_holds:
                # A box is only a box if it is NARROW -- your note: "stuck in
                # very narrow support & resistance compared to the sideways
                # trend". A wide drifting range is just a range.
                # And a rectangle is a PAUSE IN SOMETHING. Every source opens
                # with "a prior trend should exist", and your own note says the
                # same: a downtrend before the pattern, or an uptrend in the
                # case of a reversal. Without that test a random walk's
                # ordinary quiet patches all qualify, which is what was
                # happening: one every seventy bars of pure noise.
                back = max(0, lo - RECT_TREND_BARS)
                run = (float(cv[lo]) - float(cv[back])) if lo > back else 0.0
                if (band <= u * RECT_MAX_HEIGHT_ATR and span >= RECT_MIN_BARS
                        and abs(run) >= u * RECT_TREND_ATR):
                    kind, label = "rectangle", "Rectangle"
                    entry, stop = roof, floor_
                    measured = roof + band
                    detail = (f"Box between {round(floor_, 2)} and "
                              f"{round(roof, 2)} for {span} bars "
                              f"({len(roof_hits)} touches on top, "
                              f"{len(floor_hits)} below), after a "
                              f"{'rise' if run > 0 else 'fall'} of "
                              f"{abs(round(run, 2))}.")
            elif (top_flat and roof_holds and bot_rises and bots_rise
                  and len(ls) >= TRI_MIN_SLOPE):
                kind, label = "asctriangle", "Ascending triangle"
                entry = roof
                stop = min(bot_vals[-1], bot_vals[-2]) - u * 0.25
                # Your rule: "target equal to the points formed b/w the 2
                # peaks" -- the drop from the flat top to the deepest low
                # inside the pattern, projected off the breakout.
                measured = roof + height
                detail = (f"Flat top at {round(roof, 2)} with "
                          f"{len(hs)} touches, lows rising "
                          f"{round(bot_vals[0], 2)} → {round(bot_vals[-1], 2)}.")
            elif (bot_flat and floor_holds and top_falls and tops_fall
                  and len(hs) >= TRI_MIN_SLOPE):
                kind, label = "desctriangle", "Descending triangle"
                entry = stop = measured = None      # bearish: a warning, never a buy
                detail = (f"Flat floor at {round(floor_, 2)} with "
                          f"{len(ls)} touches, highs falling "
                          f"{round(top_vals[0], 2)} → {round(top_vals[-1], 2)}.")
            # A symmetrical triangle's own rule is four pivots -- two highs
            # and two lows. The "three or more" requirement belongs to the
            # ascending and descending ones, whose sloping edge is described
            # as "a series of higher lows". Applying three everywhere was me
            # over-generalising one source onto another pattern.
            elif (top_falls and bot_rises and tops_fall and bots_rise
                  and len(hs) >= TRI_MIN_FLAT and len(ls) >= TRI_MIN_FLAT):
                near = top_vals[0] - bot_vals[0]
                far = top_vals[-1] - bot_vals[-1]
                if near <= 0 or far > near * TRI_CONVERGE:
                    continue                        # not actually converging
                # The apex rule. A symmetrical triangle should break somewhere
                # between half and three-quarters of the way to where its two
                # lines meet; by the time price is squeezed into the tip the
                # pattern has spent its tension and the break means little.
                # Distance to the apex, in bars, from the converging rate:
                closing = (near - far)
                if closing > 0:
                    bars_seen = float(hs[-1][0] - hs[0][0]) or float(span)
                    to_apex = far / (closing / max(bars_seen, 1.0))
                    if to_apex > 0 and bars_seen / (bars_seen + to_apex) > TRI_MAX_APEX:
                        continue                    # already at the tip
                kind, label = "symtriangle", "Symmetrical triangle"
                entry = max(top_vals)
                stop = min(bot_vals[-1], bot_vals[-2]) - u * 0.25
                measured = entry + near             # the widest part of the wedge
                detail = (f"Highs {round(top_vals[0], 2)} → "
                          f"{round(top_vals[-1], 2)}, lows "
                          f"{round(bot_vals[0], 2)} → {round(bot_vals[-1], 2)}, "
                          f"narrowing {round(near, 2)} → {round(far, 2)}.")
            if kind is None:
                continue

            rec = {"type": kind, "label": label, "i": int(last_bar),
                   "detail": detail, "quality": float(span)}
            if entry is not None:
                rec["entry"] = round(float(entry), 2)
                rec["stop"] = round(float(stop), 2)
                rec["measured"] = round(float(measured), 2)
            found.append(rec)
            break        # the tightest span that fits wins; stop widening

    best = {}
    for f in found:
        key = (f["type"], f["i"])
        if key not in best:
            best[key] = f
    return sorted(best.values(), key=lambda f: f["i"])


# --- rounding bottom and cup with handle ------------------------------------
def scan_cups(df: pd.DataFrame, atr: np.ndarray, since: int = 0) -> list:
    """A long curved base, and the same base with a handle on it.

    The curve is tested by fitting a parabola to the closes: a real rounding
    bottom has an upward-opening one (it falls, flattens, turns up) whose
    lowest point sits somewhere near the middle. A V-shaped crash and a
    straight drift both fail that test, which is the whole point.

    Your rule for the plain rounding bottom -- buy at breakout, wide stop
    because the pattern is long, exit at the resistance above -- is kept. The
    handle version enters earlier, on the break of the handle, which is what
    makes the stop small enough to be worth taking.
    """
    n = len(df)
    if n < ROUND_MIN_BARS + 10:
        return []
    high, low, close = df["High"], df["Low"], df["Close"]
    hv, lv, cv = (high.to_numpy(float), low.to_numpy(float), close.to_numpy(float))
    unit = _unit_array(cv, atr)
    found = []

    # Striding keeps this affordable, but the LAST bar must always be a
    # candidate: without it whether today's cup is seen at all depends on
    # whether the history happens to divide by the stride.
    ends = [e for e in range(ROUND_MIN_BARS, n, ROUND_STRIDE) if e >= since]
    if n - 1 >= since and (not ends or ends[-1] != n - 1):
        ends.append(n - 1)
    for end in ends:
        for span in ROUND_SPANS:
            start = end - span
            if start < 0:
                continue
            seg = cv[start:end + 1]
            if len(seg) < ROUND_MIN_BARS or not np.isfinite(seg).all():
                continue
            u = unit[end]
            if not (np.isfinite(u) and u > 0):
                continue

            x = np.arange(len(seg), dtype=float)
            try:
                a, b, c0 = np.polyfit(x, seg, 2)
            except Exception:       # noqa: BLE001  degenerate segment
                continue
            if a <= 0:
                continue            # opens downward: that is a dome, not a cup
            fit = a * x * x + b * x + c0
            ss_res = float(((seg - fit) ** 2).sum())
            ss_tot = float(((seg - seg.mean()) ** 2).sum())
            if ss_tot <= 0:
                continue
            r2 = 1.0 - ss_res / ss_tot
            if r2 < ROUND_FIT:
                continue            # the curve does not describe the prices
            vertex = -b / (2 * a)
            if not (len(seg) * 0.25 <= vertex <= len(seg) * 0.75):
                continue            # the low has to be in the middle, not at an end

            rim = float(hv[start:end + 1].max())
            cup_low = float(lv[start:end + 1].min())
            depth = rim - cup_low
            if depth < u * ROUND_MIN_DEPTH_ATR:
                continue            # too shallow to be a base

            # A cup has two rims at roughly the same height. Without this a
            # plain decline that curls up at the end fits a parabola happily
            # and gets reported as a base.
            edge_n = max(3, int(span * ROUND_EDGE))
            left_rim = float(hv[start:start + edge_n].max())
            right_rim = float(hv[end - edge_n:end + 1].max())
            # How long the CUP ran, rim to rim -- not how wide the window was
            # that found it. The window reaches back past the left rim to see
            # the approach, so capping the window at six months quietly capped
            # the cup at rather less than that.
            cup_bars = (end - int(np.argmax(hv[start:start + edge_n])) - start)
            if abs(left_rim - right_rim) > depth * ROUND_RIM_TOL:
                continue

            # The right side has to have actually come back up. A cup still on
            # its way down is just a downtrend with a nice curve through it.
            right = cv[end - max(3, span // 10):end + 1]
            if float(right.mean()) < rim - depth * ROUND_RECOVER:
                continue

            edge = max(3, int(span * ROUND_EDGE))
            right_low = float(lv[end - edge:end + 1].min())
            found.append({
                "type": "rounding", "label": "Rounding bottom", "i": int(end),
                "entry": round(rim, 2),
                # Your note: the stop "should be big as the pattern is often
                # long term". Big, but anchored to the last real low on the
                # right-hand climb rather than all the way down in the cup --
                # a stop under the cup itself risks the entire pattern.
                "stop": round(right_low - u * 0.25, 2),
                "measured": round(rim + depth, 2),
                "quality": float(r2),
                "detail": (f"Curved base over {span} bars, low {round(cup_low, 2)}, "
                           f"rim {round(rim, 2)} (curve fit {round(r2 * 100)}%)."),
            })

            # --- the handle ---------------------------------------------
            # A shallow pullback after the price has come back to the rim.
            # Take the LONGEST handle that still qualifies, not the first. A
            # handle that has been forming for a fortnight should be reported
            # with a fortnight's low as its stop, not with day four's.
            handle = None
            for hl in (range(HANDLE_MIN_BARS, HANDLE_MAX_BARS + 1)
                       if cup_bars <= CUP_MAX_SPAN else ()):
                h_end = end + hl
                if h_end >= n:
                    break
                h_high = float(hv[end + 1:h_end + 1].max())
                h_low = float(lv[end + 1:h_end + 1].min())
                if h_high > rim + u * 0.5:
                    break           # it broke out instead of forming a handle
                if rim - h_low > depth * HANDLE_MAX_DEPTH:
                    break           # too deep: that is a second cup, not a handle
                handle = (h_end, hl, h_high, h_low)
            if handle:
                h_end, hl, h_high, h_low = handle
                found.append({
                    "type": "cuphandle", "label": "Cup and handle",
                    "i": int(h_end),
                    "entry": round(max(h_high, rim * 0.999), 2),
                    "stop": round(h_low - u * 0.25, 2),
                    "measured": round(rim + depth, 2),
                    "quality": float(r2),
                    "detail": (f"Cup low {round(cup_low, 2)} to rim "
                               f"{round(rim, 2)}, then a {hl}-bar handle down to "
                               f"{round(h_low, 2)}."),
                })
            break                   # one cup per end bar
    return sorted(found, key=lambda f: f["i"])


def scan_structures(df: pd.DataFrame, atr: np.ndarray = None,
                    thin: bool = True, since: int = 0) -> list:
    """Every flag, pennant, box, triangle and cup in the series.

    `thin` keeps the FIRST bar of each cluster, which is what the backtest
    wants -- the day the setup appeared is the day you would have acted on it.
    The live scan passes thin=False and takes the LAST instead, because today's
    view of a pause that has been forming for a fortnight is the complete one.
    """
    if atr is None:
        atr = wilder_atr(df).to_numpy(dtype=float)
    out = []
    for fn in (scan_flags, scan_boxes, scan_cups):
        try:
            out.extend(fn(df, atr, since))
        except Exception as exc:    # noqa: BLE001
            print(f"    {fn.__name__} failed: {exc}", flush=True)
    if not thin:
        return out
    gap = {"rounding": ROUND_DEDUPE, "cuphandle": ROUND_DEDUPE}
    thinned, last = [], {}
    for h in sorted(out, key=lambda x: x["i"]):
        prev = last.get(h["type"])
        if prev is not None and h["i"] - prev < gap.get(h["type"], CONT_DEDUPE):
            continue
        last[h["type"]] = h["i"]
        thinned.append(h)
    return thinned


def live_structures(df: pd.DataFrame, atr: np.ndarray = None) -> tuple:
    """The ones that are still live today, split into entries and warnings.

    "Live" means the consolidation's last bar is recent. Without that test a
    flag from eight months ago is still reported as a trade -- the same
    mistake that once had 58 of 60 stocks showing a setup.
    """
    n = len(df)
    atr_arr = (atr if atr is not None else wilder_atr(df).to_numpy(dtype=float))
    last_price = float(df["Close"].iloc[-1])
    entries, warnings = [], []
    # Newest first, one per pattern: a pause that is still forming is reported
    # as it looks TODAY, not as it looked the first day it qualified.
    # Only the tail of the series can hold a LIVE pattern, so only the tail is
    # searched. Walking five years of history to answer a question about the
    # last fortnight cost about fifty milliseconds a stock, and there are
    # fifteen hundred stock-timeframes in a run.
    since = max(0, n - 1 - (CONT_MAX_AGE + SWING_BARS + HANDLE_MAX_BARS + 2))
    fresh, seen = [], set()
    for s in sorted(scan_structures(df, atr_arr, thin=False, since=since),
                    key=lambda x: -x["i"]):
        if s["type"] in seen:
            continue
        seen.add(s["type"])
        fresh.append(s)
    for s in fresh:
        age = n - 1 - s["i"]
        # Boxes and triangles are anchored on a PIVOT, and a pivot is not
        # confirmed until SWING_BARS bars have printed after it. Holding them
        # to the same freshness as a flag would be holding them to a deadline
        # that passed before they could exist.
        limit = CONT_MAX_AGE + (SWING_BARS if s["type"] in PIVOT_PATTERNS else 0)
        if age > limit:
            continue
        rec = {
            "type": s["type"], "label": s["label"], "direction": "long",
            "date": df.index[s["i"]].strftime("%Y-%m-%d"),
            "ageBars": int(age), "detail": s["detail"],
        }
        if s["type"] in WARNINGS:
            rec["direction"] = "warn"
            warnings.append(rec)
            continue
        entry, measured = s.get("entry"), s.get("measured")
        if not entry:
            continue
        # Price already below where the stop would go: the consolidation broke
        # the wrong way. Offering it as a trade would mean offering one that
        # has already been stopped out.
        stop = s.get("stop")
        if stop is not None and last_price <= stop:
            continue
        # Already most of the way to the measured move? The trade has gone.
        if measured and measured > entry:
            if last_price > entry + (measured - entry) * CONT_LATE_FRAC:
                continue
        rec.update({"entry": entry, "stop": s.get("stop"), "measured": measured})
        entries.append(rec)
    return entries, warnings


def find_flow_divergence(df: pd.DataFrame, atr: float) -> list:
    """Money-flow divergence: a lower bottom on price, a higher one on A/D.

    The same shape as the RSI divergence this desk was built around, with one
    difference that matters: RSI is made of price alone, so an RSI divergence
    says momentum is slowing. A/D is made of price AND volume, so this one says
    somebody is buying into the fall. On the desk's own backtest RSI
    divergence barely clears break-even, which is the reason to ask the
    question with money in it rather than with price twice.
    """
    if "Volume" not in df.columns or len(df) < DIV_WINDOW // 2:
        return []
    low, high = df["Low"], df["High"]
    ad = ad_line(df)
    if not np.isfinite(ad.to_numpy(float)).any():
        return []
    n = len(df)
    start = max(0, n - DIV_WINDOW)
    lows_idx = [i for i in pivot_lows(low)
                if i >= start and np.isfinite(float(ad.iloc[i]))]
    if len(lows_idx) < 2:
        return []

    b = lows_idx[-1]
    if n - 1 - b > FLOW_MAX_AGE:
        return []                       # the bottom is old news
    lv, av = low.to_numpy(float), ad.to_numpy(float)
    unit = atr if (atr and np.isfinite(atr) and atr > 0) else float(df["Close"].iloc[-1]) * 0.01
    for a in reversed(lows_idx[:-1]):
        if lv[b] >= lv[a]:
            continue                    # not a lower bottom, keep looking back
        if av[b] <= av[a]:
            break                       # lower bottom AND weaker money: no signal
        span = high.iloc[a:b + 1]
        if not len(span):
            break
        neck = float(span.max())
        # Scale the two A/D readings by the bigger of them so the number in the
        # sentence means something on a giant and on a microcap alike.
        scale = max(abs(av[a]), abs(av[b]), 1.0)
        gain = (av[b] - av[a]) / scale * 100.0
        return [{
            "type": "flowdiv", "label": "Money-flow divergence",
            "direction": "long",
            "date": df.index[b].strftime("%Y-%m-%d"),
            "ageBars": int(n - 1 - b),
            "entry": round(neck, 2),
            "stop": round(neck - unit * ATR_MULT, 2),
            "detail": (f"Price {round(lv[a], 2)} → {round(lv[b], 2)} "
                       f"(lower bottom) while the A/D line rose "
                       f"{gain:+.1f}% — buying into the fall."),
            "marks": {
                "dateA": df.index[a].strftime("%Y-%m-%d"),
                "lowA": round(lv[a], 2),
                "dateB": df.index[b].strftime("%Y-%m-%d"),
                "lowB": round(lv[b], 2),
                "flowGain": round(gain, 1),
            },
        }]
    return []


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


def volume_profile(df: pd.DataFrame, bins: int = VP_BINS,
                   lookback: int = VP_LOOKBACK) -> dict:
    """Volume at price: where the shares actually changed hands.

    Swing highs and lows say where price TURNED. This says where it TRADED,
    which is a different and often better question. A level with a year of
    volume piled on it is a wall made of people who own stock there and would
    like their money back; a price range almost nobody traded is air.

    Each bar's volume is spread evenly across its own high-low range rather
    than dumped on its close, because a bar that ranged 5% did not do all its
    business at one price. Without intraday data that is the honest
    approximation, and it is the one most charting packages make too.
    """
    if "Volume" not in df.columns or len(df) < 30:
        return {}
    tail = df.tail(lookback)
    hi = tail["High"].to_numpy(float)
    lo = tail["Low"].to_numpy(float)
    vol = tail["Volume"].to_numpy(float)
    ok = np.isfinite(hi) & np.isfinite(lo) & np.isfinite(vol) & (vol > 0) & (hi >= lo)
    hi, lo, vol = hi[ok], lo[ok], vol[ok]
    if len(vol) < 20:
        return {}
    top, bottom = float(hi.max()), float(lo.min())
    if not (top > bottom > 0):
        return {}

    edges = np.linspace(bottom, top, bins + 1)
    width = edges[1] - edges[0]
    if width <= 0:
        return {}
    buckets = np.zeros(bins, dtype=float)
    # Spread each bar across the bins it covers. Vectorising this properly
    # needs a loop over bars, but 250 bars x 40 bins is nothing.
    for h, l, v in zip(hi, lo, vol):
        first = int(np.clip((l - bottom) // width, 0, bins - 1))
        last = int(np.clip((h - bottom) // width, 0, bins - 1))
        if last < first:
            first, last = last, first
        buckets[first:last + 1] += v / (last - first + 1)
    total = float(buckets.sum())
    if total <= 0:
        return {}

    mids = (edges[:-1] + edges[1:]) / 2
    poc_i = int(buckets.argmax())
    # Value area: grow out from the point of control until 70% of the volume
    # is inside it. That band is where the market agreed on a price.
    lo_i = hi_i = poc_i
    got = buckets[poc_i]
    while got < total * VP_VALUE_AREA and (lo_i > 0 or hi_i < bins - 1):
        down = buckets[lo_i - 1] if lo_i > 0 else -1.0
        up = buckets[hi_i + 1] if hi_i < bins - 1 else -1.0
        if up >= down:
            hi_i += 1
            got += up
        else:
            lo_i -= 1
            got += down
    shares = buckets / total
    return {
        "poc": round(float(mids[poc_i]), 2),
        "valueHigh": round(float(edges[hi_i + 1]), 2),
        "valueLow": round(float(edges[lo_i]), 2),
        "binWidth": round(float(width), 2),
        "prices": [round(float(m), 2) for m in mids],
        "shares": [round(float(s), 4) for s in shares],
        "bars": int(len(vol)),
    }


def volume_between(profile: dict, lo: float, hi: float) -> float:
    """Share of the profile's volume sitting between two prices, 0 to 1.

    This is the "runway" question: how much stock is parked between here and
    the target, waiting to be sold back to you on the way up.
    """
    if not profile or lo is None or hi is None or hi <= lo:
        return None
    prices = profile.get("prices") or []
    shares = profile.get("shares") or []
    if not prices:
        return None
    return round(float(sum(s for p, s in zip(prices, shares) if lo < p <= hi)), 4)


def volume_shelf(profile: dict, above: float, below: float,
                 min_share: float = VP_SHELF_SHARE) -> dict:
    """The heaviest price bin standing between two levels, if it is heavy.

    A shelf below the measured move is where the move is likely to stall,
    which makes it the honest target even though the formula says otherwise.
    """
    if not profile or above is None or below is None or below <= above:
        return None
    prices = profile.get("prices") or []
    shares = profile.get("shares") or []
    band = [(p, s) for p, s in zip(prices, shares) if above * 1.002 < p < below]
    if not band:
        return None
    price, share = max(band, key=lambda x: x[1])
    if share < min_share:
        return None
    return {"price": price, "share": round(share, 4)}


def breakout_volume(df: pd.DataFrame, entry: float, from_idx: int) -> dict:
    """The volume on the bar that actually cleared the entry level.

    Different question from the one `volume_state` answers. That one asks
    whether the bar the PATTERN completed on had conviction behind it. This
    asks about the bar that broke the level -- the one where buyers had to
    outbid everybody who has been trapped at that price. For a breakout trade
    it is the more relevant bar, and until now the scan never looked at it.
    """
    blank = {"breakoutVolumeRatio": None, "breakoutVolumeConfirmed": None,
             "breakoutDate": None, "breakoutBars": None, "brokeOut": False}
    if entry is None or "Volume" not in df.columns:
        return blank
    n = len(df)
    start = max(0, int(from_idx) if from_idx is not None else 0)
    close = df["Close"].to_numpy(float)
    vol = df["Volume"].to_numpy(float)
    hit = None
    # A breakout is a CLOSE above the level, not a wick through it. Every
    # source says so, and it is the difference between a breakout and the
    # false breakout everyone warns about. This used to look at the high,
    # which meant a bar that poked above and closed back under counted as
    # the breakout bar and had its volume reported as confirmation.
    for j in range(start + 1, n):
        if np.isfinite(close[j]) and close[j] > entry:
            hit = j
            break
    if hit is None:
        return blank                    # not triggered yet: nothing to measure
    lo = max(0, hit - VOL_LOOKBACK)
    window = vol[lo:hit]
    if not len(window):
        return blank
    avg = float(np.nanmean(window))
    here = float(vol[hit])
    if not (np.isfinite(avg) and avg > 0 and np.isfinite(here)):
        return blank
    return {
        "breakoutVolumeRatio": round(here / avg, 2),
        "breakoutVolumeConfirmed": bool(here >= avg * VOL_CONFIRM_MULT),
        "breakoutDate": df.index[hit].strftime("%Y-%m-%d"),
        "breakoutBars": int(n - 1 - hit),
        "brokeOut": True,
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
                  high52: float = None, profile: dict = None) -> dict:
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

    # A shelf of real traded volume between here and the target is where the
    # move is likely to stall, whatever the measured move says. Cap there and
    # say so, rather than printing a ratio that needs a wall to evaporate.
    shelf = volume_shelf(profile, entry, target)
    if shelf:
        target = shelf["price"]
        source = f"volume shelf ({round(shelf['share'] * 100)}% of a year's trade)"
        setup["shelfShare"] = shelf["share"]

    # How much stock is parked between the entry and the target: the runway.
    setup["runway"] = volume_between(profile, entry, target)

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

    vol = df["Volume"].to_numpy(float) if "Volume" in df.columns else None
    dates = [d.strftime("%Y-%m-%d") for d in df.index]

    def unit(i):
        a = atr[i]
        return a if (a and np.isfinite(a) and a > 0) else float(c.iloc[i]) * 0.01

    def vol_ok(i):
        if vol is None or i < 5:
            return None
        lo = max(0, i - VOL_LOOKBACK)
        avg = np.nanmean(vol[lo:i]) if i > lo else np.nan
        return bool(np.isfinite(avg) and avg > 0 and vol[i] >= avg * VOL_CONFIRM_MULT)

    def atr_pct(i):
        px = float(c.iloc[i])
        a = atr[i]
        return round(a / px * 100, 2) if (px > 0 and np.isfinite(a)) else None

    def emit(kind, i, entry, stop):
        """One historical occurrence, with everything the filter splits need."""
        out.append({"kind": kind, "i": i, "entry": entry, "stop": stop,
                    "date": dates[i], "volOK": vol_ok(i), "atrPct": atr_pct(i)})

    # --- candlestick setups: local, so just walk the bars -------------------
    # These must apply EXACTLY the live rules. When the replay is a loose
    # paraphrase of the detector, the hit rate on the page is a number about a
    # pattern nobody trades.
    bodies = (c - o).abs().to_numpy(float)
    opens = o.to_numpy(float)
    for i in range(TREND_BARS + 1, n):
        if not in_downtrend(c, i - 1):
            continue
        po, pc = opens[i - 1], float(c.iloc[i - 1])
        co, cc = opens[i], float(c.iloc[i])

        if pc < po and cc > co and cc >= po and co <= pc:
            lo_b = max(0, i - ENGULF_LOOKBACK)
            avg_body = np.nanmean(bodies[lo_b:i]) if i > lo_b else np.nan
            stands_out = (not np.isfinite(avg_body) or avg_body <= 0 or
                          bodies[i] >= avg_body * ENGULF_BODY_MULT)
            evol = True
            if vol is not None:
                lo_v = max(0, i - VOL_LOOKBACK)
                av = np.nanmean(vol[lo_v:i]) if i > lo_v else np.nan
                if np.isfinite(av) and av > 0:
                    evol = vol[i] >= av * ENGULF_VOL_MULT
            if stands_out and evol:
                emit("engulfing", i, float(h.iloc[i]), float(l.iloc[i]))

        tol = unit(i) * TWEEZER_TOL_ATR
        if abs(lows_arr[i] - lows_arr[i - 1]) <= tol:
            first_falls = pc < po
            second_turns = cc > co
            tvol = True
            if vol is not None and np.isfinite(vol[i - 1]) and vol[i - 1] > 0:
                tvol = vol[i] >= vol[i - 1] * TWEEZER_VOL_MIN
            if first_falls and second_turns and tvol:
                emit("tweezer", i, float(h.iloc[i]),
                     min(lows_arr[i], lows_arr[i - 1]))

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
                emit("divergence", b, neck, neck - unit(b) * ATR_MULT)
            break

    # --- money-flow divergence: lower low on price, higher low on A/D -------
    adv = ad_line(df).to_numpy(float) if "Volume" in df.columns else None
    if adv is not None and np.isfinite(adv).any():
        for x in range(1, len(plows)):
            b = plows[x]
            for y in range(x - 1, -1, -1):
                a = plows[y]
                if b - a > DIV_WINDOW:
                    break
                if lows_arr[b] >= lows_arr[a]:
                    continue
                if not (np.isfinite(adv[a]) and np.isfinite(adv[b])):
                    break
                if adv[b] > adv[a]:
                    neck = float(h.iloc[a:b + 1].max())
                    emit("flowdiv", b, neck, neck - unit(b) * ATR_MULT)
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
            emit("doublebottom", b, neck, foot - unit(b) * 0.25)
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
                emit("invhs", r, neck, vr - u * 0.25)
                break
            else:
                continue
            break

    # --- continuation and base patterns -------------------------------------
    # Exactly the detector the live scan uses, replayed across the history.
    # Sharing it is the point: a backtest of a slightly different flag would
    # be a number about a pattern you never trade.
    for st in scan_structures(df, atr):
        if st.get("entry") is None or st.get("stop") is None:
            continue                       # bearish ones carry no trade
        if st["entry"] <= st["stop"]:
            continue
        emit(st["type"], st["i"], float(st["entry"]), float(st["stop"]))

    # Risk:reward as it looked THEN. The nearest swing high already printed
    # above the entry is the wall price has to get through, which is the same
    # idea the live scan uses -- without recomputing every level from scratch
    # for each of several thousand historical signals.
    hi_at = [(j, highs_arr[j]) for j in phighs]
    highs_all = h.to_numpy(float)
    closes_all = c.to_numpy(float)
    for sig in out:
        # The bar that cleared the level, and whether IT had volume behind it.
        # Different bar from the one the pattern completed on, and for a
        # breakout trade the more relevant of the two.
        sig["bvolOK"] = None
        if vol is not None and sig["entry"]:
            for j in range(sig["i"] + 1, min(n, sig["i"] + 1 + BT_TRIGGER_BARS)):
                if closes_all[j] > sig["entry"]:      # a close, not a wick
                    lo2 = max(0, j - VOL_LOOKBACK)
                    avg2 = np.nanmean(vol[lo2:j]) if j > lo2 else np.nan
                    if np.isfinite(avg2) and avg2 > 0:
                        sig["bvolOK"] = bool(vol[j] >= avg2 * VOL_CONFIRM_MULT)
                    break
        e, st = sig["entry"], sig["stop"]
        risk = (e - st) if (e is not None and st is not None) else None
        wall = None
        if risk and risk > 0:
            above = [px for j, px in hi_at if j < sig["i"] and px > e * 1.002]
            if above:
                wall = min(above)
        sig["rr"] = round((wall - e) / risk, 2) if (wall and risk) else None
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


def sector_rank_timeline(series: dict, step: int = 21) -> list:
    """Sector terciles at monthly checkpoints across the whole history.

    Judging a 2023 signal by today's sector ranking would be look-ahead bias --
    the single easiest way to make a backtest lie. Ranks move slowly, so
    monthly checkpoints are plenty and cost a fraction of ranking per signal.
    """
    dates = sorted({d for s in series.values() for d in s.index})
    first = SECTOR_SKIP + SECTOR_LONG + 5
    out = []
    for i in range(first, len(dates), step):
        d = dates[i]
        ranked = rank_sectors(sector_returns(series, upto=d))
        if ranked:
            out.append((d.strftime("%Y-%m-%d"),
                        {r["name"]: r["tercile"] for r in ranked}))
    return out


def tercile_as_at(timeline: list, date: str, sector: str):
    """The tercile that sector was in on that date, or None before we can say."""
    lo, hi, best = 0, len(timeline) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if timeline[mid][0] <= date:
            best = timeline[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    return best.get(sector) if best else None


def backtest_patterns(frames: dict, symbols: list, meta: dict = None,
                      timeline: list = None, regime: dict = None,
                      deliv_hist: dict = None) -> dict:
    """Hit rates per pattern, and per FILTER.

    The second half is the point: every knob on the dashboard is an opinion
    until it is measured. This splits the same historical signals by whether
    each filter would have passed them, so you can see which ones move the hit
    rate and which are decoration.
    """
    stats, splits, cross = {}, {}, {}
    scanned = 0
    floor = ATRPCT_FLOOR
    # The signal currently being scored. `note` is called once per filter and
    # needs to know which pattern it belongs to; passing it through every call
    # site would be six more arguments to keep in step.
    current = {"kind": None}

    def note(key, label, tf, passed, verdict):
        rec = splits.setdefault(f"{key}|{tf}", {
            "filter": key, "label": label, "timeframe": tf,
            "with": {"win": 0, "loss": 0}, "without": {"win": 0, "loss": 0}})
        rec["with" if passed else "without"][verdict] += 1

        # ...and the same split again, once for this pattern and once for its
        # family. This is the whole point of the cross-tab: an average lift of
        # +3.4 points could be +17 in one pattern and zero everywhere else,
        # and the one-factor table cannot tell those apart.
        kind = current["kind"]
        if not kind:
            return
        for group, gkind in ((kind, "pattern"),
                             (FAMILY_OF.get(kind), "family")):
            if not group:
                continue
            # Namespaced by groupKind on purpose. The family "divergence"
            # and the pattern "divergence" share a name, so an unqualified key
            # put both increments in the same cell and counted every
            # divergence signal twice.
            cell = cross.setdefault(f"{gkind}:{group}|{key}|{tf}", {
                "group": group, "groupKind": gkind, "filter": key,
                "label": label, "timeframe": tf,
                "with": {"win": 0, "loss": 0}, "without": {"win": 0, "loss": 0}})
            cell["with" if passed else "without"][verdict] += 1

    for sym in symbols:
        daily = frames.get(sym)
        if daily is None or len(daily) < 120:
            continue
        dser = (deliv_hist or {}).get(sym) or {}
        dvals = list(dser.values())
        dmean = (sum(dvals) / len(dvals)) if dvals else None
        scanned += 1
        sector = ((meta or {}).get(sym) or {}).get("industry")
        for tf in TIMEFRAMES:
            frame = resample_tf(daily, tf)
            if len(frame) < 80:
                continue
            try:
                signals = historical_signals(frame)
            except Exception:  # noqa: BLE001
                continue
            for sig in signals:
                verdict = evaluate_signal(frame, sig["i"], sig["entry"], sig["stop"])
                if verdict not in ("win", "loss"):
                    continue
                current["kind"] = sig["kind"]
                key = f"{sig['kind']}|{tf}"
                rec = stats.setdefault(key, {"pattern": sig["kind"], "timeframe": tf,
                                             "win": 0, "loss": 0, "open": 0})
                rec[verdict] += 1

                if sig.get("volOK") is not None:
                    note("volume", "Volume confirmed on the signal bar", tf,
                         sig["volOK"], verdict)
                if sig.get("bvolOK") is not None:
                    note("bvol", "Volume on the breakout bar", tf,
                         sig["bvolOK"], verdict)
                # Delivery only reaches back as far as the cache, so most of a
                # five-year history has no reading and is simply not counted.
                # The split's own sample size says how much to trust it.
                if dser and delivery is not None:
                    dpct = delivery.on_date(dser, sig["date"])
                    if dpct is not None:
                        davg = dmean
                        note("deliv",
                             f"Delivery above {DELIV_HIGH:.0f}%", tf,
                             dpct >= DELIV_HIGH, verdict)
                        if davg:
                            note("delivspike",
                                 "Delivery above the stock's own average", tf,
                                 dpct >= davg * DELIV_SPIKE, verdict)
                if sig.get("atrPct") is not None:
                    note("atrpct", f"ATR% above {floor:.0f}", tf,
                         sig["atrPct"] > floor, verdict)
                if sig.get("rr") is not None:
                    note("rr", f"Risk:reward at least 1:{MIN_RR:.0f}", tf,
                         sig["rr"] >= MIN_RR, verdict)
                if timeline and sector:
                    terc = tercile_as_at(timeline, sig["date"], sector)
                    if terc:
                        note("sector", "Sector in the top third", tf,
                             terc == "top", verdict)
                if regime:
                    up = regime.get(sig["date"])
                    if up is not None:
                        note("regime", f"Nifty above its {REGIME_MA}-day average",
                             tf, up, verdict)

    for rec in stats.values():
        decided = rec["win"] + rec["loss"]
        rec["n"] = decided + rec["open"]
        rec["hitRate"] = round(rec["win"] / decided * 100, 1) if decided else None
    rows = sorted(stats.values(), key=lambda r: (r["timeframe"], -(r["hitRate"] or 0)))

    fil = []
    for rec in splits.values():
        for side in ("with", "without"):
            d = rec[side]
            n = d["win"] + d["loss"]
            d["n"] = n
            d["hitRate"] = round(d["win"] / n * 100, 1) if n else None
        a, b = rec["with"]["hitRate"], rec["without"]["hitRate"]
        # Too few either side and the difference is noise, not a finding.
        enough = rec["with"]["n"] >= 30 and rec["without"]["n"] >= 30
        rec["lift"] = round(a - b, 1) if (a is not None and b is not None and enough) else None
        rec["enough"] = enough
        fil.append(rec)
    fil.sort(key=lambda r: (r["timeframe"], -(r["lift"] if r["lift"] is not None else -99)))

    cells = []
    for rec in cross.values():
        for side in ("with", "without"):
            d = rec[side]
            n = d["win"] + d["loss"]
            d["n"] = n
            d["hitRate"] = round(d["win"] / n * 100, 1) if n else None
        a, b = rec["with"]["hitRate"], rec["without"]["hitRate"]
        enough = (rec["with"]["n"] >= BT_MIN_CELL and
                  rec["without"]["n"] >= BT_MIN_CELL)
        rec["lift"] = (round(a - b, 1)
                       if (a is not None and b is not None and enough) else None)
        rec["enough"] = enough
        cells.append(rec)
    cells.sort(key=lambda r: (r["timeframe"], r["groupKind"] != "family",
                              r["group"], r["filter"]))

    return {"sample": scanned, "targetR": BT_TARGET_R, "holdBars": BT_HOLD_BARS,
            "rows": rows, "filters": fil, "cross": cells,
            "minCell": BT_MIN_CELL,
            "familyLabels": FAMILY_LABEL}


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


def sector_series(frames: dict, meta: dict) -> tuple:
    """A daily index per sector, built from its members' own closes.

    Each member is normalised to its own first close so a 3,000-rupee stock
    does not drown a 30-rupee one, then the members are averaged. The result is
    a series you can measure like any other price series -- which is what lets
    the backtest ask what a sector looked like on a date five years ago instead
    of judging an old signal by today's ranking.
    """
    by_sector = {}
    for sym, df in frames.items():
        ind = (meta.get(sym) or {}).get("industry") or "Unclassified"
        if ind in ("", "Unclassified"):
            continue
        close = df["Close"].dropna()
        if len(close) < SECTOR_SKIP + 30:
            continue
        base = float(close.iloc[0])
        if not np.isfinite(base) or base <= 0:
            continue
        by_sector.setdefault(ind, []).append(close / base)

    series, counts = {}, {}
    for ind, members in by_sector.items():
        if len(members) < SECTOR_MIN_N:
            continue
        series[ind] = pd.concat(members, axis=1).mean(axis=1).dropna()
        counts[ind] = len(members)
    return series, counts


def sector_returns(series: dict, upto=None) -> dict:
    """Trailing 12-1 and 6-1 returns per sector, as at `upto` (default: today).

    Passing a past date is the whole point -- it is how a signal from 2023 gets
    judged by the sector ranking that existed in 2023.
    """
    out = {}
    for ind, s in series.items():
        s2 = s if upto is None else s[s.index <= upto]
        n = len(s2)
        if n < SECTOR_SKIP + 40:
            continue
        end = float(s2.iloc[-1 - SECTOR_SKIP])          # skip the last month
        if not np.isfinite(end):
            continue
        def ret(span):
            if n < span + SECTOR_SKIP + 1:
                return None
            start = float(s2.iloc[-1 - SECTOR_SKIP - span])
            if not np.isfinite(start) or start <= 0:
                return None
            v = (end / start - 1) * 100
            return round(v, 2) if np.isfinite(v) else None
        r12, r6 = ret(SECTOR_LONG), ret(SECTOR_SHORT)
        if r12 is None and r6 is None:
            continue
        out[ind] = {"r12": r12, "r6": r6}
    return out


def rank_sectors(rets: dict, counts: dict = None) -> list:
    """Rank by the 12-month reading and split into thirds."""
    rows = [{"name": k, "r12": v["r12"], "r6": v["r6"],
             "members": (counts or {}).get(k)}
            for k, v in rets.items() if v["r12"] is not None]
    rows.sort(key=lambda r: -r["r12"])
    n = len(rows)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
        r["of"] = n
        r["tercile"] = "top" if i < n / 3 else ("bottom" if i >= 2 * n / 3 else "mid")
    return rows


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
            bench: pd.Series = None, deliv: dict = None) -> dict:
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
    if "flowdiv" in SETUPS:
        raw.extend(find_flow_divergence(df, last_atr))

    # Continuation and base patterns. These come back already carrying their
    # own entry, stop and measured move, because for a consolidation those
    # three numbers ARE the pattern -- the edge it broke, the edge it held,
    # and the move that ran into it.
    cont_entries, cont_warnings = live_structures(df, df["ATR"].to_numpy(float))
    raw.extend([c for c in cont_entries if c["type"] in SETUPS])

    # Order matters. Ground the stop on real support FIRST, because every other
    # number -- quantity, deployment, amount at risk, risk:reward -- is derived
    # from the distance between entry and stop. Size it, then measure the reward
    # against a price the chart can actually reach.
    profile = volume_profile(df)
    setups = []
    for s in raw:
        s.setdefault("direction", "long")
        ground_stop(s, last_atr, levels)
        size_setup(s, last_price)
        attach_reward(s, levels, last_price, high_52, profile)
        age = s.get("ageBars")
        sig_idx = (len(df) - 1 - age) if isinstance(age, int) else len(df) - 1
        s.update(volume_state(df, sig_idx))
        # ...and the OTHER volume question: the bar that cleared the level.
        s.update(breakout_volume(df, s.get("entry"), sig_idx))
        # Delivery on the day the setup formed: volume says shares moved,
        # delivery says somebody kept them.
        if deliv and delivery is not None:
            s["delivOnSignal"] = delivery.on_date(deliv, s.get("date"))
        setups.append(s)

    # --- topping patterns: warnings, never entries -------------------------
    warnings = []
    if "doubletop" in WARNINGS:
        warnings.extend(find_double_top(df, last_atr))
    if "hs" in WARNINGS:
        warnings.extend(find_head_shoulders(df, last_atr))
    warnings.extend([c for c in cont_warnings if c["type"] in WARNINGS])
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
        "breakoutVolumeConfirmed": (primary or {}).get("breakoutVolumeConfirmed"),
        "breakoutVolumeRatio": (primary or {}).get("breakoutVolumeRatio"),
        "breakoutDate": (primary or {}).get("breakoutDate"),
        "breakoutBars": (primary or {}).get("breakoutBars"),
        "brokeOut": bool((primary or {}).get("brokeOut")),
        "runway": (primary or {}).get("runway"),
        "shelfShare": (primary or {}).get("shelfShare"),
        # The profile itself, minus the forty-bucket histogram: the page wants
        # the three prices, not a year of buckets on every one of 750 rows.
        "delivOnSignal": (primary or {}).get("delivOnSignal"),
        "poc": (profile or {}).get("poc"),
        "valueHigh": (profile or {}).get("valueHigh"),
        "valueLow": (profile or {}).get("valueLow"),
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
                 benches: dict, deliv: dict = None) -> dict:
    """One stock across every timeframe, as a single record."""
    results, complete = {}, {}
    for tf in TIMEFRAMES:
        frame = resample_tf(daily, tf)
        complete[tf] = bar_is_complete(daily, frame, tf)
        try:
            results[tf] = analyse(symbol, name, meta, frame, benches.get(tf), deliv)
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

    # Delivery is a daily-only figure -- NSE publishes one number per session --
    # so it describes the stock, not the candle size you are looking at.
    if delivery is not None:
        rec.update(delivery.metrics(deliv or {}))

    # Recent daily bars for the journal: [high, low, close] per session,
    # aligned to the payload's shared date axis.
    tail = daily.tail(BAR_HISTORY)
    rec["barDates"] = [d.strftime("%Y-%m-%d") for d in tail.index]
    rec["bars"] = [[round(float(h), 2), round(float(l), 2), round(float(c), 2)]
                   for h, l, c in zip(tail["High"], tail["Low"], tail["Close"])]
    # Weekly closes so the sparkline spans the same candles as the rest of the
    # row. A chart showing five weeks while you trade weekly would mislead.
    # Closes only -- a sparkline needs the shape, not the highs and lows.
    wk = resample_tf(daily, "weekly").tail(BAR_HISTORY)
    rec["closesW"] = [round(float(c), 2) for c in wk["Close"]]

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
    # Phase timings, so a slow run can be diagnosed from the log instead of
    # guessed at. Nearly all of it is normally the download.
    clock = {}
    t_start = time.time()
    def phase(name, t0):
        clock[name] = time.time() - t0
        print(f"  [{clock[name]:6.1f}s] {name}", flush=True)
        return time.time()

    t = time.time()
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

    t = phase("constituent lists", t)
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

    # --- delivery percentage ------------------------------------------------
    # NSE publishes one file per session and there is no bulk download, so the
    # first run walks back a year (a few minutes, once) and every run after it
    # collects the single day it is missing. Bounded by a time budget: a slow
    # NSE evening must not be able to hang the whole scan.
    deliv_hist = {}
    if DELIVERY and delivery is not None:
        try:
            stat = delivery.ensure_history(days=DELIVERY_DAYS)
            hist = delivery.load_history(days=DELIVERY_DAYS)
            print(f"  delivery: {stat['cached']} of {stat['days']} weekdays cached "
                  f"({stat['fetched']} fetched this run, {stat['holidays']} holidays, "
                  f"{stat['seconds']}s)", flush=True)
            if hist:
                wanted = {row["symbol"] for row in watchlist}
                for sym in wanted:
                    ser = delivery.for_symbol(hist, sym)
                    if ser:
                        deliv_hist[sym] = ser
                print(f"  delivery: history for {len(deliv_hist)} of {len(wanted)} "
                      f"stocks across {len(hist)} sessions", flush=True)
            else:
                print("  delivery: nothing cached yet -- the page hides the "
                      "column until there is", flush=True)
        except Exception as exc:  # noqa: BLE001
            # Delivery is an extra, not a dependency. If NSE will not talk to
            # us the scan still has to produce a page.
            print(f"  delivery unavailable ({exc}) -- carrying on without it",
                  flush=True)
        t = phase("delivery percentage", t)

    frames, missing = fetch_frames([row["symbol"] for row in watchlist])
    t = phase("price downloads", t)
    if not frames:
        print("No price data came back at all -- leaving data.json untouched.", file=sys.stderr)
        return 1

    rows = []
    for row in watchlist:
        sym = row["symbol"]
        if sym not in frames:
            continue
        try:
            rows.append(build_record(sym, row["name"], row, frames[sym], benches,
                                     deliv_hist.get(sym)))
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

    t = phase("indicator + pattern analysis", t)

    # --- sector strength ---------------------------------------------------
    print("measuring sector strength", flush=True)
    sec_series, sec_counts = sector_series(frames, meta)
    sectors_ranked = rank_sectors(sector_returns(sec_series), sec_counts)
    for r in sectors_ranked[:3] + (["..."] if len(sectors_ranked) > 6 else []) + sectors_ranked[-3:]:
        if r == "...":
            print("   ...", flush=True); continue
        print(f"  {r['rank']:>2}. {r['name'][:28]:<28} 12m {str(r['r12']):>7}%  "
              f"6m {str(r['r6']):>7}%  ({r['members']} stocks, {r['tercile']})", flush=True)
    terc_now = {r["name"]: r["tercile"] for r in sectors_ranked}
    rank_now = {r["name"]: r["rank"] for r in sectors_ranked}
    for row in rows:
        ind = row.get("industry")
        row["sectorRank"] = rank_now.get(ind)
        row["sectorTercile"] = terc_now.get(ind)

    # --- market regime ------------------------------------------------------
    regime_map, regime_now = {}, None
    if bench is not None and len(bench) > REGIME_MA:
        ma = bench.rolling(REGIME_MA).mean()
        above = (bench > ma).dropna()
        regime_map = {d.strftime("%Y-%m-%d"): bool(v) for d, v in above.items()}
        regime_now = bool(above.iloc[-1])
        print(f"market regime: Nifty is {'ABOVE' if regime_now else 'BELOW'} its "
              f"{REGIME_MA}-day average", flush=True)

    t = phase("sector strength + regime", t)

    backtest = None
    if BACKTEST:
        # Evenly spaced across the watchlist so the sample spans large, mid,
        # small and micro caps rather than whichever names sort first.
        have = [r["symbol"] for r in watchlist if r["symbol"] in frames]
        if BT_SAMPLE and BT_SAMPLE < len(have):
            step = max(1, len(have) // BT_SAMPLE)
            sample = have[::step][:BT_SAMPLE]
        else:
            sample = have
        print(f"backtesting {len(sample)} of {len(have)} stocks "
              f"({BT_TARGET_R:.0f}R target, {BT_HOLD_BARS}-bar horizon)", flush=True)
        t0 = time.time()
        timeline = sector_rank_timeline(sec_series)
        backtest = backtest_patterns(frames, sample, meta, timeline, regime_map,
                                     deliv_hist)
        print(f"  took {time.time() - t0:.0f}s", flush=True)
        for rec in backtest["rows"]:
            rate = f"{rec['hitRate']}%" if rec["hitRate"] is not None else "n/a"
            print(f"  {rec['pattern']:<14} {rec['timeframe']:<7} "
                  f"{rate:>6} of {rec['win'] + rec['loss']:>5} decided", flush=True)
        print("  --- do the filters earn their place? ---", flush=True)
        for rec in backtest["filters"]:
            lift = f"{rec['lift']:+.1f} pts" if rec["lift"] is not None else "too few"
            print(f"  {rec['filter']:<8} {rec['timeframe']:<7} "
                  f"with {str(rec['with']['hitRate']):>5}% (n={rec['with']['n']:>5})  "
                  f"without {str(rec['without']['hitRate']):>5}% (n={rec['without']['n']:>5})  "
                  f"-> {lift}", flush=True)

    t = phase("backtest", t)

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
        "sectorStrength": sectors_ranked,
        "sectorWindows": {"long": SECTOR_LONG, "short": SECTOR_SHORT,
                          "skip": SECTOR_SKIP},
        "regime": {"above": regime_now, "ma": REGIME_MA},
        # The page hides its delivery column entirely when this says nothing
        # has been collected yet, rather than printing a column of dashes on
        # the first run while the backfill is still catching up.
        "delivery": {
            "stocks": len(deliv_hist),
            "sessions": len({d for ser in deliv_hist.values() for d in ser}),
            "high": DELIV_HIGH, "spike": DELIV_SPIKE,
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
    payload = json_safe(payload)

    # Compact separators, not indent=1. At ~750 stocks the pretty version is
    # about 1.2 MB and this one about 800 KB, for identical content -- and the
    # file is re-committed every trading day, so the saving compounds.
    with open("data.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"), allow_nan=False)

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
                # Risk first, always 1. The e-mail digest was still printing
                # this the other way round while the page had been fixed.
                rr = f", risk:reward 1:{v['rr']:.2f}" if v.get("rr") else ""
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

    phase("writing data.json + alerts", t)
    print(f"\ntotal {time.time() - t_start:.0f}s  "
          f"({', '.join(f'{k} {v:.0f}s' for k, v in sorted(clock.items(), key=lambda x: -x[1])[:3])})",
          flush=True)
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
