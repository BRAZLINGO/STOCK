"""Does every detector find its own pattern, in the right place, and only there?

These are the checks that caught the real bugs while this scanner was being
built -- a "flag" that was a pole with no flag on it, a trendline that touched
exactly one point by construction, a neckline drawn through an unrelated spike.
They lived outside the repository for a while, which meant none of them could
stop a bad change. They live here now, and CI runs them before every scan.

Everything is synthetic and offline: each case is a chart drawn on purpose so
the right answer is known in advance.

Run:  python -m pytest test_patterns.py -q
"""
import numpy as np
import pandas as pd
import pytest

import scanner as S


# --- drawing tools ----------------------------------------------------------
# A fixed seed per drawing. NOT hash(name): Python randomises string hashing
# per process, so a suite keyed on it passes and fails at random between runs,
# which is worse than no suite at all.
SEEDS = {"flag": 11, "pennant": 12, "rectangle": 13, "asctriangle": 14,
         "desctriangle": 15, "symtriangle": 16, "rounding": 17, "cuphandle": 18}


def rng_for(case):
    """One random stream per case.

    A single shared stream means editing one drawing shifts the noise in every
    drawing after it, and unrelated cases change verdict -- which makes the
    suite useless for telling a real regression from a reshuffle.
    """
    return np.random.default_rng(1000 + case)


def frame(closes, vols=None, rng=None, wiggle=0.004):
    """OHLCV around a close path, with a small honest intrabar range."""
    rng = rng or np.random.default_rng(0)
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    idx = pd.bdate_range("2021-01-04", periods=n)
    op = np.r_[closes[0], closes[:-1]] + rng.normal(0, wiggle, n) * closes
    hi = np.maximum(op, closes) * (1 + abs(rng.normal(0, wiggle, n)))
    lo = np.minimum(op, closes) * (1 - abs(rng.normal(0, wiggle, n)))
    if vols is None:
        vols = np.full(n, 120_000.0)
    v = np.asarray(vols, dtype=float) * (1 + rng.normal(0, 0.12, n))
    return pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": closes,
                         "Volume": np.abs(v)}, index=idx)


def drift(rng, start, n, pct_total, jitter=0.004):
    step = (start * (1 + pct_total) - start) / max(n, 1)
    path = start + step * np.arange(1, n + 1)
    return path * (1 + rng.normal(0, jitter, n))


def zig(centre, n, amp, cycles, tilt_lo=0.0, tilt_hi=0.0):
    """A saw-tooth that touches a roof and a floor, each allowed to tilt."""
    t = np.arange(n)
    wave = np.sin(2 * np.pi * cycles * t / max(n - 1, 1))
    roof = centre + amp + tilt_hi * t
    floor = centre - amp + tilt_lo * t
    mid, half = (roof + floor) / 2, (roof - floor) / 2
    return mid + wave * half


def flat(rng, n=120, level=100.0):
    return level * (1 + rng.normal(0, 0.004, n)).cumprod()


def vols(*parts):
    return np.concatenate([np.full(n, lvl, dtype=float) for n, lvl in parts])


# --- one drawing per pattern ------------------------------------------------
def draw(name, seed=None):
    """Returns (frame, the bar range the pattern occupies)."""
    r = np.random.default_rng(seed if seed is not None else 1000 + SEEDS[name])
    if name == "flag":
        # quiet base, loud sharp pole, then a tight channel drifting DOWN
        path = np.r_[flat(r, 80, 100), drift(r, 100, 10, 0.22),
                     zig(118, 14, 2.5, 1.5, tilt_lo=-0.25, tilt_hi=-0.25)]
        return frame(path, vols((80, 1e5), (10, 4.2e5), (14, 9e4)), r), (90, 103)
    if name == "pennant":
        # the same pole, then converging lines that do NOT cross
        path = np.r_[flat(r, 80, 100), drift(r, 100, 10, 0.22),
                     zig(116, 14, 4.0, 2.0, tilt_lo=0.21, tilt_hi=-0.21)]
        return frame(path, vols((80, 1e5), (10, 4.2e5), (14, 9e4)), r), (90, 103)
    if name == "rectangle":
        path = np.r_[flat(r, 60, 90), drift(r, 90, 20, 0.15),
                     zig(103, 60, 1.6, 4.0)]
        return frame(path, None, r), (80, 139)
    if name == "asctriangle":
        return frame(np.r_[flat(r, 60, 90), zig(96, 70, 4.0, 3.0, tilt_lo=0.085)],
                     None, r), (60, 129)
    if name == "desctriangle":
        return frame(np.r_[flat(r, 60, 110), zig(104, 70, 4.0, 3.0, tilt_hi=-0.085)],
                     None, r), (60, 129)
    if name == "symtriangle":
        # Ninety bars and four swings a side, not seventy and three. A short
        # triangle is genuinely marginal -- detection depended on where the
        # noise fell, which is a fact about real charts as much as this test.
        return frame(np.r_[flat(r, 60, 100),
                           zig(100, 90, 6.0, 4.0, tilt_lo=0.05, tilt_hi=-0.05)],
                     None, r), (60, 149)
    x = np.linspace(-1, 1, 130)
    cup = 100 - 22 * (1 - x ** 2)
    if name == "rounding":
        return frame(np.r_[flat(r, 50, 100), cup], None, r), (50, 179)
    if name == "cuphandle":
        # a handle that STEADIES; one still falling on the last bar has failed.
        # Shallow, because a handle retracing more than a third of the cup is
        # a second leg down and the detector is right to refuse it.
        path = np.r_[flat(r, 50, 100), cup, drift(r, 100, 5, -0.035),
                     drift(r, 96.5, 6, 0.02)]
        return frame(path, None, r), (50, 191)
    raise ValueError(name)


ALL = ["flag", "pennant", "rectangle", "asctriangle", "desctriangle",
       "symtriangle", "rounding", "cuphandle"]


@pytest.mark.parametrize("name", ALL)
def test_pattern_is_found_where_it_was_drawn(name):
    """Across EIGHT different noise streams, not one.

    A single drawing tests whether one lucky arrangement of noise happens to
    be detectable. Eight measures whether the detector is robust, which is the
    thing worth knowing -- and it stops a passing test from being an accident
    of the seed.
    """
    found = 0
    misses = []
    for k in range(8):
        df, (lo, hi) = draw(name, seed=500 + k)
        atr = S.wilder_atr(df).to_numpy(float)
        hit = [s for s in S.scan_structures(df, atr)
               if s["type"] == name and lo <= s["i"] <= hi + 6]
        if hit:
            found += 1
        else:
            misses.append(500 + k)
    assert found >= 7, (
        f"{name} found in only {found} of 8 noise streams "
        f"(missed on seeds {misses}) -- too fragile to trust")


@pytest.mark.parametrize("name", ALL)
def test_every_signal_is_tradeable(name):
    """A stop above its entry is not a trade, and a 40% stop is not a stop."""
    df, _ = draw(name)
    atr = S.wilder_atr(df).to_numpy(float)
    for s in S.scan_structures(df, atr):
        if s.get("entry") is None:
            continue                       # bearish ones carry no trade
        assert s["stop"] < s["entry"], f"{s['type']}: stop at or above entry"
        risk = (s["entry"] - s["stop"]) / s["entry"] * 100
        assert risk <= 40, f"{s['type']}: risks {risk:.0f}% of the entry"


def test_bearish_triangle_never_offers_a_trade():
    df, _ = draw("desctriangle")
    atr = S.wilder_atr(df).to_numpy(float)
    found = [s for s in S.scan_structures(df, atr) if s["type"] == "desctriangle"]
    assert found, "descending triangle not detected"
    for s in found:
        assert s.get("entry") is None, "a breakdown pattern must carry no entry"
    assert "desctriangle" in S.WARNINGS


def test_a_pole_with_no_flag_is_not_a_flag():
    """The bug that started the rewrite.

    A sharp run followed by a tight drift UPWARD is a pole with nothing on the
    end of it. The old detector measured the pause as a box, allowed its high
    to sit above the pole's top, and never checked that price pulled back at
    all -- so this qualified.
    """
    r = rng_for(41)
    path = np.r_[flat(r, 80, 100), drift(r, 100, 10, 0.22),
                 drift(r, 122, 10, 0.02)]          # still creeping up: no flag
    df = frame(path, vols((80, 1e5), (10, 4.2e5), (10, 9e4)), r)
    atr = S.wilder_atr(df).to_numpy(float)
    late = [s for s in S.scan_structures(df, atr)
            if s["type"] in ("flag", "pennant") and s["i"] >= 90]
    assert not late, f"a pole with no pause was reported as {late}"


def test_trendline_contains_the_points_it_is_drawn_through():
    """Two earlier versions of this failed, both plausibly.

    A least-squares fit slid above every high touches exactly one point, so a
    "two touches" rule can never pass. A line through the two HIGHEST points
    drops too steeply in a falling channel and leaves later highs above it.
    The hull edge is the one a ruler finds.
    """
    highs = np.array([10.0, 9.4, 9.8, 9.0, 9.4, 8.6, 9.0, 8.2])
    slope, b, touches = S._hull_line(highs, True, 0.05)
    assert slope is not None
    line = slope * np.arange(len(highs)) + b
    assert (highs <= line + 1e-9).all(), "highs poke above their own trendline"
    assert touches >= 2, "a trendline needs at least two touches"

    lows = highs - 1.0
    slope2, b2, touches2 = S._hull_line(lows, False, 0.05)
    line2 = slope2 * np.arange(len(lows)) + b2
    assert (lows >= line2 - 1e-9).all(), "lows fall through their own trendline"
    assert touches2 >= 2


def test_neckline_ignores_a_spike_outside_the_rallies():
    """The inverse head-and-shoulders bug.

    The neckline joins the two reaction highs either side of the head. Taking
    the highest high anywhere in the formation put the entry far above the
    level price actually has to clear.
    """
    highs = pd.Series(np.array(
        [10, 35, 10, 10, 10, 20, 12, 11, 10, 11, 12, 13, 16, 20, 15, 14, 13],
        dtype=float))
    neck = S._neckline(highs, 1, 10, 16, invert=True)
    assert neck == pytest.approx(20.0, abs=0.5), (
        f"neckline {neck} should sit on the two rallies at 20, not the spike at 35")


def test_neckline_never_projects_below_its_own_points():
    """A steep neckline projected far enough lands under the pattern, and an
    entry beneath the pattern's own rallies is a price already passed."""
    highs = pd.Series(np.array(
        [10, 10, 10, 10, 10, 30, 12, 11, 10, 11, 12, 13, 18, 15, 14, 13, 12],
        dtype=float))
    neck = S._neckline(highs, 1, 10, 16, invert=True)
    assert neck >= 18.0, f"neckline {neck} projected below the lower reaction high"


def _inverse_hs_with_a_tall_left_rally():
    """An inverse H&S whose LEFT rally is much taller than its right one.

    On this chart the two rules disagree loudly: the neckline joining the two
    reaction highs sits near 105, while "the highest high anywhere in the
    formation" says 118.
    """
    seq = ([104] * 8 + [100, 98, 100] + [118] * 3 + [104, 100]
           + [96, 92, 90, 92, 96]
           + [105] * 3 + [104, 102]
           + [100, 98, 100] + [102] * 10)
    c = np.array(seq, dtype=float)
    idx = pd.bdate_range("2023-01-02", periods=len(c))
    return pd.DataFrame({"Open": c, "High": c * 1.004, "Low": c * 0.996,
                         "Close": c, "Volume": np.full(len(c), 1e6)}, index=idx)


def test_inverse_hs_entry_uses_the_neckline_not_the_tallest_bar():
    """End to end, not just the helper.

    An earlier version of this suite tested `_neckline` on its own and passed
    while `find_inverse_hs` still called the old rule -- the unit was right
    and the detector never used it. Mutation testing found that hole, so this
    one goes through the detector.
    """
    df = _inverse_hs_with_a_tall_left_rally()
    atr = S.wilder_atr(df).to_numpy(float)
    found = S.find_inverse_hs(df, float(atr[-1]))
    assert found, "the inverse head and shoulders was not detected at all"
    entry = found[0]["entry"]
    tallest = float(df["High"].max())
    assert entry < tallest - 5, (
        f"entry {entry} is at the tallest bar ({tallest:.1f}) -- the neckline "
        f"must join the two reaction highs, not the highest high")
    assert 103 <= entry <= 109, f"entry {entry} is not on the neckline"


def test_noise_does_not_produce_a_pattern_every_few_bars():
    """A smoke test. Random walks DO contain ranges and curves, so some
    detections are correct; the real check that noise carries no EDGE is
    test_lookahead.py, which measures hit rates rather than counts."""
    r = rng_for(9)
    df = frame(flat(r, 400, 100), None, r)
    atr = S.wilder_atr(df).to_numpy(float)
    found = S.scan_structures(df, atr)
    assert len(found) <= 30, (
        f"{len(found)} signals in 400 bars of noise is too eager")
