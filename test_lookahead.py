"""The look-ahead test: noise must not carry an edge.

Random walks contain no information about their own future. So a detector run
over them should score at the break-even rate and no better. Anything well
ABOVE break-even means the detector is reading bars it should not be able to
see -- a window that includes one bar too many, an entry taken from a high
that had not printed yet. That class of bug is invisible in ordinary testing
and fatal in a backtest, because it produces numbers that look wonderful and
cannot be traded.

This is the check that matters most in the whole suite. It is also the one
that says nothing about whether the patterns are PROFITABLE -- only that the
measurement is honest.

Run:  python -m pytest test_lookahead.py -q
"""
import numpy as np
import pandas as pd
import pytest

import scanner as S

N_STOCKS = 40          # enough for the aggregate; CI runs this on every push
N_BARS = 1300          # about five years of daily bars


def random_stock(seed):
    """A believable midcap that nonetheless knows nothing about its future."""
    r = np.random.default_rng(seed)
    vol = 0.018 * (1 + 0.5 * np.sin(np.arange(N_BARS) / 90.0))   # clustered
    close = 100 * np.exp(np.cumsum(r.normal(0.0004, 1.0, N_BARS) * vol))
    intraday = np.abs(r.normal(0, 0.008, N_BARS)) * close
    op = np.r_[close[0], close[:-1]] * (1 + r.normal(0, 0.003, N_BARS))
    idx = pd.bdate_range("2021-01-04", periods=N_BARS)
    return pd.DataFrame({"Open": op,
                         "High": np.maximum(op, close) + intraday,
                         "Low": np.minimum(op, close) - intraday,
                         "Close": close,
                         "Volume": r.integers(50_000, 500_000, N_BARS)}, index=idx)


@pytest.fixture(scope="module")
def noise_results():
    """Every pattern's hit rate on pure noise, measured once for all tests."""
    stats, fires = {}, {}
    for k in range(N_STOCKS):
        df = random_stock(2000 + k)
        for sig in S.historical_signals(df):
            kind = sig["kind"]
            fires[kind] = fires.get(kind, 0) + 1
            v = S.evaluate_signal(df, sig["i"], sig["entry"], sig["stop"])
            if v in ("win", "loss"):
                rec = stats.setdefault(kind, [0, 0])
                rec[0 if v == "win" else 1] += 1
    return stats, fires


def break_even():
    return 100.0 / (1.0 + S.BT_TARGET_R)


def test_no_pattern_beats_break_even_on_noise(noise_results):
    stats, _ = noise_results
    be = break_even()
    offenders = []
    for kind, (w, l) in sorted(stats.items()):
        n = w + l
        if n < 80:
            continue                      # too few to judge either way
        hit = 100.0 * w / n
        if hit - be > 8:
            offenders.append(f"{kind} {hit:.1f}% on {n} trades ({hit - be:+.1f})")
    assert not offenders, (
        "these scored well above break-even on data with no edge in it, "
        "which means they are reading the future: " + "; ".join(offenders))


def test_patterns_actually_fire(noise_results):
    """The mirror of the test above. A detector that never fires cannot read
    the future either, and would pass the look-ahead check by doing nothing."""
    _, fires = noise_results
    for kind in ("flag", "doublebottom", "invhs", "rectangle", "tweezer"):
        assert fires.get(kind, 0) > 0, f"{kind} never fired on {N_STOCKS} stocks"


def test_entries_are_above_their_stops():
    df = random_stock(99)
    for sig in S.historical_signals(df):
        if sig["entry"] is None or sig["stop"] is None:
            continue
        assert sig["entry"] > sig["stop"], f"{sig['kind']} entry at or below stop"


def test_evaluate_signal_scores_a_same_bar_touch_as_a_loss():
    """When one bar touches both the stop and the target there is no way to
    know which came first without intraday data. The backtest must assume the
    worse outcome; guessing the better one quietly inflates every number."""
    idx = pd.bdate_range("2025-01-01", periods=6)
    # bar 2 spans both levels: entry 100, stop 95, 2R target 110
    rows = [[100, 100, 100], [101, 99, 100], [115, 90, 100],
            [100, 100, 100], [100, 100, 100], [100, 100, 100]]
    df = pd.DataFrame(rows, columns=["High", "Low", "Close"], index=idx)
    df["Open"] = df["Close"]
    df["Volume"] = 1.0
    assert S.evaluate_signal(df, 0, 100.0, 95.0) == "loss"


def test_an_entry_that_never_triggers_is_not_counted():
    idx = pd.bdate_range("2025-01-01", periods=8)
    df = pd.DataFrame({"High": np.full(8, 90.0), "Low": np.full(8, 88.0),
                       "Close": np.full(8, 89.0), "Open": np.full(8, 89.0),
                       "Volume": np.ones(8)}, index=idx)
    # entry far above anything that prints: no trade, not a loss
    assert S.evaluate_signal(df, 0, 150.0, 140.0) == ""
