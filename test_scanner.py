"""Offline checks for the scanner's indicators and divergence rule.

Run:  python test_scanner.py
No network needed -- it builds synthetic price series with known answers.
"""
import numpy as np
import pandas as pd

import scanner as S

META = {"industry": "Test Sector", "universes": ["TESTIDX"]}


def frame(close, high=None, low=None):
    n = len(close)
    idx = pd.bdate_range("2025-01-01", periods=n)
    close = np.asarray(close, dtype=float)
    high = close * 1.01 if high is None else np.asarray(high, dtype=float)
    low = close * 0.99 if low is None else np.asarray(low, dtype=float)
    return pd.DataFrame({"Open": close, "High": high, "Low": low,
                         "Close": close, "Volume": np.ones(n)}, index=idx)


def test_rsi_matches_wilder_reference():
    # Wilder's own worked example (New Concepts in Technical Trading Systems).
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
              46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41,
              46.22, 45.64]
    rsi = S.wilder_rsi(pd.Series(closes))
    got = round(float(rsi.iloc[14]), 2)
    assert 70.0 < got < 70.9, f"RSI(14) at bar 15 should be ~70.5, got {got}"
    print(f"  RSI reference bar = {got} (expected ~70.46)")


def test_rsi_bounds_and_extremes():
    rising = S.wilder_rsi(pd.Series(np.arange(1, 60, dtype=float)))
    assert round(float(rising.iloc[-1]), 1) == 100.0, "all-up series must pin RSI at 100"
    falling = S.wilder_rsi(pd.Series(np.arange(60, 1, -1, dtype=float)))
    assert float(falling.iloc[-1]) < 1.0, "all-down series must pin RSI near 0"
    print("  RSI extremes ok (100 / ~0)")


def test_atr_constant_range():
    n = 60
    close = np.full(n, 100.0)
    df = frame(close, high=close + 3, low=close - 3)   # true range is always 6
    atr = S.wilder_atr(df)
    assert abs(float(atr.iloc[-1]) - 6.0) < 1e-6, f"ATR should be 6.0, got {atr.iloc[-1]}"
    print("  ATR on a constant 6-point range = 6.00")


def test_pivots():
    s = pd.Series([10, 9, 8, 7, 6, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 5, 6, 7, 8, 9])
    lows = S.pivot_lows(s, k=3)
    assert 5 in lows, f"expected a pivot low at index 5, got {lows}"
    assert 16 in lows, f"expected a pivot low at index 16, got {lows}"
    print(f"  pivot lows found at {lows}")


def bullish_divergence_series():
    """Price makes a LOWER bottom; RSI makes a HIGHER bottom.

    Shape: deep fast crash to bottom A, strong bounce, then a slow shallow
    drift to bottom B slightly below A. The second decline is gentler, so
    Wilder RSI reads higher at B than at A -- textbook bullish divergence.
    """
    seq = []
    seq += [100] * 20                       # flat base
    seq += list(np.linspace(100, 60, 18))   # violent drop -> bottom A (deep RSI)
    seq += list(np.linspace(60, 88, 14))    # sharp bounce -> the resistance
    seq += list(np.linspace(88, 58, 26))    # slow grind -> bottom B (lower price)
    seq += list(np.linspace(58, 75, 12))    # recovery off bottom B
    return np.array(seq, dtype=float)


def test_divergence_detected():
    close = bullish_divergence_series()
    df = frame(close)
    out = S.analyse("TEST", "Test Co", META, df)
    assert out["divergence"] is True, f"divergence should be detected, got {out}"
    m = out["marks"]
    assert m["lowB"] < m["lowA"], f"bottom B must be lower: {m}"
    assert m["rsiB"] > m["rsiA"], f"RSI at B must be higher: {m}"
    assert out["resistance"] is not None and out["resistance"] > m["lowB"], \
        "resistance must sit above the second bottom"
    print(f"  divergence: price {m['lowA']} -> {m['lowB']} (lower), "
          f"RSI {m['rsiA']} -> {m['rsiB']} (higher), resistance {out['resistance']}")
    print(f"  status={out['status']} stop={out['stop']} qty={out['qty']}")
    return out


def test_no_divergence_on_clean_downtrend():
    close = np.linspace(200, 80, 140)       # price and RSI both fall
    out = S.analyse("DOWN", "Falling Co", META, frame(close))
    assert out["divergence"] is False, "a clean downtrend is not a divergence"
    print(f"  clean downtrend -> divergence={out['divergence']}, status={out['status']}")


def test_sizing_math():
    out = test_divergence_detected()
    rpt = S.TOTAL_RISK / S.RPT_DIVISOR
    expected_risk = round(out["atr"] * S.ATR_MULT, 2)
    assert abs(out["riskPerShare"] - expected_risk) < 0.02, "risk per share = ATR x mult"
    assert out["qty"] == int(rpt // out["riskPerShare"]), "qty = RPT / risk per share"
    assert abs((out["entry"] - out["stop"]) - out["riskPerShare"]) < 0.02, \
        "stop must sit exactly one risk unit below entry"
    at_risk = out["qty"] * out["riskPerShare"]
    assert at_risk <= rpt + 0.01, f"never risk more than the RPT: {at_risk} vs {rpt}"
    print(f"  RPT={rpt:.0f}  risk/share={out['riskPerShare']}  qty={out['qty']}  "
          f"actually at risk={at_risk:.0f}")


def test_triggered_state():
    """Clearing the divergence's resistance must flip it to triggered.

    The resistance is anchored to the second bottom, so it must NOT drift
    upward as price rises -- otherwise a breakout would never register.
    """
    close = list(bullish_divergence_series())
    out = S.analyse("T", "T", META, frame(np.array(close)))
    res = out["resistance"]
    close += list(np.linspace(close[-1], res * 1.03, 10))
    out2 = S.analyse("T", "T", META, frame(np.array(close)))
    print(f"  armed -> {out['status']} (resistance {res}); "
          f"after breakout -> {out2['status']} "
          f"(price {out2['price']} vs resistance {out2['resistance']})")
    assert out["status"] == "armed"
    assert out2["resistance"] == res, \
        f"resistance drifted from {res} to {out2['resistance']} as price rose"
    assert out2["status"] == "triggered" and out2["broke"] is True


def test_short_history_is_skipped():
    out = S.analyse("TINY", "Tiny", META, frame(np.full(20, 100.0)))
    assert out["status"] == "nodata"
    print("  short history handled without crashing")



def test_selection_measures_present():
    """ATR%, turnover and the risk/reward targets must come through."""
    out = S.analyse("SEL", "Sel Co", META, frame(bullish_divergence_series()))
    assert out["industry"] == "Test Sector" and out["universes"] == ["TESTIDX"]
    assert out["atrPct"] is not None and 0 < out["atrPct"] < 100
    assert out["turnover"] is not None
    assert len(out["targets"]) == len(S.RR_TARGETS)
    for mult, tgt in zip(S.RR_TARGETS, out["targets"]):
        expected = out["entry"] + out["riskPerShare"] * mult
        assert abs(tgt - expected) < 0.02, f"target 1:{mult} wrong"
    assert out["targets"][0] > out["entry"] > out["stop"], "target/entry/stop out of order"
    print(f"  ATR%={out['atrPct']}  targets={out['targets']}  entry={out['entry']}  stop={out['stop']}")


def test_relative_strength():
    """Beating the index reads as outperforming; lagging it does not."""
    n = 300
    idx = pd.Series(np.linspace(100, 130, n), index=pd.bdate_range("2025-01-01", periods=n))
    strong = pd.Series(np.linspace(100, 220, n), index=idx.index)
    weak = pd.Series(np.linspace(100, 104, n), index=idx.index)
    up, gap = S.comparative_strength(strong, idx)
    dn, gap2 = S.comparative_strength(weak, idx)
    assert up is True, f"a stock far outpacing the index must read outperforming ({gap})"
    assert dn is False, f"a lagging stock must read under ({gap2})"
    print(f"  strong: outperforming (+{gap}%)   weak: lagging ({gap2}%)")


def test_relative_strength_without_benchmark():
    out = S.comparative_strength(pd.Series([1.0] * 50), None)
    assert out == (None, None), "no benchmark must degrade to blank, not crash"
    print("  missing benchmark handled")


def ohlc(rows):
    """rows = list of (open, high, low, close)."""
    a = np.array(rows, dtype=float)
    idx = pd.bdate_range("2025-01-01", periods=len(a))
    return pd.DataFrame({"Open": a[:,0], "High": a[:,1], "Low": a[:,2],
                         "Close": a[:,3], "Volume": np.full(len(a), 1e6)}, index=idx)


def downtrend_rows(n=40, start=200.0, step=2.0):
    """A clean stair-step decline: each candle red, each low lower."""
    rows = []
    p = start
    for _ in range(n):
        o = p; c = p - step; rows.append((o, o + 0.4, c - 0.4, c)); p = c
    return rows


def test_engulfing_detected():
    rows = downtrend_rows()
    o_prev, c_prev = rows[-1][0], rows[-1][3]           # last red candle
    # green candle that swallows it whole: opens below its close, closes above its open
    rows.append((c_prev - 1.0, o_prev + 4.0, c_prev - 2.0, o_prev + 2.0))
    df = ohlc(rows)
    found = S.find_engulfing(df)
    assert found, "a textbook engulfing after a downtrend must be found"
    e = found[-1]
    assert e["type"] == "engulfing"
    assert abs(e["entry"] - float(df["High"].iloc[-1])) < 1e-6, "entry = the green candle's high"
    assert abs(e["stop"] - float(df["Low"].iloc[-1])) < 1e-6, "stop = the green candle's low"
    assert e["ageBars"] == 0
    print(f"  engulfing at {e['date']}: buy above {e['entry']}, stop {e['stop']}")


def test_engulfing_needs_a_downtrend():
    """The same candle pair inside an UPtrend must not count."""
    rows = []
    p = 100.0
    for _ in range(40):
        o = p; c = p + 2.0; rows.append((o, c + 0.4, o - 0.4, c)); p = c
    rows.append((p, p + 0.3, p - 3.0, p - 2.5))          # red
    rows.append((p - 3.5, p + 4.0, p - 4.0, p + 1.0))    # green, engulfs
    assert not S.find_engulfing(ohlc(rows)), "no downtrend in front -> not a valid setup"
    print("  engulfing correctly rejected inside an uptrend")


def test_engulfing_goes_stale():
    """A pattern older than FRESH_BARS must drop off the list."""
    rows = downtrend_rows()
    o_prev, c_prev = rows[-1][0], rows[-1][3]
    rows.append((c_prev - 1.0, o_prev + 4.0, c_prev - 2.0, o_prev + 2.0))
    flat = rows[-1][3]
    for _ in range(S.FRESH_BARS + 3):                    # drift on, doing nothing
        rows.append((flat, flat + 0.2, flat - 0.2, flat))
    assert not S.find_engulfing(ohlc(rows)), "a stale pattern must not be reported"
    print(f"  engulfing dropped after {S.FRESH_BARS} sessions")


def test_tweezer_detected():
    rows = downtrend_rows()
    low = rows[-1][2]
    rows.append((low + 3.0, low + 3.4, low, low + 0.5))          # bottoms at `low`
    rows.append((low + 0.6, low + 4.0, low + 0.02, low + 3.0))   # bottoms again
    df = ohlc(rows)
    atr = float(S.wilder_atr(df).iloc[-1])
    found = S.find_tweezer(df, atr)
    assert found, "two candles bottoming together after a downtrend must be found"
    t = found[-1]
    assert t["type"] == "tweezer"
    assert abs(t["stop"] - low) < 0.05, f"stop sits on the shared low, got {t['stop']}"
    assert t["entry"] > t["stop"]
    print(f"  tweezer at {t['date']}: shared low {t['stop']}, entry {t['entry']}")


def test_tweezer_rejects_mismatched_lows():
    """Two bottoms at clearly different levels are not a tweezer.

    Every low in the fresh window is kept distinct, so nothing can pair up.
    """
    rows = downtrend_rows()
    low = rows[-1][2]
    rows.append((low + 3.0, low + 3.4, low + 2.5, low + 3.0))   # low well above
    rows.append((low + 3.1, low + 4.0, low - 6.0, low + 3.5))   # low well below
    df = ohlc(rows)
    atr = float(S.wilder_atr(df).iloc[-1])
    found = S.find_tweezer(df, atr)
    assert not found, f"lows that differ a lot are not a tweezer, got {found}"
    print("  tweezer correctly rejected when the lows do not match")


def test_touch_levels_rank_by_touches():
    """A price revisited many times must outrank one touched once."""
    rows = []
    for _ in range(6):                      # oscillate between 90 and 110 repeatedly
        for p in (90, 100, 110, 100):
            rows.append((p, p + 1.0, p - 1.0, p))
            rows += [(p, p + 0.5, p - 0.5, p)] * 5
    rows.append((95, 140, 94, 96))          # one lone spike to 140
    df = ohlc(rows)
    atr = float(S.wilder_atr(df).iloc[-1])
    levels = S.touch_levels(df, atr)
    assert levels, "levels should be found"
    top = levels[0]
    assert top["touches"] >= 2, "the strongest level must have multiple touches"
    spike = [l for l in levels if l["price"] > 130]
    if spike:
        assert spike[0]["touches"] < top["touches"], \
            "a one-off spike must not outrank a repeatedly-touched level"
    print(f"  strongest level {top['price']} with {top['touches']} touches; "
          f"{len(levels)} levels total")


def test_setups_carry_their_own_entry_and_stop():
    """Each setup sizes off ITS own rule, and the row leads with the tightest."""
    rows = downtrend_rows(n=60)
    o_prev, c_prev = rows[-1][0], rows[-1][3]
    rows.append((c_prev - 1.0, o_prev + 4.0, c_prev - 2.0, o_prev + 2.0))
    df = ohlc(rows)
    out = S.analyse("MIX", "Mixed Co", META, df)
    assert out["setups"], "at least the engulfing should be present"
    for s in out["setups"]:
        assert s["entry"] > s["stop"], f"{s['type']}: entry must sit above the stop"
        assert s["riskPerShare"] is None or abs(
            s["riskPerShare"] - (s["entry"] - s["stop"])) < 0.02
        if s["qty"]:
            assert s["qty"] * s["riskPerShare"] <= S.TOTAL_RISK / S.RPT_DIVISOR + 0.01
    sized = [s for s in out["setups"] if s["riskPerShare"]]
    if len(sized) > 1:
        tightest = min(s["riskPerShare"] for s in sized)
        lead = [s for s in sized if s["type"] == out["primary"]][0]
        assert abs(lead["riskPerShare"] - tightest) < 0.02, "row must lead with the tightest stop"
    print(f"  setups found: {out['setupTypes']}, leading with {out['primary']}")

if __name__ == "__main__":
    checks = [
        ("Wilder RSI vs published reference", test_rsi_matches_wilder_reference),
        ("RSI extremes", test_rsi_bounds_and_extremes),
        ("ATR on known range", test_atr_constant_range),
        ("Pivot detection", test_pivots),
        ("Bullish divergence detected", test_divergence_detected),
        ("No false positive on downtrend", test_no_divergence_on_clean_downtrend),
        ("Position sizing math", test_sizing_math),
        ("Armed -> triggered on breakout", test_triggered_state),
        ("Short history", test_short_history_is_skipped),
        ("Selection measures", test_selection_measures_present),
        ("Relative strength vs Nifty", test_relative_strength),
        ("Relative strength w/o benchmark", test_relative_strength_without_benchmark),
        ("Bullish engulfing detected", test_engulfing_detected),
        ("Engulfing needs a downtrend", test_engulfing_needs_a_downtrend),
        ("Engulfing goes stale", test_engulfing_goes_stale),
        ("Tweezer bottom detected", test_tweezer_detected),
        ("Tweezer rejects mismatched lows", test_tweezer_rejects_mismatched_lows),
        ("Levels rank by touch count", test_touch_levels_rank_by_touches),
        ("Setups carry own entry/stop", test_setups_carry_their_own_entry_and_stop),
    ]
    failed = 0
    for label, fn in checks:
        try:
            print(f"\n{label}")
            fn()
            print("  PASS")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL: {exc}")
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    raise SystemExit(1 if failed else 0)
