"""Is the thing the scan publishes actually safe for the page to read?

Every one of these checks exists because something went wrong once.

The sharpest example: a NaN reached data.json. NaN is valid Python and
INVALID JSON, so `json.dump` wrote it happily and the browser's JSON.parse
refused the whole file. The page went blank with no error anybody could see,
and it stayed blank until the file itself was inspected. Five lines here
would have caught it before the commit.

Run:  python -m pytest test_payload.py -q
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

import scanner as S
import test_lookahead as L


@pytest.fixture(scope="module")
def payload():
    """A real analyse() run over synthetic prices, shaped like data.json."""
    rows = []
    for i in range(6):
        daily = L.random_stock(4000 + i)
        meta = {"industry": "Test Sector", "universes": ["TESTIDX"]}
        rows.append(S.build_record(f"T{i:03d}", f"Test {i}", meta, daily,
                                   {tf: None for tf in S.TIMEFRAMES}))
    return {"asOf": "2026-09-20", "rows": rows,
            "settings": {"totalRisk": S.TOTAL_RISK}}


def test_payload_survives_a_round_trip_through_json(payload):
    """The NaN incident, as a test.

    allow_nan=False is what makes this fail loudly here instead of silently
    in a browser three hours later.
    """
    text = json.dumps(S.json_safe(payload), allow_nan=False)
    back = json.loads(text)
    assert back["rows"], "the payload lost its rows in the round trip"


def test_no_non_finite_numbers_anywhere(payload):
    bad = []

    def walk(node, path):
        if isinstance(node, float) and not math.isfinite(node):
            bad.append(path)
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(S.json_safe(payload), "payload")
    assert not bad, f"non-finite numbers would break JSON.parse at: {bad[:5]}"


def test_json_safe_converts_numpy_and_nan():
    messy = {"a": np.float64("nan"), "b": np.float64(1.5), "c": np.int64(3),
             "d": np.bool_(True), "e": [float("inf"), 2.0],
             "f": {"g": float("-inf")}}
    clean = S.json_safe(messy)
    json.dumps(clean, allow_nan=False)          # must not raise
    assert clean["a"] is None and clean["e"][0] is None and clean["f"]["g"] is None
    assert clean["b"] == 1.5 and clean["c"] == 3 and clean["d"] is True


def test_every_setup_is_a_tradeable_shape(payload):
    """A stop at or above its entry means a negative risk, which means a
    quantity computed from a negative number. It must never be published."""
    for row in payload["rows"]:
        for tf, v in (row.get("tf") or {}).items():
            for s in v.get("setups") or []:
                e, st = s.get("entry"), s.get("stop")
                if e is None or st is None:
                    continue
                assert e > st, f"{row['symbol']}/{tf}/{s['type']}: stop >= entry"
                assert s.get("riskPerShare") is None or s["riskPerShare"] > 0


def test_quantities_never_exceed_the_risk_budget(payload):
    """Position size is the one number that can lose real money if it is
    wrong, so it gets checked against the rule that produced it."""
    rpt = S.TOTAL_RISK / S.RPT_DIVISOR
    for row in payload["rows"]:
        for tf, v in (row.get("tf") or {}).items():
            for s in v.get("setups") or []:
                qty, rps = s.get("qty"), s.get("riskPerShare")
                if not qty or not rps:
                    continue
                assert qty * rps <= rpt + 1e-6, (
                    f"{row['symbol']}: {qty} shares x Rs {rps} risk = "
                    f"Rs {qty * rps:.0f}, over the Rs {rpt:.0f} budget")


def test_warnings_never_carry_an_entry(payload):
    """Topping and breakdown patterns are warnings. If one ever arrives with
    an entry and a quantity, the desk is recommending a long into a
    breakdown."""
    for row in payload["rows"]:
        for tf, v in (row.get("tf") or {}).items():
            for w in v.get("warnings") or []:
                assert w.get("entry") is None, f"{w.get('type')} carries an entry"
                assert w.get("qty") in (None, 0)
            for t in v.get("warningTypes") or []:
                assert t in S.WARNINGS, f"{t} is published as a warning but is not one"


def test_setup_types_are_all_known(payload):
    """A typo in a detector's `type` renders as an empty badge on the page
    rather than an error, so it has to be caught here."""
    known = set(S.SETUPS) | set(S.WARNINGS)
    for row in payload["rows"]:
        for tf, v in (row.get("tf") or {}).items():
            for t in (v.get("setupTypes") or []):
                assert t in known, f"unknown setup type published: {t}"


def test_rows_carry_what_the_page_reads(payload):
    """The page reads these by name. A rename in the scanner that is not
    mirrored in index.html shows up as a column of dashes, not an error."""
    needed = {"symbol", "name", "industry", "universes", "tf"}
    per_tf = {"price", "status", "setups", "setupTypes", "asOf"}
    for row in payload["rows"]:
        missing = needed - set(row)
        assert not missing, f"{row.get('symbol')}: row is missing {missing}"
        for tf, v in (row["tf"] or {}).items():
            if v.get("status") == "nodata":
                continue
            gap = per_tf - set(v)
            assert not gap, f"{row['symbol']}/{tf} is missing {gap}"


def test_a_scan_with_no_history_is_skipped_not_published():
    idx = pd.bdate_range("2025-01-01", periods=12)
    thin = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0,
                         "Close": 100.0, "Volume": 1000.0}, index=idx)
    out = S.analyse("TINY", "Tiny", {}, thin)
    assert out["status"] == "nodata"
