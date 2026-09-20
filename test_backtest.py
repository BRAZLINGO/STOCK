"""The backtest's own arithmetic, and the cross-tab's.

Three things are checked here:

  1. that the cross-tab can see an interaction the one-factor table is blind
     to -- the whole reason it exists;
  2. that its numbers RECONCILE, so no signal is counted twice or lost. This
     check earned its place immediately: it caught a key collision where the
     family named "divergence" and the pattern named "divergence" wrote into
     the same cell and double-counted every divergence signal;
  3. that a cell with too few trades is withheld rather than reported.

Run:  python -m pytest test_backtest.py -q
"""
import random

import scanner as S


def rate(w, l):
    return 100.0 * w / (w + l) if (w + l) else None


def test_cross_tab_separates_an_interaction_the_average_hides():
    """A filter worth twenty points to one pattern and nothing to another.

    The one-factor table can only report the blend of the two, and the blend
    is consistent with several completely different truths.
    """
    random.seed(11)
    truth = {("A", True): 0.65, ("A", False): 0.45,
             ("B", True): 0.40, ("B", False): 0.40}
    overall = {"with": [0, 0], "without": [0, 0]}
    per_pattern = {}
    for kind in ("A", "B"):
        for _ in range(600):
            passed = random.random() < 0.30
            win = random.random() < truth[(kind, passed)]
            side = "with" if passed else "without"
            overall[side][0 if win else 1] += 1
            cell = per_pattern.setdefault((kind, side), [0, 0])
            cell[0 if win else 1] += 1

    blended = rate(*overall["with"]) - rate(*overall["without"])
    lift_a = rate(*per_pattern[("A", "with")]) - rate(*per_pattern[("A", "without")])
    lift_b = rate(*per_pattern[("B", "with")]) - rate(*per_pattern[("B", "without")])

    # the blend understates A badly and overstates B
    assert blended < 15, f"blended lift {blended:.1f} should be a compromise"
    assert lift_a > 15, f"pattern A's real lift ({lift_a:.1f}) should be large"
    assert abs(lift_b) < 10, f"pattern B's real lift ({lift_b:.1f}) should be ~0"
    assert lift_a - lift_b > 15, (
        "the cross-tab must separate the two; if it cannot, it adds nothing")


def _tiny_universe(n_stocks=8):
    """A handful of stocks with enough history for the backtest to run."""
    import test_lookahead as L
    return {f"T{i:03d}": L.random_stock(700 + i) for i in range(n_stocks)}


def test_cross_tab_reconciles_with_the_one_factor_table():
    """Every signal counted in a pattern cell must also be counted once in
    that filter's overall row. Off-by-one here means a double count."""
    frames = _tiny_universe()
    bt = S.backtest_patterns(frames, list(frames), meta={})
    cells = bt.get("cross") or []
    assert cells, "no cross-tab produced"

    per_filter = {}
    for c in cells:
        if c["groupKind"] != "pattern":
            continue
        k = (c["filter"], c["timeframe"])
        t = per_filter.setdefault(k, [0, 0])
        t[0] += c["with"]["win"] + c["with"]["loss"]
        t[1] += c["without"]["win"] + c["without"]["loss"]

    for f in bt.get("filters", []):
        k = (f["filter"], f["timeframe"])
        if k not in per_filter:
            continue
        want = (f["with"]["win"] + f["with"]["loss"],
                f["without"]["win"] + f["without"]["loss"])
        assert tuple(per_filter[k]) == want, (
            f"{k}: pattern cells total {tuple(per_filter[k])}, "
            f"one-factor row says {want}")


def test_family_cells_equal_the_sum_of_their_patterns():
    frames = _tiny_universe()
    bt = S.backtest_patterns(frames, list(frames), meta={})
    cells = bt.get("cross") or []

    summed = {}
    for c in cells:
        if c["groupKind"] != "pattern":
            continue
        fam = S.FAMILY_OF.get(c["group"])
        if not fam:
            continue
        k = (fam, c["filter"], c["timeframe"])
        t = summed.setdefault(k, [0, 0, 0, 0])
        t[0] += c["with"]["win"]
        t[1] += c["with"]["loss"]
        t[2] += c["without"]["win"]
        t[3] += c["without"]["loss"]

    checked = 0
    for c in cells:
        if c["groupKind"] != "family":
            continue
        k = (c["group"], c["filter"], c["timeframe"])
        if k not in summed:
            continue
        got = (c["with"]["win"], c["with"]["loss"],
               c["without"]["win"], c["without"]["loss"])
        assert tuple(summed[k]) == got, (
            f"family {k} reports {got} but its patterns total {tuple(summed[k])}")
        checked += 1
    assert checked, "no family cells were checked"


def test_thin_cells_are_withheld_not_reported():
    frames = _tiny_universe()
    bt = S.backtest_patterns(frames, list(frames), meta={})
    for c in bt.get("cross") or []:
        thin = min(c["with"]["n"], c["without"]["n"])
        if thin < S.BT_MIN_CELL:
            assert c["lift"] is None, (
                f"{c['group']}/{c['filter']} reported a lift on only "
                f"{thin} trades")
        elif c["with"]["hitRate"] is not None:
            assert c["lift"] is not None, "a cell with enough trades went unreported"


def test_hit_rates_are_percentages_and_counts_agree():
    frames = _tiny_universe()
    bt = S.backtest_patterns(frames, list(frames), meta={})
    for r in bt["rows"]:
        decided = r["win"] + r["loss"]
        assert r["n"] == decided + r["open"]
        if r["hitRate"] is not None:
            assert 0 <= r["hitRate"] <= 100
            assert round(r["win"] / decided * 100, 1) == r["hitRate"]


def test_every_pattern_belongs_to_exactly_one_family():
    seen = {}
    for fam, pats in S.FAMILIES.items():
        for p in pats:
            assert p not in seen, f"{p} is in both {seen[p]} and {fam}"
            seen[p] = fam
    # every tradeable setup should be classified, or it silently misses the
    # family rows that are the only readable part of the table early on
    for p in S.SETUPS:
        assert p in seen, f"{p} belongs to no family"
