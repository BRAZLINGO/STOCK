"""NSE delivery percentage -- who actually meant it.

Volume says shares changed hands. Delivery percentage says how many of those
shares were carried home rather than squared off the same afternoon. It is the
difference between "there was activity" and "somebody bought", and it is
published only by the Indian exchanges, so no imported screener has it.

NSE puts out one file a day, after the close:

    https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv
    SYMBOL, SERIES, DATE1, ... , TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES,
    DELIV_QTY, DELIV_PER

One file per day is the only way to get history -- there is no bulk download --
so this module keeps its own cache, ONE SLIM FILE PER DATE:

    cache/delivery/2026-09-18.csv      SYMBOL,DELIV_PER

Per date rather than one big file on purpose. A single long file would be
rewritten every evening, and git would store a fresh five-megabyte copy of it
with every commit; a year of that is a repository nobody wants. One small file
per date means each day's commit adds about thirty kilobytes and the history
stays flat.

The first run backfills a year (about 250 requests, a few minutes, once).
Every run after that fetches the one file it is missing.
"""
from __future__ import annotations

import csv
import io
import os
import time
from datetime import date, datetime, timedelta

CACHE_DIR = os.path.join("cache", "delivery")
BASE = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_"

# Series worth keeping. EQ is the normal rolling segment; BE is trade-to-trade,
# where delivery is compulsory -- worth keeping precisely because a BE stock
# showing low delivery would mean the data is wrong.
KEEP_SERIES = {"EQ", "BE"}

BACKFILL_DAYS = 260      # calendar weekdays reached back to on a first run
FETCH_BUDGET = 420       # seconds of fetching allowed per run, backfill or not
MAX_FETCH = 320          # hard cap on requests per run
PAUSE = 0.9              # between requests, so NSE does not start refusing
STALE_AFTER = 7          # a 404 older than this many days is a market holiday,
                         # not an outage, so stop asking about it
GIVE_UP_AFTER = 5        # consecutive failures that mean NSE is refusing us


def _path(d: date) -> str:
    return os.path.join(CACHE_DIR, d.strftime("%Y-%m-%d") + ".csv")


def _url(d: date) -> str:
    return BASE + d.strftime("%d%m%Y") + ".csv"


def _weekdays(upto: date, days: int) -> list:
    """The last `days` weekdays, newest first. Holidays fall out as 404s."""
    out, cur = [], upto
    while len(out) < days:
        if cur.weekday() < 5:
            out.append(cur)
        cur -= timedelta(days=1)
    return out


def parse(text: str) -> dict:
    """{SYMBOL: delivery percent} from one bhavcopy.

    NSE writes the header as "SYMBOL, SERIES, DATE1, ..." -- with a space after
    every comma -- so both keys and values need stripping. Rows with no
    delivery figure carry a bare "-", which is not a number and must not become
    a zero: a zero would read as "nobody took delivery" when the truth is
    "the exchange did not say".
    """
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        clean = {(k or "").strip().upper(): (v or "").strip()
                 for k, v in row.items()}
        if clean.get("SERIES") not in KEEP_SERIES:
            continue
        sym = clean.get("SYMBOL")
        raw = clean.get("DELIV_PER", "")
        if not sym or raw in ("", "-"):
            continue
        try:
            pct = float(raw)
        except ValueError:
            continue
        if 0.0 <= pct <= 100.0:
            out[sym] = round(pct, 2)
    return out


def _write(d: date, rows: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_path(d), "w", encoding="utf-8", newline="") as fh:
        fh.write("SYMBOL,DELIV_PER\n")
        for sym in sorted(rows):
            fh.write(f"{sym},{rows[sym]}\n")


def _write_holiday(d: date) -> None:
    """A deliberate empty marker, so a closed market is asked about once."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_path(d), "w", encoding="utf-8") as fh:
        fh.write("SYMBOL,DELIV_PER\n")      # header only: no trading that day


def read_cached(d: date) -> dict | None:
    path = _path(d)
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, encoding="utf-8") as fh:
        next(fh, None)                        # header
        for line in fh:
            sym, _, raw = line.partition(",")
            try:
                out[sym.strip()] = float(raw)
            except ValueError:
                continue
    return out                                # {} means "market was shut"


def _session():
    """Reuse the NSE session the universe loader already warms up."""
    import universes
    return universes._get_session()           # noqa: SLF001


def fetch(d: date) -> tuple:
    """One day's file, as (rows, status).

    The status matters more than it looks. A 404 means NSE is certain there
    was no trading -- a holiday -- and that answer can be cached for ever. A
    timeout or a refused connection means we do not know, and caching THAT as
    a holiday would quietly delete a real trading day from the history and
    never look at it again. They are different answers and must stay
    different: "none" is a fact, "error" is an absence of one.
    """
    try:
        r = _session().get(_url(d), timeout=30)
    except Exception as exc:                  # noqa: BLE001
        print(f"    delivery {d}: {exc}", flush=True)
        return None, "error"
    if r.status_code == 404:
        return None, "none"                   # holiday, or not published yet
    if r.status_code != 200:
        return None, "error"
    if len(r.text) < 500:
        return None, "error"                  # a stub or an error page
    rows = parse(r.text)
    return (rows, "ok") if rows else (None, "error")


def ensure_history(upto: date = None, days: int = BACKFILL_DAYS,
                   budget: float = FETCH_BUDGET, cap: int = MAX_FETCH) -> dict:
    """Fill in whatever the cache is missing, newest dates first.

    Newest first matters: if the budget runs out mid-backfill the scan still
    has today's delivery and yesterday's, which is what the live page needs.
    The older end fills itself in over the following runs.
    """
    upto = upto or date.today()
    wanted = _weekdays(upto, days)
    started = time.time()
    got = fetched = holidays = errors = 0
    misses = 0
    for d in wanted:
        if read_cached(d) is not None:
            got += 1
            continue
        if fetched >= cap or (time.time() - started) > budget:
            break
        if misses >= GIVE_UP_AFTER:
            # NSE has refused several in a row. Keep hammering it and we get
            # rate-limited properly; stop, and the next run picks up where
            # this one left off with nothing lost.
            print(f"    delivery: {misses} failures in a row -- stopping for "
                  f"this run", flush=True)
            break
        rows, status = fetch(d)
        fetched += 1
        if status == "ok":
            _write(d, rows)
            got += 1
            misses = 0
        elif status == "none":
            misses = 0                        # a definite answer, not a failure
            if (upto - d).days > STALE_AFTER:
                # Old and still nothing: the market was shut that day. Record
                # it, or every future run re-asks about every holiday.
                _write_holiday(d)
                holidays += 1
        else:
            errors += 1
            misses += 1                       # unknown: leave it to be retried
        time.sleep(PAUSE)
    return {"days": len(wanted), "cached": got, "fetched": fetched,
            "holidays": holidays, "errors": errors,
            "seconds": round(time.time() - started, 1)}


def load_history(upto: date = None, days: int = BACKFILL_DAYS) -> dict:
    """{'YYYY-MM-DD': {SYMBOL: pct}} for every cached trading day we have."""
    upto = upto or date.today()
    out = {}
    for d in _weekdays(upto, days):
        rows = read_cached(d)
        if rows:                              # skip holidays and missing days
            out[d.strftime("%Y-%m-%d")] = rows
    return out


def for_symbol(history: dict, symbol: str) -> dict:
    """{'YYYY-MM-DD': pct} for one stock, oldest first."""
    return {d: rows[symbol] for d, rows in sorted(history.items())
            if symbol in rows}


def metrics(series: dict, window: int = 20) -> dict:
    """Latest delivery, its own recent average, and the ratio between them.

    The ratio is the useful one. A stock that normally delivers 40% and today
    delivered 70% has had something happen; a stock that always delivers 70%
    has not. An absolute threshold alone would just rank the sleepy stocks.
    """
    blank = {"delivPct": None, "delivAvg": None, "delivRatio": None,
             "delivDate": None, "delivDays": 0}
    if not series:
        return blank
    dates = sorted(series)
    last = dates[-1]
    recent = [series[d] for d in dates[-(window + 1):-1]]
    avg = (sum(recent) / len(recent)) if recent else None
    pct = series[last]
    return {
        "delivPct": round(pct, 1),
        "delivAvg": round(avg, 1) if avg else None,
        "delivRatio": round(pct / avg, 2) if (avg and avg > 0) else None,
        "delivDate": last,
        "delivDays": len(dates),
    }


def on_date(series: dict, when: str, tolerance: int = 7):
    """Delivery on a given date, or the closest trading day before it.

    A weekly signal is labelled with its Friday, and a Friday can be a holiday,
    so an exact-match lookup would silently drop a chunk of the history.
    """
    if not series or not when:
        return None
    if when in series:
        return series[when]
    try:
        target = datetime.strptime(when, "%Y-%m-%d").date()
    except ValueError:
        return None
    for back in range(1, tolerance + 1):
        key = (target - timedelta(days=back)).strftime("%Y-%m-%d")
        if key in series:
            return series[key]
    return None
