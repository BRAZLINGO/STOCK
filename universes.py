"""Index universes — which pool of stocks the scan runs over.

Step 1 of the system: "select a wide universe of groups". Rather than freezing
a list that goes stale every time NSE rebalances, this pulls the official
constituent file for each index straight from NSE and caches it in the repo.
If NSE is unreachable the scan falls back to the last cached copy, so a bad
fetch day never empties your watchlist.

Add or remove universes in scanner.py's UNIVERSES setting.
"""
from __future__ import annotations

import csv
import io
import os
import time

import requests

CACHE_DIR = "cache"

# Universes that can fall back to the list bundled in tickers.py when neither
# NSE nor a cached copy is available.
BUILTIN = {"NIFTYMIDCAP100"}
NSE_BASE = "https://nsearchives.nseindia.com/content/indices/"

# Every universe the scan can run over. `file` is NSE's published constituent
# CSV; `label` is what the dashboard shows.
UNIVERSES = {
    "NIFTY50":        {"label": "Nifty 50",          "file": "ind_nifty50list.csv"},
    "NIFTYNEXT50":    {"label": "Nifty Next 50",     "file": "ind_niftynext50list.csv"},
    "NIFTY100":       {"label": "Nifty 100",         "file": "ind_nifty100list.csv"},
    "NIFTYMIDCAP100": {"label": "Nifty Midcap 100",  "file": "ind_niftymidcap100list.csv"},
    "NIFTYMIDCAP150": {"label": "Nifty Midcap 150",  "file": "ind_niftymidcap150list.csv"},
    "NIFTYSMLCAP100": {"label": "Nifty Smallcap 100","file": "ind_niftysmallcap100list.csv"},
    "NIFTY500":       {"label": "Nifty 500",         "file": "ind_nifty500list.csv"},
    "NIFTYBANK":      {"label": "Nifty Bank",        "file": "ind_niftybanklist.csv"},
    "NIFTYIT":        {"label": "Nifty IT",          "file": "ind_niftyitlist.csv"},
    "NIFTYPHARMA":    {"label": "Nifty Pharma",      "file": "ind_niftypharmalist.csv"},
    "NIFTYAUTO":      {"label": "Nifty Auto",        "file": "ind_niftyautolist.csv"},
    "NIFTYFMCG":      {"label": "Nifty FMCG",        "file": "ind_niftyfmcglist.csv"},
    "NIFTYMETAL":     {"label": "Nifty Metal",       "file": "ind_niftymetallist.csv"},
    "NIFTYENERGY":    {"label": "Nifty Energy",      "file": "ind_niftyenergylist.csv"},
    "NIFTYREALTY":    {"label": "Nifty Realty",      "file": "ind_niftyrealtylist.csv"},
    "CUSTOM":         {"label": "My own list",       "file": None},
}

# NSE serves these only to something that looks like a browser and has first
# visited the site, so the session picks up a cookie before asking for the CSV.
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
    "Accept": "text/csv,application/csv,*/*",
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/live-market-indices",
}


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.csv")


def _parse(text: str) -> list:
    """NSE's CSV carries Company Name, Industry, Symbol, Series, ISIN Code."""
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        clean = { (k or "").strip().lower(): (v or "").strip() for k, v in row.items() }
        symbol = clean.get("symbol", "")
        if not symbol:
            continue
        rows.append({
            "symbol": symbol,
            "name": clean.get("company name", symbol),
            "industry": clean.get("industry", "") or "Unclassified",
        })
    return rows


def _download(key: str) -> list | None:
    spec = UNIVERSES.get(key)
    if not spec or not spec["file"]:
        return None
    url = NSE_BASE + spec["file"]
    for attempt in range(3):
        try:
            s = requests.Session()
            s.headers.update(_HEADERS)
            s.get("https://www.nseindia.com", timeout=15)      # pick up the cookie
            r = s.get(url, timeout=25)
            r.raise_for_status()
            rows = _parse(r.text)
            if len(rows) >= 5:
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(_cache_path(key), "w", encoding="utf-8") as fh:
                    fh.write(r.text)
                return rows
        except Exception as exc:  # noqa: BLE001
            print(f"  {key}: fetch attempt {attempt + 1} failed ({exc})", flush=True)
            time.sleep(3 * (attempt + 1))
    return None


def load_universe(key: str) -> tuple:
    """Return (rows, source). Live NSE data when reachable, else the cache."""
    if key == "CUSTOM":
        try:
            from tickers import TICKERS
            return ([{"symbol": s, "name": n, "industry": "Unclassified"}
                     for s, n, _ in TICKERS], "tickers.py")
        except Exception:  # noqa: BLE001
            return ([], "tickers.py (missing)")

    rows = _download(key)
    if rows:
        return rows, "nseindia.com"

    path = _cache_path(key)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            cached = _parse(fh.read())
        if cached:
            print(f"  {key}: using cached list ({len(cached)} stocks)", flush=True)
            return cached, "cache"

    # Last resort: the list bundled inside tickers.py. This is why the scan
    # still works with no cache/ folder and no reachable NSE.
    if key in BUILTIN:
        try:
            from tickers import TICKERS
            rows = [{"symbol": s, "name": n, "industry": "Unclassified"}
                    for s, n, _ in TICKERS]
            if rows:
                print(f"  {key}: using the built-in list ({len(rows)} stocks)", flush=True)
                return rows, "built-in"
        except Exception as exc:  # noqa: BLE001
            print(f"  {key}: built-in list unavailable ({exc})", flush=True)

    print(f"  {key}: no data, no cache, no built-in list — skipped", flush=True)
    return [], "unavailable"


def build_watchlist(keys: list) -> tuple:
    """Merge several universes. A stock in two indexes keeps both tags."""
    merged, sources = {}, {}
    for key in keys:
        rows, src = load_universe(key)
        sources[key] = {"source": src, "count": len(rows),
                        "label": UNIVERSES.get(key, {}).get("label", key)}
        for row in rows:
            sym = row["symbol"]
            if sym not in merged:
                merged[sym] = {"symbol": sym, "name": row["name"],
                               "industry": row["industry"], "universes": []}
            if key not in merged[sym]["universes"]:
                merged[sym]["universes"].append(key)
            if merged[sym]["industry"] in ("", "Unclassified") and row["industry"]:
                merged[sym]["industry"] = row["industry"]
    return list(merged.values()), sources


def yahoo_symbol(nse_symbol: str) -> str:
    """NSE symbol -> Yahoo Finance ticker."""
    try:
        from tickers import YAHOO_OVERRIDES
    except Exception:  # noqa: BLE001
        YAHOO_OVERRIDES = {}
    if nse_symbol in YAHOO_OVERRIDES:
        return YAHOO_OVERRIDES[nse_symbol]
    return nse_symbol + ".NS"
