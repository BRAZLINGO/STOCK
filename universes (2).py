"""Index universes -- which pool of stocks the scan runs over.

Step 1 of the system: "select a wide universe of groups". Rather than freezing
a list that goes stale every time NSE rebalances, this pulls the official
constituent file for each index straight from NSE and caches it in the repo.

Three levels of fallback, so a bad fetch day never empties your watchlist:

  1. NSE's published CSV (the live, correct answer)
  2. cache/<KEY>.csv -- the last good copy, committed back by the workflow
  3. the list bundled in tickers.py (Midcap 100 only)

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

# How the dashboard groups the index picker. Order here is the order shown.
GROUPS = [
    ("broad",  "Broad market"),
    ("size",   "By size"),
    ("bank",   "Banking"),
    ("sector", "Sector"),
    ("custom", "My own"),
]

# Every universe the scan can run over.
#   label  -- what the dashboard shows
#   group  -- which heading it sits under
#   files  -- candidate NSE filenames, tried in order. NSE is not consistent
#             about its own naming (most are "ind_nifty<x>list.csv" but a few
#             use "ind_nifty<x>_list.csv"), and it has renamed files before,
#             so each entry may list more than one spelling.
UNIVERSES = {
    # --- broad market -------------------------------------------------------
    "NIFTY50": {
        "label": "Nifty 50", "group": "broad",
        "files": ["ind_nifty50list.csv"]},
    "NIFTYNEXT50": {
        "label": "Nifty Next 50", "group": "broad",
        "files": ["ind_niftynext50list.csv"]},
    "NIFTY100": {
        "label": "Nifty 100", "group": "broad",
        "files": ["ind_nifty100list.csv"]},
    "NIFTY200": {
        "label": "Nifty 200", "group": "broad",
        "files": ["ind_nifty200list.csv"]},
    "NIFTY500": {
        "label": "Nifty 500", "group": "broad",
        "files": ["ind_nifty500list.csv"]},

    # --- by size ------------------------------------------------------------
    "NIFTYMIDCAP50": {
        "label": "Midcap 50", "group": "size",
        "files": ["ind_niftymidcap50list.csv"]},
    "NIFTYMIDCAP100": {
        "label": "Midcap 100", "group": "size",
        "files": ["ind_niftymidcap100list.csv"]},
    "NIFTYMIDCAP150": {
        "label": "Midcap 150", "group": "size",
        "files": ["ind_niftymidcap150list.csv"]},
    "NIFTYMIDCAPSELECT": {
        "label": "Midcap Select", "group": "size",
        "files": ["ind_niftymidcapselect_list.csv",
                  "ind_niftymidcapselectlist.csv"]},
    "NIFTYSMLCAP50": {
        "label": "Smallcap 50", "group": "size",
        "files": ["ind_niftysmallcap50list.csv"]},
    "NIFTYSMLCAP100": {
        "label": "Smallcap 100", "group": "size",
        "files": ["ind_niftysmallcap100list.csv"]},
    "NIFTYSMLCAP250": {
        "label": "Smallcap 250", "group": "size",
        "files": ["ind_niftysmallcap250list.csv"]},
    "NIFTYMICROCAP250": {
        "label": "Microcap 250", "group": "size",
        "files": ["ind_niftymicrocap250_list.csv",
                  "ind_niftymicrocap250list.csv"]},

    # --- banking ------------------------------------------------------------
    "NIFTYBANK": {
        "label": "Nifty Bank", "group": "bank",
        "files": ["ind_niftybanklist.csv"]},
    "NIFTYPSUBANK": {
        "label": "PSU Bank", "group": "bank",
        "files": ["ind_niftypsubanklist.csv"]},
    "NIFTYPVTBANK": {
        "label": "Private Bank", "group": "bank",
        "files": ["ind_nifty_privatebanklist.csv",
                  "ind_niftyprivatebanklist.csv"]},
    "NIFTYFINSERVICE": {
        "label": "Financial Services", "group": "bank",
        "files": ["ind_niftyfinancelist.csv"]},

    # --- sector (available, not scanned unless you add them to UNIVERSES) ----
    "NIFTYIT": {
        "label": "IT", "group": "sector",
        "files": ["ind_niftyitlist.csv"]},
    "NIFTYPHARMA": {
        "label": "Pharma", "group": "sector",
        "files": ["ind_niftypharmalist.csv"]},
    "NIFTYAUTO": {
        "label": "Auto", "group": "sector",
        "files": ["ind_niftyautolist.csv"]},
    "NIFTYFMCG": {
        "label": "FMCG", "group": "sector",
        "files": ["ind_niftyfmcglist.csv"]},
    "NIFTYMETAL": {
        "label": "Metal", "group": "sector",
        "files": ["ind_niftymetallist.csv"]},
    "NIFTYENERGY": {
        "label": "Energy", "group": "sector",
        "files": ["ind_niftyenergylist.csv"]},
    "NIFTYREALTY": {
        "label": "Realty", "group": "sector",
        "files": ["ind_niftyrealtylist.csv"]},
    "NIFTYINFRA": {
        "label": "Infrastructure", "group": "sector",
        "files": ["ind_niftyinfralist.csv"]},

    # --- your own list ------------------------------------------------------
    "CUSTOM": {
        "label": "My own list", "group": "custom", "files": []},
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

# One session for the whole run, so the NSE cookie is picked up once rather
# than once per index. With 14 universes that is 14 fewer handshakes.
_session = None


def _get_session():
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update(_HEADERS)
        try:
            s.get("https://www.nseindia.com", timeout=15)      # pick up the cookie
        except Exception as exc:  # noqa: BLE001
            print(f"  NSE handshake failed ({exc}) -- trying the files anyway",
                  flush=True)
        _session = s
    return _session


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.csv")


def _parse(text: str) -> list:
    """NSE's CSV carries Company Name, Industry, Symbol, Series, ISIN Code."""
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        clean = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
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
    if not spec or not spec.get("files"):
        return None

    for attempt in range(3):
        s = _get_session()
        for fname in spec["files"]:
            try:
                r = s.get(NSE_BASE + fname, timeout=25)
                if r.status_code == 404:
                    continue                      # wrong spelling, try the next
                r.raise_for_status()
                rows = _parse(r.text)
                if len(rows) >= 5:
                    os.makedirs(CACHE_DIR, exist_ok=True)
                    with open(_cache_path(key), "w", encoding="utf-8") as fh:
                        fh.write(r.text)
                    return rows
            except Exception as exc:  # noqa: BLE001
                print(f"  {key}: {fname} attempt {attempt + 1} failed ({exc})",
                      flush=True)
        time.sleep(2 * (attempt + 1))
    return None


def load_universe(key: str) -> tuple:
    """Return (rows, source). Live NSE data when reachable, else a fallback."""
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
                print(f"  {key}: using the built-in list ({len(rows)} stocks)",
                      flush=True)
                return rows, "built-in"
        except Exception as exc:  # noqa: BLE001
            print(f"  {key}: built-in list unavailable ({exc})", flush=True)

    print(f"  {key}: no data, no cache, no built-in list -- skipped", flush=True)
    return [], "unavailable"


def build_watchlist(keys: list) -> tuple:
    """Merge several universes. A stock in five indexes is fetched once and
    keeps all five tags, which is what lets the page switch instantly."""
    merged, sources = {}, {}
    for key in keys:
        rows, src = load_universe(key)
        spec = UNIVERSES.get(key, {})
        sources[key] = {"source": src, "count": len(rows),
                        "label": spec.get("label", key),
                        "group": spec.get("group", "other")}
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
