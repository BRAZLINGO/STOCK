# Midcap Reversal Desk

A self-hosted RSI-divergence scanner for the NIFTY Midcap 100. It runs on
GitHub's free tier — no server, no API key, no subscription to anything.

Every weekday after the NSE close, GitHub Actions pulls a year of daily prices,
computes Wilder RSI(14) and ATR(14), looks for bullish divergences, and
publishes an updated dashboard.

## Choosing your universe

Step 1 of the system — which pool of stocks to hunt in. Set it at the top of
`scanner.py`:

```python
UNIVERSES = ["NIFTYMIDCAP100"]          # one index
UNIVERSES = ["NIFTY50", "NIFTYMIDCAP100", "NIFTYSMLCAP100"]   # or several
```

Available keys live in `universes.py`: `NIFTY50`, `NIFTYNEXT50`, `NIFTY100`,
`NIFTYMIDCAP100`, `NIFTYMIDCAP150`, `NIFTYSMLCAP100`, `NIFTY500`, the sector
indexes (`NIFTYBANK`, `NIFTYIT`, `NIFTYPHARMA`, `NIFTYAUTO`, `NIFTYFMCG`,
`NIFTYMETAL`, `NIFTYENERGY`, `NIFTYREALTY`), and `CUSTOM` for your own list in
`tickers.py`.

The constituents are pulled from NSE's own published files on every run, so
when NSE rebalances an index your watchlist follows without you editing
anything. Each fetch is cached into `cache/`; if NSE is unreachable that day
the scan falls back to the last good copy rather than emptying your list. The
dashboard says which happened, and a stock in two indexes keeps both tags so
you can switch between them on the page.

*Caveat worth knowing:* NSE sometimes refuses automated requests. The fallback
means a refusal is harmless, but if a universe never fetches, download its CSV
from nseindia.com by hand and drop it in `cache/<KEY>.csv`. The repo ships with
the Midcap 100 list already cached, so it works out of the box.

## The three setups

**RSI divergence.** Price prints a *lower bottom* while RSI prints a *higher*
one. The peak price made between those two bottoms is the resistance that has
to break — nothing is a buy until price clears it. Stop is `ATR(14) × 1.5`
below that level.

**Bullish engulfing.** After a downtrend, a green candle swallows the previous
red candle's body whole. Buy above the green candle's high; its low is the
stop — exactly as the notes have it.

**Tweezer bottom.** After a downtrend, two sessions bottom at the same level.
Buy above the second candle's high; the shared low is the stop.

Each setup sizes off **its own** entry and stop, so the ATR multiplier moves
the divergence stop while a candlestick stop stays pinned to the candle. Risk
per trade = total risk ÷ 50, quantity = RPT ÷ risk per share, targets at 1:2
and 1:3.

When several fire on one stock the row leads with the **widest** stop. That is
deliberate: risk per trade is fixed, so a tight stop means a large share count —
on the sample data the tightest rule wanted 60–99% of the capital base in a
single position, which cannot work alongside the four-open-positions rule. Set
`PRIMARY_RULE = "tightest"` in `scanner.py` to flip it. A setup you cannot size
to even one share (a 1.5-lakh-rupee stock against a ₹2,000 risk budget) is
labelled **too big** rather than quietly reported as a quantity of zero, and the
drawer says what risk capital one share would need.

**Max per trade ₹** on the dashboard is an optional ceiling on how much capital
one position may use — the notes' "investment is of 10%" rule. Leave it blank
for no cap. It caps on whichever is higher, the breakout level or today's price,
so the limit holds even when price has already run past the entry.

A setup is **armed** once found and **triggered** when price clears its entry.
Candlestick patterns go stale after 5 sessions, matching the exit-on-the-fifth-
candle rule, and the drawer lists every setup a stock is showing.

Pick which setups to hunt with the chips on the page. Choose more than one and
you get the stocks where **all** of them agree; when nothing satisfies all of
them, the list falls back to every stock where any fired, with a Setup column
naming which indicator found it.

## Support and resistance

Levels are found by clustering every pivot high and low into bands, so a level
is a price **many touches agree on** rather than one swing high — the notes'
rule that a level must join many points and sit on a major level. The number
beside "to break" is how many times price respected that level. The divergence
trigger is anchored to the bounce peak between the two bottoms, so it does not
drift upward as price rises.

## The selection filters

Three measures ride alongside every stock so you can narrow the list the way
the notes describe, and each is a toggle on the dashboard:

**ATR % (velocity)** — the day's average range as a percentage of price. Above
3% marks the stock as having enough movement to be worth trading; the threshold
is `ATRPCT_FLOOR` in `scanner.py`.

**Comparative strength vs Nifty 50** — the stock-to-index ratio measured against
its own 100-session average. Above the line it is outperforming the index,
below it is lagging. This is the "is it worth buying at all" screen, and it
reads `over +8.1%` or `under -6.2%` in the table.

**Turnover** — average traded value over 20 sessions, as a liquidity proxy.
True bid-ask spread needs live order-book depth that free end-of-day data does
not carry, so this stands in for it: high turnover means you can get filled.

Sector comes from NSE's own classification, so you can also narrow to one
industry when a sector is running.

## One-time setup (about 10 minutes)

1. **Create the repo.** On GitHub, click *New repository*, name it anything
   (`midcap-reversal` works), make it **Public**, and create it.
   Public matters: Actions minutes are unlimited on public repos, and GitHub
   Pages is free there.
2. **Upload these files.** On the empty repo page choose
   *uploading an existing file*, drag in everything from this folder — keep the
   `.github/workflows/` folder structure intact — and commit.
   *(If the drag-and-drop drops the hidden `.github` folder, create the file
   manually: Add file → Create new file → type `.github/workflows/scan.yml` as
   the name, paste the contents, commit.)*
3. **Run the scan once.** Actions tab → *Midcap reversal scan* → *Run workflow*.
   If Actions asks you to enable workflows on a fresh repo, say yes. The run
   takes two to four minutes and commits `data.json`.
4. **Turn on the website.** Settings → Pages → Source: *Deploy from a branch*,
   Branch: `main`, folder `/ (root)` → Save. A minute later your dashboard is at
   `https://<your-username>.github.io/<repo-name>/`.

That's it. It now updates itself every trading day, forever, for free.

## Optional: the daily e-mail

Settings → Secrets and variables → Actions → *New repository secret*, add three:

| Secret | Value |
|---|---|
| `MAIL_USERNAME` | your Gmail address |
| `MAIL_PASSWORD` | a Gmail **App Password** (Google Account → Security → 2-Step Verification → App passwords). Your normal password will not work. |
| `MAIL_TO` | where the digest should go |

You get an e-mail only on days something is armed or triggered. Without these
secrets the step is skipped and everything else still works.

## Changing what it watches

- **Stocks:** set `UNIVERSES` in `scanner.py` (see above), or edit `tickers.py` and use the `CUSTOM` key.
- **Strategy knobs:** the constants at the top of `scanner.py` — `ATR_MULT`,
  `TOTAL_RISK`, `SWING_BARS` (how pronounced a swing low must be), `DIV_WINDOW`
  (how far back to hunt for the two bottoms), `RR_TARGETS`, `ATRPCT_FLOOR`, `CRS_PERIOD`.
- **Risk capital on the fly:** change it directly on the dashboard. That, your
  resistance overrides and your notes are stored in your browser, so they
  survive every scan.

## On a phone

Below 700px the watchlist stops being a table and becomes a list of cards, one
per stock, so nothing hides off the side of the screen: symbol, state, which
setups fired, RSI, ATR%, relative strength, then close, entry, stop, quantity,
deployment and the distance to the trigger. Tap a card to expand the same
detail the desktop drawer shows.

The filter block folds away behind a one-line summary ("Nifty Midcap 100 · DIV
+TWZ · 2 filters"), so the live triggers sit near the top where they belong on a
phone. The list loads twenty at a time with a Show more button. Rotate to
landscape or open it on a desktop and it switches back to the full table
automatically.

## Things worth knowing

- **Prices come from Yahoo Finance** via `yfinance`. It is free and unofficial;
  it occasionally rate-limits or renames a ticker. If the dashboard footer says
  a stock had no data, look it up on finance.yahoo.com and add its ticker to
  `YAHOO_OVERRIDES` in `tickers.py`. Newly listed names are the usual suspects.
- **Split and bonus adjustments** follow Yahoo's own data. A very recent
  corporate action can distort RSI for a few sessions.
- **GitHub pauses scheduled workflows** in repos with no activity for 60 days.
  The daily commit normally counts as activity, but if GitHub e-mails you about
  it, one click re-enables the schedule.
- **The scan reads closes, not intraday.** A breakout that happens and fails
  inside one session shows up as whatever the close did.
- **Wilder's definitions** are used for both RSI and ATR, so the numbers match
  TradingView and Kite rather than drifting from them. `test_scanner.py` checks
  this against Wilder's own published worked example — run `python
  test_scanner.py` any time you change the maths.

## Files

| File | What it is |
|---|---|
| `scanner.py` | Fetches prices, computes the indicators, finds setups, writes `data.json` |
| `universes.py` | The index catalogue and the NSE constituent fetch |
| `tickers.py` | The CUSTOM list and any Yahoo ticker overrides |
| `cache/` | Last good copy of each index's constituents |
| `index.html` | The dashboard (reads `data.json`, no build step) |
| `test_scanner.py` | Offline checks for the indicator and divergence logic |
| `.github/workflows/scan.yml` | The daily schedule |
| `data.json` | Written by each scan |
| `alerts.md` | The digest the e-mail step sends |

Not investment advice. It automates a strategy you defined; verify every level
on your own chart before trading.
