# 🎯 AI ACCUMULATION & BREAKOUT SCANNER — V8

World → India → Sectors → Market-cap & quality filters → Stocks (incl. a long-term Cup Base setup) → Top 5 →
Reward:Risk plan → Backtest → Journal.

## Files (all 4 must be in the GitHub repo root)
`app.py` (screen) · `engine.py` (all logic) · `requirements.txt` · `README.md`
Replace the old V7 files, commit, then **Reboot** the Streamlit app.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Scan AFTER market close (after ~4 PM IST). Today's incomplete candle is ignored by default.

## Data and timestamps (as asked)
* **Free, end-of-day only** (yfinance / Yahoo Finance). There is no live/intraday feed anywhere in this app.
* After every scan the screen shows **"Price data as of <date>"** (the last completed daily candle across your
  universe) and the wall-clock time the scan ran. If a stock's data is stale or missing it goes to the
  "Skipped" list, not into a fabricated signal.
* Universe lists come only from NSE's own published index files (Nifty 500 / Nifty 200) or NSE's full equity
  list (EQUITY_L.csv) — never an unverified "Nifty 1000"-style list.

## What is new in V8

### 1. Simple mode (default ON)
A new toggle at the top of the sidebar. ON: only Data source, Universe, History, Capital and Risk% are shown —
everything else runs on sensible presets. OFF: every slider from V7 is visible and tunable. Both modes run the
exact same engine and safety gates; Simple mode only hides sliders, it never skips a check.

### 2. New setup: Cup Base Breakout (long-term)
Built from the long-base "cup" breakout idea you shared (multi-year base, breakout with a volume surge,
10%/20%/30% staged targets). Implemented as a real, tested pattern on properly resampled **monthly** bars:
* An old high (the neckline) at least **3 years** back (configurable 3-10y) that the stock fell at least **15%**
  below before recovering — a real base, not sideways noise.
* A **fresh** monthly close back above that neckline, within **4%** of it (the entry filter from the video), with
  a **strong bullish candle** and **2x+ monthly volume** vs the trailing 6 months.
* Scored and gated exactly like the other setups (same Stop/Entry/Target/position-sizing framework, same
  Backtest tab) so it stands on the same evidence bar as everything else — not on anecdote.
* The card also shows the **video's own 10%/20%/30% target levels** from entry, separately labelled as an
  unverified reference, alongside the app's own structural Stop/Target.

**Honesty note on the source video:** it is promotional content for a paid course, built on a handful of
hand-picked historical winners shown on monthly charts with a flat 2% stop — that stop is unrealistic for daily
execution (an ordinary down day can gap past it) and a few cherry-picked examples is not a backtest. What's
implemented here is the underlying technical idea (a long, deep base followed by a high-volume breakout) with
real risk control and a real multi-year/multi-stock Backtest so you can judge it on this app's own numbers, not
the video's.

### 3. Longer history options
History now includes 10y and max (besides 2y/3y/5y) — the Cup setup needs years of data to find a genuine base.
Shorter setups (Trendline, 63D Breakout, Sweep, Pullback, Weekly Accumulation) work fine on 2y/3y and scan faster.

## Everything from V5-V7 is unchanged
Market-cap filter (replaces "must be Nifty 500"), minimum price, circuit-day gate, World/India regime, sector
strength + top-sector trading, trendline/63D-breakout/sweep/pullback/weekly-accumulation setups, the
reward:risk resistance gate, 3 entry styles × 2 exit styles backtest comparison, position sizing capped to your
capital, fundamentals check, journal. See the in-app "📖 वापरण्याची पद्धत" expander for the week-by-week routine.

## Important
* Score is NOT a probability. Only the Backtest tab's per-setup Win% / Avg R, built from real history on YOUR
  universe, tells you that — check the Cup setup's own row before trusting it, the same as any other setup.
* Backtest limits: today's universe membership (survivorship bias), free data, no slippage beyond the cost
  setting; regime, sector and market-cap rules are NOT replayed in the backtest (no history for them).
* yfinance is unofficial and can be delayed, rate-limited, or missing data for some symbols — especially 10y+
  history for smaller/newer-listed companies.
* Educational tool, not investment advice.
