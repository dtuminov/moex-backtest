"""Known market-wide/security-specific data-quality events for the MOEX universe.

These are not adjustments to the OHLCV series itself (Finam's Bars API
carries no split/dividend-adjustment field, per `FinamClient`'s docstring,
and neither event below is a data error -- both are real, verified market
events, see each constant's own docstring). They are markers for
cross-sectional research code to exclude specific *return* observations from
rolling-volatility windows (realized vol, a low-vol factor's ranking
lookback) where treating them as an ordinary single trading day's return
would distort the estimate -- while leaving the underlying price series, and
any momentum/level-based calculation, completely untouched.

**Do not use these to exclude data from a momentum/return-level lookback.**
A momentum factor is supposed to see the real (possibly large) return; only a
*volatility* estimator is distorted by a single extreme or artificially
compressed observation the way this module corrects for.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

# MOEX suspended equity trading market-wide after Russia's invasion of
# Ukraine on 2022-02-24. The last trading day before the halt was
# 2022-02-25; trading resumed 2022-03-24 (~19 trading days / ~27 calendar
# days later). Because the halted days simply have no rows in
# `data/raw/*.parquet`, the row-to-row `pct_change()` between these two
# dates computes a single "daily" return that actually compresses ~1
# calendar month of price action into what looks like one ordinary trading
# day -- verified directly against the cached data on 2026-09-10: GAZP
# +13.4%, SBER +3.9% for this one row. Both are below the project's
# existing >=30% single-jump flag (see `scripts/import_universe.py`), so
# this gap was NOT caught by that check -- it is still not a genuine
# one-day return, and any rolling window (realized vol, a lookback that
# counts trading days) spanning it silently treats a month of information
# as a single day.
HALT_2022_LAST_TRADING_DAY = date(2022, 2, 25)
HALT_2022_REOPEN_DAY = date(2022, 3, 24)

# CBOM: +54.0% on 2026-04-13. Confirmed (2026-09-09/10) as a genuine
# corporate event, not a data artifact: volume that day was 764.8M vs.
# roughly 0.6-15M on the preceding sessions (>50x), and stayed elevated
# (292M/183M/184M/296M) for the following four sessions -- a data glitch
# would not produce sustained elevated volume after the fact. Public
# reporting corroborates a real catalyst: a shareholder buyback offer at
# 10.35 RUB/share tied to a 2026-04-21 AGM, alongside a Q1 2026 net profit
# beat (RUB 2.3bn -> 14.3bn y/y). Kept in the price series (it is real,
# tradable price history) -- excluded only from rolling realized-volatility
# windows, the same treatment as the 2022 halt reopening above, since a
# single-name corporate-action jump this large otherwise dominates any
# vol estimate for a window that contains it.
CBOM_CORPORATE_EVENT_DAY = date(2026, 4, 13)
CBOM_SYMBOL = "CBOM"


def excluded_return_dates(symbol: str) -> frozenset[date]:
    """Dates whose per-period return should be excluded from rolling
    realized-volatility windows for `symbol` (see module docstring for what
    "excluded" means and why momentum/level calculations should NOT use
    this).

    The 2022 halt reopening applies market-wide (every symbol); CBOM's
    2026-04-13 event only to `CBOM_SYMBOL` itself.
    """
    dates = {HALT_2022_REOPEN_DAY}
    if symbol == CBOM_SYMBOL:
        dates.add(CBOM_CORPORATE_EVENT_DAY)
    return frozenset(dates)


def mask_returns_for_vol_estimation(returns: pd.Series, symbol: str) -> pd.Series:
    """Returns a copy of `returns` with the dates from
    `excluded_return_dates(symbol)` zeroed out -- use only when feeding a
    rolling *volatility*/dispersion estimator, never a momentum or
    return-level calculation (see module docstring).

    `returns.index` must be convertible to dates (a `pd.DatetimeIndex`, or
    anything `pd.DatetimeIndex(...)` accepts). A no-op (returns `returns`
    unchanged, no copy) if none of `symbol`'s excluded dates fall within
    `returns.index` -- expected and harmless for any window that doesn't
    span the relevant date (e.g. a post-2022 walk-forward slice, or any
    non-CBOM symbol's ordinary window).
    """
    exclude = excluded_return_dates(symbol)
    dates = pd.DatetimeIndex(returns.index).date
    hit = pd.Series(dates, index=returns.index).isin(exclude)
    if not hit.any():
        return returns
    out = returns.copy()
    out[hit] = 0.0
    return out
