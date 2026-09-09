"""Cross-sectional momentum / low-volatility signal construction and
dollar-neutral quintile-portfolio backtesting.

See `strategies/CrossSectionalFactors/PREREGISTRATION.md` for the fixed
hypotheses, universe, trial grid, and cost assumptions this module
implements -- config values (lookback, skip, n_per_leg, cost) are always
passed in by the caller (`run_cycle.py`), never hardcoded here, so the
pre-registered grid is the single source of truth for what actually ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from moex_backtest.data.known_events import mask_returns_for_vol_estimation

Direction = Literal["high_long", "low_long"]


def mask_daily_returns_for_vol(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Applies `mask_returns_for_vol_estimation` per ticker column -- see
    that function's docstring. Use the *unmasked* `daily_returns`/
    `monthly_close` for everything except a volatility lookback.
    """
    masked = {
        ticker: mask_returns_for_vol_estimation(daily_returns[ticker], ticker)
        for ticker in daily_returns.columns
    }
    return pd.DataFrame(masked, index=daily_returns.index)


def momentum_signal(
    monthly_close: pd.DataFrame, lookback_months: int, skip_months: int
) -> pd.DataFrame:
    """Cross-sectional J-K momentum: trailing return from `lookback_months`
    ago to `skip_months` ago, evaluated at every row of `monthly_close`
    (NaN wherever there isn't `lookback_months` of history yet).

    ``skip_months=1`` skips the most recent month (the standard "12-1"
    construction, avoiding short-term-reversal contamination);
    ``skip_months=0`` uses the full window through the formation date
    itself.
    """
    if lookback_months <= skip_months:
        raise ValueError(
            f"lookback_months ({lookback_months}) must be > skip_months ({skip_months})"
        )
    recent = monthly_close.shift(skip_months)
    past = monthly_close.shift(lookback_months)
    return recent / past - 1.0


def low_vol_signal(
    masked_daily_returns: pd.DataFrame, formation_dates: pd.DatetimeIndex, lookback_months: int
) -> pd.DataFrame:
    """Trailing realized daily-return volatility as of each formation date
    (approximated as `lookback_months * 21` trading days), on the
    *vol-masked* return panel (see `mask_daily_returns_for_vol`).
    """
    window = lookback_months * 21
    vol = masked_daily_returns.rolling(window=window, min_periods=window // 2).std()
    # `formation_dates` are calendar month-end labels (from
    # `panel.monthly_close_panel`'s `.resample("ME")`), which don't
    # generally coincide with an actual trading day in `vol`'s daily index
    # -- ffill picks each formation date's most recent actual trading day's
    # rolling estimate instead of leaving every row NaN on an exact-match
    # reindex.
    return vol.reindex(formation_dates, method="ffill")


def forward_monthly_returns(monthly_close: pd.DataFrame) -> pd.DataFrame:
    """Return earned holding from each formation date to the next one:
    row ``t`` holds ``price[t+1] / price[t] - 1`` (real price return, never
    vol-masked -- a holding-period P&L must reflect the actual return,
    including any day a volatility estimator would exclude).
    """
    return monthly_close.pct_change().shift(-1)


def _quintile_legs(
    signal_row: pd.Series, n_per_leg: int, direction: Direction
) -> tuple[frozenset[str], frozenset[str]]:
    """Returns (long_set, short_set) of ticker names, or two empty
    frozensets if fewer than ``2 * n_per_leg`` names have a non-NaN signal
    this month (skip the rebalance rather than trade a degenerate universe).
    """
    valid = signal_row.dropna()
    if len(valid) < 2 * n_per_leg:
        return frozenset(), frozenset()
    ranked = valid.sort_values()
    bottom = frozenset(ranked.index[:n_per_leg])
    top = frozenset(ranked.index[-n_per_leg:])
    if direction == "high_long":
        return top, bottom
    return bottom, top


def _leg_turnover(
    prev_long: frozenset[str],
    prev_short: frozenset[str],
    long_set: frozenset[str],
    short_set: frozenset[str],
    n_per_leg: int,
) -> float:
    """Fraction of the ``2 * n_per_leg`` total positions that are newly
    entered this rebalance (a name that stays in the same leg costs
    nothing; a name that enters -- including every name on the very first
    rebalance, when the previous legs are empty -- pays the round-trip
    cost priced in `backtest_factor`).
    """
    entries = len(long_set - prev_long) + len(short_set - prev_short)
    total_positions = 2 * n_per_leg
    return entries / total_positions if total_positions else 0.0


@dataclass(frozen=True, slots=True)
class FactorBacktestResult:
    returns: pd.Series
    """Monthly net (after-cost) portfolio returns, indexed by formation
    date, covering the holding period starting at that date. Rebalances
    where fewer than ``2 * n_per_leg`` names had a valid signal are
    dropped (not zero-filled), so the index may be a strict subset of the
    requested formation dates.
    """
    n_rebalances: int
    n_skipped: int
    """Requested formation dates that were dropped for lacking enough
    valid-signal names (see `returns`'s docstring)."""


def backtest_factor(
    signal: pd.DataFrame,
    monthly_close: pd.DataFrame,
    formation_dates: pd.DatetimeIndex,
    *,
    direction: Direction,
    n_per_leg: int,
    cost_bps: float,
) -> FactorBacktestResult:
    """Runs the dollar-neutral quintile long/short backtest over
    `formation_dates` (a subset of `signal`'s and `monthly_close`'s shared
    index): at each date, rank `signal`'s cross-section, go long/short the
    top/bottom `n_per_leg` names per `direction` (see `_quintile_legs`),
    equal-weight within each leg, hold one month, and net out `cost_bps`
    (one-way, in basis points) on every name that enters a leg this
    rebalance (see `_leg_turnover`).
    """
    fwd_ret = forward_monthly_returns(monthly_close)
    values: list[float] = []
    index: list[pd.Timestamp] = []
    n_skipped = 0
    prev_long: frozenset[str] = frozenset()
    prev_short: frozenset[str] = frozenset()

    for t in formation_dates:
        long_set, short_set = _quintile_legs(signal.loc[t], n_per_leg, direction)
        if not long_set or not short_set:
            n_skipped += 1
            continue

        fwd_row = fwd_ret.loc[t]
        long_ret = float(fwd_row[list(long_set)].mean())
        short_ret = float(fwd_row[list(short_set)].mean())
        gross = long_ret - short_ret

        turnover = _leg_turnover(prev_long, prev_short, long_set, short_set, n_per_leg)
        cost = turnover * (cost_bps / 10_000.0)

        if not np.isnan(gross):
            values.append(gross - cost)
            index.append(t)
        else:
            n_skipped += 1
        prev_long, prev_short = long_set, short_set

    returns = pd.Series(values, index=pd.DatetimeIndex(index), dtype=float)
    return FactorBacktestResult(returns=returns, n_rebalances=len(returns), n_skipped=n_skipped)
