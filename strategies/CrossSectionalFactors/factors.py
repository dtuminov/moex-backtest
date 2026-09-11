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


def composite_momentum_signal(
    monthly_close: pd.DataFrame, horizons: tuple[int, ...], skip_months: int
) -> pd.DataFrame:
    """Multi-horizon composite momentum: for each `horizons` value, computes
    `momentum_signal` and converts it to a cross-sectional percentile rank
    (`DataFrame.rank(axis=1, pct=True)`, in (0, 1], NaN-aware) -- percentile
    rank, not the raw return, so a longer horizon's naturally larger-
    magnitude return doesn't dominate a shorter horizon's in the average
    (see `strategies/CrossSectionalFactors/PREREGISTRATION_cycle2_momentum_ensemble.md`
    for why this specific construction, and why `horizons` is a fixed
    literature convention rather than swept as a grid).

    The composite score at a given (date, ticker) is the mean percentile
    rank across `horizons`, defined **only** where every horizon has a
    valid (non-NaN) `momentum_signal` value for that cell -- in practice
    this means the composite is gated by the longest horizon's own history
    requirement, exactly like a single-horizon signal at that longest
    horizon; no NaN cell gets a partial composite score built from fewer
    horizons than the rest of the panel.
    """
    if len(horizons) < 2:
        raise ValueError(f"horizons needs at least 2 values to be a composite, got {horizons}")
    ranks = [
        momentum_signal(monthly_close, j, skip_months).rank(axis=1, pct=True) for j in horizons
    ]
    stacked = pd.concat(ranks, keys=range(len(ranks)))
    composite = stacked.groupby(level=1).mean()
    valid_count = stacked.notna().groupby(level=1).sum()
    return composite.where(valid_count == len(ranks))


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


def daily_dollar_volume(close_panel: pd.DataFrame, volume_panel: pd.DataFrame) -> pd.DataFrame:
    """Daily traded value in rubles per ticker (``VOLUME * CLOSE``), the
    input `illiquidity_signal` averages over a trailing window (cycle 3,
    H-ILLIQ). ``close_panel`` should be `panel.load_close_panel`'s ffilled
    panel (price continuity across isolated small gaps); ``volume_panel``
    `panel.load_volume_panel`'s zero-filled panel (a no-trade day really
    traded zero rubles, unlike a stale-but-plausible carried-forward price)
    -- both built from the same per-ticker date union, so their indices
    align without reindexing.
    """
    return close_panel * volume_panel


def illiquidity_signal(
    daily_dollar_volume_panel: pd.DataFrame,
    formation_dates: pd.DatetimeIndex,
    lookback_months: int,
) -> pd.DataFrame:
    """Trailing mean daily traded value (rubles) as of each formation date
    (window = ``lookback_months * 21`` trading days, matching
    `low_vol_signal`'s convention) -- a HIGHER value means MORE liquid. Pass
    ``direction="low_long"`` to `backtest_factor` to go long the
    least-liquid leg: the illiquidity-premium construction is compensation
    for trading friction, not a misreaction story, so it is deliberately
    the same portfolio-construction direction as `low_vol_signal` (long the
    low value) even though the underlying mechanism is different -- see
    `strategies/CrossSectionalFactors/PREREGISTRATION_cycle3_illiquidity.md`.

    Deliberately **not** masked for the 2022-03-24 halt-reopening day or
    CBOM's 2026-04-13 event (contrast `low_vol_signal`, which masks both
    from realized-*volatility* windows via
    `known_events.mask_returns_for_vol_estimation`): a volume spike distorts
    a 12-month rolling *mean* linearly and boundedly (at most ~1/252nd of a
    12-month window), unlike the quadratic distortion an extreme *return*
    has on a variance estimate. See the pre-registration's "Deliberately
    not masking..." section for the full reasoning.
    """
    window = lookback_months * 21
    mean_dollar_volume = daily_dollar_volume_panel.rolling(
        window=window, min_periods=window // 2
    ).mean()
    return mean_dollar_volume.reindex(formation_dates, method="ffill")


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
