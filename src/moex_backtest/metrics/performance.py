"""Risk/return metrics for a periodic-return series.

Every function takes plain period returns (e.g. daily simple returns,
``equity.pct_change()``), so the same functions work for any bar size — the
caller supplies ``periods_per_year`` to annualize (252 for daily, 12 for
monthly, etc). All functions raise ``ValueError`` on an empty series instead
of silently returning ``NaN``: a metric computed on no data is a bug at the
call site.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _require_non_empty(returns: pd.Series) -> None:
    if returns.empty:
        raise ValueError("returns series is empty")


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Geometric annualized return: ``(1 + total_return) ** (periods_per_year / n) - 1``.

    Raises ``ValueError`` if the compounded growth factor is ``<= 0`` (i.e.
    the total return over the period is ``<= -100%``). The product of
    ``(1 + returns)`` can only reach ``<= 0`` if *at least one period's own*
    return is ``<= -100%`` — every well-formed per-period return above that
    floor is a strictly positive factor, so no amount of compounding across
    *moderate* losses can drive the product negative on its own. A single
    period that bad typically means equity crossed zero within one bar (e.g.
    a leveraged short gapping through zero) — a raw
    ``float(growth ** fractional_exponent)`` would otherwise raise an opaque
    ``TypeError`` (Python refuses to convert the resulting complex number to
    a float) instead of explaining what happened. See ``Portfolio``'s equity
    floor for how the engine limits how much *further* damage such a bar can
    do, even though it can't undo that one bar's own return.
    """
    _require_non_empty(returns)
    n = len(returns)
    growth = float(np.prod((1.0 + returns).to_numpy(dtype=float)))
    if growth <= 0.0:
        raise ValueError(
            "cannot annualize a return series with total return <= -100% "
            f"(compounded growth factor = {growth!r}); equity was wiped out or went "
            "negative over the period, so a geometric/CAGR-style return is undefined"
        )
    return float(growth ** (periods_per_year / n)) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Sample standard deviation of returns, scaled by ``sqrt(periods_per_year)``."""
    _require_non_empty(returns)
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """Annualized Sharpe ratio: mean(excess) / std(excess) * sqrt(periods_per_year).

    ``risk_free_rate`` is an annual rate, converted to a per-period rate by
    dividing by ``periods_per_year`` and subtracted from every return before
    taking the mean/std. This is the textbook definition (Sharpe 1994) —
    per-period arithmetic mean and sample standard deviation, not annualized
    return over annualized volatility, which is a different (and for
    volatile series, systematically lower) number. Returns ``inf``/``-inf``
    if the standard deviation is zero (a constant return series).
    """
    _require_non_empty(returns)
    period_rf = risk_free_rate / periods_per_year
    excess = returns - period_rf
    std = float(excess.std(ddof=1))
    if std == 0.0:
        return float("inf") if excess.mean() >= 0 else float("-inf")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """Like :func:`sharpe_ratio`, but the denominator only penalizes downside deviation.

    Downside deviation is ``sqrt(mean(min(excess, 0) ** 2))``, averaged over
    all ``N`` periods: a period with non-negative excess return contributes
    0 to the sum but still counts toward ``N`` (the van der Meer/Sortino
    definition). Returns ``inf`` if no period has a shortfall against
    ``risk_free_rate`` and the mean excess return is non-negative.
    """
    _require_non_empty(returns)
    period_rf = risk_free_rate / periods_per_year
    excess = returns - period_rf
    mean_excess = float(excess.mean())
    shortfall = excess.clip(upper=0.0)
    downside_dev = float(np.sqrt((shortfall**2).mean()))
    if downside_dev == 0.0:
        return float("inf") if mean_excess >= 0 else float("-inf")
    return float(mean_excess / downside_dev * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Deepest peak-to-trough decline of an equity/NAV curve, as a negative fraction.

    E.g. ``-0.25`` means the curve fell 25% from its running high at the worst point.
    """
    _require_non_empty(equity_curve)
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized return divided by the magnitude of the max drawdown.

    Returns ``inf`` if the equity curve never drew down (max drawdown is zero).
    """
    _require_non_empty(returns)
    mdd = max_drawdown(equity_curve)
    if mdd == 0.0:
        return float("inf")
    return annualized_return(returns, periods_per_year) / abs(mdd)


def historical_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical (non-parametric) Value at Risk at level ``alpha``, as a positive loss fraction.

    E.g. ``historical_var(returns, alpha=0.05) == 0.03`` means: over this
    sample, losses exceeded 3% in the worst 5% of periods. The quantile is
    estimated with pandas' default linear interpolation between the two
    nearest observed returns (``Series.quantile``'s ``interpolation="linear"``).
    """
    _require_non_empty(returns)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return float(-returns.quantile(alpha))


def historical_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical Conditional VaR / Expected Shortfall: mean loss in the worst ``alpha``
    tail, as a positive loss fraction. Always ``>= historical_var`` at the same ``alpha``.
    The tail cutoff uses the same linearly-interpolated quantile as
    :func:`historical_var`.
    """
    _require_non_empty(returns)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    threshold = returns.quantile(alpha)
    tail = returns[returns <= threshold]
    return float(-tail.mean())
