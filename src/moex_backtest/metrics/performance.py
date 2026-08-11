"""Risk/return metrics for a periodic-return series.

Every function takes plain period returns (e.g. daily simple returns,
``equity.pct_change()``) rather than prices, so the same functions work for
any bar size — the caller supplies ``periods_per_year`` to annualize (252
for daily, 12 for monthly, etc). All functions raise ``ValueError`` on an
empty series instead of silently returning ``NaN``: a metric computed on no
data is a bug at the call site, not a valid (if degenerate) result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _require_non_empty(returns: pd.Series) -> None:
    if returns.empty:
        raise ValueError("returns series is empty")


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Geometric annualized return: ``(1 + total_return) ** (periods_per_year / n) - 1``."""
    _require_non_empty(returns)
    n = len(returns)
    growth = float(np.prod((1.0 + returns).to_numpy(dtype=float)))
    return float(growth ** (periods_per_year / n)) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Sample standard deviation of returns, scaled by ``sqrt(periods_per_year)``."""
    _require_non_empty(returns)
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """Annualized Sharpe ratio using a constant annual risk-free rate.

    ``(annualized_return - risk_free_rate) / annualized_volatility``. Returns
    ``inf``/``-inf`` if volatility is zero (a constant return series).
    """
    _require_non_empty(returns)
    excess = annualized_return(returns, periods_per_year) - risk_free_rate
    vol = annualized_volatility(returns, periods_per_year)
    if vol == 0.0:
        return float("inf") if excess >= 0 else float("-inf")
    return excess / vol


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """Like :func:`sharpe_ratio`, but only penalizes downside deviation (return < 0).

    Returns ``inf`` if there are no negative returns in the sample (no
    downside risk observed) and the excess return is non-negative.
    """
    _require_non_empty(returns)
    excess = annualized_return(returns, periods_per_year) - risk_free_rate
    downside = returns[returns < 0]
    if downside.empty:
        return float("inf") if excess >= 0 else float("-inf")
    downside_vol = float(np.sqrt((downside**2).mean()) * np.sqrt(periods_per_year))
    if downside_vol == 0.0:
        return float("inf") if excess >= 0 else float("-inf")
    return excess / downside_vol


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
    sample, losses exceeded 3% in the worst 5% of periods.
    """
    _require_non_empty(returns)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return float(-returns.quantile(alpha))


def historical_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical Conditional VaR / Expected Shortfall: mean loss in the worst ``alpha``
    tail, as a positive loss fraction. Always ``>= historical_var`` at the same ``alpha``.
    """
    _require_non_empty(returns)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    threshold = returns.quantile(alpha)
    tail = returns[returns <= threshold]
    return float(-tail.mean())
