"""Stationary-bootstrap significance test for a Sharpe ratio.

Percentile-bootstraps the observed per-period return series: resample it
``n_simulations`` times with
:func:`moex_backtest.validation._bootstrap.stationary_bootstrap_resample`
(block bootstrap, preserves autocorrelation — a plain i.i.d. resample of daily
returns would understate the true sampling uncertainty for a
serially-correlated series), compute the annualized Sharpe ratio on each
resampled path, and report:

- a one-sided p-value: the fraction of bootstrap Sharpes ``<= 0`` — "how often
  would a series with this same empirical distribution of returns, resampled
  with replacement, fail to show a positive risk-adjusted return".
- a 95% percentile confidence interval for the Sharpe ratio.

2000 simulations by default, to match the "Rule Significance Test" convention
already used on this project's crypto/Jesse track (see
``memory/jesse-trade-algo-strategy.md``) — consistent practice across tracks,
not a claim that 2000 is the statistically necessary number; a caller with a
specific precision requirement (e.g. resolving p to 3 significant figures)
should raise it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from moex_backtest.metrics.performance import sharpe_ratio
from moex_backtest.validation._bootstrap import (
    default_expected_block_length,
    stationary_bootstrap_resample,
)

_DEFAULT_N_SIMULATIONS = 2000


@dataclass(frozen=True, slots=True)
class SignificanceResult:
    """See module docstring. ``bootstrap_sharpes`` is the raw simulated
    distribution (length ``n_simulations``) — kept on the result so callers
    can build their own diagnostics/plots without re-running the bootstrap.
    """

    observed_sharpe: float
    p_value: float
    ci_low: float
    ci_high: float
    n_simulations: int
    expected_block_length: float
    bootstrap_sharpes: np.ndarray


def significance_test(
    returns: pd.Series,
    *,
    n_simulations: int = _DEFAULT_N_SIMULATIONS,
    expected_block_length: float | None = None,
    ci_level: float = 0.95,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    rng: np.random.Generator | None = None,
) -> SignificanceResult:
    """Stationary-bootstrap test of ``H0: true Sharpe <= 0`` against the
    observed per-period ``returns`` series.

    ``expected_block_length`` defaults to
    :func:`moex_backtest.validation._bootstrap.default_expected_block_length`
    (``len(returns) ** (1/3)``) if not given. ``ci_level`` controls the
    reported confidence interval (e.g. 0.95 -> [2.5th, 97.5th] percentile of
    the bootstrap Sharpe distribution).

    Raises ``ValueError`` if ``returns`` is empty or has fewer than 2
    observations (a Sharpe ratio needs a sample standard deviation).
    """
    if len(returns) < 2:
        raise ValueError(f"need at least 2 return observations, got {len(returns)}")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    if n_simulations < 1:
        raise ValueError(f"n_simulations must be >= 1, got {n_simulations}")

    values = returns.to_numpy(dtype=float)
    block_length = (
        default_expected_block_length(len(values))
        if expected_block_length is None
        else expected_block_length
    )
    generator = rng if rng is not None else np.random.default_rng()

    resampled = stationary_bootstrap_resample(values, n_simulations, block_length, generator)
    period_rf = risk_free_rate / periods_per_year
    excess = resampled - period_rf
    mean = excess.mean(axis=1)
    std = excess.std(axis=1, ddof=1)
    # A resampled path can (rarely, for very short/degenerate series) have
    # zero variance; treat it the same way performance.sharpe_ratio does.
    bootstrap_sharpes = np.where(
        std == 0.0,
        np.where(mean >= 0, np.inf, -np.inf),
        mean / np.where(std == 0.0, 1.0, std) * np.sqrt(periods_per_year),
    )

    observed = sharpe_ratio(returns, risk_free_rate, periods_per_year)
    p_value = float(np.mean(bootstrap_sharpes <= 0.0))
    alpha = 1.0 - ci_level
    ci_low, ci_high = np.quantile(bootstrap_sharpes, [alpha / 2.0, 1.0 - alpha / 2.0])

    return SignificanceResult(
        observed_sharpe=observed,
        p_value=p_value,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_simulations=n_simulations,
        expected_block_length=block_length,
        bootstrap_sharpes=bootstrap_sharpes,
    )
