"""Monte Carlo placement: where does the actually-observed Sharpe fall within
the distribution of Sharpes a return-resampling process would produce?

Two resampling schemes, both against the same observed per-period return
series:

- :func:`monte_carlo_iid` — i.i.d. resample with replacement. Destroys serial
  correlation; answers "if these returns were independent draws from the same
  empirical distribution, how lucky/unlucky was this particular ordering's
  Sharpe".
- :func:`monte_carlo_block_bootstrap` — stationary block bootstrap (see
  :mod:`moex_backtest.validation._bootstrap`). Preserves short-range serial
  dependence; the more informative of the two for daily financial returns.

A ``percentile_rank`` near 50 means the observed result sits at the median of
what resampling this same data would produce — "not in the tail of luck". A
``percentile_rank`` close to 100 (observed clearly above nearly all resampled
scenarios) is the red flag this function exists to catch: it means the
observed Sharpe is higher than all but a few of many plausible reshufflings
of the same data, which is the signature of a result that got lucky rather
than a durable edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from moex_backtest.metrics.performance import sharpe_ratio
from moex_backtest.validation._bootstrap import (
    default_expected_block_length,
    iid_resample,
    stationary_bootstrap_resample,
)

_DEFAULT_N_SIMULATIONS = 2000


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    observed_sharpe: float
    simulated_sharpes: np.ndarray
    percentile_rank: float
    """Percentage of ``simulated_sharpes`` that are ``<= observed_sharpe``
    (0-100). 50 = observed sits at the median of the simulated distribution;
    100 = observed is at or above every simulated scenario."""
    method: str


def _simulated_sharpes(
    resampled: np.ndarray, risk_free_rate: float, periods_per_year: int
) -> np.ndarray:
    period_rf = risk_free_rate / periods_per_year
    excess = resampled - period_rf
    mean = excess.mean(axis=1)
    std = excess.std(axis=1, ddof=1)
    return np.where(
        std == 0.0,
        np.where(mean >= 0, np.inf, -np.inf),
        mean / np.where(std == 0.0, 1.0, std) * np.sqrt(periods_per_year),
    )


def _percentile_rank(observed: float, simulated: np.ndarray) -> float:
    return float(np.mean(simulated <= observed) * 100.0)


def monte_carlo_iid(
    returns: pd.Series,
    *,
    n_simulations: int = _DEFAULT_N_SIMULATIONS,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    rng: np.random.Generator | None = None,
) -> MonteCarloResult:
    """I.i.d. resample-with-replacement Monte Carlo (see module docstring).
    Raises ``ValueError`` if ``returns`` has fewer than 2 observations.
    """
    if len(returns) < 2:
        raise ValueError(f"need at least 2 return observations, got {len(returns)}")
    if n_simulations < 1:
        raise ValueError(f"n_simulations must be >= 1, got {n_simulations}")

    values = returns.to_numpy(dtype=float)
    generator = rng if rng is not None else np.random.default_rng()
    resampled = iid_resample(values, n_simulations, generator)
    simulated = _simulated_sharpes(resampled, risk_free_rate, periods_per_year)
    observed = sharpe_ratio(returns, risk_free_rate, periods_per_year)

    return MonteCarloResult(
        observed_sharpe=observed,
        simulated_sharpes=simulated,
        percentile_rank=_percentile_rank(observed, simulated),
        method="iid_bootstrap",
    )


def monte_carlo_block_bootstrap(
    returns: pd.Series,
    *,
    n_simulations: int = _DEFAULT_N_SIMULATIONS,
    expected_block_length: float | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    rng: np.random.Generator | None = None,
) -> MonteCarloResult:
    """Stationary block-bootstrap Monte Carlo (see module docstring).
    ``expected_block_length`` defaults to
    :func:`moex_backtest.validation._bootstrap.default_expected_block_length`.
    Raises ``ValueError`` if ``returns`` has fewer than 2 observations.
    """
    if len(returns) < 2:
        raise ValueError(f"need at least 2 return observations, got {len(returns)}")
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
    simulated = _simulated_sharpes(resampled, risk_free_rate, periods_per_year)
    observed = sharpe_ratio(returns, risk_free_rate, periods_per_year)

    return MonteCarloResult(
        observed_sharpe=observed,
        simulated_sharpes=simulated,
        percentile_rank=_percentile_rank(observed, simulated),
        method="stationary_block_bootstrap",
    )
