"""Tests for moex_backtest.validation.monte_carlo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from moex_backtest.validation.monte_carlo import monte_carlo_block_bootstrap, monte_carlo_iid


def test_constant_positive_returns_place_observed_exactly_at_the_top_percentile() -> None:
    # Every i.i.d. resample of a constant series is that same constant:
    # simulated Sharpe == observed Sharpe (+inf) for every simulation, so
    # observed is <= every simulated value -> percentile_rank is exactly 100.
    # Exact 0.0 (rather than e.g. 0.01) sidesteps floating-point summation
    # noise that can make a "constant" series' sample std a tiny nonzero
    # epsilon instead of bit-exact 0.0 for some repeat counts.
    returns = pd.Series([0.0] * 10)

    result = monte_carlo_iid(returns, n_simulations=200, rng=np.random.default_rng(0))

    assert result.observed_sharpe == float("inf")
    assert result.percentile_rank == 100.0
    assert result.method == "iid_bootstrap"
    assert len(result.simulated_sharpes) == 200


def test_block_bootstrap_constant_positive_returns_place_observed_at_the_top_percentile() -> None:
    returns = pd.Series([0.0] * 10)

    result = monte_carlo_block_bootstrap(returns, n_simulations=200, rng=np.random.default_rng(0))

    assert result.observed_sharpe == float("inf")
    assert result.percentile_rank == 100.0
    assert result.method == "stationary_block_bootstrap"


def test_iid_percentile_rank_is_bounded_and_reproducible() -> None:
    returns = pd.Series([0.02, -0.01, 0.03, 0.0, -0.02, 0.015, 0.005, -0.005, 0.01, -0.015])

    result_a = monte_carlo_iid(returns, n_simulations=300, rng=np.random.default_rng(5))
    result_b = monte_carlo_iid(returns, n_simulations=300, rng=np.random.default_rng(5))

    assert 0.0 <= result_a.percentile_rank <= 100.0
    np.testing.assert_array_equal(result_a.simulated_sharpes, result_b.simulated_sharpes)
    assert result_a.percentile_rank == result_b.percentile_rank


def test_block_bootstrap_percentile_rank_is_bounded_and_reproducible() -> None:
    returns = pd.Series([0.02, -0.01, 0.03, 0.0, -0.02, 0.015, 0.005, -0.005, 0.01, -0.015])

    result_a = monte_carlo_block_bootstrap(returns, n_simulations=300, rng=np.random.default_rng(5))
    result_b = monte_carlo_block_bootstrap(returns, n_simulations=300, rng=np.random.default_rng(5))

    assert 0.0 <= result_a.percentile_rank <= 100.0
    np.testing.assert_array_equal(result_a.simulated_sharpes, result_b.simulated_sharpes)


@pytest.mark.parametrize("fn", [monte_carlo_iid, monte_carlo_block_bootstrap])
def test_rejects_a_series_shorter_than_two_observations(fn: object) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        fn(pd.Series([0.01]))  # type: ignore[operator]


@pytest.mark.parametrize("fn", [monte_carlo_iid, monte_carlo_block_bootstrap])
def test_rejects_non_positive_n_simulations(fn: object) -> None:
    returns = pd.Series([0.01, -0.01, 0.02])
    with pytest.raises(ValueError, match="n_simulations"):
        fn(returns, n_simulations=0)  # type: ignore[operator]
