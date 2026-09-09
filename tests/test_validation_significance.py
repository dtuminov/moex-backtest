"""Tests for moex_backtest.validation.significance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from moex_backtest.validation.significance import significance_test


def test_clearly_positive_returns_have_a_zero_p_value_and_a_positive_ci() -> None:
    # Small deterministic alternating noise around a mean far above zero:
    # every plausible bootstrap resample still has a clearly positive mean,
    # so p (fraction of bootstrap Sharpes <= 0) must be exactly 0.
    returns = pd.Series([0.02, 0.018, 0.022, 0.019, 0.021, 0.017, 0.023, 0.02] * 5)

    result = significance_test(returns, n_simulations=500, rng=np.random.default_rng(0))

    assert result.p_value == 0.0
    assert result.ci_low > 0.0
    assert result.observed_sharpe > 0.0
    assert result.n_simulations == 500
    assert len(result.bootstrap_sharpes) == 500


def test_clearly_negative_returns_have_a_high_p_value() -> None:
    returns = pd.Series([-0.02, -0.018, -0.022, -0.019, -0.021, -0.017, -0.023, -0.02] * 5)

    result = significance_test(returns, n_simulations=500, rng=np.random.default_rng(0))

    assert result.p_value > 0.9
    assert result.observed_sharpe < 0.0


def test_constant_positive_returns_give_an_infinite_sharpe_and_zero_p_value() -> None:
    # Every resample of a constant series is that same constant: std == 0
    # everywhere, so every bootstrap Sharpe is +inf (see performance.sharpe_ratio's
    # own convention for a zero-std, non-negative-mean series). Using exact
    # 0.0 (rather than e.g. 0.01) sidesteps floating-point summation noise
    # that can make a "constant" series' sample std a tiny nonzero epsilon
    # instead of bit-exact 0.0 for some repeat counts.
    returns = pd.Series([0.0] * 10)

    result = significance_test(returns, n_simulations=100, rng=np.random.default_rng(0))

    assert result.p_value == 0.0
    assert result.observed_sharpe == float("inf")
    assert np.all(np.isinf(result.bootstrap_sharpes))


def test_is_reproducible_with_the_same_seed() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, 0.0, -0.01, 0.02, -0.015, 0.025] * 3)

    result_a = significance_test(returns, n_simulations=200, rng=np.random.default_rng(11))
    result_b = significance_test(returns, n_simulations=200, rng=np.random.default_rng(11))

    np.testing.assert_array_equal(result_a.bootstrap_sharpes, result_b.bootstrap_sharpes)
    assert result_a.p_value == result_b.p_value


def test_rejects_a_series_shorter_than_two_observations() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        significance_test(pd.Series([0.01]))


def test_rejects_ci_level_outside_open_unit_interval() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, -0.02])
    with pytest.raises(ValueError, match="ci_level"):
        significance_test(returns, ci_level=1.5)


def test_rejects_non_positive_n_simulations() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, -0.02])
    with pytest.raises(ValueError, match="n_simulations"):
        significance_test(returns, n_simulations=0)


def test_default_block_length_matches_the_shared_heuristic() -> None:
    from moex_backtest.validation._bootstrap import default_expected_block_length

    returns = pd.Series(np.linspace(-0.01, 0.01, 27))

    result = significance_test(returns, n_simulations=10, rng=np.random.default_rng(0))

    assert result.expected_block_length == default_expected_block_length(27)
