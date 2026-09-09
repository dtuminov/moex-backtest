"""Tests for moex_backtest.validation._bootstrap — resampling primitives."""

from __future__ import annotations

import numpy as np
import pytest

from moex_backtest.validation._bootstrap import (
    default_expected_block_length,
    iid_resample,
    stationary_bootstrap_resample,
)


def test_iid_resample_has_the_requested_shape() -> None:
    returns = np.array([0.01, -0.02, 0.03, 0.0, -0.01])
    rng = np.random.default_rng(0)

    result = iid_resample(returns, n_simulations=7, rng=rng)

    assert result.shape == (7, 5)


def test_iid_resample_only_draws_values_present_in_the_original_series() -> None:
    returns = np.array([1.0, 2.0, 3.0])
    rng = np.random.default_rng(1)

    result = iid_resample(returns, n_simulations=50, rng=rng)

    assert set(np.unique(result)).issubset({1.0, 2.0, 3.0})


def test_iid_resample_is_reproducible_with_the_same_seed() -> None:
    returns = np.array([0.01, -0.02, 0.03, 0.0, -0.01, 0.02])

    result_a = iid_resample(returns, n_simulations=10, rng=np.random.default_rng(42))
    result_b = iid_resample(returns, n_simulations=10, rng=np.random.default_rng(42))

    np.testing.assert_array_equal(result_a, result_b)


def test_iid_resample_rejects_empty_series() -> None:
    with pytest.raises(ValueError, match="empty"):
        iid_resample(np.array([]), n_simulations=5, rng=np.random.default_rng(0))


def test_stationary_bootstrap_resample_has_the_requested_shape() -> None:
    returns = np.arange(20, dtype=float)
    rng = np.random.default_rng(2)

    result = stationary_bootstrap_resample(
        returns, n_simulations=5, expected_block_length=3.0, rng=rng
    )

    assert result.shape == (5, 20)


def test_stationary_bootstrap_with_huge_block_length_is_a_circular_rotation() -> None:
    # expected_block_length so large that p = 1/expected_block_length is
    # effectively 0: with overwhelming probability every position after the
    # first continues the same block, so each simulated row is just a
    # circular rotation of the original series. Using arange(10) as the
    # source series makes a rotation trivially checkable: row[i] must equal
    # (row[0] + i) % 10 for every i.
    returns = np.arange(10, dtype=float)
    rng = np.random.default_rng(3)

    result = stationary_bootstrap_resample(
        returns, n_simulations=20, expected_block_length=1e12, rng=rng
    )

    for row in result:
        expected = (row[0] + np.arange(10)) % 10
        np.testing.assert_array_equal(row, expected)


def test_stationary_bootstrap_resample_only_draws_values_present_in_the_original_series() -> None:
    returns = np.array([1.0, 2.0, 3.0, 4.0])
    rng = np.random.default_rng(4)

    result = stationary_bootstrap_resample(
        returns, n_simulations=30, expected_block_length=2.0, rng=rng
    )

    assert set(np.unique(result)).issubset({1.0, 2.0, 3.0, 4.0})


def test_stationary_bootstrap_resample_is_reproducible_with_the_same_seed() -> None:
    returns = np.array([0.01, -0.02, 0.03, 0.0, -0.01, 0.02, 0.04, -0.03])

    result_a = stationary_bootstrap_resample(
        returns, n_simulations=10, expected_block_length=2.5, rng=np.random.default_rng(7)
    )
    result_b = stationary_bootstrap_resample(
        returns, n_simulations=10, expected_block_length=2.5, rng=np.random.default_rng(7)
    )

    np.testing.assert_array_equal(result_a, result_b)


def test_stationary_bootstrap_resample_rejects_block_length_below_one() -> None:
    with pytest.raises(ValueError, match="expected_block_length"):
        stationary_bootstrap_resample(
            np.array([1.0, 2.0]),
            n_simulations=1,
            expected_block_length=0.5,
            rng=np.random.default_rng(0),
        )


@pytest.mark.parametrize(("n", "expected"), [(1, 1.0), (8, 2.0), (1000, 10.0), (27, 3.0)])
def test_default_expected_block_length_matches_cube_root_heuristic(n: int, expected: float) -> None:
    assert default_expected_block_length(n) == pytest.approx(expected)


def test_default_expected_block_length_rejects_non_positive_n() -> None:
    with pytest.raises(ValueError, match="n_observations"):
        default_expected_block_length(0)
