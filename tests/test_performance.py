"""Tests for moex_backtest.metrics.performance — analytical examples plus edge cases."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from moex_backtest.metrics.performance import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    historical_cvar,
    historical_var,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


def test_annualized_return_compounds_two_periods() -> None:
    returns = pd.Series([0.1, 0.1])

    result = annualized_return(returns, periods_per_year=2)

    assert result == pytest.approx(1.1 * 1.1 - 1.0)


def test_annualized_volatility_matches_hand_computed_sample_std() -> None:
    returns = pd.Series([0.0, 0.02])  # mean 0.01, sample std (ddof=1) = sqrt(0.0002)

    result = annualized_volatility(returns, periods_per_year=1)

    assert result == pytest.approx(math.sqrt(0.0002))


def test_annualized_volatility_scales_with_sqrt_periods() -> None:
    returns = pd.Series([0.0, 0.02, -0.01, 0.03])

    vol_1 = annualized_volatility(returns, periods_per_year=1)
    vol_4 = annualized_volatility(returns, periods_per_year=4)

    assert vol_4 == pytest.approx(vol_1 * 2.0)


def test_sharpe_ratio_is_infinite_for_constant_returns() -> None:
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])

    assert sharpe_ratio(returns) == float("inf")


def test_sharpe_ratio_is_negative_infinite_for_constant_negative_returns() -> None:
    returns = pd.Series([-0.01, -0.01, -0.01])

    assert sharpe_ratio(returns) == float("-inf")


def test_sortino_ratio_is_infinite_with_no_downside_returns() -> None:
    returns = pd.Series([0.01, 0.02, 0.03])

    assert sortino_ratio(returns) == float("inf")


def test_sortino_ratio_matches_a_hand_computed_formula() -> None:
    returns = pd.Series([0.02, -0.03, 0.01, -0.01])
    growth = 1.02 * 0.97 * 1.01 * 0.99
    expected_excess = growth ** (1 / 4) - 1.0
    downside = [-0.03, -0.01]
    expected_downside_vol = math.sqrt(sum(x**2 for x in downside) / len(downside))

    result = sortino_ratio(returns, periods_per_year=1)

    assert result == pytest.approx(expected_excess / expected_downside_vol)


def test_max_drawdown_finds_the_deepest_trough() -> None:
    equity = pd.Series([100.0, 110.0, 90.0, 95.0, 120.0])

    result = max_drawdown(equity)

    assert result == pytest.approx(90.0 / 110.0 - 1.0)


def test_max_drawdown_is_zero_for_monotonically_increasing_curve() -> None:
    equity = pd.Series([100.0, 101.0, 105.0, 110.0])

    assert max_drawdown(equity) == 0.0


def test_calmar_ratio_is_infinite_with_no_drawdown() -> None:
    equity = pd.Series([100.0, 101.0, 105.0, 110.0])
    returns = equity.pct_change().dropna()

    assert calmar_ratio(returns, equity) == float("inf")


def test_calmar_ratio_matches_return_over_drawdown() -> None:
    equity = pd.Series([100.0, 110.0, 90.0, 95.0, 120.0])
    returns = equity.pct_change().dropna()

    result = calmar_ratio(returns, equity, periods_per_year=1)

    expected = annualized_return(returns, periods_per_year=1) / abs(max_drawdown(equity))
    assert result == pytest.approx(expected)


def test_historical_var_and_cvar_on_a_quantile_aligned_sample() -> None:
    # 11 symmetric points -> alpha=0.1 quantile lands exactly on sorted[1] = -0.08,
    # no interpolation ambiguity.
    returns = pd.Series([-0.10, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10])

    var = historical_var(returns, alpha=0.1)
    cvar = historical_cvar(returns, alpha=0.1)

    assert var == pytest.approx(0.08)
    assert cvar == pytest.approx(0.09)  # mean(-0.10, -0.08) tail, sign-flipped


def test_cvar_is_at_least_as_large_as_var() -> None:
    returns = pd.Series([-0.2, -0.05, -0.01, 0.0, 0.01, 0.02, 0.03, 0.1])

    assert historical_cvar(returns, alpha=0.25) >= historical_var(returns, alpha=0.25)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_var_rejects_alpha_outside_open_unit_interval(alpha: float) -> None:
    returns = pd.Series([0.01, -0.01, 0.02])

    with pytest.raises(ValueError, match="alpha"):
        historical_var(returns, alpha=alpha)


@pytest.mark.parametrize(
    "fn",
    [
        annualized_return,
        annualized_volatility,
        sharpe_ratio,
        sortino_ratio,
        max_drawdown,
        historical_var,
        historical_cvar,
    ],
)
def test_metrics_reject_empty_series(fn: object) -> None:
    with pytest.raises(ValueError, match="empty"):
        fn(pd.Series([], dtype=float))  # type: ignore[operator]
