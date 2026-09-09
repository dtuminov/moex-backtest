"""Tests for moex_backtest.validation.walk_forward."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from moex_backtest.validation.walk_forward import walk_forward_analysis


def _dated_series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=index)


def test_splits_into_the_expected_number_of_non_overlapping_windows() -> None:
    # is_window=4, oos_window=2 -> each window pair consumes 6 points;
    # step defaults to oos_window (2), so windows start at 0, 2, 4, ...
    # 20 points -> starts 0,2,4,6,8,10,12 that still fit a 6-point segment
    # (last valid start is 20-6=14) -> starts 0,2,4,...,14 => 8 windows.
    returns = _dated_series([0.01 * ((-1) ** i) for i in range(20)])

    result = walk_forward_analysis(returns, is_window=4, oos_window=2)

    assert len(result.windows) == 8


def test_hand_computed_single_window_is_sharpe_and_oos_sharpe() -> None:
    is_values = [0.02, -0.01, 0.03, 0.0]
    oos_values = [0.01, -0.02]
    returns = _dated_series(is_values + oos_values)

    result = walk_forward_analysis(returns, is_window=4, oos_window=2)

    assert len(result.windows) == 1
    window = result.windows[0]

    is_mean = sum(is_values) / len(is_values)
    is_std = math.sqrt(sum((v - is_mean) ** 2 for v in is_values) / (len(is_values) - 1))
    expected_is_sharpe = is_mean / is_std * math.sqrt(252)
    assert window.is_sharpe == pytest.approx(expected_is_sharpe)

    oos_mean = sum(oos_values) / len(oos_values)
    oos_std = math.sqrt(sum((v - oos_mean) ** 2 for v in oos_values) / (len(oos_values) - 1))
    expected_oos_sharpe = oos_mean / oos_std * math.sqrt(252)
    assert window.oos_sharpe == pytest.approx(expected_oos_sharpe)

    assert window.is_start == returns.index[0]
    assert window.is_end == returns.index[3]
    assert window.oos_start == returns.index[4]
    assert window.oos_end == returns.index[5]


def test_efficiency_is_none_when_mean_is_sharpe_is_non_positive() -> None:
    # IS segment has a negative mean (Sharpe < 0); OOS segment does too but
    # less badly -- naively dividing would give a positive-looking ratio,
    # exactly the footgun this gate exists to prevent (see module docstring).
    returns = _dated_series([-0.02, -0.03, -0.025, -0.02, -0.01, -0.005])

    result = walk_forward_analysis(returns, is_window=4, oos_window=2)

    assert result.mean_is_sharpe < 0.0
    assert result.efficiency is None


def test_efficiency_is_the_ratio_of_means_when_is_sharpe_is_positive() -> None:
    returns = _dated_series([0.02, 0.018, 0.022, 0.019, 0.021, 0.017, 0.023, 0.02, 0.01, 0.005])

    result = walk_forward_analysis(returns, is_window=4, oos_window=2)

    assert result.mean_is_sharpe > 0.0
    assert result.efficiency == pytest.approx(result.mean_oos_sharpe / result.mean_is_sharpe)


def test_smaller_step_than_oos_window_produces_overlapping_oos_segments() -> None:
    returns = _dated_series([0.01 * ((-1) ** i) for i in range(10)])

    result = walk_forward_analysis(returns, is_window=4, oos_window=2, step=1)

    # segment length 6, step 1, n=10 -> starts 0..4 inclusive -> 5 windows
    assert len(result.windows) == 5
    assert result.windows[0].oos_start == returns.index[4]
    assert result.windows[1].oos_start == returns.index[5]  # overlaps window 0's OOS tail


def test_rejects_window_sizes_below_two() -> None:
    returns = _dated_series([0.01] * 10)
    with pytest.raises(ValueError, match="is_window and oos_window"):
        walk_forward_analysis(returns, is_window=1, oos_window=2)


def test_rejects_a_series_too_short_for_a_single_window() -> None:
    returns = _dated_series([0.01] * 4)
    with pytest.raises(ValueError, match="at least"):
        walk_forward_analysis(returns, is_window=4, oos_window=2)


def test_rejects_non_positive_step() -> None:
    returns = _dated_series([0.01] * 10)
    with pytest.raises(ValueError, match="step"):
        walk_forward_analysis(returns, is_window=4, oos_window=2, step=0)
