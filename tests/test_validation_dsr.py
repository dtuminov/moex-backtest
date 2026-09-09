"""Tests for moex_backtest.validation.dsr — hand-computed against the
Bailey & Lopez de Prado (2014) formula (see module docstring for the
citation and the three steps), independently re-derived here rather than
re-reading the implementation, plus the qualitative monotonicity properties
the formula must have regardless of the exact numbers.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from moex_backtest.validation.dsr import deflated_sharpe_ratio

_EULER_MASCHERONI = 0.5772156649015329
_PERIODS_PER_YEAR = 252


def _hand_computed_dsr(
    candidate_returns: list[float], trial_sharpes_annual: list[float]
) -> tuple[float, float, float]:
    """Independent re-implementation of the three-step formula, used as the
    known-answer oracle for test_matches_hand_computed_formula_with_the_full_trial_list.
    Returns (dsr, observed_sharpe_annual, expected_max_annual).
    """
    excess = np.array(candidate_returns, dtype=float)
    t = len(excess)
    std = float(np.std(excess, ddof=1))
    sr_hat = float(np.mean(excess)) / std
    skew = float(stats.skew(excess, bias=True))
    kurtosis = float(stats.kurtosis(excess, fisher=False, bias=True))

    sr_var = (1.0 / (t - 1)) * (1.0 - skew * sr_hat + ((kurtosis - 1.0) / 4.0) * sr_hat**2)

    trials_period = np.array(trial_sharpes_annual, dtype=float) / math.sqrt(_PERIODS_PER_YEAR)
    n = len(trials_period)
    trial_var = float(np.var(trials_period, ddof=1))

    expected_max = math.sqrt(trial_var) * (
        (1.0 - _EULER_MASCHERONI) * stats.norm.ppf(1.0 - 1.0 / n)
        + _EULER_MASCHERONI * stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    )
    dsr = float(stats.norm.cdf((sr_hat - expected_max) / math.sqrt(sr_var)))
    annualize = math.sqrt(_PERIODS_PER_YEAR)
    return dsr, sr_hat * annualize, expected_max * annualize


def test_matches_hand_computed_formula_with_the_full_trial_list() -> None:
    candidate = [0.02, -0.01, 0.03, 0.01, -0.02, 0.015, 0.005, -0.005]
    trial_sharpes = [0.1, 0.2, -0.1, 0.3, 0.0, 0.15]

    expected_dsr, expected_observed, expected_max = _hand_computed_dsr(candidate, trial_sharpes)

    result = deflated_sharpe_ratio(pd.Series(candidate), trial_sharpes=trial_sharpes)

    assert result.dsr == pytest.approx(expected_dsr)
    assert result.observed_sharpe == pytest.approx(expected_observed)
    assert result.expected_max_sharpe_null == pytest.approx(expected_max)
    assert result.n_trials == 6
    assert result.variance_source == "full_trial_distribution"
    assert result.is_optimistic_proxy is False


def test_n_trials_and_variance_proxy_matches_the_full_list_when_variance_matches() -> None:
    candidate = [0.02, -0.01, 0.03, 0.01, -0.02, 0.015, 0.005, -0.005]
    trial_sharpes = [0.1, 0.2, -0.1, 0.3, 0.0, 0.15]

    full_result = deflated_sharpe_ratio(pd.Series(candidate), trial_sharpes=trial_sharpes)

    # Convert the same trial distribution's variance into "annualized units"
    # (Var(X_annual) = periods_per_year * Var(X_period), see dsr.py docstring)
    # to feed the proxy path and confirm it reproduces the exact same result.
    trials_period = np.array(trial_sharpes, dtype=float) / math.sqrt(_PERIODS_PER_YEAR)
    trial_variance_annual = float(np.var(trials_period, ddof=1)) * _PERIODS_PER_YEAR

    proxy_result = deflated_sharpe_ratio(
        pd.Series(candidate), n_trials=6, trial_sharpe_variance=trial_variance_annual
    )

    assert proxy_result.dsr == pytest.approx(full_result.dsr)
    assert proxy_result.variance_source == "n_and_variance_proxy"
    assert proxy_result.is_optimistic_proxy is True


def test_dsr_decreases_as_n_trials_increases_holding_variance_fixed() -> None:
    # The core deflation property: more trials searched -> a higher bar to
    # clear -> lower DSR for the same observed result and the same
    # trial-Sharpe variance.
    candidate = pd.Series([0.02, -0.005, 0.015, 0.01, -0.008, 0.012, 0.006, 0.003, -0.002, 0.018])

    small_n = deflated_sharpe_ratio(candidate, n_trials=2, trial_sharpe_variance=0.25)
    large_n = deflated_sharpe_ratio(candidate, n_trials=2471, trial_sharpe_variance=0.25)

    assert large_n.dsr <= small_n.dsr


def test_dsr_increases_with_the_candidates_own_observed_sharpe() -> None:
    low = pd.Series([0.001, -0.002, 0.0015, 0.0005, -0.001, 0.002, 0.0008, -0.0005, 0.001, 0.0003])
    high = low + 0.02  # same shape/dispersion, shifted mean up -> higher Sharpe

    low_result = deflated_sharpe_ratio(low, n_trials=50, trial_sharpe_variance=0.1)
    high_result = deflated_sharpe_ratio(high, n_trials=50, trial_sharpe_variance=0.1)

    assert high_result.observed_sharpe > low_result.observed_sharpe
    assert high_result.dsr >= low_result.dsr


def test_rejects_fewer_than_three_return_observations() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        deflated_sharpe_ratio(pd.Series([0.01, 0.02]), n_trials=5, trial_sharpe_variance=0.1)


def test_rejects_both_trial_sharpes_and_proxy_given_together() -> None:
    candidate = pd.Series([0.01, -0.01, 0.02, 0.0, 0.015])
    with pytest.raises(ValueError, match="not both"):
        deflated_sharpe_ratio(
            candidate, trial_sharpes=[0.1, 0.2], n_trials=5, trial_sharpe_variance=0.1
        )


def test_rejects_neither_trial_sharpes_nor_proxy_given() -> None:
    candidate = pd.Series([0.01, -0.01, 0.02, 0.0, 0.015])
    with pytest.raises(ValueError, match="must pass"):
        deflated_sharpe_ratio(candidate)


def test_rejects_fewer_than_two_trial_sharpes() -> None:
    candidate = pd.Series([0.01, -0.01, 0.02, 0.0, 0.015])
    with pytest.raises(ValueError, match="at least 2 trials"):
        deflated_sharpe_ratio(candidate, trial_sharpes=[0.1])


def test_rejects_n_trials_below_two() -> None:
    candidate = pd.Series([0.01, -0.01, 0.02, 0.0, 0.015])
    with pytest.raises(ValueError, match="n_trials"):
        deflated_sharpe_ratio(candidate, n_trials=1, trial_sharpe_variance=0.1)


def test_rejects_negative_trial_sharpe_variance() -> None:
    candidate = pd.Series([0.01, -0.01, 0.02, 0.0, 0.015])
    with pytest.raises(ValueError, match="trial_sharpe_variance"):
        deflated_sharpe_ratio(candidate, n_trials=5, trial_sharpe_variance=-0.1)


def test_rejects_zero_variance_candidate_returns() -> None:
    candidate = pd.Series([0.01, 0.01, 0.01, 0.01])
    with pytest.raises(ValueError, match="zero variance"):
        deflated_sharpe_ratio(candidate, n_trials=5, trial_sharpe_variance=0.1)
