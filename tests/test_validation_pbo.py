"""Tests for moex_backtest.validation.pbo.

`test_matches_a_from_scratch_brute_force_reimplementation` is the main
correctness check: an independent CSCV implementation, re-derived from the
module docstring's algorithm description (not by reading `pbo.py`'s
block-summary-statistics optimization), operating directly on sliced return
arrays instead of precomputed per-block sum/sum-of-squares -- the two should
agree exactly since they compute the same thing two different ways (mirrors
the approach in `test_validation_dsr.py`).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from moex_backtest.validation.pbo import probability_of_backtest_overfitting


def _brute_force_pbo(
    trial_returns: pd.DataFrame, n_blocks: int, periods_per_year: int = 252
) -> tuple[float, np.ndarray, np.ndarray]:
    values = trial_returns.to_numpy(dtype=float)
    n_obs, n_trials = values.shape
    edges = np.linspace(0, n_obs, n_blocks + 1)
    starts = np.round(edges[:-1]).astype(int)
    ends = np.round(edges[1:]).astype(int)
    boundaries = list(zip(starts.tolist(), ends.tolist(), strict=True))

    def _sharpe(col: np.ndarray) -> float:
        mean, std = float(col.mean()), float(col.std(ddof=1))
        if std == 0.0:
            return float("inf") if mean >= 0.0 else float("-inf")
        return mean / std * math.sqrt(periods_per_year)

    half = n_blocks // 2
    combos = list(itertools.combinations(range(n_blocks), half))
    logits = np.empty(len(combos), dtype=float)
    selection_counts = np.zeros(n_trials, dtype=int)

    for i, train_idx in enumerate(combos):
        test_idx = sorted(set(range(n_blocks)) - set(train_idx))
        train_rows = np.concatenate(
            [values[boundaries[b][0] : boundaries[b][1]] for b in train_idx]
        )
        test_rows = np.concatenate([values[boundaries[b][0] : boundaries[b][1]] for b in test_idx])

        train_sharpe = np.array([_sharpe(train_rows[:, j]) for j in range(n_trials)])
        test_sharpe = np.array([_sharpe(test_rows[:, j]) for j in range(n_trials)])

        n_star = int(np.argmax(train_sharpe))
        selection_counts[n_star] += 1
        rank = float(stats.rankdata(test_sharpe, method="average")[n_star])
        omega = rank / (n_trials + 1)
        logits[i] = math.log(omega / (1.0 - omega))

    pbo = float(np.mean(logits <= 0.0))
    return pbo, logits, selection_counts


def test_matches_a_from_scratch_brute_force_reimplementation() -> None:
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0.0005, scale=0.01, size=(96, 7))
    # Give a couple of columns a genuine edge so selection isn't degenerate.
    data[:, 2] += 0.002
    data[:, 5] += 0.001
    trial_returns = pd.DataFrame(data, columns=[f"trial_{j}" for j in range(7)])

    expected_pbo, expected_logits, expected_counts = _brute_force_pbo(trial_returns, n_blocks=8)
    result = probability_of_backtest_overfitting(trial_returns, n_blocks=8)

    assert result.pbo == pytest.approx(expected_pbo)
    np.testing.assert_allclose(result.logits, expected_logits)
    np.testing.assert_array_equal(result.selection_counts, expected_counts)
    assert result.n_combinations == math.comb(8, 4)
    assert result.n_trials == 7
    assert result.n_blocks == 8


def test_n_combinations_matches_the_binomial_coefficient() -> None:
    rng = np.random.default_rng(1)
    trial_returns = pd.DataFrame(rng.normal(size=(64, 3)))

    result = probability_of_backtest_overfitting(trial_returns, n_blocks=16)

    assert result.n_combinations == math.comb(16, 8)
    assert len(result.logits) == result.n_combinations
    assert int(result.selection_counts.sum()) == result.n_combinations


def test_a_consistently_dominant_trial_gets_pbo_zero_and_is_always_selected() -> None:
    # trial 0 beats every other trial by a wide, consistent margin in every
    # block -- it should win in-sample in literally every combination, and
    # since it's genuinely superior, also rank top out-of-sample every time:
    # PBO == 0.0 exactly (no combination has a non-positive logit).
    n_obs = 64
    i = np.arange(n_obs)
    dominant = 0.01 + 0.0005 * np.sin(i)
    noise_a = 0.0005 * np.sin(i * 1.3)
    noise_b = 0.0005 * np.sin(i * 2.1 + 1.0)
    noise_c = 0.0005 * np.sin(i * 0.7 + 2.0)
    trial_returns = pd.DataFrame(
        {"dominant": dominant, "noise_a": noise_a, "noise_b": noise_b, "noise_c": noise_c}
    )

    result = probability_of_backtest_overfitting(trial_returns, n_blocks=8)

    assert result.pbo == 0.0
    assert result.selection_counts[0] == result.n_combinations


def test_a_regime_flip_between_two_trials_produces_nonzero_pbo() -> None:
    # trial A dominates the first half of the sample, trial B dominates the
    # second half -- whichever one an in-sample-only train split favors is,
    # for many splits, exactly the one that then underperforms out-of-sample.
    # This is the canonical overfitting signature CSCV is built to catch.
    first_half = np.array([0.02, 0.02] * 16)
    second_half = np.array([0.0, 0.0] * 16)
    trial_a = np.concatenate([first_half, second_half])
    trial_b = np.concatenate([second_half, first_half])
    trial_returns = pd.DataFrame({"a": trial_a, "b": trial_b})

    result = probability_of_backtest_overfitting(trial_returns, n_blocks=4)

    assert result.pbo > 0.0


def test_rejects_fewer_than_two_trial_columns() -> None:
    trial_returns = pd.DataFrame({"only_trial": [0.01, -0.01, 0.02, 0.0, 0.015, 0.005, 0.01, 0.02]})
    with pytest.raises(ValueError, match="at least 2 trial columns"):
        probability_of_backtest_overfitting(trial_returns, n_blocks=4)


def test_rejects_odd_n_blocks() -> None:
    trial_returns = pd.DataFrame(np.random.default_rng(0).normal(size=(32, 3)))
    with pytest.raises(ValueError, match="even"):
        probability_of_backtest_overfitting(trial_returns, n_blocks=7)


def test_rejects_n_blocks_below_four() -> None:
    trial_returns = pd.DataFrame(np.random.default_rng(0).normal(size=(32, 3)))
    with pytest.raises(ValueError, match=">= 4"):
        probability_of_backtest_overfitting(trial_returns, n_blocks=2)


def test_rejects_more_blocks_than_observations() -> None:
    trial_returns = pd.DataFrame(np.random.default_rng(0).normal(size=(6, 3)))
    with pytest.raises(ValueError, match="at least n_blocks"):
        probability_of_backtest_overfitting(trial_returns, n_blocks=8)
