"""Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
Cross-Validation (CSCV).

Source: Bailey, D. H., Borwein, J., Lopez de Prado, M. & Zhu, Q. J., "The
Probability of Backtest Overfitting", *Journal of Computational Finance*.
https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf

**What this catches that :mod:`dsr` doesn't.** Deflated Sharpe Ratio corrects
for "how many trials were run" given a trial-Sharpe log -- but it takes the
IS-winning trial as *given*, from wherever it came from. It has no opinion on
*how* that winner was picked, and in particular doesn't see a screening step
that happens before any Sharpe is even computed (e.g. this project's
cointegration pair search: rank ~1000 candidate pairs by an Engle-Granger
p-value, then backtest only the survivor -- the p-value screen and the
backtested Sharpe are different tests, and DSR's ``n_trials`` has no honest
way to absorb the screen). PBO instead asks the question directly and
empirically: split the *entire available return history* of every candidate
into many independent train/test partitions, and check how often "the
best-looking candidate in-sample" turns out to be a middling-or-worse
performer out-of-sample. A high PBO is direct evidence that whatever
selection process produced the "winner" (screening, grid search, manual
cherry-picking -- CSCV doesn't need to know which) doesn't generalize.

**Algorithm** (``quant-validation-methodology.md`` section 2, matching the
paper):

1. Split the ``T``-period return history into ``n_blocks`` contiguous,
   equal-ish-length blocks (default 16, must be even).
2. For every way to choose exactly half of those blocks as the "train" set
   (the other half is "test") -- ``C(n_blocks, n_blocks // 2)`` combinations,
   e.g. 12,870 for the default 16 -- do steps 3-4.
3. On the train blocks only, compute every candidate's Sharpe ratio and pick
   the argmax (the trial that *would* have been selected as "the winner" had
   this been the only data available).
4. Find that same candidate's *rank* among all candidates' Sharpe ratios on
   the complementary test blocks: ``omega = rank / (n_trials + 1)``,
   ``logit = ln(omega / (1 - omega))``.
5. ``PBO = fraction of combinations where logit <= 0``, i.e. the IS-winner
   ranked at or below the OOS median -- backtest overfitting in its most
   literal sense: a configuration selected for looking best in-sample that is
   not even average out-of-sample.

**Implementation note**: rather than re-slicing and re-computing each trial's
full return array per combination (``O(combinations x n_trials x T)``), each
trial's per-block sum/sum-of-squares/count of *excess* returns are
precomputed once (``O(n_trials x T)``), and every combination's train/test
Sharpe is then a cheap aggregation over ``n_blocks`` block-level statistics --
this is the only reason 12,870 combinations x wide trial grids stays fast
enough to run inside a normal research cycle on this machine (see
``memory/heavy-job-disk-memory-safety.md``).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True, slots=True)
class PBOResult:
    pbo: float
    """Fraction of train/test combinations where the in-sample-best trial's
    out-of-sample rank was at or below the median (``logit <= 0``). In
    [0, 1]; conventionally read as a red flag above ~0.5, and as reasonably
    safe below ~0.2 (no universally agreed hard threshold in the source
    paper -- unlike DSR's 0.95, treat this as a continuous warning signal,
    not a pass/fail gate on its own)."""
    n_combinations: int
    n_trials: int
    n_blocks: int
    logits: np.ndarray
    """Length ``n_combinations``; the full ``logit`` distribution behind
    ``pbo``, kept for callers who want to plot/inspect it directly (e.g. a
    logit-degradation histogram) rather than only the summary fraction."""
    selection_counts: np.ndarray
    """Length ``n_trials``; how many of the ``n_combinations`` train splits
    picked each trial column as the in-sample winner. A count concentrated
    on very few trials (or spread almost uniformly across all of them) is
    itself diagnostic: this project's crypto/Jesse track found that the same
    "good-looking" candidate kept winning in-sample across walk-forward
    windows that included both passing and failing OOS windows
    (``memory/jesse-trade-algo-strategy.md``, cycle 2) -- a concentrated
    selection count here is the same failure mode made visible directly."""


def _block_boundaries(n_obs: int, n_blocks: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n_obs, n_blocks + 1)
    starts = np.round(edges[:-1]).astype(int)
    ends = np.round(edges[1:]).astype(int)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _aggregate_sharpe(
    sum_excess: np.ndarray, sumsq_excess: np.ndarray, n_obs: int, periods_per_year: int
) -> np.ndarray:
    """Vectorized Sharpe ratio from summary statistics (sum, sum-of-squares,
    count of *excess* returns) instead of a raw return array -- see module
    docstring's "Implementation note". Same zero-variance convention as
    :func:`moex_backtest.metrics.performance.sharpe_ratio`
    (``+inf``/``-inf`` rather than ``nan``).
    """
    mean = sum_excess / n_obs
    # ddof=1 sample variance from sufficient statistics: Var = (sumsq - n*mean^2) / (n-1).
    var = (sumsq_excess - n_obs * mean**2) / (n_obs - 1)
    var = np.clip(var, 0.0, None)  # guard tiny negative values from float roundoff
    std = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = mean / std * math.sqrt(periods_per_year)
    return np.where(std == 0.0, np.where(mean >= 0.0, np.inf, -np.inf), sharpe)


def probability_of_backtest_overfitting(
    trial_returns: pd.DataFrame,
    *,
    n_blocks: int = 16,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> PBOResult:
    """Computes PBO via CSCV (see module docstring) over `trial_returns`, a
    ``T x n_trials`` frame of per-period simple returns -- one column per
    candidate/trial (parameter combination, screened pair, architecture,
    whatever the "many things compared, one picked as best" process was),
    all aligned to the same period index (e.g. the same calendar dates).

    `n_blocks` must be even and >= 4 (need at least 2 train + 2 test blocks
    to make "in-sample" and "out-of-sample" meaningful); the paper's own
    examples default to 16 (12,870 combinations), used here as the default
    too. Raises `ValueError` if `trial_returns` has fewer than 2 columns
    (need >=2 trials to rank), fewer rows than `n_blocks` (each block needs
    at least 1 observation), or `n_blocks` is odd or < 4.
    """
    n_obs, n_trials = trial_returns.shape
    if n_trials < 2:
        raise ValueError(f"trial_returns needs at least 2 trial columns, got {n_trials}")
    if n_blocks < 4 or n_blocks % 2 != 0:
        raise ValueError(f"n_blocks must be even and >= 4, got {n_blocks}")
    if n_obs < n_blocks:
        raise ValueError(
            f"trial_returns has {n_obs} rows, need at least n_blocks={n_blocks} "
            "(one observation per block)"
        )

    period_rf = risk_free_rate / periods_per_year
    excess = trial_returns.to_numpy(dtype=float) - period_rf  # (n_obs, n_trials)
    boundaries = _block_boundaries(n_obs, n_blocks)

    block_sum = np.stack([excess[s:e].sum(axis=0) for s, e in boundaries])  # (n_blocks, n_trials)
    block_sumsq = np.stack([(excess[s:e] ** 2).sum(axis=0) for s, e in boundaries])
    block_n = np.array([e - s for s, e in boundaries])
    if (block_n < 1).any():
        raise ValueError(
            f"n_blocks={n_blocks} split {n_obs} observations into at least one empty block; "
            "reduce n_blocks"
        )

    half = n_blocks // 2
    all_blocks = set(range(n_blocks))
    combinations = list(itertools.combinations(range(n_blocks), half))

    logits = np.empty(len(combinations), dtype=float)
    selection_counts = np.zeros(n_trials, dtype=int)

    for i, train_idx in enumerate(combinations):
        test_idx = sorted(all_blocks - set(train_idx))

        train_sum = block_sum[list(train_idx)].sum(axis=0)
        train_sumsq = block_sumsq[list(train_idx)].sum(axis=0)
        train_n = int(block_n[list(train_idx)].sum())
        train_sharpe = _aggregate_sharpe(train_sum, train_sumsq, train_n, periods_per_year)

        test_sum = block_sum[test_idx].sum(axis=0)
        test_sumsq = block_sumsq[test_idx].sum(axis=0)
        test_n = int(block_n[test_idx].sum())
        test_sharpe = _aggregate_sharpe(test_sum, test_sumsq, test_n, periods_per_year)

        n_star = int(np.argmax(train_sharpe))
        selection_counts[n_star] += 1

        rank = float(stats.rankdata(test_sharpe, method="average")[n_star])
        omega = rank / (n_trials + 1)
        logits[i] = math.log(omega / (1.0 - omega))

    pbo = float(np.mean(logits <= 0.0))

    return PBOResult(
        pbo=pbo,
        n_combinations=len(combinations),
        n_trials=n_trials,
        n_blocks=n_blocks,
        logits=logits,
        selection_counts=selection_counts,
    )
