"""Deflated Sharpe Ratio (DSR) — Bailey & Lopez de Prado (2014).

Source: Bailey, D. H. & Lopez de Prado, M., "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality",
*Journal of Portfolio Management* 40(5):94-107.
https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

Answers: given that ``n_trials`` Sharpe ratios were computed (a parameter
sweep, a pair search, N architectures — any process that eventually reports
"the best one") and the best-performing one had Sharpe ``SR̂``, what is the
probability that ``SR̂`` is genuinely positive, once the fact that it was
*selected as the best of many* is priced in? A single lucky trial among many
independent, genuinely-zero-Sharpe candidates will, by construction, tend to
show a high Sharpe purely from sampling noise — DSR corrects for exactly
this, plus the estimation-error inflation from non-normal (skewed,
fat-tailed) returns.

**Three steps**, computed here in per-period (not annualized) Sharpe units
internally — the paper's formulas are derived for whatever periodicity the
underlying returns are sampled at, and mixing per-period and annualized
quantities inside the nonlinear terms of step 2 would silently give a wrong
answer, whereas re-annualizing the *final* comparison (steps 1 and 3 together)
after computing everything consistently in one unit is exact. The public
:class:`DeflatedSharpeResult` reports annualized Sharpe values for
readability (matching :func:`moex_backtest.metrics.performance.sharpe_ratio`'s
convention), but ``dsr`` itself is a probability and doesn't depend on the
unit choice as long as steps 1-3 are internally consistent, which this
implementation guarantees by doing all three in per-period units:

1. Expected maximum Sharpe under the null that all ``n_trials`` trials have
   true Sharpe 0 (an extreme-value / Gumbel approximation for the max of
   ``N`` correlated-or-not Gaussian order statistics)::

       E[max SR_N] ~= sqrt(V[{SR_n}]) * [(1-gamma)*Phi^-1(1 - 1/N)
                                          + gamma*Phi^-1(1 - 1/(N*e))]

   ``gamma`` ~= 0.5772156649 (Euler-Mascheroni constant). ``V[{SR_n}]`` is
   the variance of Sharpe **across all N trials** — the full distribution,
   not just the winner or the top few. See ``trial_sharpes`` vs.
   ``n_trials``/``trial_sharpe_variance`` below for what happens when only a
   summary is available.

2. Variance of the Sharpe *estimator itself*, corrected for the candidate's
   own return skewness/kurtosis (non-normal returns inflate the estimator's
   variance beyond the textbook ``1/T`` result)::

       V[SR_hat] = (1/(T-1)) * (1 - gamma_3*SR_hat + ((gamma_4-1)/4)*SR_hat^2)

   ``T`` = number of return observations for the candidate; ``gamma_3``,
   ``gamma_4`` = sample skewness and (raw, normal=3) kurtosis of the
   candidate's own per-period returns.

3. ``DSR = Phi((SR_hat - E[max SR_N]) / sqrt(V[SR_hat]))``. Conventional
   significance threshold: ``DSR > 0.95``.

**On the two ways to supply trial information**: passing the full
``trial_sharpes`` array (every trial's Sharpe, not just the winner's) is the
methodologically correct source for ``V[{SR_n}]`` in step 1, per the paper.
When the full per-trial log isn't available, ``n_trials`` +
``trial_sharpe_variance`` is accepted as a **less rigorous, more optimistic
proxy** — supplying a plausible variance without the actual per-trial
distribution behind it tends to understate ``V[{SR_n}]`` relative to what a
real (often fat-tailed, not clean-Gaussian) set of trial Sharpes would show,
which *understates* ``E[max SR_N]`` and therefore *overstates* DSR. This
mirrors an identical, explicitly-flagged limitation on this project's
crypto/Jesse track (``memory/jesse-trade-algo-strategy.md``, cycle 4: "Jesse
API отдаёт только top-20 кандидатов... использовала доступные консервативные
прокси, которые смещают DSR в оптимистичную сторону"). ``variance_source`` on
the result says which path was used; treat a proxy-based DSR as an upper
bound on the true DSR, not a point estimate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

_EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True, slots=True)
class DeflatedSharpeResult:
    observed_sharpe: float
    """Annualized Sharpe of the candidate (same convention as
    :func:`moex_backtest.metrics.performance.sharpe_ratio`)."""
    expected_max_sharpe_null: float
    """``E[max SR_N]`` under the all-trials-null-Sharpe-0 hypothesis,
    annualized for comparability with ``observed_sharpe``."""
    dsr: float
    """``Phi((SR_hat - E[max SR_N]) / sqrt(V[SR_hat]))``, in [0, 1].
    Conventional significance threshold: > 0.95."""
    n_trials: int
    variance_source: Literal["full_trial_distribution", "n_and_variance_proxy"]
    is_optimistic_proxy: bool
    """``True`` when ``variance_source == "n_and_variance_proxy"`` — see
    module docstring: treat ``dsr`` as an upper bound, not a point estimate,
    in that case."""


def deflated_sharpe_ratio(
    candidate_returns: pd.Series,
    *,
    trial_sharpes: Sequence[float] | None = None,
    n_trials: int | None = None,
    trial_sharpe_variance: float | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> DeflatedSharpeResult:
    """Computes the Deflated Sharpe Ratio for ``candidate_returns`` (the
    winning trial's per-period simple returns) against a search of
    ``n_trials`` (see module docstring for the two ways to supply that).

    Exactly one of ``trial_sharpes`` or (``n_trials`` and
    ``trial_sharpe_variance``) must be given. ``trial_sharpes`` (when given)
    must be **annualized** Sharpes computed the same way as
    :func:`moex_backtest.metrics.performance.sharpe_ratio` (same
    ``periods_per_year``) — this function converts internally to per-period
    units for the actual computation. Likewise ``trial_sharpe_variance`` (when
    given instead) must be the variance of annualized trial Sharpes.

    Both trial-count paths require at least 2 trials (the variance of a
    single trial's Sharpe is undefined) — raises ``ValueError`` otherwise.
    Also raises ``ValueError`` if ``candidate_returns`` has fewer than 3
    observations (skewness/kurtosis need at least a handful of points to be
    defined at all), or if the step-2 variance formula evaluates to <= 0
    (possible for a pathological combination of extreme skew/kurtosis and a
    very small ``T``, where the asymptotic approximation breaks down).
    """
    t = len(candidate_returns)
    if t < 3:
        raise ValueError(f"need at least 3 return observations, got {t}")

    have_full = trial_sharpes is not None
    have_proxy = n_trials is not None or trial_sharpe_variance is not None
    if have_full and have_proxy:
        raise ValueError(
            "pass exactly one of trial_sharpes or (n_trials, trial_sharpe_variance), not both"
        )
    if not have_full and not have_proxy:
        raise ValueError("must pass trial_sharpes or (n_trials, trial_sharpe_variance)")
    if have_proxy and (n_trials is None or trial_sharpe_variance is None):
        raise ValueError("n_trials and trial_sharpe_variance must both be given together")

    if have_full:
        assert trial_sharpes is not None  # narrows for mypy; guaranteed by have_full
        n = len(trial_sharpes)
        if n < 2:
            raise ValueError(f"trial_sharpes needs at least 2 trials, got {n}")
        trials_period = np.asarray(trial_sharpes, dtype=float) / math.sqrt(periods_per_year)
        trial_variance_period = float(np.var(trials_period, ddof=1))
        variance_source: Literal["full_trial_distribution", "n_and_variance_proxy"] = (
            "full_trial_distribution"
        )
    else:
        assert n_trials is not None and trial_sharpe_variance is not None
        if n_trials < 2:
            raise ValueError(f"n_trials must be >= 2, got {n_trials}")
        if trial_sharpe_variance < 0.0:
            raise ValueError(f"trial_sharpe_variance must be >= 0, got {trial_sharpe_variance}")
        n = n_trials
        trial_variance_period = trial_sharpe_variance / periods_per_year
        variance_source = "n_and_variance_proxy"

    excess = candidate_returns.to_numpy(dtype=float) - risk_free_rate / periods_per_year
    std = float(np.std(excess, ddof=1))
    if std == 0.0:
        raise ValueError("candidate_returns has zero variance; Sharpe/DSR are undefined")
    sr_hat_period = float(np.mean(excess)) / std
    skew = float(stats.skew(excess, bias=True))
    kurtosis = float(stats.kurtosis(excess, fisher=False, bias=True))

    sr_variance_period = (1.0 / (t - 1)) * (
        1.0 - skew * sr_hat_period + ((kurtosis - 1.0) / 4.0) * sr_hat_period**2
    )
    if sr_variance_period <= 0.0:
        raise ValueError(
            f"step-2 Sharpe-estimator variance evaluated to {sr_variance_period!r} (<= 0); "
            "the asymptotic approximation is unreliable for this combination of T, skew and "
            "kurtosis -- DSR cannot be computed for this candidate"
        )

    expected_max_period = math.sqrt(trial_variance_period) * (
        (1.0 - _EULER_MASCHERONI) * stats.norm.ppf(1.0 - 1.0 / n)
        + _EULER_MASCHERONI * stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    )

    z_score = (sr_hat_period - expected_max_period) / math.sqrt(sr_variance_period)
    dsr = float(stats.norm.cdf(z_score))

    annualize = math.sqrt(periods_per_year)
    return DeflatedSharpeResult(
        observed_sharpe=sr_hat_period * annualize,
        expected_max_sharpe_null=expected_max_period * annualize,
        dsr=dsr,
        n_trials=n,
        variance_source=variance_source,
        is_optimistic_proxy=(variance_source == "n_and_variance_proxy"),
    )
