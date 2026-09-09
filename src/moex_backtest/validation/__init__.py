"""Statistical validation layer for strategy research results.

Reusable across projects that depend on ``moex-backtest`` (currently
``moex-pairs-trading``, and future cross-sectional-universe work) — every
function here takes a plain ``pandas.Series`` of per-period returns (or, for
:func:`parameter_sensitivity`, a caller-supplied metric callback) rather than
anything specific to a particular strategy engine.

Pipeline, in the order a research result should normally pass through it:

1. :func:`significance_test` — is the mean return significantly above zero
   (stationary-bootstrap p-value + Sharpe confidence interval)?
2. :func:`walk_forward_analysis` — does in-sample performance survive
   out-of-sample, across multiple rolling windows? (Use
   :func:`lock_config`/:func:`require_locked_config` from :mod:`lock` to
   physically gate this step and everything after it behind a config that
   was fixed on in-sample data alone — see that module's docstring.)
3. :func:`parameter_sensitivity` — does performance degrade smoothly around
   the chosen parameters (plateau), or is the choice an isolated spike?
4. :func:`monte_carlo_iid` / :func:`monte_carlo_block_bootstrap` — where does
   the observed result sit within the distribution of plausible resampled
   outcomes ("not in the tail of luck")?
5. :func:`deflated_sharpe_ratio` — Bailey & Lopez de Prado's Deflated Sharpe
   Ratio, correcting for how many trials were searched before this one was
   picked.
6. :func:`probability_of_backtest_overfitting` — CSCV/PBO, a second and
   independent check on the *selection process* itself (screening, grid
   search, cherry-picking) that DSR's trial count alone doesn't fully
   capture — see :mod:`pbo`'s docstring for exactly what this adds over
   step 5.

None of these functions execute a backtest themselves; they all operate on
already-computed return series (or, for step 3, repeatedly call a
caller-supplied backtest closure; for step 6, a ``T x n_trials`` frame of
every candidate's returns). See each module's docstring for the underlying
formula/method and its source.
"""

from __future__ import annotations

from moex_backtest.validation.dsr import DeflatedSharpeResult, deflated_sharpe_ratio
from moex_backtest.validation.lock import (
    ConfigNotLockedError,
    LockedConfig,
    lock_config,
    require_locked_config,
)
from moex_backtest.validation.monte_carlo import (
    MonteCarloResult,
    monte_carlo_block_bootstrap,
    monte_carlo_iid,
)
from moex_backtest.validation.pbo import PBOResult, probability_of_backtest_overfitting
from moex_backtest.validation.sensitivity import (
    ParameterSensitivityResult,
    SensitivityPoint,
    parameter_sensitivity,
)
from moex_backtest.validation.significance import SignificanceResult, significance_test
from moex_backtest.validation.walk_forward import (
    WalkForwardResult,
    WalkForwardWindow,
    walk_forward_analysis,
)

__all__ = [
    "ConfigNotLockedError",
    "DeflatedSharpeResult",
    "LockedConfig",
    "MonteCarloResult",
    "PBOResult",
    "ParameterSensitivityResult",
    "SensitivityPoint",
    "SignificanceResult",
    "WalkForwardResult",
    "WalkForwardWindow",
    "deflated_sharpe_ratio",
    "lock_config",
    "monte_carlo_block_bootstrap",
    "monte_carlo_iid",
    "parameter_sensitivity",
    "probability_of_backtest_overfitting",
    "require_locked_config",
    "significance_test",
    "walk_forward_analysis",
]
