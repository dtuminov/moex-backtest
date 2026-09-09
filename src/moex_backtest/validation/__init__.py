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
   out-of-sample, across multiple rolling windows?
3. :func:`parameter_sensitivity` — does performance degrade smoothly around
   the chosen parameters (plateau), or is the choice an isolated spike?
4. :func:`monte_carlo_iid` / :func:`monte_carlo_block_bootstrap` — where does
   the observed result sit within the distribution of plausible resampled
   outcomes ("not in the tail of luck")?
5. :func:`deflated_sharpe_ratio` — Bailey & Lopez de Prado's Deflated Sharpe
   Ratio, correcting for how many trials were searched before this one was
   picked.

None of these functions execute a backtest themselves; they all operate on
already-computed return series (or, for step 3, repeatedly call a
caller-supplied backtest closure). See each module's docstring for the
underlying formula/method and its source.
"""

from __future__ import annotations

from moex_backtest.validation.dsr import DeflatedSharpeResult, deflated_sharpe_ratio
from moex_backtest.validation.monte_carlo import (
    MonteCarloResult,
    monte_carlo_block_bootstrap,
    monte_carlo_iid,
)
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
    "DeflatedSharpeResult",
    "MonteCarloResult",
    "ParameterSensitivityResult",
    "SensitivityPoint",
    "SignificanceResult",
    "WalkForwardResult",
    "WalkForwardWindow",
    "deflated_sharpe_ratio",
    "monte_carlo_block_bootstrap",
    "monte_carlo_iid",
    "parameter_sensitivity",
    "significance_test",
    "walk_forward_analysis",
]
