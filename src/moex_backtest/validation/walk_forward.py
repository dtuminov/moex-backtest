"""Rolling walk-forward validation: does an in-sample Sharpe survive out of sample.

Slides a fixed-size in-sample (IS) window and a following out-of-sample (OOS)
window across a return series, computing the Sharpe ratio of each. Reports
per-window results plus an aggregate "walk-forward efficiency"
(``mean(OOS Sharpe) / mean(IS Sharpe)``) — **but only when the mean IS Sharpe
is positive**.

If the mean IS Sharpe is ``<= 0``, dividing by it produces a misleadingly
"healthy"-looking positive percentage even when OOS performance is just as
bad or worse (e.g. IS -0.5 / OOS -0.6 -> a nonsensical +120%). This exact
footgun was caught on this project's crypto/Jesse track
(``memory/jesse-trade-algo-strategy.md``, cycle 8: a 137.4% "efficiency" that
actually meant "OOS lost money slightly faster than IS already was"). When
gated off, :class:`WalkForwardResult` still carries the raw per-window IS/OOS
Sharpes — report those directly instead of a percentage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from moex_backtest.metrics.performance import sharpe_ratio


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    is_sharpe: float
    oos_sharpe: float


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    windows: list[WalkForwardWindow]
    mean_is_sharpe: float
    mean_oos_sharpe: float
    efficiency: float | None
    """``mean_oos_sharpe / mean_is_sharpe``, or ``None`` when
    ``mean_is_sharpe <= 0`` (see module docstring) — in that case, report
    ``mean_is_sharpe``/``mean_oos_sharpe`` (or the per-window values in
    ``windows``) directly instead."""


def walk_forward_analysis(
    returns: pd.Series,
    is_window: int,
    oos_window: int,
    *,
    step: int | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> WalkForwardResult:
    """Slides an ``is_window``-then-``oos_window`` pair of consecutive,
    non-overlapping segments across ``returns``, advancing the start of each
    successive window pair by ``step`` periods (default: ``oos_window``, i.e.
    OOS segments across windows don't overlap each other either — the
    standard non-overlapping-OOS rolling walk-forward scheme). Passing a
    smaller ``step`` produces overlapping OOS segments across windows
    (rolling in the stricter sense); passing ``step >= is_window + oos_window``
    produces non-overlapping windows entirely (expanding/anchored-style
    splits are not implemented here — pass one call per anchor point instead).

    Returns as many windows as fit; raises ``ValueError`` if not even one
    full ``is_window + oos_window`` segment fits in ``returns``.
    """
    if is_window < 2 or oos_window < 2:
        raise ValueError(
            f"is_window and oos_window must each be >= 2 (need a sample std), "
            f"got is_window={is_window}, oos_window={oos_window}"
        )
    step_size = oos_window if step is None else step
    if step_size < 1:
        raise ValueError(f"step must be >= 1, got {step_size}")

    n = len(returns)
    segment_length = is_window + oos_window
    if n < segment_length:
        raise ValueError(
            f"returns has {n} observations, need at least is_window + oos_window "
            f"= {segment_length} for a single walk-forward window"
        )

    windows: list[WalkForwardWindow] = []
    start = 0
    while start + segment_length <= n:
        is_slice = returns.iloc[start : start + is_window]
        oos_slice = returns.iloc[start + is_window : start + segment_length]
        windows.append(
            WalkForwardWindow(
                is_start=is_slice.index[0],
                is_end=is_slice.index[-1],
                oos_start=oos_slice.index[0],
                oos_end=oos_slice.index[-1],
                is_sharpe=sharpe_ratio(is_slice, risk_free_rate, periods_per_year),
                oos_sharpe=sharpe_ratio(oos_slice, risk_free_rate, periods_per_year),
            )
        )
        start += step_size

    mean_is = float(np.mean([w.is_sharpe for w in windows]))
    mean_oos = float(np.mean([w.oos_sharpe for w in windows]))
    efficiency = (mean_oos / mean_is) if mean_is > 0.0 else None

    return WalkForwardResult(
        windows=windows,
        mean_is_sharpe=mean_is,
        mean_oos_sharpe=mean_oos,
        efficiency=efficiency,
    )
