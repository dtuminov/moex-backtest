"""Parameter sensitivity: does a strategy's performance metric degrade
smoothly around its chosen parameter values ("plateau"), or is the chosen
value an isolated lucky spike ("cliff")?

Generic over the strategy: the caller supplies a ``metric_fn(parameter_value)
-> float`` closure that runs a full backtest at that parameter value and
returns whichever scalar metric is being checked (typically Sharpe). This
module only evaluates the grid and classifies the resulting curve — it has no
opinion on what a strategy or its parameters are.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

_DEFAULT_RELATIVE_STEPS: tuple[float, ...] = (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3)
_DEFAULT_SPIKE_RATIO = 3.0


@dataclass(frozen=True, slots=True)
class SensitivityPoint:
    relative_change: float
    """0.0 for the base/chosen value; e.g. -0.2 for a point 20% below it."""
    parameter_value: float
    metric_value: float


@dataclass(frozen=True, slots=True)
class ParameterSensitivityResult:
    parameter_name: str
    points: list[SensitivityPoint]
    """Sorted by ``parameter_value`` ascending, including the base point
    (``relative_change == 0.0``)."""
    max_adjacent_jump: float
    median_adjacent_jump: float
    is_plateau: bool
    """``True`` if the metric changes smoothly across the grid (see
    ``max_adjacent_jump`` / ``median_adjacent_jump`` and the
    ``spike_ratio`` threshold in :func:`parameter_sensitivity`), ``False`` if
    one adjacent step is a disproportionate jump relative to the others —
    evidence the base point is an isolated spike rather sitting on a plateau
    of similar-performing neighbors."""


def parameter_sensitivity(
    parameter_name: str,
    base_value: float,
    metric_fn: Callable[[float], float],
    *,
    relative_steps: Sequence[float] = _DEFAULT_RELATIVE_STEPS,
    spike_ratio: float = _DEFAULT_SPIKE_RATIO,
) -> ParameterSensitivityResult:
    """Evaluates ``metric_fn`` at ``base_value`` and at
    ``base_value * (1 + s)`` for each ``s`` in ``relative_steps``, then
    classifies the resulting metric-vs-parameter curve as a plateau or a
    spike.

    **Classification heuristic** (not a formal statistical test — a
    practical screen, tune ``spike_ratio`` if it misfires on a particular
    curve shape): sort all points (base + perturbed) by ``parameter_value``,
    take the absolute differences between metric values at adjacent points,
    and flag a spike if the single largest adjacent jump exceeds
    ``spike_ratio`` times the median adjacent jump. A smoothly degrading
    curve has adjacent jumps of a similar order of magnitude (ratio close to
    1); an isolated spike produces one jump much larger than the rest.

    ``base_value`` must be nonzero (relative perturbation is undefined at 0).
    Raises ``ValueError`` if ``relative_steps`` is empty, contains ``0.0``
    (redundant with the base point), or would perturb ``base_value`` to two
    points sharing the same value (e.g. duplicate steps).
    """
    if base_value == 0.0:
        raise ValueError("base_value must be nonzero (relative perturbation is undefined at 0)")
    if not relative_steps:
        raise ValueError("relative_steps must be non-empty")
    if any(s == 0.0 for s in relative_steps):
        raise ValueError(
            "relative_steps must not include 0.0 (the base point is added automatically)"
        )
    if spike_ratio <= 0.0:
        raise ValueError(f"spike_ratio must be > 0, got {spike_ratio}")

    all_steps = [0.0, *relative_steps]
    parameter_values = [base_value * (1.0 + s) for s in all_steps]
    if len(set(parameter_values)) != len(parameter_values):
        raise ValueError("relative_steps produced duplicate parameter values")

    raw_points = [
        SensitivityPoint(relative_change=s, parameter_value=v, metric_value=metric_fn(v))
        for s, v in zip(all_steps, parameter_values, strict=True)
    ]
    points = sorted(raw_points, key=lambda p: p.parameter_value)

    metric_values = np.array([p.metric_value for p in points])
    adjacent_jumps = np.abs(np.diff(metric_values))
    max_jump = float(np.max(adjacent_jumps))
    median_jump = float(np.median(adjacent_jumps))

    # median_jump == 0 (all jumps equal, most degenerately: all zero, i.e. a
    # perfectly flat curve) can't produce a finite ratio; a flat curve is by
    # definition a plateau (max_jump would also be 0 in that all-zero case).
    is_plateau = max_jump <= spike_ratio * median_jump if median_jump > 0.0 else max_jump == 0.0

    return ParameterSensitivityResult(
        parameter_name=parameter_name,
        points=points,
        max_adjacent_jump=max_jump,
        median_adjacent_jump=median_jump,
        is_plateau=is_plateau,
    )
