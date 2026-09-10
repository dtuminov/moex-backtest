"""Parameter sensitivity: does a strategy's performance metric degrade
smoothly around its chosen parameter values ("plateau"), or is the chosen
value an isolated lucky spike ("cliff")?

Generic over the strategy: the caller supplies a ``metric_fn(parameter_value)
-> float`` closure that runs a full backtest at that parameter value and
returns whichever scalar metric is being checked (typically Sharpe). This
module only evaluates the grid and classifies the resulting curve — it has no
opinion on what a strategy or its parameters are.

**``is_plateau`` looks only at the two jumps touching the base point** (its
nearest tested neighbor on each side), not the largest jump anywhere in the
swept grid. An earlier version compared the single largest adjacent jump
*anywhere in the sorted grid* against the median jump — this mislabeled a
genuinely smooth neighborhood around the base value as a SPIKE whenever the
grid's largest jump happened to fall somewhere else entirely (e.g. at the far
edge of the swept range, across an unrelated regime boundary that has nothing
to do with the base value's own stability). This is not hypothetical: it is
exactly what happened on
``strategies/CrossSectionalFactors/reports/momentum_sensitivity_diagnostic.md``
in this project's MOEX track — cycle 1 locked a momentum lookback of 12
months, and the default ±10/20/30% grid's largest jump landed between the
*unrelated* 8-month and 10-month points (a real regime boundary a few months
below the base value, not a symptom of fragility at 12 itself), which was
enough to flag the whole curve as a SPIKE even though 12's immediate
neighbors (11 and 13 months) were close to it. A follow-up dense scan (2..20
months) confirmed the base value sits on a broad, genuinely smooth local hump
peaking around 10-14 months. ``max_adjacent_jump``/``median_adjacent_jump``
(whole-grid statistics) are kept on the result as informational context —
still useful for noticing "the plateau ends abruptly somewhere in the tested
range" — but no longer drive the plateau/spike verdict; ``max_local_jump`` is
the field that does.
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
    """Largest jump between any two adjacent points anywhere in the sorted
    grid (not necessarily touching the base point) — informational context
    only, see module docstring; does not drive ``is_plateau``."""
    median_adjacent_jump: float
    """Median jump across the whole sorted grid — the "typical variability"
    scale that both ``max_adjacent_jump`` and ``max_local_jump`` are compared
    against."""
    max_local_jump: float
    """Largest jump between the base point and its nearest tested neighbor
    on either side (only one side if the base point sits at the edge of the
    grid). This is what ``is_plateau`` is actually computed from — see
    module docstring for why."""
    is_plateau: bool
    """``True`` if the metric changes smoothly in the base point's immediate
    neighborhood (``max_local_jump`` small relative to ``median_adjacent_jump``,
    per the ``spike_ratio`` threshold in :func:`parameter_sensitivity`),
    ``False`` if a step immediately adjacent to the base value is a
    disproportionate jump — evidence the base point itself is an isolated
    spike, not evidence that the grid is bumpy somewhere else."""


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
    and flag a spike if the jump from the base point to its nearest tested
    neighbor (on whichever side, or both) exceeds ``spike_ratio`` times the
    median adjacent jump *across the whole grid* (the median is still a
    whole-grid statistic — a reasonable "typical variability" scale even
    though only the base's own local jump is compared against it; see module
    docstring for why the *numerator* is local, not global). A smoothly
    degrading neighborhood around the base has a local jump of a similar
    order of magnitude to the grid's typical jump (ratio close to 1); an
    isolated spike at the base produces a local jump much larger than that.

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

    # Local jump: only the jump(s) touching the base point (relative_change
    # == 0.0), on whichever side(s) it has a neighbor — see module docstring
    # for why this, and not the grid-wide max, is what should drive
    # is_plateau. adjacent_jumps[i] is the jump between points[i] and
    # points[i+1], so the base's own local jump(s) are adjacent_jumps at
    # index (base_index - 1) (jump to its left neighbor, if any) and/or
    # base_index (jump to its right neighbor, if any).
    base_index = next(i for i, p in enumerate(points) if p.relative_change == 0.0)
    local_jumps = [
        adjacent_jumps[i]
        for i in (base_index - 1, base_index)
        if 0 <= i < len(adjacent_jumps)
    ]
    max_local_jump = float(max(local_jumps))

    # median_jump == 0 (all jumps equal, most degenerately: all zero, i.e. a
    # perfectly flat curve) can't produce a finite ratio; a flat curve is by
    # definition a plateau (max_local_jump would also be 0 in that case).
    is_plateau = (
        max_local_jump <= spike_ratio * median_jump if median_jump > 0.0 else max_local_jump == 0.0
    )

    return ParameterSensitivityResult(
        parameter_name=parameter_name,
        points=points,
        max_adjacent_jump=max_jump,
        median_adjacent_jump=median_jump,
        max_local_jump=max_local_jump,
        is_plateau=is_plateau,
    )
