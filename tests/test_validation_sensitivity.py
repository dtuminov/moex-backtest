"""Tests for moex_backtest.validation.sensitivity."""

from __future__ import annotations

import pytest

from moex_backtest.validation.sensitivity import parameter_sensitivity


def test_smoothly_varying_metric_is_classified_as_a_plateau() -> None:
    # A smooth, gently-sloped function of the parameter: adjacent jumps are
    # all comparable in size, nothing exceeds spike_ratio * median.
    def metric_fn(value: float) -> float:
        return 1.0 - 0.01 * (value - 10.0) ** 2

    result = parameter_sensitivity("window", base_value=10.0, metric_fn=metric_fn)

    assert result.is_plateau is True
    assert len(result.points) == 7  # base + 6 default relative_steps


def test_isolated_spike_at_the_base_value_is_classified_as_not_a_plateau() -> None:
    # Every perturbed point is flat at 0.0; only the exact base value scores
    # high -- a textbook "lucky single point", not a plateau.
    def metric_fn(value: float) -> float:
        return 5.0 if value == 10.0 else 0.0

    result = parameter_sensitivity("window", base_value=10.0, metric_fn=metric_fn)

    assert result.is_plateau is False


def test_points_are_sorted_by_parameter_value_and_include_the_base_point() -> None:
    calls: list[float] = []

    def metric_fn(value: float) -> float:
        calls.append(value)
        return value

    result = parameter_sensitivity(
        "entry_z", base_value=2.0, metric_fn=metric_fn, relative_steps=(0.5, -0.5)
    )

    values = [p.parameter_value for p in result.points]
    assert values == sorted(values)
    assert 2.0 in values
    base_point = next(p for p in result.points if p.relative_change == 0.0)
    assert base_point.parameter_value == 2.0
    assert base_point.metric_value == 2.0
    assert set(calls) == {1.0, 2.0, 3.0}


def test_hand_computed_max_and_median_adjacent_jump() -> None:
    # base=10 with steps (-0.5, 0.5) -> parameter values 5, 10, 15 (sorted),
    # metric values chosen so the jumps are exactly 1 and 9.
    def metric_fn(value: float) -> float:
        return {5.0: 0.0, 10.0: 1.0, 15.0: 10.0}[value]

    result = parameter_sensitivity(
        "param", base_value=10.0, metric_fn=metric_fn, relative_steps=(-0.5, 0.5)
    )

    assert result.max_adjacent_jump == pytest.approx(9.0)
    assert result.median_adjacent_jump == pytest.approx(5.0)  # median of [1.0, 9.0]


def test_flat_curve_is_a_plateau_even_with_zero_median_jump() -> None:
    def metric_fn(value: float) -> float:
        return 1.0

    result = parameter_sensitivity("param", base_value=10.0, metric_fn=metric_fn)

    assert result.median_adjacent_jump == 0.0
    assert result.max_adjacent_jump == 0.0
    assert result.is_plateau is True


def test_rejects_zero_base_value() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        parameter_sensitivity("param", base_value=0.0, metric_fn=lambda v: v)


def test_rejects_empty_relative_steps() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        parameter_sensitivity("param", base_value=1.0, metric_fn=lambda v: v, relative_steps=())


def test_rejects_a_zero_relative_step() -> None:
    with pytest.raises(ValueError, match="must not include"):
        parameter_sensitivity(
            "param", base_value=1.0, metric_fn=lambda v: v, relative_steps=(0.1, 0.0)
        )


def test_rejects_duplicate_resulting_parameter_values() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parameter_sensitivity(
            "param", base_value=1.0, metric_fn=lambda v: v, relative_steps=(0.1, 0.1)
        )


def test_rejects_non_positive_spike_ratio() -> None:
    with pytest.raises(ValueError, match="spike_ratio"):
        parameter_sensitivity("param", base_value=1.0, metric_fn=lambda v: v, spike_ratio=0.0)
