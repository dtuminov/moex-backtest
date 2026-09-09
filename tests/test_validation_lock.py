"""Tests for moex_backtest.validation.lock -- this is a defensive/process
control, not a statistical function: the tests exist to confirm it fails
loudly (not silently) in exactly the situations it's meant to prevent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from moex_backtest.validation.lock import (
    ConfigNotLockedError,
    lock_config,
    require_locked_config,
)


def test_lock_then_require_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "cycle1.lock.json"

    locked = lock_config(
        path,
        cycle_name="cycle1",
        config={"lookback_months": 6, "skip_month": True},
        is_metric_name="is_sharpe",
        is_metric_value=1.23,
    )
    fetched = require_locked_config(path, cycle_name="cycle1")

    assert path.exists()
    assert fetched.cycle_name == "cycle1"
    assert fetched.config == {"lookback_months": 6, "skip_month": True}
    assert fetched.is_metric_name == "is_sharpe"
    assert fetched.is_metric_value == pytest.approx(1.23)
    assert fetched == locked


def test_require_before_lock_raises(tmp_path: Path) -> None:
    path = tmp_path / "never_locked.json"

    with pytest.raises(ConfigNotLockedError, match="does not exist"):
        require_locked_config(path, cycle_name="cycle1")


def test_require_with_a_different_cycle_name_raises(tmp_path: Path) -> None:
    path = tmp_path / "cycle1.lock.json"
    lock_config(
        path, cycle_name="cycle1", config={}, is_metric_name="is_sharpe", is_metric_value=1.0
    )

    with pytest.raises(ConfigNotLockedError, match="cycle1"):
        require_locked_config(path, cycle_name="cycle2")


def test_locking_twice_at_the_same_path_raises_without_overwriting(tmp_path: Path) -> None:
    path = tmp_path / "cycle1.lock.json"
    lock_config(
        path, cycle_name="cycle1", config={"a": 1}, is_metric_name="is_sharpe", is_metric_value=1.0
    )

    with pytest.raises(FileExistsError, match="already exists"):
        lock_config(
            path,
            cycle_name="cycle1",
            config={"a": 2},  # a different config -- must NOT silently replace the lock
            is_metric_name="is_sharpe",
            is_metric_value=2.0,
        )

    # The original lock is untouched.
    fetched = require_locked_config(path, cycle_name="cycle1")
    assert fetched.config == {"a": 1}


def test_lock_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "experiments" / "nested" / "cycle1.lock.json"

    lock_config(
        path, cycle_name="cycle1", config={}, is_metric_name="is_sharpe", is_metric_value=0.5
    )

    assert path.exists()
