"""Tests for SmaCrossoverStrategy."""

from __future__ import annotations

import pandas as pd
import pytest

from moex_backtest.engine.events import Bar
from moex_backtest.strategy.sma_crossover import SmaCrossoverStrategy


def _bars(closes: list[float]) -> list[Bar]:
    return [
        Bar(pd.Timestamp("2026-01-01") + pd.Timedelta(days=i), "SBER", c, c, c, c, 1000.0)
        for i, c in enumerate(closes)
    ]


def test_stays_flat_before_enough_history_accumulates() -> None:
    strategy = SmaCrossoverStrategy(fast_window=3, slow_window=10)
    bars = _bars(list(range(1, 6)))  # only 5 bars, need 10

    signal = strategy.generate_signals(bars[-1], bars)[0]

    assert signal.target_weight == 0.0


def test_goes_long_when_fast_sma_is_above_slow_sma() -> None:
    strategy = SmaCrossoverStrategy(fast_window=3, slow_window=10)
    bars = _bars(list(range(1, 11)))  # uptrend: recent closes pull the fast SMA up

    signal = strategy.generate_signals(bars[-1], bars)[0]

    assert signal.target_weight == 1.0


def test_stays_flat_when_fast_sma_is_below_slow_sma() -> None:
    strategy = SmaCrossoverStrategy(fast_window=3, slow_window=10)
    bars = _bars(list(range(10, 0, -1)))  # downtrend: recent closes pull the fast SMA down

    signal = strategy.generate_signals(bars[-1], bars)[0]

    assert signal.target_weight == 0.0


def test_computes_as_soon_as_history_exactly_fills_the_slow_window() -> None:
    strategy = SmaCrossoverStrategy(fast_window=3, slow_window=10)
    bars = _bars(list(range(1, 11)))  # exactly 10 bars

    signal = strategy.generate_signals(bars[-1], bars)[0]

    assert signal.target_weight == 1.0


@pytest.mark.parametrize(
    "fast_window,slow_window",
    [(0, 10), (3, 0), (10, 10), (11, 10)],
)
def test_rejects_invalid_windows(fast_window: int, slow_window: int) -> None:
    with pytest.raises(ValueError):
        SmaCrossoverStrategy(fast_window=fast_window, slow_window=slow_window)
