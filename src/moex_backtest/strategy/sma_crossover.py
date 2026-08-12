"""A simple moving-average crossover — the engine's smoke-test strategy."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean

from moex_backtest.engine.events import Bar, SignalEvent


class SmaCrossoverStrategy:
    """Long-only trend follower: fully long while the `fast`-period SMA of
    closes is above the `slow`-period SMA, flat otherwise. Flat until `slow`
    bars of history exist.

    Long-only on purpose: the broker doesn't model borrow cost, so a short
    position here would look free when it costs real money to hold. This
    strategy exists to exercise the engine end-to-end on real data — see the
    Limitations section of the top-level README before trading it as-is.
    """

    def __init__(self, fast_window: int, slow_window: int) -> None:
        if fast_window <= 0 or slow_window <= 0:
            raise ValueError("window sizes must be positive")
        if fast_window >= slow_window:
            raise ValueError(f"fast_window ({fast_window}) must be < slow_window ({slow_window})")
        self._fast_window = fast_window
        self._slow_window = slow_window

    def generate_signals(self, bar: Bar, history: Sequence[Bar]) -> list[SignalEvent]:
        if len(history) < self._slow_window:
            return [SignalEvent(bar.timestamp, bar.symbol, target_weight=0.0)]

        closes = [b.close for b in history[-self._slow_window :]]
        fast_sma = fmean(closes[-self._fast_window :])
        slow_sma = fmean(closes)
        target_weight = 1.0 if fast_sma > slow_sma else 0.0
        return [SignalEvent(bar.timestamp, bar.symbol, target_weight=target_weight)]
