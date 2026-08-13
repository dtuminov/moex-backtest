"""The strategy interface: pure signal generation, no portfolio/execution concerns."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from moex_backtest.engine.events import Bar, SignalEvent


class Strategy(Protocol):
    """A strategy only ever sees bars up to and including the current one.

    ``history`` includes ``bar`` as its last element, e.g. ``history[-20:]``
    for a moving average.
    """

    def generate_signals(self, bar: Bar, history: Sequence[Bar]) -> list[SignalEvent]: ...
