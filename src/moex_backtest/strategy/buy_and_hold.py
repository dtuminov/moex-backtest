"""The standard baseline every other strategy should be measured against."""

from __future__ import annotations

from collections.abc import Sequence

from moex_backtest.engine.events import Bar, SignalEvent


class BuyAndHoldStrategy:
    """Goes fully long on the first bar, then emits no further signals.

    ``len(history) == 1`` identifies the first bar of the run. Emitting the
    signal only once matters: a signal re-emitted every bar would get
    resized against current equity each time, and ordinary price drift
    would read as a rebalance.
    """

    def generate_signals(self, bar: Bar, history: Sequence[Bar]) -> list[SignalEvent]:
        if len(history) > 1:
            return []
        return [SignalEvent(bar.timestamp, bar.symbol, target_weight=1.0)]
