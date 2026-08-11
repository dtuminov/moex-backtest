"""The standard baseline every other strategy should be measured against."""

from __future__ import annotations

from collections.abc import Sequence

from moex_backtest.engine.events import Bar, SignalEvent


class BuyAndHoldStrategy:
    """Fully long from the first bar onward, then holds — no rebalancing signal."""

    def generate_signals(self, bar: Bar, history: Sequence[Bar]) -> list[SignalEvent]:
        return [SignalEvent(bar.timestamp, bar.symbol, target_weight=1.0)]
