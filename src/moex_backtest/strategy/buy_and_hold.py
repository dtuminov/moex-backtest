"""The standard baseline every other strategy should be measured against."""

from __future__ import annotations

from collections.abc import Sequence

from moex_backtest.engine.events import Bar, SignalEvent


class BuyAndHoldStrategy:
    """Goes fully long on the first bar, then emits no further signals.

    ``history`` includes ``bar`` as its last element (see
    :class:`moex_backtest.strategy.base.Strategy`), so ``len(history) == 1``
    identifies the first bar of the run. A signal re-emitted on every bar
    still gets resized against the bar's current equity each time the
    portfolio sees it, so ordinary price drift between bars reads as a
    rebalance and generates a small phantom trade at real cost. Emitting the
    signal once avoids that.
    """

    def generate_signals(self, bar: Bar, history: Sequence[Bar]) -> list[SignalEvent]:
        if len(history) > 1:
            return []
        return [SignalEvent(bar.timestamp, bar.symbol, target_weight=1.0)]
