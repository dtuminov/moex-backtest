"""Single-symbol portfolio: cash/position bookkeeping and target-weight sizing.

v1 scope is deliberately single-symbol. Cross-asset risk limits (gross/net
exposure across a book, correlation-aware sizing) are a separate design
problem — see the Roadmap in the top-level README.
"""

from __future__ import annotations

import pandas as pd

from moex_backtest.engine.events import Bar, FillEvent, OrderEvent, SignalEvent

_MIN_ORDER_QUANTITY = 1e-9  # avoid emitting economically meaningless dust orders


class Portfolio:
    """Tracks cash and one symbol's position; turns signals into sized orders."""

    def __init__(self, initial_cash: float, max_gross_exposure: float = 1.0) -> None:
        if initial_cash <= 0:
            raise ValueError(f"initial_cash must be > 0, got {initial_cash}")
        if max_gross_exposure <= 0:
            raise ValueError(f"max_gross_exposure must be > 0, got {max_gross_exposure}")
        self.cash = initial_cash
        self.position = 0.0
        self._max_gross_exposure = max_gross_exposure
        self._equity_timestamps: list[pd.Timestamp] = []
        self._equity_values: list[float] = []
        self.fills: list[FillEvent] = []

    def equity(self, price: float) -> float:
        return self.cash + self.position * price

    def orders_from_signals(self, signals: list[SignalEvent], bar: Bar) -> list[OrderEvent]:
        """Size the latest signal for ``bar.symbol`` against current equity at ``bar.close``.

        If several signals for the same bar are given, only the last one
        applies (a strategy emitting more than one signal per symbol per bar
        is expressing indecision, not two trades).
        """
        relevant = [s for s in signals if s.symbol == bar.symbol]
        if not relevant:
            return []
        target_weight = relevant[-1].target_weight
        clipped_weight = max(
            -self._max_gross_exposure, min(self._max_gross_exposure, target_weight)
        )

        current_equity = self.equity(bar.close)
        target_quantity = clipped_weight * current_equity / bar.close
        delta = target_quantity - self.position
        if abs(delta) < _MIN_ORDER_QUANTITY:
            return []
        return [OrderEvent(timestamp=bar.timestamp, symbol=bar.symbol, quantity=delta)]

    def apply_fill(self, fill: FillEvent) -> None:
        self.cash -= fill.quantity * fill.price + fill.commission
        self.position += fill.quantity
        self.fills.append(fill)

    def mark_to_market(self, bar: Bar) -> None:
        self._equity_timestamps.append(bar.timestamp)
        self._equity_values.append(self.equity(bar.close))

    def equity_curve(self) -> pd.Series:
        return pd.Series(
            self._equity_values,
            index=pd.Index(self._equity_timestamps, name="timestamp"),
            name="equity",
        )
