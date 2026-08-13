"""Single-symbol portfolio: cash/position bookkeeping and target-weight sizing.

v1 scope is deliberately single-symbol. Cross-asset risk limits (gross/net
exposure across a book, correlation-aware sizing) are a separate design
problem — see the Roadmap in the top-level README.
"""

from __future__ import annotations

import pandas as pd

from moex_backtest.engine.events import (
    Bar,
    FillEvent,
    OrderEvent,
    SignalEvent,
    require_tradeable_price,
)

_MIN_ORDER_QUANTITY = 1e-9  # avoid emitting economically meaningless dust orders


class Portfolio:
    """Tracks cash and one symbol's position; turns signals into sized orders.

    Equity floor / forced liquidation: this engine has no margin-call
    simulation, but shorts and unbounded gross exposure are officially
    supported, so equity *can* legitimately reach zero or go negative (e.g.
    a short position against a large adverse move). The first time
    :meth:`mark_to_market` observes equity ``<= 0``, the position is
    force-closed (flattened at that bar's price) and the portfolio is
    permanently halted: :meth:`orders_from_signals` returns no further
    orders from that point on. This is a deliberately simple circuit
    breaker ("if equity <= 0, stop trading and hold flat"), not a realistic
    liquidation model.
    """

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
        self.halted = False

    def equity(self, price: float) -> float:
        return self.cash + self.position * price

    def orders_from_signals(self, signals: list[SignalEvent], bar: Bar) -> list[OrderEvent]:
        """Size the latest signal for ``bar.symbol`` against current equity at ``bar.close``.

        If several signals for the same bar are given, only the last one
        applies — it's treated as the strategy's settled decision for that
        bar. Returns no orders once the portfolio has been :attr:`halted` by
        the equity floor.
        """
        if self.halted:
            return []
        relevant = [s for s in signals if s.symbol == bar.symbol]
        if not relevant:
            return []
        require_tradeable_price(bar.close, context="Portfolio.orders_from_signals: bar.close")
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
        equity = self.equity(bar.close)
        if not self.halted and equity <= 0.0:
            # Equity wiped out (or worse): force-close the position so it can
            # no longer swing with price, and stop trading. See the equity
            # floor note in the class docstring.
            self.position = 0.0
            self.cash = equity
            self.halted = True
        self._equity_timestamps.append(bar.timestamp)
        self._equity_values.append(equity)

    def equity_curve(self) -> pd.Series:
        return pd.Series(
            self._equity_values,
            index=pd.Index(self._equity_timestamps, name="timestamp"),
            name="equity",
        )
