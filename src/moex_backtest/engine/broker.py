"""Simulated execution: turns orders into fills at a modelled cost.

Orders are filled at the *bar's open*, never at the close/price the order
was sized on — see :mod:`moex_backtest.engine.backtester` for why that
matters (it's what keeps the backtest free of lookahead bias).
"""

from __future__ import annotations

from moex_backtest.engine.events import Bar, FillEvent, OrderEvent, require_tradeable_price


class SimulatedBroker:
    """Fills orders at the bar open plus a fixed linear slippage and commission.

    Both cost models are deliberately simple: percentage commission, fixed
    basis-point slippage against the trade direction. That's enough to make
    turnover show up as a real cost in the summary metrics; it doesn't model
    a limit order book, which this engine has no data for.
    """

    def __init__(self, commission_rate: float = 0.0005, slippage_bps: float = 5.0) -> None:
        if commission_rate < 0:
            raise ValueError(f"commission_rate must be >= 0, got {commission_rate}")
        if slippage_bps < 0:
            raise ValueError(f"slippage_bps must be >= 0, got {slippage_bps}")
        self._commission_rate = commission_rate
        self._slippage_bps = slippage_bps

    def execute(self, orders: list[OrderEvent], bar: Bar) -> list[FillEvent]:
        """Fill each order at the bar's open, then apply slippage and commission in that order.

        Slippage moves the price first; commission is then charged on that
        already-slipped fill price, not on the bar's raw open — so the two
        costs compound rather than being computed independently.
        """
        fills = []
        for order in orders:
            if order.quantity == 0:
                continue
            require_tradeable_price(bar.open, context="SimulatedBroker.execute: bar.open")
            direction = 1.0 if order.quantity > 0 else -1.0
            fill_price = bar.open * (1.0 + direction * self._slippage_bps / 10_000)
            commission = abs(order.quantity) * fill_price * self._commission_rate
            fills.append(
                FillEvent(
                    timestamp=bar.timestamp,
                    symbol=order.symbol,
                    quantity=order.quantity,
                    price=fill_price,
                    commission=commission,
                )
            )
        return fills
