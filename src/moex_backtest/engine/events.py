"""Typed events that flow through the backtest loop: Bar -> Signal -> Order -> Fill.

Each event is an immutable, timestamped record. Distinct types (versus raw
dicts/tuples) make the strategy/portfolio/broker boundary in
:mod:`moex_backtest.engine.backtester` enforceable and testable in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV observation for a single symbol."""

    timestamp: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """A strategy's desired position, expressed as a target fraction of portfolio equity.

    ``target_weight = 1.0`` means "fully long this symbol", ``-1.0`` means
    "fully short", ``0.0`` means "flat". The portfolio is responsible for
    turning this into an actual order quantity and for respecting exposure
    limits — the strategy never sees cash or share counts.
    """

    timestamp: pd.Timestamp
    symbol: str
    target_weight: float


@dataclass(frozen=True, slots=True)
class OrderEvent:
    """An instruction to trade a signed quantity of a symbol."""

    timestamp: pd.Timestamp
    symbol: str
    quantity: float  # positive = buy, negative = sell


@dataclass(frozen=True, slots=True)
class FillEvent:
    """A broker's report of an executed order, including simulated cost."""

    timestamp: pd.Timestamp
    symbol: str
    quantity: float  # positive = bought, negative = sold
    price: float
    commission: float


def require_tradeable_price(price: float, *, context: str) -> None:
    """Raise ``ValueError`` if ``price`` is not safe to divide or multiply by.

    A zero, negative, or NaN price is a legitimate shape for bad/degenerate
    market data to take — e.g. ``moex_backtest.data.moex_iss`` warns that most
    FORTS option-days carry no real trades — and letting one flow into a
    sizing division or a fill-price multiplication produces a silent
    ``ZeroDivisionError``, an infinite order size, or a NaN that poisons the
    rest of the equity curve. Fail loudly at the point of use instead.
    """
    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"{context}: price must be a finite number > 0, got {price!r}")
