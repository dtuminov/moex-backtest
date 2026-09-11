"""Typed order/account records for live execution against the Finam Trade API.

Enum values are the wire values the REST gateway actually sends and accepts
(``proto/grpc/tradeapi/v1/orders/orders_service.proto`` in
https://github.com/FinamWeb/finam-trade-api), not our own names for them, so
a value can be sent or compared without a translation table in between.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final


class Side(StrEnum):
    BUY = "SIDE_BUY"
    SELL = "SIDE_SELL"


class OrderType(StrEnum):
    MARKET = "ORDER_TYPE_MARKET"
    LIMIT = "ORDER_TYPE_LIMIT"
    STOP = "ORDER_TYPE_STOP"
    STOP_LIMIT = "ORDER_TYPE_STOP_LIMIT"


class TimeInForce(StrEnum):
    DAY = "TIME_IN_FORCE_DAY"
    GOOD_TILL_CANCEL = "TIME_IN_FORCE_GOOD_TILL_CANCEL"


class OrderStatus(StrEnum):
    """Every status the gateway can report, verbatim from ``OrderStatus``.

    Kept complete rather than collapsed into "open/closed": the broker-side
    rejection states (:attr:`DENIED_BY_BROKER`, :attr:`REJECTED_BY_EXCHANGE`,
    :attr:`FAILED`) are operationally different from a clean
    :attr:`CANCELED`, and a live trading loop that lumps them together loses
    exactly the signal it needs to stop trading.
    """

    UNSPECIFIED = "ORDER_STATUS_UNSPECIFIED"
    NEW = "ORDER_STATUS_NEW"
    PARTIALLY_FILLED = "ORDER_STATUS_PARTIALLY_FILLED"
    FILLED = "ORDER_STATUS_FILLED"
    DONE_FOR_DAY = "ORDER_STATUS_DONE_FOR_DAY"
    CANCELED = "ORDER_STATUS_CANCELED"
    REPLACED = "ORDER_STATUS_REPLACED"
    PENDING_CANCEL = "ORDER_STATUS_PENDING_CANCEL"
    REJECTED = "ORDER_STATUS_REJECTED"
    SUSPENDED = "ORDER_STATUS_SUSPENDED"
    PENDING_NEW = "ORDER_STATUS_PENDING_NEW"
    EXPIRED = "ORDER_STATUS_EXPIRED"
    FAILED = "ORDER_STATUS_FAILED"
    FORWARDING = "ORDER_STATUS_FORWARDING"
    WAIT = "ORDER_STATUS_WAIT"
    DENIED_BY_BROKER = "ORDER_STATUS_DENIED_BY_BROKER"
    REJECTED_BY_EXCHANGE = "ORDER_STATUS_REJECTED_BY_EXCHANGE"
    WATCHING = "ORDER_STATUS_WATCHING"
    EXECUTED = "ORDER_STATUS_EXECUTED"
    DISABLED = "ORDER_STATUS_DISABLED"
    LINK_WAIT = "ORDER_STATUS_LINK_WAIT"
    SL_GUARD_TIME = "ORDER_STATUS_SL_GUARD_TIME"
    SL_EXECUTED = "ORDER_STATUS_SL_EXECUTED"
    SL_FORWARDING = "ORDER_STATUS_SL_FORWARDING"
    TP_GUARD_TIME = "ORDER_STATUS_TP_GUARD_TIME"
    TP_EXECUTED = "ORDER_STATUS_TP_EXECUTED"
    TP_CORRECTION = "ORDER_STATUS_TP_CORRECTION"
    TP_FORWARDING = "ORDER_STATUS_TP_FORWARDING"
    TP_CORR_GUARD_TIME = "ORDER_STATUS_TP_CORR_GUARD_TIME"


#: Statuses after which the order will never trade again. Anything NOT in
#: here may still fill, so a caller that stops polling on one of those is
#: risking an unnoticed position.
TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.EXECUTED,
        OrderStatus.SL_EXECUTED,
        OrderStatus.TP_EXECUTED,
        OrderStatus.CANCELED,
        OrderStatus.REJECTED,
        OrderStatus.REJECTED_BY_EXCHANGE,
        OrderStatus.DENIED_BY_BROKER,
        OrderStatus.EXPIRED,
        OrderStatus.FAILED,
        OrderStatus.DONE_FOR_DAY,
        OrderStatus.REPLACED,
        OrderStatus.DISABLED,
    }
)

#: Terminal statuses that mean the order did NOT do what was asked. Worth
#: separating from a fill, because a strategy that silently treats a
#: rejection as "position established" trades a book it does not own.
REJECTION_STATUSES: Final[frozenset[str]] = frozenset(
    {
        OrderStatus.REJECTED,
        OrderStatus.REJECTED_BY_EXCHANGE,
        OrderStatus.DENIED_BY_BROKER,
        OrderStatus.FAILED,
        OrderStatus.EXPIRED,
        OrderStatus.DISABLED,
    }
)


def is_terminal(status: str) -> bool:
    """True if the order is finished and will not trade further."""
    return status in TERMINAL_STATUSES


def is_rejection(status: str) -> bool:
    """True if the order ended without doing what it was asked to do."""
    return status in REJECTION_STATUSES


@dataclass(frozen=True, slots=True)
class Position:
    """An open position on the trading account, as the broker sees it."""

    symbol: str
    quantity: float  # signed: positive = long, negative = short
    average_price: float
    current_price: float
    unrealized_pnl: float
    daily_pnl: float


@dataclass(frozen=True, slots=True)
class CashBalance:
    """One currency's free cash on the account."""

    currency: str
    amount: float


@dataclass(frozen=True, slots=True)
class AccountState:
    """Account equity, cash and open positions at a point in time."""

    account_id: str
    equity: float
    unrealized_profit: float
    cash: tuple[CashBalance, ...]
    positions: tuple[Position, ...]

    def position_in(self, symbol: str) -> float:
        """Signed quantity held in `symbol`, or 0.0 if flat.

        Reconciling against this — rather than against an internally
        tracked position — is what keeps a restarted or crashed strategy
        from doubling a position it already holds.
        """
        for position in self.positions:
            if position.symbol == symbol:
                return position.quantity
        return 0.0


@dataclass(frozen=True, slots=True)
class OrderState:
    """The broker's view of one order.

    ``executed_quantity`` is the number that matters for reconciliation: an
    order can sit in :attr:`OrderStatus.PARTIALLY_FILLED` indefinitely, and
    the position it has already established is real regardless of what
    happens to the remainder.
    """

    order_id: str
    client_order_id: str
    status: str
    symbol: str
    side: str
    quantity: float
    executed_quantity: float
    remaining_quantity: float
    limit_price: float | None
    transact_at: datetime | None

    @property
    def is_terminal(self) -> bool:
        return is_terminal(self.status)

    @property
    def is_rejection(self) -> bool:
        return is_rejection(self.status)
