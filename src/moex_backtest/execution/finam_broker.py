"""Live order execution and account/position sync against the Finam Trade API.

This is the counterpart to :class:`moex_backtest.engine.broker.SimulatedBroker`:
same idea (turn an intended trade into a fill), but against a real account,
so the failure modes are different in kind rather than in degree. Two of them
drive the whole design:

**A retried order is a doubled order.** :class:`~moex_backtest.data.finam.FinamClient`
retries on 429/5xx/transport errors, which is right for reading bars and
wrong for placing orders: a POST that reached the matching engine but whose
response was lost would be sent again. So :meth:`FinamBroker.place_order`
disables that retry loop and, when the outcome is genuinely unknown, raises
:class:`OrderSubmissionUncertainError` instead of guessing. Every order carries a
``client_order_id`` we generate, so the caller can resolve the ambiguity by
asking the broker what happened (:meth:`find_by_client_order_id`) rather than
by re-sending and hoping.

**The broker, not our bookkeeping, is the source of truth.** Positions come
from :meth:`account`, not from an internal tally of what we think we sent —
a crash, a restart, a partial fill or a manual intervention on the account
all silently break an internal tally, and the first symptom is a doubled
position.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final
from uuid import uuid4

import httpx

from moex_backtest.data.finam import FinamAPIError, FinamClient
from moex_backtest.execution.types import (
    AccountState,
    CashBalance,
    OrderState,
    OrderType,
    Position,
    Side,
    TimeInForce,
)

# The API caps `client_order_id` at 20 characters (swagger: "Уникальный
# идентификатор заявки, максимум 20 символов"). `mb` + 18 hex chars fits
# exactly and keeps enough entropy to never collide in practice.
_CLIENT_ORDER_ID_MAX: Final = 20
_CLIENT_ORDER_ID_PREFIX: Final = "mb"


class OrderSubmissionUncertainError(FinamAPIError):
    """Raised when an order may or may not have been accepted.

    The request failed in a way that cannot distinguish "never arrived" from
    "arrived, executed, and the response was lost" — a timeout, a dropped
    connection, a 5xx. **Do not re-send the order on this exception.** Look
    up :attr:`client_order_id` via
    :meth:`FinamBroker.find_by_client_order_id` and act on what the broker
    actually has.
    """

    def __init__(self, client_order_id: str, cause: str) -> None:
        super().__init__(
            f"order submission outcome unknown (client_order_id={client_order_id}): {cause}. "
            "Do not resend; reconcile with find_by_client_order_id() first."
        )
        self.client_order_id = client_order_id


class FinamBroker:
    """Order placement, cancellation and account/position sync for one account."""

    def __init__(self, client: FinamClient, account_id: str) -> None:
        if not account_id:
            raise ValueError("account_id must not be empty")
        self._client = client
        self._account_id = account_id

    @property
    def account_id(self) -> str:
        return self._account_id

    # -- account state -----------------------------------------------------------

    def account(self) -> AccountState:
        """Current equity, cash and open positions, as the broker sees them."""
        payload = self._client._authed_request("GET", f"/v1/accounts/{self._account_id}")
        return _parse_account(payload, fallback_account_id=self._account_id)

    def position_in(self, symbol: str) -> float:
        """Signed quantity currently held in `symbol` (0.0 if flat)."""
        return self.account().position_in(symbol)

    # -- orders ------------------------------------------------------------------

    def orders(self) -> list[OrderState]:
        """Every order the broker currently reports for this account."""
        payload = self._client._authed_request("GET", f"/v1/accounts/{self._account_id}/orders")
        raw = payload.get("orders")
        if raw is None:
            raise FinamAPIError(f"unexpected orders payload shape: missing 'orders' ({payload!r})")
        return [_parse_order_state(item) for item in raw]

    def get_order(self, order_id: str) -> OrderState:
        """Current state of one order."""
        payload = self._client._authed_request(
            "GET", f"/v1/accounts/{self._account_id}/orders/{order_id}"
        )
        return _parse_order_state(payload)

    def find_by_client_order_id(self, client_order_id: str) -> OrderState | None:
        """The order carrying `client_order_id`, or ``None`` if the broker has none.

        This is the recovery path after :class:`OrderSubmissionUncertainError`:
        ``None`` means the order never landed and is safe to send again;
        anything else means it did land, and re-sending would double it.
        """
        for order in self.orders():
            if order.client_order_id == client_order_id:
                return order
        return None

    def place_order(
        self,
        symbol: str,
        quantity: float,
        *,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: str | None = None,
        comment: str = "",
    ) -> OrderState:
        """Place one order and return the broker's acknowledgement.

        `quantity` is **signed** — positive buys, negative sells — matching
        :class:`moex_backtest.engine.events.OrderEvent`, so a strategy's
        sizing output can be passed through without a sign convention
        changing hands (and being flipped) on the way.

        Never retried at the transport level: see the module docstring, and
        :class:`OrderSubmissionUncertainError` for what to do when the outcome is
        unknown.
        """
        if quantity == 0:
            raise ValueError("quantity must be non-zero: a zero-quantity order is not a trade")
        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and limit_price is None:
            raise ValueError(f"limit_price is required for {order_type}")
        if order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and stop_price is None:
            raise ValueError(f"stop_price is required for {order_type}")

        order_id = client_order_id or _new_client_order_id()
        if len(order_id) > _CLIENT_ORDER_ID_MAX:
            raise ValueError(
                f"client_order_id must be at most {_CLIENT_ORDER_ID_MAX} characters, "
                f"got {len(order_id)}"
            )

        body: dict[str, Any] = {
            "account_id": self._account_id,
            "symbol": symbol,
            "quantity": {"value": _format_quantity(abs(quantity))},
            "side": Side.BUY if quantity > 0 else Side.SELL,
            "type": order_type,
            "time_in_force": time_in_force,
            "client_order_id": order_id,
        }
        if limit_price is not None:
            body["limit_price"] = {"value": _format_price(limit_price)}
        if stop_price is not None:
            body["stop_price"] = {"value": _format_price(stop_price)}
        if comment:
            body["comment"] = comment

        try:
            payload = self._client._authed_request(
                "POST", f"/v1/accounts/{self._account_id}/orders", json=body, max_retries=0
            )
        except httpx.TransportError as exc:  # never reached a response
            raise OrderSubmissionUncertainError(order_id, f"transport error: {exc}") from exc
        except FinamAPIError as exc:
            # A 4xx is a definite rejection: the gateway refused it, nothing
            # was placed. A 429/5xx may have been applied server-side before
            # the failure, so it is genuinely ambiguous.
            if _is_definite_rejection(str(exc)):
                raise
            raise OrderSubmissionUncertainError(order_id, str(exc)) from exc

        return _parse_order_state(payload)

    def cancel_order(self, order_id: str) -> OrderState:
        """Cancel one order and return its resulting state.

        Safe to retry, unlike placement: cancelling an already-cancelled or
        already-filled order changes nothing that a second attempt could
        double.
        """
        payload = self._client._authed_request(
            "DELETE", f"/v1/accounts/{self._account_id}/orders/{order_id}"
        )
        return _parse_order_state(payload)

    def cancel_all(self) -> list[OrderState]:
        """Cancel every order that can still trade. Returns the resulting states.

        The end-of-session and kill-switch primitive: leaving a working
        order behind over a weekend is how a flat book becomes an
        unsupervised position.
        """
        return [
            self.cancel_order(order.order_id) for order in self.orders() if not order.is_terminal
        ]


# -- parsing ---------------------------------------------------------------------


def _new_client_order_id() -> str:
    width = _CLIENT_ORDER_ID_MAX - len(_CLIENT_ORDER_ID_PREFIX)
    return f"{_CLIENT_ORDER_ID_PREFIX}{uuid4().hex[:width]}"


def _format_quantity(value: float) -> str:
    return f"{value:g}"


def _format_price(value: float) -> str:
    return f"{value:g}"


def _is_definite_rejection(message: str) -> bool:
    """True if the error message names a 4xx other than the retryable 429."""
    return any(f"with {code}" in message for code in range(400, 429)) or any(
        f"with {code}" in message for code in range(430, 500)
    )


def _decimal(field: Any, *, context: str) -> float:
    """Parse a ``google.type.Decimal`` (``{"value": "1.23"}``) or a bare number."""
    raw = field.get("value") if isinstance(field, dict) else field
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise FinamAPIError(f"unexpected decimal shape for {context}: {field!r}") from exc


def _optional_decimal(field: Any, *, context: str) -> float | None:
    if field is None:
        return None
    return _decimal(field, context=context)


def _money(field: Any, *, context: str) -> tuple[str, float]:
    """Parse a cash entry into ``(currency, amount)``.

    Accepts both shapes the gateway is documented to use for money: a
    ``google.type.Money`` (``units``/``nanos``/``currency_code``) and the
    simpler ``{"value": ...}`` decimal wrapper used elsewhere in the API.
    """
    if not isinstance(field, dict):
        raise FinamAPIError(f"unexpected money shape for {context}: {field!r}")
    currency = str(field.get("currency_code") or field.get("currency") or "")
    if "units" in field or "nanos" in field:
        units = float(field.get("units") or 0)
        nanos = float(field.get("nanos") or 0)
        return currency, units + nanos / 1e9
    return currency, _decimal(field, context=context)


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_order_state(payload: dict[str, Any]) -> OrderState:
    """Turn an ``OrderState`` payload into the typed record.

    The submitted order's own fields live in a nested ``order`` object
    (proto ``OrderState.order``), while ids, status and executed quantity
    live at the top level.
    """
    order = payload.get("order") or {}
    if not isinstance(order, dict):
        raise FinamAPIError(f"unexpected order payload shape: 'order' is not an object ({order!r})")

    quantity = _optional_decimal(order.get("quantity"), context="order.quantity")
    if quantity is None:
        quantity = _optional_decimal(payload.get("initial_quantity"), context="initial_quantity")

    return OrderState(
        order_id=str(payload.get("order_id", "")),
        client_order_id=str(order.get("client_order_id", "") or ""),
        status=str(payload.get("status", "")),
        symbol=str(order.get("symbol", "") or ""),
        side=str(order.get("side", "") or ""),
        quantity=quantity if quantity is not None else 0.0,
        executed_quantity=_decimal(
            payload.get("executed_quantity", 0), context="executed_quantity"
        ),
        remaining_quantity=_decimal(
            payload.get("remaining_quantity", 0), context="remaining_quantity"
        ),
        limit_price=_optional_decimal(order.get("limit_price"), context="order.limit_price"),
        transact_at=_parse_timestamp(payload.get("transact_at")),
    )


def _parse_position(payload: dict[str, Any]) -> Position:
    return Position(
        symbol=str(payload.get("symbol", "")),
        quantity=_decimal(payload.get("quantity", 0), context="position.quantity"),
        average_price=_decimal(payload.get("average_price", 0), context="position.average_price"),
        current_price=_decimal(payload.get("current_price", 0), context="position.current_price"),
        unrealized_pnl=_decimal(
            payload.get("unrealized_pnl", 0), context="position.unrealized_pnl"
        ),
        daily_pnl=_decimal(payload.get("daily_pnl", 0), context="position.daily_pnl"),
    )


def _parse_account(payload: dict[str, Any], *, fallback_account_id: str) -> AccountState:
    raw_positions = payload.get("positions") or []
    raw_cash = payload.get("cash") or []
    cash: list[CashBalance] = []
    for entry in raw_cash:
        currency, amount = _money(entry, context="account.cash")
        cash.append(CashBalance(currency=currency, amount=amount))
    return AccountState(
        account_id=str(payload.get("account_id") or fallback_account_id),
        equity=_decimal(payload.get("equity", 0), context="account.equity"),
        unrealized_profit=_decimal(
            payload.get("unrealized_profit", 0), context="account.unrealized_profit"
        ),
        cash=tuple(cash),
        positions=tuple(_parse_position(item) for item in raw_positions),
    )
