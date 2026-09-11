"""Tests for FinamBroker. All HTTP is mocked via respx — no network access.

The cases that matter most here are not the happy paths but the ambiguous
ones: an order whose fate is unknown must never be silently re-sent, and a
definite rejection must never be mistaken for one.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
import respx

from moex_backtest.data.finam import FinamAPIError, FinamClient
from moex_backtest.execution import (
    FinamBroker,
    OrderStatus,
    OrderSubmissionUncertainError,
    OrderType,
    Side,
    TimeInForce,
)

_FAKE_TOKEN = "fake.jwt.token"
_ACCOUNT = "TEST-ACC"
_BASE = "https://api.finam.ru"
_ACCOUNT_URL = f"{_BASE}/v1/accounts/{_ACCOUNT}"
_ORDERS_URL = f"{_ACCOUNT_URL}/orders"


def _auth_route() -> None:
    respx.post(f"{_BASE}/v1/sessions").mock(
        return_value=httpx.Response(200, json={"token": _FAKE_TOKEN})
    )
    expires_at = (datetime.now(UTC) + timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
    respx.post(f"{_BASE}/v1/sessions/details").mock(
        return_value=httpx.Response(200, json={"expires_at": expires_at, "readonly": False})
    )


def _broker() -> FinamBroker:
    client = FinamClient(secret="fake-secret", request_delay=0.0, retry_backoff=0.0)
    return FinamBroker(client, _ACCOUNT)


def _order_state(
    *,
    order_id: str = "ORD-1",
    client_order_id: str = "mb0123456789abcdef01",
    status: str = OrderStatus.NEW,
    symbol: str = "SBER@MISX",
    side: str = Side.BUY,
    quantity: str = "10",
    executed: str = "0",
    remaining: str = "10",
) -> dict[str, Any]:
    return {
        "order_id": order_id,
        "exec_id": "EX-1",
        "status": status,
        "order": {
            "account_id": _ACCOUNT,
            "symbol": symbol,
            "quantity": {"value": quantity},
            "side": side,
            "type": OrderType.MARKET,
            "time_in_force": TimeInForce.DAY,
            "client_order_id": client_order_id,
        },
        "transact_at": "2026-09-10T09:30:00Z",
        "initial_quantity": {"value": quantity},
        "executed_quantity": {"value": executed},
        "remaining_quantity": {"value": remaining},
    }


# -- account ---------------------------------------------------------------------


@respx.mock
def test_account_parses_equity_positions_and_cash() -> None:
    _auth_route()
    respx.get(_ACCOUNT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "account_id": _ACCOUNT,
                "equity": {"value": "3000000.5"},
                "unrealized_profit": {"value": "-1200.25"},
                "cash": [{"currency_code": "RUB", "units": "2500000", "nanos": 500000000}],
                "positions": [
                    {
                        "symbol": "SBER@MISX",
                        "quantity": {"value": "100"},
                        "average_price": {"value": "250.5"},
                        "current_price": {"value": "260.0"},
                        "unrealized_pnl": {"value": "950.0"},
                        "daily_pnl": {"value": "120.0"},
                    }
                ],
            },
        )
    )
    account = _broker().account()

    assert account.equity == pytest.approx(3_000_000.5)
    assert account.unrealized_profit == pytest.approx(-1200.25)
    assert account.cash[0].currency == "RUB"
    assert account.cash[0].amount == pytest.approx(2_500_000.5)
    assert account.position_in("SBER@MISX") == pytest.approx(100.0)
    assert account.position_in("GAZP@MISX") == 0.0


@respx.mock
def test_account_accepts_decimal_shaped_cash() -> None:
    """Cash may arrive as the plain decimal wrapper rather than google.type.Money."""
    _auth_route()
    respx.get(_ACCOUNT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "account_id": _ACCOUNT,
                "equity": {"value": "100"},
                "unrealized_profit": {"value": "0"},
                "cash": [{"currency_code": "RUB", "value": "42.5"}],
                "positions": [],
            },
        )
    )
    account = _broker().account()
    assert account.cash[0].amount == pytest.approx(42.5)


# -- placement -------------------------------------------------------------------


@respx.mock
def test_place_order_derives_side_and_absolute_quantity_from_sign() -> None:
    _auth_route()
    route = respx.post(_ORDERS_URL).mock(
        return_value=httpx.Response(200, json=_order_state(side=Side.SELL, quantity="7"))
    )
    state = _broker().place_order("SBER@MISX", -7)

    body = json.loads(route.calls[0].request.read())
    assert body["side"] == Side.SELL
    assert body["quantity"] == {"value": "7"}  # absolute value, not -7
    assert body["type"] == OrderType.MARKET
    assert body["time_in_force"] == TimeInForce.DAY
    assert state.order_id == "ORD-1"
    assert state.status == OrderStatus.NEW
    assert not state.is_terminal


@respx.mock
def test_place_order_sends_generated_client_order_id_within_length_limit() -> None:
    _auth_route()
    route = respx.post(_ORDERS_URL).mock(return_value=httpx.Response(200, json=_order_state()))
    _broker().place_order("SBER@MISX", 1)

    client_order_id = json.loads(route.calls[0].request.read())["client_order_id"]
    assert 0 < len(client_order_id) <= 20


def test_place_order_rejects_zero_quantity() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        _broker().place_order("SBER@MISX", 0)


def test_place_order_requires_limit_price_for_limit_order() -> None:
    with pytest.raises(ValueError, match="limit_price"):
        _broker().place_order("SBER@MISX", 1, order_type=OrderType.LIMIT)


def test_place_order_rejects_overlong_client_order_id() -> None:
    with pytest.raises(ValueError, match="at most 20"):
        _broker().place_order("SBER@MISX", 1, client_order_id="x" * 21)


# -- the ambiguous cases ---------------------------------------------------------


@respx.mock
def test_place_order_does_not_retry_on_server_error() -> None:
    """A 5xx POST must be sent exactly once: a retry could double the position."""
    _auth_route()
    route = respx.post(_ORDERS_URL).mock(return_value=httpx.Response(503))

    with pytest.raises(OrderSubmissionUncertainError) as exc_info:
        _broker().place_order("SBER@MISX", 5)

    assert route.call_count == 1
    assert exc_info.value.client_order_id.startswith("mb")
    assert "Do not resend" in str(exc_info.value)


@respx.mock
def test_place_order_treats_transport_failure_as_uncertain() -> None:
    _auth_route()
    respx.post(_ORDERS_URL).mock(side_effect=httpx.ConnectTimeout("timed out"))

    with pytest.raises(OrderSubmissionUncertainError):
        _broker().place_order("SBER@MISX", 5)


@respx.mock
def test_place_order_reports_a_4xx_as_a_definite_rejection() -> None:
    """A 400 never reached the matching engine — it is not ambiguous."""
    _auth_route()
    respx.post(_ORDERS_URL).mock(return_value=httpx.Response(400))

    with pytest.raises(FinamAPIError) as exc_info:
        _broker().place_order("SBER@MISX", 5)

    assert not isinstance(exc_info.value, OrderSubmissionUncertainError)


@respx.mock
def test_find_by_client_order_id_resolves_an_uncertain_submission() -> None:
    _auth_route()
    respx.get(_ORDERS_URL).mock(
        return_value=httpx.Response(
            200, json={"orders": [_order_state(client_order_id="mbAAAA", order_id="ORD-9")]}
        )
    )
    broker = _broker()

    assert broker.find_by_client_order_id("mbAAAA").order_id == "ORD-9"  # type: ignore[union-attr]
    assert broker.find_by_client_order_id("mbNOPE") is None


# -- lifecycle -------------------------------------------------------------------


@respx.mock
def test_partially_filled_order_is_not_terminal_and_reports_executed_quantity() -> None:
    _auth_route()
    respx.get(f"{_ORDERS_URL}/ORD-1").mock(
        return_value=httpx.Response(
            200,
            json=_order_state(status=OrderStatus.PARTIALLY_FILLED, executed="4", remaining="6"),
        )
    )
    state = _broker().get_order("ORD-1")

    assert state.executed_quantity == pytest.approx(4.0)
    assert state.remaining_quantity == pytest.approx(6.0)
    assert not state.is_terminal


@respx.mock
def test_rejected_order_is_terminal_and_flagged_as_a_rejection() -> None:
    _auth_route()
    respx.get(f"{_ORDERS_URL}/ORD-1").mock(
        return_value=httpx.Response(
            200, json=_order_state(status=OrderStatus.REJECTED_BY_EXCHANGE)
        )
    )
    state = _broker().get_order("ORD-1")

    assert state.is_terminal
    assert state.is_rejection


@respx.mock
def test_cancel_all_only_touches_orders_that_can_still_trade() -> None:
    _auth_route()
    respx.get(_ORDERS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "orders": [
                    _order_state(order_id="LIVE-1", status=OrderStatus.NEW),
                    _order_state(order_id="DONE-1", status=OrderStatus.FILLED),
                    _order_state(order_id="LIVE-2", status=OrderStatus.PARTIALLY_FILLED),
                ]
            },
        )
    )
    cancelled = respx.delete(url__regex=rf"{_ORDERS_URL}/.*").mock(
        return_value=httpx.Response(200, json=_order_state(status=OrderStatus.CANCELED))
    )
    states = _broker().cancel_all()

    assert cancelled.call_count == 2
    cancelled_ids = {str(call.request.url).rsplit("/", 1)[-1] for call in cancelled.calls}
    assert cancelled_ids == {"LIVE-1", "LIVE-2"}
    assert all(state.status == OrderStatus.CANCELED for state in states)
