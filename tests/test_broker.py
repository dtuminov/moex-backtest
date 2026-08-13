"""Tests for SimulatedBroker."""

from __future__ import annotations

import pandas as pd
import pytest

from moex_backtest.engine.broker import SimulatedBroker
from moex_backtest.engine.events import Bar, OrderEvent

_BAR = Bar(
    timestamp=pd.Timestamp("2026-01-05"),
    symbol="SBER",
    open=100.0,
    high=101.0,
    low=99.0,
    close=100.5,
    volume=1000.0,
)


def test_buy_order_fills_above_open_with_slippage() -> None:
    broker = SimulatedBroker(commission_rate=0.0, slippage_bps=10.0)  # 10bps = 0.1%

    fills = broker.execute([OrderEvent(_BAR.timestamp, "SBER", quantity=10.0)], _BAR)

    assert len(fills) == 1
    assert fills[0].price == pytest.approx(100.0 * 1.001)
    assert fills[0].commission == 0.0


def test_sell_order_fills_below_open_with_slippage() -> None:
    broker = SimulatedBroker(commission_rate=0.0, slippage_bps=10.0)

    fills = broker.execute([OrderEvent(_BAR.timestamp, "SBER", quantity=-10.0)], _BAR)

    assert fills[0].price == pytest.approx(100.0 * 0.999)


def test_commission_is_proportional_to_notional() -> None:
    broker = SimulatedBroker(commission_rate=0.001, slippage_bps=0.0)

    fills = broker.execute([OrderEvent(_BAR.timestamp, "SBER", quantity=10.0)], _BAR)

    assert fills[0].commission == pytest.approx(10.0 * 100.0 * 0.001)


def test_zero_quantity_orders_produce_no_fill() -> None:
    broker = SimulatedBroker()

    fills = broker.execute([OrderEvent(_BAR.timestamp, "SBER", quantity=0.0)], _BAR)

    assert fills == []


@pytest.mark.parametrize("bad_kwargs", [{"commission_rate": -0.01}, {"slippage_bps": -1.0}])
def test_rejects_negative_cost_parameters(bad_kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SimulatedBroker(**bad_kwargs)


@pytest.mark.parametrize("bad_open", [0.0, -1.0, float("nan")])
def test_execute_rejects_non_tradeable_open_price_for_a_nonzero_order(bad_open: float) -> None:
    broker = SimulatedBroker()
    bad_bar = Bar(_BAR.timestamp, "SBER", bad_open, 101.0, 99.0, 100.5, 1000.0)

    with pytest.raises(ValueError, match="price must be a finite number > 0"):
        broker.execute([OrderEvent(bad_bar.timestamp, "SBER", quantity=10.0)], bad_bar)


@pytest.mark.parametrize("bad_open", [0.0, -1.0, float("nan")])
def test_execute_does_not_validate_price_when_there_is_nothing_to_fill(bad_open: float) -> None:
    # A degenerate price on a bar we're not trading (all orders zero-quantity)
    # must not raise: there is no division/multiplication happening.
    broker = SimulatedBroker()
    bad_bar = Bar(_BAR.timestamp, "SBER", bad_open, 101.0, 99.0, 100.5, 1000.0)

    fills = broker.execute([OrderEvent(bad_bar.timestamp, "SBER", quantity=0.0)], bad_bar)

    assert fills == []
