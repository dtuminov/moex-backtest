"""Tests for Portfolio: sizing, fills, exposure clipping, mark-to-market."""

from __future__ import annotations

import pandas as pd
import pytest

from moex_backtest.engine.events import Bar, FillEvent, SignalEvent
from moex_backtest.engine.portfolio import Portfolio

_BAR = Bar(
    timestamp=pd.Timestamp("2026-01-05"),
    symbol="SBER",
    open=100.0,
    high=101.0,
    low=99.0,
    close=100.0,
    volume=1000.0,
)


def test_full_long_signal_sizes_order_to_all_equity() -> None:
    portfolio = Portfolio(initial_cash=10_000.0)

    orders = portfolio.orders_from_signals(
        [SignalEvent(_BAR.timestamp, "SBER", target_weight=1.0)], _BAR
    )

    assert len(orders) == 1
    assert orders[0].quantity == pytest.approx(100.0)  # 10_000 / 100.0


def test_flat_signal_on_an_existing_position_orders_a_full_exit() -> None:
    portfolio = Portfolio(initial_cash=10_000.0)
    portfolio.apply_fill(
        FillEvent(_BAR.timestamp, "SBER", quantity=50.0, price=100.0, commission=0.0)
    )

    orders = portfolio.orders_from_signals(
        [SignalEvent(_BAR.timestamp, "SBER", target_weight=0.0)], _BAR
    )

    assert orders[0].quantity == pytest.approx(-50.0)


def test_signal_weight_beyond_max_exposure_is_clipped() -> None:
    portfolio = Portfolio(initial_cash=10_000.0, max_gross_exposure=0.5)

    orders = portfolio.orders_from_signals(
        [SignalEvent(_BAR.timestamp, "SBER", target_weight=1.0)], _BAR
    )

    assert orders[0].quantity == pytest.approx(50.0)  # 0.5 * 10_000 / 100.0


def test_no_signal_for_symbol_produces_no_order() -> None:
    portfolio = Portfolio(initial_cash=10_000.0)

    orders = portfolio.orders_from_signals(
        [SignalEvent(_BAR.timestamp, "OTHER", target_weight=1.0)], _BAR
    )

    assert orders == []


def test_already_at_target_produces_no_order() -> None:
    portfolio = Portfolio(initial_cash=10_000.0)
    portfolio.apply_fill(
        FillEvent(_BAR.timestamp, "SBER", quantity=100.0, price=100.0, commission=0.0)
    )

    orders = portfolio.orders_from_signals(
        [SignalEvent(_BAR.timestamp, "SBER", target_weight=1.0)], _BAR
    )

    assert orders == []


def test_apply_fill_updates_cash_and_position_including_commission() -> None:
    portfolio = Portfolio(initial_cash=10_000.0)

    portfolio.apply_fill(
        FillEvent(_BAR.timestamp, "SBER", quantity=10.0, price=100.0, commission=5.0)
    )

    assert portfolio.position == pytest.approx(10.0)
    assert portfolio.cash == pytest.approx(10_000.0 - 1_000.0 - 5.0)


def test_mark_to_market_records_equity_curve_points() -> None:
    portfolio = Portfolio(initial_cash=10_000.0)
    portfolio.apply_fill(
        FillEvent(_BAR.timestamp, "SBER", quantity=10.0, price=100.0, commission=0.0)
    )

    portfolio.mark_to_market(_BAR)
    next_bar = Bar(pd.Timestamp("2026-01-06"), "SBER", 101, 102, 100, 105.0, 1000.0)
    portfolio.mark_to_market(next_bar)

    curve = portfolio.equity_curve()
    assert list(curve.values) == pytest.approx([10_000.0, 9_000.0 + 10 * 105.0])


@pytest.mark.parametrize("bad_kwargs", [{"initial_cash": 0.0}, {"initial_cash": -1.0}])
def test_rejects_non_positive_initial_cash(bad_kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        Portfolio(**bad_kwargs)
