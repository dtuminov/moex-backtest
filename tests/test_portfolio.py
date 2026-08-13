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


def test_equity_floor_force_closes_position_and_halts_when_equity_hits_zero_or_below() -> None:
    # Short 100 shares at 100 (cash goes to 20_000), then the price gaps 10x
    # against the short: equity = 20_000 - 100 * 1_000 = -80_000.
    portfolio = Portfolio(initial_cash=10_000.0)
    portfolio.apply_fill(
        FillEvent(_BAR.timestamp, "SBER", quantity=-100.0, price=100.0, commission=0.0)
    )
    assert portfolio.cash == pytest.approx(20_000.0)

    adverse_bar = Bar(pd.Timestamp("2026-01-06"), "SBER", 1000.0, 1010.0, 990.0, 1000.0, 1000.0)
    portfolio.mark_to_market(adverse_bar)

    assert portfolio.halted is True
    assert portfolio.position == 0.0
    assert portfolio.cash == pytest.approx(-80_000.0)  # floored at (not below) the observed equity
    assert list(portfolio.equity_curve().values) == pytest.approx([-80_000.0])


def test_halted_portfolio_emits_no_further_orders_regardless_of_signal() -> None:
    portfolio = Portfolio(initial_cash=10_000.0)
    portfolio.apply_fill(
        FillEvent(_BAR.timestamp, "SBER", quantity=-100.0, price=100.0, commission=0.0)
    )
    adverse_bar = Bar(pd.Timestamp("2026-01-06"), "SBER", 1000.0, 1010.0, 990.0, 1000.0, 1000.0)
    portfolio.mark_to_market(adverse_bar)
    assert portfolio.halted is True

    orders = portfolio.orders_from_signals(
        [SignalEvent(adverse_bar.timestamp, "SBER", target_weight=1.0)], adverse_bar
    )

    assert orders == []


def test_equity_does_not_deteriorate_further_once_halted() -> None:
    portfolio = Portfolio(initial_cash=10_000.0)
    portfolio.apply_fill(
        FillEvent(_BAR.timestamp, "SBER", quantity=-100.0, price=100.0, commission=0.0)
    )
    bar_2 = Bar(pd.Timestamp("2026-01-06"), "SBER", 1000.0, 1010.0, 990.0, 1000.0, 1000.0)
    portfolio.mark_to_market(bar_2)
    assert portfolio.halted is True

    # Even a further, more extreme adverse move must not move equity anymore:
    # the position was force-closed, so there's nothing left to mark.
    bar_3 = Bar(pd.Timestamp("2026-01-07"), "SBER", 5000.0, 5100.0, 4900.0, 5000.0, 1000.0)
    portfolio.mark_to_market(bar_3)

    assert list(portfolio.equity_curve().values) == pytest.approx([-80_000.0, -80_000.0])


@pytest.mark.parametrize("bad_close", [0.0, -1.0, float("nan")])
def test_orders_from_signals_rejects_non_tradeable_close_price(bad_close: float) -> None:
    portfolio = Portfolio(initial_cash=10_000.0)
    bad_bar = Bar(_BAR.timestamp, "SBER", 100.0, 101.0, 99.0, bad_close, 1000.0)

    with pytest.raises(ValueError, match="price must be a finite number > 0"):
        portfolio.orders_from_signals(
            [SignalEvent(bad_bar.timestamp, "SBER", target_weight=1.0)], bad_bar
        )
