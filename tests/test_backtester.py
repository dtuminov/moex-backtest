"""End-to-end tests for Backtester — in particular, that it cannot look ahead."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import pytest

from moex_backtest.engine.backtester import Backtester
from moex_backtest.engine.broker import SimulatedBroker
from moex_backtest.engine.events import Bar, SignalEvent
from moex_backtest.strategy.buy_and_hold import BuyAndHoldStrategy


class _GoLongFromIndex:
    """Flat until `switch_at`, then fully long — used to pin down exactly which
    bar's price a signal gets sized on and which bar's price it fills at."""

    def __init__(self, switch_at: int) -> None:
        self._switch_at = switch_at

    def generate_signals(self, bar: Bar, history: Sequence[Bar]) -> list[SignalEvent]:
        weight = 1.0 if len(history) - 1 >= self._switch_at else 0.0
        return [SignalEvent(bar.timestamp, bar.symbol, target_weight=weight)]


def _bars_frame(rows: list[tuple[str, float, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
    ).assign(TRADEDATE=lambda d: pd.to_datetime(d["TRADEDATE"]))


_NO_COST_BROKER = SimulatedBroker(commission_rate=0.0, slippage_bps=0.0)


def test_signal_fills_one_bar_later_at_the_open_not_the_signal_bars_close() -> None:
    # Signal turns long on bar index 1 (sized off its CLOSE=102); the fill must
    # happen on bar index 2, at its OPEN=105 — never at 102 and never on bar 1.
    bars = _bars_frame(
        [
            ("2026-01-05", 100.0, 101.0, 99.0, 100.0, 1000.0),
            ("2026-01-06", 101.0, 102.0, 100.0, 102.0, 1000.0),
            ("2026-01-07", 105.0, 106.0, 104.0, 106.0, 1000.0),
        ]
    )
    backtester = Backtester(
        _GoLongFromIndex(switch_at=1), initial_cash=10_000.0, broker=_NO_COST_BROKER
    )

    result = backtester.run(bars, symbol="SBER")

    assert len(result.fills) == 1
    fill = result.fills[0]
    expected_quantity = 10_000.0 / 102.0  # sized off bar 1's close, the bar the signal fired on
    assert fill.quantity == pytest.approx(expected_quantity)
    assert fill.price == pytest.approx(105.0)  # filled at bar 2's open, not bar 1's close (102)

    expected_final_equity = (10_000.0 - expected_quantity * 105.0) + expected_quantity * 106.0
    assert result.equity_curve.iloc[-1] == pytest.approx(expected_final_equity)


def test_signal_on_the_last_bar_never_fills() -> None:
    bars = _bars_frame(
        [
            ("2026-01-05", 100.0, 101.0, 99.0, 100.0, 1000.0),
            ("2026-01-06", 101.0, 102.0, 100.0, 102.0, 1000.0),
        ]
    )
    backtester = Backtester(
        _GoLongFromIndex(switch_at=1), initial_cash=10_000.0, broker=_NO_COST_BROKER
    )

    result = backtester.run(bars, symbol="SBER")

    assert result.fills == []
    assert list(result.equity_curve.values) == pytest.approx([10_000.0, 10_000.0])


def test_run_raises_on_missing_columns() -> None:
    bars = pd.DataFrame({"TRADEDATE": ["2026-01-05"], "CLOSE": [100.0]})
    backtester = Backtester(BuyAndHoldStrategy(), initial_cash=10_000.0)

    with pytest.raises(ValueError, match="missing required columns"):
        backtester.run(bars, symbol="SBER")


def test_run_raises_on_empty_bars() -> None:
    bars = _bars_frame([])
    backtester = Backtester(BuyAndHoldStrategy(), initial_cash=10_000.0)

    with pytest.raises(ValueError, match="no bars"):
        backtester.run(bars, symbol="SBER")


class _AlwaysFullyShort:
    """Constant target_weight=-1.0 — used to blow a short position through
    the equity floor via a large adverse (upward) price move."""

    def generate_signals(self, bar: Bar, history: Sequence[Bar]) -> list[SignalEvent]:
        return [SignalEvent(bar.timestamp, bar.symbol, target_weight=-1.0)]


def test_short_position_plus_10x_adverse_move_does_not_crash_and_halts_the_engine() -> None:
    # Reproduction of the original bug: a short position against a ~10x
    # adverse price move over 6 bars used to make Portfolio.equity go
    # negative, which in turn made annualized_return() raise an uncaught
    # TypeError (float() on a complex number) instead of failing cleanly.
    #
    # Now: the equity floor in Portfolio force-closes the position and halts
    # further trading the first time equity hits zero or below, so
    # Backtester.run() itself never crashes. The one bar where equity
    # actually crosses zero still has a return <= -100% by construction (an
    # equity curve cannot go from positive to <= 0 any other way), so
    # summary() raises a clear ValueError from annualized_return's
    # defense-in-depth guard rather than an opaque TypeError.
    prices = [100.0, 100.0, 110.0, 180.0, 300.0, 500.0, 800.0, 1000.0]  # ~10x over 6 bars
    bars = _bars_frame(
        [(f"2026-01-{i + 1:02d}", p, p + 1.0, p - 1.0, p, 1000.0) for i, p in enumerate(prices)]
    )
    backtester = Backtester(_AlwaysFullyShort(), initial_cash=10_000.0, broker=_NO_COST_BROKER)

    result = backtester.run(bars, symbol="SBER")  # must not raise

    equity = list(result.equity_curve.values)
    assert min(equity) <= 0.0  # equity did go to/below zero at some point
    assert equity[-1] == equity[-2]  # flat (halted) for the last two bars

    with pytest.raises(ValueError, match="total return <= -100%"):
        result.summary()


def test_backtest_over_a_single_bar_raises_a_clear_error_on_summary() -> None:
    bars = _bars_frame([("2026-01-05", 100.0, 101.0, 99.0, 100.0, 1000.0)])
    backtester = Backtester(BuyAndHoldStrategy(), initial_cash=10_000.0, broker=_NO_COST_BROKER)

    result = backtester.run(bars, symbol="SBER")

    with pytest.raises(ValueError, match="need at least 2 bars to compute returns"):
        result.summary()


def test_summary_reports_the_expected_metric_keys() -> None:
    bars = _bars_frame(
        [
            ("2026-01-0" + str(d), 100.0 + d, 101.0 + d, 99.0 + d, 100.0 + d, 1000.0)
            for d in range(1, 6)
        ]
    )
    backtester = Backtester(BuyAndHoldStrategy(), initial_cash=10_000.0, broker=_NO_COST_BROKER)

    result = backtester.run(bars, symbol="SBER")
    summary = result.summary()

    assert set(summary) == {
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "num_trades",
    }
    assert summary["num_trades"] == float(len(result.fills))
