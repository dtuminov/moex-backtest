"""Tests for BuyAndHoldStrategy."""

from __future__ import annotations

import pandas as pd

from moex_backtest.engine.events import Bar
from moex_backtest.strategy.buy_and_hold import BuyAndHoldStrategy


def test_always_signals_fully_long() -> None:
    strategy = BuyAndHoldStrategy()
    early_bar = Bar(pd.Timestamp("2026-01-01"), "SBER", 100, 100, 100, 100, 1000.0)
    later_bar = Bar(pd.Timestamp("2026-06-01"), "SBER", 150, 150, 150, 150, 1000.0)

    early_signal = strategy.generate_signals(early_bar, [early_bar])[0]
    later_signal = strategy.generate_signals(later_bar, [early_bar, later_bar])[0]

    assert early_signal.target_weight == 1.0
    assert later_signal.target_weight == 1.0
