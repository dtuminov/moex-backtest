"""Tests for moex_backtest.data.known_events."""

from __future__ import annotations

import pandas as pd

from moex_backtest.data.known_events import (
    CBOM_CORPORATE_EVENT_DAY,
    CBOM_SYMBOL,
    HALT_2022_REOPEN_DAY,
    excluded_return_dates,
    mask_returns_for_vol_estimation,
)


def test_excluded_return_dates_is_market_wide_for_a_generic_symbol() -> None:
    assert excluded_return_dates("SBER") == frozenset({HALT_2022_REOPEN_DAY})


def test_excluded_return_dates_adds_the_cbom_event_only_for_cbom() -> None:
    assert excluded_return_dates(CBOM_SYMBOL) == frozenset(
        {HALT_2022_REOPEN_DAY, CBOM_CORPORATE_EVENT_DAY}
    )


def test_mask_zeroes_the_halt_reopen_day_for_any_symbol() -> None:
    index = pd.to_datetime(["2022-03-23", "2022-03-24", "2022-03-25"])
    returns = pd.Series([0.01, 0.134, -0.02], index=index)

    masked = mask_returns_for_vol_estimation(returns, "GAZP")

    assert masked.loc["2022-03-23"] == 0.01
    assert masked.loc["2022-03-24"] == 0.0
    assert masked.loc["2022-03-25"] == -0.02
    # The original series is untouched (a copy was returned).
    assert returns.loc["2022-03-24"] == 0.134


def test_mask_also_zeroes_the_cbom_specific_event_day() -> None:
    index = pd.to_datetime(["2026-04-10", "2026-04-13", "2026-04-14"])
    returns = pd.Series([0.0, 0.54, -0.05], index=index)

    masked = mask_returns_for_vol_estimation(returns, CBOM_SYMBOL)

    assert masked.loc["2026-04-13"] == 0.0


def test_mask_does_not_zero_the_cbom_event_day_for_other_symbols() -> None:
    index = pd.to_datetime(["2026-04-13"])
    returns = pd.Series([0.54], index=index)

    masked = mask_returns_for_vol_estimation(returns, "SBER")

    assert masked.loc["2026-04-13"] == 0.54


def test_mask_is_a_no_op_when_no_excluded_date_is_present() -> None:
    index = pd.to_datetime(["2023-01-01", "2023-01-02"])
    returns = pd.Series([0.01, -0.01], index=index)

    masked = mask_returns_for_vol_estimation(returns, "SBER")

    pd.testing.assert_series_equal(masked, returns)
