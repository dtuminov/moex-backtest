"""Event-driven backtest loop.

Orders sized from bar *t*'s signal are executed at bar *t+1*'s open — never
at bar *t*'s own close. That one-bar delay is what keeps the loop free of
lookahead bias: nothing a strategy decides after seeing bar *t* can be
filled at a price that was not yet known at the moment it decided.
A consequence worth knowing: the signal generated on the *last* bar has no
following bar to fill at, so it never executes — the final equity value
reflects the position held *before* that last, unfilled decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from moex_backtest.engine.broker import SimulatedBroker
from moex_backtest.engine.events import Bar, FillEvent, OrderEvent
from moex_backtest.engine.portfolio import Portfolio
from moex_backtest.metrics import performance
from moex_backtest.strategy.base import Strategy

_REQUIRED_COLUMNS = ("OPEN", "HIGH", "LOW", "CLOSE", "VOLUME")


@dataclass(frozen=True, slots=True)
class BacktestResult:
    equity_curve: pd.Series
    fills: list[FillEvent]

    def returns(self) -> pd.Series:
        return self.equity_curve.pct_change().dropna()

    def summary(self, periods_per_year: int = 252, risk_free_rate: float = 0.0) -> dict[str, float]:
        """Headline metrics computed from this result via :mod:`moex_backtest.metrics`."""
        returns = self.returns()
        return {
            "annualized_return": performance.annualized_return(returns, periods_per_year),
            "annualized_volatility": performance.annualized_volatility(returns, periods_per_year),
            "sharpe_ratio": performance.sharpe_ratio(returns, risk_free_rate, periods_per_year),
            "sortino_ratio": performance.sortino_ratio(returns, risk_free_rate, periods_per_year),
            "max_drawdown": performance.max_drawdown(self.equity_curve),
            "calmar_ratio": performance.calmar_ratio(returns, self.equity_curve, periods_per_year),
            "num_trades": float(len(self.fills)),
        }


class Backtester:
    """Runs one strategy against one symbol's daily bars.

    v1 is single-symbol by design — see :mod:`moex_backtest.engine.portfolio`.
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_cash: float = 1_000_000.0,
        broker: SimulatedBroker | None = None,
        max_gross_exposure: float = 1.0,
    ) -> None:
        self._strategy = strategy
        self._initial_cash = initial_cash
        self._broker = broker or SimulatedBroker()
        self._max_gross_exposure = max_gross_exposure

    def run(self, bars: pd.DataFrame, symbol: str) -> BacktestResult:
        bar_series = _bars_from_frame(bars, symbol)
        if not bar_series:
            raise ValueError("no bars to run the backtest on")

        portfolio = Portfolio(self._initial_cash, self._max_gross_exposure)
        pending_orders: list[OrderEvent] = []

        for i, bar in enumerate(bar_series):
            for fill in self._broker.execute(pending_orders, bar):
                portfolio.apply_fill(fill)

            portfolio.mark_to_market(bar)

            signals = self._strategy.generate_signals(bar, bar_series[: i + 1])
            pending_orders = portfolio.orders_from_signals(signals, bar)

        return BacktestResult(equity_curve=portfolio.equity_curve(), fills=portfolio.fills)


def _bars_from_frame(frame: pd.DataFrame, symbol: str) -> list[Bar]:
    missing = [c for c in (*_REQUIRED_COLUMNS, "TRADEDATE") if c not in frame.columns]
    if missing:
        raise ValueError(f"bars frame is missing required columns: {missing}")

    ordered = frame.sort_values("TRADEDATE")
    timestamps = pd.to_datetime(ordered["TRADEDATE"])
    opens = ordered["OPEN"].to_numpy(dtype=float)
    highs = ordered["HIGH"].to_numpy(dtype=float)
    lows = ordered["LOW"].to_numpy(dtype=float)
    closes = ordered["CLOSE"].to_numpy(dtype=float)
    volumes = ordered["VOLUME"].to_numpy(dtype=float)

    return [
        Bar(timestamp=ts, symbol=symbol, open=o, high=h, low=low, close=c, volume=v)
        for ts, o, h, low, c, v in zip(timestamps, opens, highs, lows, closes, volumes, strict=True)
    ]
