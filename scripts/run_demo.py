"""End-to-end demo: real MOEX data -> cached fetch -> SMA-crossover backtest -> metrics.

Runs on the IMOEX index, not a single equity: anonymous ISS access limits
equity history to ~35 trading days, and a SMA(20, 100) crossover needs years
of bars — see `moex_backtest.data.moex_iss.MoexISSClient` for the access-depth
details.

Run with: uv run python scripts/run_demo.py
"""

from __future__ import annotations

from datetime import date

from moex_backtest.data import MoexISSClient, ParquetCache
from moex_backtest.engine import Backtester, SimulatedBroker
from moex_backtest.strategy import SmaCrossoverStrategy

SYMBOL = "IMOEX"
START = date(2018, 1, 1)
END = date.today()


def main() -> None:
    cache = ParquetCache()
    with MoexISSClient() as client:
        bars = cache.get_or_fetch(
            key=f"index_{SYMBOL}_{START}_{END}",
            fetch=lambda: client.index_history(SYMBOL, START, END),
        )

    if bars.empty:
        raise SystemExit(f"no data returned for {SYMBOL} in [{START}, {END}]")

    span = f"{bars['TRADEDATE'].min().date()} .. {bars['TRADEDATE'].max().date()}"
    print(f"Fetched {len(bars)} daily bars for {SYMBOL} ({span})")

    strategy = SmaCrossoverStrategy(fast_window=20, slow_window=100)
    backtester = Backtester(
        strategy,
        initial_cash=1_000_000.0,
        broker=SimulatedBroker(commission_rate=0.0004, slippage_bps=5.0),
    )
    result = backtester.run(bars, symbol=SYMBOL)

    print(f"\n{len(result.fills)} trades executed\n")
    for metric, value in result.summary().items():
        print(f"{metric:>22}: {value:10.4f}")


if __name__ == "__main__":
    main()
