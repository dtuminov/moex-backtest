"""End-to-end demo: real MOEX data -> cached fetch -> SMA-crossover backtest -> metrics.

Runs on the IMOEX index rather than a single tradable instrument on purpose:
indices have deep, unrestricted history on the ISS API, unlike individual
equities (capped at ~35 days for anonymous access) — see
`moex_backtest.data.moex_iss`. A SMA(20, 100) crossover needs a few years of
daily bars to say anything at all, so an index is what makes this demo
possible without an authenticated session.

This proves the pipeline works end-to-end on live numbers. Picking an
actual tradable instrument and a strategy with a real edge is the next
step, once this engine gets wired up to a Finam Arena entry.

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
