"""Generates the report assets in reports/assets/: a parameter grid for SMA
crossover on IMOEX, a buy-and-hold benchmark, and a with/without transaction
costs comparison, on real MOEX ISS data. Writes results.csv (all metrics per
strategy) and equity_curve.png (equity + drawdown for the two headline
strategies).

Run with: uv run python scripts/generate_report_assets.py
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moex_backtest.data import MoexISSClient, ParquetCache
from moex_backtest.engine import Backtester, BacktestResult, SimulatedBroker
from moex_backtest.strategy import BuyAndHoldStrategy, SmaCrossoverStrategy
from moex_backtest.strategy.base import Strategy

SYMBOL = "IMOEX"
START = date(2018, 1, 1)
END = date.today()
INITIAL_CASH = 1_000_000.0
REALISTIC_BROKER = SimulatedBroker(commission_rate=0.0004, slippage_bps=5.0)
ZERO_COST_BROKER = SimulatedBroker(commission_rate=0.0, slippage_bps=0.0)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "reports" / "assets"


def fetch_bars() -> pd.DataFrame:
    cache = ParquetCache()
    with MoexISSClient() as client:
        return cache.get_or_fetch(
            key=f"index_{SYMBOL}_{START}_{END}",
            fetch=lambda: client.index_history(SYMBOL, START, END),
        )


def run(strategy: Strategy, broker: SimulatedBroker, bars: pd.DataFrame) -> BacktestResult:
    backtester = Backtester(strategy, initial_cash=INITIAL_CASH, broker=broker)
    return backtester.run(bars, symbol=SYMBOL)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    bars = fetch_bars()
    span = f"{bars['TRADEDATE'].min().date()} .. {bars['TRADEDATE'].max().date()}"
    print(f"{SYMBOL}: {len(bars)} daily bars, {span}")

    rows = []

    bh_result = run(BuyAndHoldStrategy(), REALISTIC_BROKER, bars)
    rows.append(("Buy & Hold", bh_result))

    param_grid = [(10, 50), (20, 100), (50, 200)]
    best_result: BacktestResult | None = None
    for fast, slow in param_grid:
        result = run(SmaCrossoverStrategy(fast, slow), REALISTIC_BROKER, bars)
        rows.append((f"SMA({fast},{slow}) crossover", result))
        if fast == 20 and slow == 100:
            best_result = result

    assert best_result is not None
    zero_cost_result = run(SmaCrossoverStrategy(20, 100), ZERO_COST_BROKER, bars)
    rows.append(("SMA(20,100) — zero costs (illustrative)", zero_cost_result))

    csv_path = ASSETS_DIR / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", *best_result.summary().keys()])
        for name, result in rows:
            summary = result.summary()
            writer.writerow([name, *(f"{v:.4f}" for v in summary.values())])
            print(f"\n{name}")
            for metric, value in summary.items():
                print(f"  {metric:>22}: {value:10.4f}")
    print(f"\nWrote {csv_path}")

    png_path = ASSETS_DIR / "equity_curve.png"
    _plot_equity_and_drawdown(bh_result, best_result, span, png_path)
    print(f"Wrote {png_path}")


def _drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def _plot_equity_and_drawdown(
    bh_result: BacktestResult, best_result: BacktestResult, span: str, png_path: Path
) -> None:
    """Equity curves on top, drawdown of each strategy below, shared x-axis."""
    fig, (ax_equity, ax_dd) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    series = [
        (bh_result, "Buy & Hold"),
        (best_result, "SMA(20,100) crossover"),
    ]
    for result, label in series:
        equity = result.equity_curve
        drawdown_pct = _drawdown(equity).to_numpy(dtype=float) * 100
        ax_equity.plot(equity.index, equity.values, label=label, linewidth=1.2)
        line = ax_dd.plot(equity.index, drawdown_pct, linewidth=1.0)[0]
        ax_dd.fill_between(equity.index, drawdown_pct, 0, alpha=0.15, color=line.get_color())

    ax_equity.set_title(f"{SYMBOL}: equity curve and drawdown, {span}")
    ax_equity.set_ylabel("Portfolio value, RUB")
    ax_equity.legend()
    ax_equity.grid(alpha=0.3)

    ax_dd.set_ylabel("Drawdown, %")
    ax_dd.set_xlabel("Date")
    ax_dd.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)


if __name__ == "__main__":
    main()
