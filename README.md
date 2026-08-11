# moex-backtest

An event-driven backtesting engine and strategy research toolkit for the
Russian market (MOEX), built from scratch: a typed MOEX ISS data client, a
lookahead-safe backtest loop, cost-aware execution simulation, and the
standard risk/return metric set (Sharpe, Sortino, max drawdown, CVaR,
Calmar).

## Why this exists

This is the infrastructure layer for a strategy I plan to enter into
[Финам Арена](https://arena.finam.ru/) — the engine here is built so the
same `Strategy` protocol and cost model can later be pointed at a live
Trade API execution backend instead of the simulated broker, without
touching the strategy code.

## Design decisions worth reading before the code

**No lookahead, by construction, not by convention.** A strategy sees bars
only up to and including the current one (`generate_signals(bar, history)`).
Orders sized from bar *t*'s signal are executed at bar *t+1*'s **open** —
never at bar *t*'s own close. This one-bar delay is enforced by the loop
itself (`engine/backtester.py`), not left as a discipline the strategy
author has to remember. `tests/test_backtester.py` pins this down with an
explicit assertion: a signal sized off bar *N*'s close fills at bar *N+1*'s
open, never at bar *N*'s own price.

**Costs are modelled, not ignored.** Every fill pays a proportional
commission plus fixed-bps slippage against the trade direction
(`engine/broker.py`). A strategy that trades often will visibly bleed
performance to costs in the summary metrics — turnover isn't free here.

**MOEX ISS has an anonymous-access quirk that matters for data selection,**
found by probing the API directly rather than assuming: equities
(`stock`/`shares`, board `TQBR`) return only the last ~35 trading days to
unauthenticated requests, no matter how far back you ask. Indices and FORTS
futures/options are *not* depth-limited. `data/moex_iss.py` documents this
on the relevant methods; `scripts/run_demo.py` deliberately backtests an
index rather than a single stock so it can run over several years without
an authenticated session.

**v1 is single-symbol on purpose.** Cross-asset portfolio construction
(gross/net exposure across a book, correlation-aware sizing) is a real
design problem that deserves its own iteration, not a half-built afterthought
bolted onto this one — see Roadmap.

## Architecture

```
data/       MoexISSClient (typed, paginated, retrying HTTP client) + ParquetCache
metrics/    Sharpe, Sortino, max drawdown, Calmar, historical VaR/CVaR — pure functions
engine/     events (Bar/Signal/Order/Fill) -> Portfolio -> SimulatedBroker -> Backtester
strategy/   Strategy protocol + SmaCrossoverStrategy (the engine's smoke test)
```

Data flow per bar, inside `Backtester.run`:

```
pending orders (from bar t-1) --[SimulatedBroker]--> fills --> Portfolio.apply_fill
                                                                       |
                                                          Portfolio.mark_to_market
                                                                       |
                                          Strategy.generate_signals(bar t, history<=t)
                                                                       |
                                          Portfolio.orders_from_signals --> pending orders (for bar t+1)
```

## Results (real IMOEX data, 2018–2026)

![Equity curve: Buy & Hold vs SMA(20,100) crossover](reports/assets/equity_curve.png)

SMA(20,100) barely participates in the early-2022 crash and gives up
upside during the 2020-2021 rally in exchange — the standard trend-following
trade-off between drawdown and participation. Full numbers (Sharpe, Sortino,
Calmar, cost sensitivity) are in `reports/report.pdf`.

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

Run the end-to-end demo (fetches real IMOEX history via MOEX ISS, caches it
to `data/raw/`, backtests a SMA(20, 100) crossover, prints metrics):

```bash
uv run python scripts/run_demo.py
```

Minimal example against your own data:

```python
from moex_backtest.engine import Backtester, SimulatedBroker
from moex_backtest.strategy import SmaCrossoverStrategy

strategy = SmaCrossoverStrategy(fast_window=20, slow_window=100)
backtester = Backtester(strategy, initial_cash=1_000_000.0, broker=SimulatedBroker())
result = backtester.run(
    bars, symbol="IMOEX"
)  # bars: DataFrame with TRADEDATE/OPEN/HIGH/LOW/CLOSE/VOLUME

print(result.summary())  # annualized_return, sharpe_ratio, max_drawdown, ...
```

## Development

```bash
uv run pytest        # tests + coverage (configured in pyproject.toml)
uv run ruff check .  # lint
uv run ruff format .
uv run mypy src tests  # strict mode
```

## Roadmap

- Multi-symbol portfolio with gross/net exposure and CVaR-based risk limits
  (the risk-oriented portfolio construction from my coursework project is
  the natural next step here).
- Execution adapter for the Finam Trade API, so the same `Strategy` can run
  live in Финам Арена instead of against `SimulatedBroker`.
- A strategy actually worth trading — `SmaCrossoverStrategy` exists to
  exercise the engine, not to make money.
