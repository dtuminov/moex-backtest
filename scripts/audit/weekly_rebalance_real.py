"""Step 1 of the T-vs-costs audit: REAL weekly-rebalanced momentum on the
REAL 47-stock daily panel already on disk. No synthetic data here -- this
just measures what actually happens (turnover, gross/net Sharpe, T) if the
locked cycle-1 momentum construction (12-1, high_long, n_per_leg=9,
cost_bps=15) is rebalanced weekly on daily bars instead of monthly.

lookback/skip are re-expressed in weeks: 52 weeks (~12 months) lookback,
4 weeks (~1 month) skip -- same economic window, finer sampling grid, not a
new hypothesis or a parameter search (single config, no grid).

Reads only strategies/CrossSectionalFactors/{panel,factors}.py (the real,
already-reviewed cycle-1 code) plus moex_backtest.metrics -- writes nothing
to experiments/grid_search_log.jsonl, no lock file: this is diagnostic
characterization, not a new trial (matches memory precedent: diagnostic
work on already-collected/already-licensed data does not count toward N).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Resolved from this file's location rather than hardcoded: the only edit
# made to these scripts after the run that produced the reported tables.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "strategies" / "CrossSectionalFactors"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factors import _leg_turnover, _quintile_legs  # noqa: E402
from panel import load_close_panel  # noqa: E402

from moex_backtest.metrics.performance import sharpe_ratio  # noqa: E402

OOS_START = pd.Timestamp("2022-01-01")
OOS_END = pd.Timestamp("2026-09-09")
IS_START = pd.Timestamp("2018-01-01")
IS_END = pd.Timestamp("2021-12-31")

N_PER_LEG = 9
COST_BPS = 15.0
LOOKBACK_WEEKS = 52
SKIP_WEEKS = 4


def weekly_momentum_signal(weekly_close: pd.DataFrame) -> pd.DataFrame:
    recent = weekly_close.shift(SKIP_WEEKS)
    past = weekly_close.shift(LOOKBACK_WEEKS)
    return recent / past - 1.0


def backtest_weekly(
    signal: pd.DataFrame, weekly_close: pd.DataFrame, formation_dates: pd.DatetimeIndex
) -> tuple[pd.Series, pd.Series]:
    """Returns (net_returns, turnover_series) -- turnover instrumented
    directly via the real `_leg_turnover`/`_quintile_legs` helpers cycle 1
    already uses, so this is the exact same cost/turnover accounting, just
    called from a local loop instead of `factors.backtest_factor` (which
    doesn't expose turnover per rebalance).
    """
    fwd_ret = weekly_close.pct_change().shift(-1)
    net_values: list[float] = []
    turnovers: list[float] = []
    index: list[pd.Timestamp] = []
    prev_long: frozenset[str] = frozenset()
    prev_short: frozenset[str] = frozenset()

    for t in formation_dates:
        long_set, short_set = _quintile_legs(signal.loc[t], N_PER_LEG, "high_long")
        if not long_set or not short_set:
            continue
        fwd_row = fwd_ret.loc[t]
        long_ret = float(fwd_row[list(long_set)].mean())
        short_ret = float(fwd_row[list(short_set)].mean())
        gross = long_ret - short_ret
        turnover = _leg_turnover(prev_long, prev_short, long_set, short_set, N_PER_LEG)
        cost = turnover * (COST_BPS / 10_000.0)
        if not np.isnan(gross):
            net_values.append(gross - cost)
            turnovers.append(turnover)
            index.append(t)
        prev_long, prev_short = long_set, short_set

    net = pd.Series(net_values, index=pd.DatetimeIndex(index), dtype=float)
    turn = pd.Series(turnovers, index=pd.DatetimeIndex(index), dtype=float)
    return net, turn


def main() -> None:
    close_panel = load_close_panel(cache_dir=REPO_ROOT / "data" / "raw")
    weekly_close = close_panel.resample("W-FRI").last()
    print(f"Weekly panel: {len(weekly_close)} weeks, "
          f"{weekly_close.index[0].date()} .. {weekly_close.index[-1].date()}")

    signal = weekly_momentum_signal(weekly_close)
    all_dates = weekly_close.index
    net, turn = backtest_weekly(signal, weekly_close, all_dates)

    is_mask = (net.index >= IS_START) & (net.index <= IS_END)
    oos_mask = (net.index >= OOS_START) & (net.index <= OOS_END)

    is_net = net[is_mask]
    oos_net = net[oos_mask]
    oos_turn = turn[oos_mask]

    PERIODS_PER_YEAR_WEEKLY = 52.18

    print(f"\nIS weeks: {len(is_net)}, OOS weeks: {len(oos_net)}")
    print(f"Mean weekly turnover (OOS): {oos_turn.mean():.4%} of positions/week")
    print(f"  -> implied monthly-equivalent turnover: {oos_turn.mean() * 4.348:.2%} "
          "positions/month-equivalent (for comparison to cycle-1's 23.2%/month)")

    # Gross (no cost) vs net Sharpe, to isolate cost drag precisely
    gross_oos = oos_net + oos_turn * (COST_BPS / 10_000.0)
    sr_gross = sharpe_ratio(gross_oos, periods_per_year=PERIODS_PER_YEAR_WEEKLY)
    sr_net = sharpe_ratio(oos_net, periods_per_year=PERIODS_PER_YEAR_WEEKLY)
    print(f"\nOOS annualized Sharpe, weekly rebalance: gross={sr_gross:+.3f}  net={sr_net:+.3f}")
    print(f"Cost drag on annualized Sharpe: {sr_gross - sr_net:+.3f}")

    annual_cost_drag_bps = oos_turn.mean() * (COST_BPS / 10_000.0) * PERIODS_PER_YEAR_WEEKLY * 10_000
    print(f"Annualized cost drag: {annual_cost_drag_bps:.1f} bps/year "
          "(cycle-1 monthly reference: ~42 bps/year)")

    print(f"\nMonthly locked config reference (cycle 1, for comparison): "
          f"OOS Sharpe (net) = +0.562, T_OOS=56, turnover=23.2%/month")
    print(f"Weekly config here: OOS Sharpe (net) = {sr_net:+.3f}, T_OOS={len(oos_net)}")

    # Save the real weekly OOS net-return series + IS series for step 2 (power audit)
    out = Path(__file__).parent
    oos_net.to_frame("weekly_momentum_net").to_csv(out / "weekly_oos_net_returns.csv")
    is_net.to_frame("weekly_momentum_net").to_csv(out / "weekly_is_net_returns.csv")
    print(f"\nSaved weekly IS/OOS net-return series to {out}")


if __name__ == "__main__":
    main()
