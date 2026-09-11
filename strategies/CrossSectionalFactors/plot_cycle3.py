"""Render cycle 3's equity curve for the README.

Two lines, deliberately: the locked 47-name construction and the same rule
with the 15 sanctioned megacaps removed. The gap between them *is* the
"roughly a quarter of the edge is a segmentation bet" finding from the
cycle 3 report -- showing only the headline curve would hide it.

Run with: uv run python strategies/CrossSectionalFactors/plot_cycle3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import run_cycle3_illiquidity as c3  # noqa: E402
from factors import backtest_factor, illiquidity_signal  # noqa: E402

from moex_backtest.metrics.performance import sharpe_ratio  # noqa: E402

OUT = HERE.parent.parent / "reports" / "assets" / "cycle3_illiquidity.png"

# The 15 directly-sanctioned megacaps / systemic banks, verbatim from the
# cycle 3 report's Objection A test.
SANCTIONED = (
    "GAZP", "LKOH", "ROSN", "NVTK", "TATN", "SNGS", "SIBN",
    "GMKN", "NLMK", "CHMF", "MAGN", "RUAL", "PLZL", "SBER", "VTBR",
)

# Categorical slots 1 and 2, in fixed order, from the dataviz reference palette.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#1a1a19"
MUTED = "#6b6b64"
GRID = "#e3e3dd"


def _curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod()


def main() -> None:
    data = c3._load_data()
    dates = data.formation_dates[data.formation_dates >= c3.IS_START]

    locked = c3._illiq_returns(data, 12, dates)

    keep = [c for c in data.monthly_close.columns if c not in SANCTIONED]
    n_per_leg = len(keep) // 5
    signal = illiquidity_signal(data.dollar_volume[keep], data.formation_dates, 12)
    signal = signal.reindex(data.monthly_close.index)
    ex_mega = backtest_factor(
        signal, data.monthly_close[keep], dates,
        direction="low_long", n_per_leg=n_per_leg, cost_bps=c3.COST_BPS,
    ).returns

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Label the line ends with OOS Sharpe, not the growth multiple: total growth
    # spans IS+OOS and would read backwards here (removing the megacaps *raises*
    # cumulative growth while *lowering* the risk-adjusted OOS result, which is
    # the quantity the cycle 3 verdict is actually about).
    for series, color, label in (
        (locked, BLUE, f"Locked rule, {len(data.monthly_close.columns)}-name universe"),
        (ex_mega, ORANGE, f"Same rule, 15 sanctioned megacaps removed ({len(keep)} names)"),
    ):
        curve = _curve(series)
        oos = series.loc[series.index >= c3.OOS_START]
        oos_sharpe = sharpe_ratio(oos, periods_per_year=12)
        ax.plot(curve.index, curve.to_numpy(), color=color, linewidth=2.0, label=label,
                solid_capstyle="round")
        ax.annotate(
            f"OOS Sharpe {oos_sharpe:+.2f}",
            xy=(curve.index[-1], curve.iloc[-1]), xytext=(6, 0),
            textcoords="offset points", va="center", ha="left",
            color=INK, fontsize=9.5,
        )
        print(f"{label}: OOS Sharpe {oos_sharpe:+.4f}, final {curve.iloc[-1]:.3f}x")

    ax.axvline(c3.OOS_START, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    # Anchored to the axes bottom, not to a y-value: keeps it clear of the
    # legend, and stays put if the data range changes.
    ax.annotate(
        "config locked  →  out-of-sample",
        xy=(c3.OOS_START, 0.02), xycoords=("data", "axes fraction"),
        xytext=(6, 0), textcoords="offset points",
        color=MUTED, fontsize=9, va="bottom", ha="left",
    )

    ax.set_title(
        "Cross-sectional illiquidity premium on MOEX, long/short, monthly rebalance",
        color=INK, fontsize=12.5, pad=14, loc="left",
    )
    ax.set_ylabel("Growth of 1 (net of 15bps one-way)", color=MUTED, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.margins(x=0.06)

    legend = ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(INK)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor="white", bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
