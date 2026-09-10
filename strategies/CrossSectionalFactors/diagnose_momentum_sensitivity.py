"""Diagnostic (not a new alpha-search cycle): why is H-MOM's parameter
sensitivity a spike, not a plateau, around the locked lookback=12/skip=1?

Cycle 1 (`run_cycle.py`, see `reports/cycle1.md`) locked H-MOM at
lookback=12/skip=1 (IS Sharpe +1.624) and, in the post-lock verdict stage,
found `parameter_sensitivity` classifies the OOS curve around that point as
a SPIKE -- one of the two reasons (with DSR) the cycle's final verdict was
FAIL. This script asks *why*, using only already-computed-shape data plus
additional points computed purely to trace the curve's shape -- **no new
alpha trial, no re-optimization, no re-lock**.

Per the user's explicit instruction: every additional lookback point
computed here is logged to `experiments/grid_search_log.jsonl` with
`"stage": "diagnostic_curve"` and `"counts_toward_n": false` -- it is a
measurement of an already-fixed hypothesis's shape, not a new candidate
competing to be selected, exactly the same distinction the crypto/Jesse
track drew for its cycle 7 (`RegimeWindowDiagnostic`,
`memory/jesse-trade-algo-strategy.md`). The MOEX-track cumulative DSR-N
stays at 9 after this script runs. `experiments/cycle1_momentum.lock.json`
is read-only here and is never rewritten.

Four questions, matching what was asked:

1. Curve shape (dense lookback grid, skip=1 fixed) -- single peak, plateau-
   with-cliff, or sawtooth? Does the peak sit exactly at 12, or is it
   somewhere else and 12 only looks good because of the ±10/20/30% grid
   cycle 1 happened to sample?
2. Is the peak location stable across sub-periods (IS 2018-2021, OOS first
   half, OOS second half) or does it wander?
3. How big is the sampling noise on each curve point, relative to the
   differences between adjacent points? (Same Sharpe-estimator variance
   formula as `validation.dsr`'s step 2, applied descriptively here, not as
   a hypothesis test.)
4. Autocorrelation of the locked (12, 1) monthly return series, and the
   resulting rough effective sample size -- are ~48-56 "monthly
   observations" actually that many independent data points?

Run with: uv run python strategies/CrossSectionalFactors/diagnose_momentum_sensitivity.py
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from factors import backtest_factor, momentum_signal
from panel import load_close_panel, monthly_close_panel

from moex_backtest.metrics.performance import sharpe_ratio

HERE = Path(__file__).parent
LEDGER_PATH = HERE / "experiments" / "grid_search_log.jsonl"
LOCK_PATH = HERE / "experiments" / "cycle1_momentum.lock.json"
REPORT_PATH = HERE / "reports" / "momentum_sensitivity_diagnostic.md"
CHART_PATH = HERE / "reports" / "momentum_lookback_curves.png"

IS_START = pd.Timestamp("2018-01-01")
IS_END = pd.Timestamp("2021-12-31")
OOS_START = pd.Timestamp("2022-01-01")
OOS_END = pd.Timestamp("2026-09-09")

N_PER_LEG = 9
COST_BPS = 15.0
PERIODS_PER_YEAR = 12
LOCKED_SKIP = 1  # from cycle1_momentum.lock.json -- read and asserted below, not hardcoded blindly
LOOKBACK_RANGE = range(2, 21)  # skip=1, so lookback must be >= 2


def _append_diagnostic(period: str, lookback: int, sharpe: float, n_obs: int, se: float) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "hypothesis": "momentum",
        "stage": "diagnostic_curve",
        "counts_toward_n": False,
        "note": "curve-shape diagnostic on the already-locked config; not a trial",
        "period": period,
        "config": {"lookback": lookback, "skip": LOCKED_SKIP},
        "is_sharpe": sharpe,
        "n_observations": n_obs,
        "sharpe_se_annual": se,
    }
    with LEDGER_PATH.open("a") as f:
        f.write(json.dumps(payload) + "\n")


@dataclass
class Data:
    monthly_close: pd.DataFrame
    formation_dates: pd.DatetimeIndex
    is_dates: pd.DatetimeIndex
    oos_dates: pd.DatetimeIndex
    oos_h1_dates: pd.DatetimeIndex
    oos_h2_dates: pd.DatetimeIndex


def _load_data() -> Data:
    close_panel = load_close_panel(cache_dir=HERE.parent.parent / "data" / "raw")
    monthly_close = monthly_close_panel(close_panel)
    formation_dates = monthly_close.index
    is_dates = formation_dates[(formation_dates >= IS_START) & (formation_dates <= IS_END)]
    oos_dates = formation_dates[(formation_dates >= OOS_START) & (formation_dates <= OOS_END)]
    mid = len(oos_dates) // 2
    return Data(
        monthly_close, formation_dates, is_dates, oos_dates, oos_dates[:mid], oos_dates[mid:]
    )


def _momentum_returns(data: Data, lookback: int, dates: pd.DatetimeIndex) -> pd.Series:
    signal = momentum_signal(data.monthly_close, lookback, LOCKED_SKIP)
    result = backtest_factor(
        signal, data.monthly_close, dates,
        direction="high_long", n_per_leg=N_PER_LEG, cost_bps=COST_BPS,
    )
    return result.returns


def _sharpe_se_annual(returns: pd.Series, periods_per_year: int) -> float:
    """Same asymptotic Sharpe-estimator variance formula used internally by
    `validation.dsr`'s step 2 (Bailey & Lopez de Prado 2014) -- applied here
    purely as a descriptive error bar per curve point, not as part of any
    hypothesis test or DSR computation. NaN if too few observations or a
    degenerate (zero-variance) series.
    """
    excess = returns.to_numpy(dtype=float)
    t = len(excess)
    if t < 3:
        return float("nan")
    std = float(np.std(excess, ddof=1))
    if std == 0.0:
        return float("nan")
    sr_period = float(np.mean(excess)) / std
    skew = float(stats.skew(excess, bias=True))
    kurt = float(stats.kurtosis(excess, fisher=False, bias=True))
    var_period = (1.0 / (t - 1)) * (1.0 - skew * sr_period + ((kurt - 1.0) / 4.0) * sr_period**2)
    if var_period <= 0.0:
        return float("nan")
    return math.sqrt(var_period) * math.sqrt(periods_per_year)


@dataclass
class CurvePoint:
    lookback: int
    sharpe: float
    n_obs: int
    se: float


def _trace_curve(data: Data, period_name: str, dates: pd.DatetimeIndex) -> list[CurvePoint]:
    points: list[CurvePoint] = []
    for lookback in LOOKBACK_RANGE:
        returns = _momentum_returns(data, lookback, dates)
        if len(returns) < 3:
            continue
        sharpe = sharpe_ratio(returns, periods_per_year=PERIODS_PER_YEAR)
        se = _sharpe_se_annual(returns, PERIODS_PER_YEAR)
        points.append(CurvePoint(lookback, sharpe, len(returns), se))
        _append_diagnostic(period_name, lookback, sharpe, len(returns), se)
    return points


def _classify_curve(points: list[CurvePoint]) -> str:
    """Rough, descriptive classification -- not the formal spike_ratio test
    from `validation.sensitivity` (that one already ran, on 7 points, in
    cycle 1; this is a qualitative read of the full dense curve)."""
    sharpes = np.array([p.sharpe for p in points])
    jumps = np.abs(np.diff(sharpes))
    if len(jumps) == 0:
        return "insufficient points"
    max_jump, median_jump = float(np.max(jumps)), float(np.median(jumps))
    ratio = max_jump / median_jump if median_jump > 0 else float("inf")
    peak_idx = int(np.argmax(sharpes))
    is_interior_peak = 0 < peak_idx < len(points) - 1
    if is_interior_peak and ratio > 3.0:
        return "single isolated peak (spike)"
    if is_interior_peak and ratio <= 3.0:
        return "smooth interior peak (plateau-like)"
    return "monotonic / edge peak (no interior maximum in range)"


def _autocorr_summary(returns: pd.Series, label: str) -> tuple[float, float, float]:
    rho1 = float(returns.autocorr(lag=1)) if len(returns) > 2 else float("nan")
    t = len(returns)
    rho1_valid = not math.isnan(rho1) and rho1 not in (-1.0, 1.0)
    n_eff = t * (1.0 - rho1) / (1.0 + rho1) if rho1_valid else float("nan")
    return rho1, float(t), n_eff


def main() -> None:
    assert LOCK_PATH.exists(), "cycle1_momentum.lock.json missing -- run_cycle.py must run first"
    locked = json.loads(LOCK_PATH.read_text())
    assert locked["config"] == {"lookback": 12, "skip": 1}, (
        f"lock file config changed unexpectedly: {locked['config']} -- refusing to run a "
        "diagnostic against a config this script doesn't recognize as the one from cycle1.md"
    )
    print(f"Locked config (read-only, not modified): {locked['config']}, "
          f"IS Sharpe={locked['is_metric_value']:+.3f}")

    data = _load_data()
    periods = {
        "IS_2018_2021": data.is_dates,
        "OOS_full_2022_2026": data.oos_dates,
        "OOS_H1": data.oos_h1_dates,
        "OOS_H2": data.oos_h2_dates,
    }
    print(f"OOS_H1: {data.oos_h1_dates[0].date()} .. {data.oos_h1_dates[-1].date()} "
          f"({len(data.oos_h1_dates)} dates)")
    print(f"OOS_H2: {data.oos_h2_dates[0].date()} .. {data.oos_h2_dates[-1].date()} "
          f"({len(data.oos_h2_dates)} dates)")
    print()

    curves: dict[str, list[CurvePoint]] = {}
    for name, dates in periods.items():
        curves[name] = _trace_curve(data, name, dates)
        peak = max(curves[name], key=lambda p: p.sharpe)
        shape = _classify_curve(curves[name])
        print(f"[{name}] peak at lookback={peak.lookback} (Sharpe={peak.sharpe:+.3f}, "
              f"SE~{peak.se:.3f}, n={peak.n_obs})  shape={shape}")
        at_12 = next((p for p in curves[name] if p.lookback == 12), None)
        if at_12:
            print(f"           lookback=12: Sharpe={at_12.sharpe:+.3f}  SE~{at_12.se:.3f}  "
                  f"n={at_12.n_obs}")

    print()
    locked_returns_oos = _momentum_returns(data, 12, data.oos_dates)
    locked_returns_is = _momentum_returns(data, 12, data.is_dates)
    rho1_oos, t_oos, neff_oos = _autocorr_summary(locked_returns_oos, "OOS")
    rho1_is, t_is, neff_is = _autocorr_summary(locked_returns_is, "IS")
    print(f"Locked-config (12,1) autocorrelation: IS lag-1={rho1_is:+.3f} (T={t_is:.0f}, "
          f"N_eff~{neff_is:.1f}); OOS lag-1={rho1_oos:+.3f} (T={t_oos:.0f}, N_eff~{neff_oos:.1f})")

    _write_report(curves, rho1_is, t_is, neff_is, rho1_oos, t_oos, neff_oos)
    _write_chart(curves)
    print(f"\nReport: {REPORT_PATH}")
    print(f"Chart:  {CHART_PATH}")
    print(f"Diagnostic points appended to ledger: {LEDGER_PATH} "
          "(stage=diagnostic_curve, counts_toward_n=false)")


def _write_report(
    curves: dict[str, list[CurvePoint]],
    rho1_is: float, t_is: float, neff_is: float,
    rho1_oos: float, t_oos: float, neff_oos: float,
) -> None:
    lines: list[str] = []
    lines.append("# Diagnostic: is H-MOM's lookback=12 sensitivity spike structural or noise?")
    lines.append("")
    lines.append(f"Run: {datetime.now(UTC).isoformat()}")
    lines.append(
        "**Not a new alpha-search cycle.** Every point below is logged to "
        "`experiments/grid_search_log.jsonl` with `stage=diagnostic_curve`, "
        "`counts_toward_n=false`. Cumulative MOEX-track DSR-N stays at **9** "
        "(unchanged from `reports/cycle1.md`). `experiments/cycle1_momentum.lock.json` "
        "was read, never rewritten."
    )
    lines.append("")
    lines.append("## Curves (skip=1 fixed, lookback swept 2..20 months)")
    lines.append("")
    for name, points in curves.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| lookback | Sharpe | SE(Sharpe) | n_obs |")
        lines.append("|---|---|---|---|")
        for p in points:
            marker = " **<- locked**" if p.lookback == 12 else ""
            lines.append(f"| {p.lookback} | {p.sharpe:+.3f} | {p.se:.3f} | {p.n_obs} |{marker}")
        peak = max(points, key=lambda p: p.sharpe)
        lines.append("")
        lines.append(f"Peak: lookback={peak.lookback} (Sharpe={peak.sharpe:+.3f}). "
                      f"Shape: {_classify_curve(points)}.")
        lines.append("")

    lines.append("## Autocorrelation of the locked (12, 1) return series")
    lines.append("")
    lines.append(f"IS: lag-1 autocorr={rho1_is:+.3f}, T={t_is:.0f} months, "
                  f"rough N_eff~{neff_is:.1f}")
    lines.append(f"OOS: lag-1 autocorr={rho1_oos:+.3f}, T={t_oos:.0f} months, "
                  f"rough N_eff~{neff_oos:.1f}")
    lines.append("")
    lines.append(
        "N_eff = T(1-rho)/(1+rho), the standard AR(1) effective-sample-size "
        "approximation -- indicative, not exact (monthly momentum returns are not "
        "a clean AR(1) process)."
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def _write_chart(curves: dict[str, list[CurvePoint]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (name, points) in zip(axes.flat, curves.items(), strict=True):
        lookbacks = [p.lookback for p in points]
        sharpes = [p.sharpe for p in points]
        ses = [p.se for p in points]
        ax.errorbar(lookbacks, sharpes, yerr=ses, marker="o", capsize=3, linewidth=1)
        ax.axvline(12, color="red", linestyle="--", linewidth=1, label="locked (12)")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title(name)
        ax.set_xlabel("lookback (months)")
        ax.set_ylabel("annualized Sharpe")
        ax.legend(fontsize=8)
    fig.suptitle("H-MOM lookback sensitivity, skip=1 fixed (error bars: Sharpe-estimator SE)")
    fig.tight_layout()
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_PATH, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
