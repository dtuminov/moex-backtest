"""Cycle 3 of the MOEX cross-sectional-factors research track: H-ILLIQ,
a cross-sectional illiquidity-premium signal.

Full protocol, mechanism-based economic rationale, circularity check, and
falsification criteria are fixed in
`PREREGISTRATION_cycle3_illiquidity.md` -- this script implements that
document exactly.

Unlike cycles 1-2, DSR's ``n_trials`` is **not** a hardcoded constant here:
it is computed from `trial_ledger.mechanism_n("illiquidity")` (the
mechanism-scoped convention adopted in `N_CONVENTION.md` after
`reports/why_no_alpha.md`), which reads the ledger and returns 0 for this
mechanism's prior history -- this cycle's own trials are the entire N.
Captured once at the very start of `main()`, before this cycle writes any
ledger rows, so a later call to the same function mid-run cannot
accidentally include this cycle's own not-yet-selected trials.

Run with: uv run python strategies/CrossSectionalFactors/run_cycle3_illiquidity.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from factors import backtest_factor, daily_dollar_volume, illiquidity_signal
from panel import load_close_panel, load_volume_panel, monthly_close_panel
from trial_ledger import mechanism_n, project_wide_n

from moex_backtest.metrics.performance import sharpe_ratio
from moex_backtest.validation import (
    deflated_sharpe_ratio,
    lock_config,
    monte_carlo_block_bootstrap,
    parameter_sensitivity,
    probability_of_backtest_overfitting,
    require_locked_config,
    significance_test,
    walk_forward_analysis,
)

HERE = Path(__file__).parent
LEDGER_PATH = HERE / "experiments" / "grid_search_log.jsonl"
LOCK_PATH = HERE / "experiments" / "cycle3_illiquidity.lock.json"
REPORT_PATH = HERE / "reports" / "cycle3_illiquidity.md"

IS_START = pd.Timestamp("2018-01-01")
IS_END = pd.Timestamp("2021-12-31")
OOS_START = pd.Timestamp("2022-01-01")
OOS_END = pd.Timestamp("2026-09-09")

N_PER_LEG = 9  # fixed a priori, matches cycles 1-2 -- see PREREGISTRATION
COST_BPS = 15.0
ROBUSTNESS_COST_POINTS_BPS = (30.0, 50.0, 100.0)  # post-lock only, not counted toward N
N_SIMULATIONS = 2000
PERIODS_PER_YEAR = 12
SIGNIFICANCE_ALPHA = 0.10

ILLIQ_GRID: tuple[int, ...] = (3, 6, 12)
ILLIQ_DEFAULT = 12
HYPOTHESIS = "illiquidity"


@dataclass
class TrialRecord:
    stage: str  # "phase_a1" | "grid"
    config: dict[str, int]
    is_sharpe: float
    is_p_value: float | None
    n_observations: int


def _append_ledger(record: TrialRecord) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "hypothesis": HYPOTHESIS,
        "stage": record.stage,
        "counts_toward_n": True,
        "config": record.config,
        "is_sharpe": record.is_sharpe,
        "is_p_value": record.is_p_value,
        "n_observations": record.n_observations,
    }
    with LEDGER_PATH.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def _append_diagnostic(cost_bps: float, oos_sharpe: float, n_obs: int) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "hypothesis": HYPOTHESIS,
        "stage": "diagnostic_curve",
        "counts_toward_n": False,
        "note": "cost-sensitivity robustness check on the already-locked candidate; not a trial",
        "config": {"lookback": ILLIQ_DEFAULT, "cost_bps": cost_bps},
        "oos_sharpe": oos_sharpe,
        "n_observations": n_obs,
    }
    with LEDGER_PATH.open("a") as f:
        f.write(json.dumps(payload) + "\n")


@dataclass
class Data:
    monthly_close: pd.DataFrame
    dollar_volume: pd.DataFrame
    formation_dates: pd.DatetimeIndex
    is_dates: pd.DatetimeIndex
    oos_dates: pd.DatetimeIndex


def _load_data() -> Data:
    cache_dir = HERE.parent.parent / "data" / "raw"
    close_panel = load_close_panel(cache_dir=cache_dir)
    volume_panel = load_volume_panel(cache_dir=cache_dir)
    monthly_close = monthly_close_panel(close_panel)
    dollar_volume = daily_dollar_volume(close_panel, volume_panel)

    formation_dates = monthly_close.index
    is_dates = formation_dates[(formation_dates >= IS_START) & (formation_dates <= IS_END)]
    oos_dates = formation_dates[(formation_dates >= OOS_START) & (formation_dates <= OOS_END)]
    return Data(monthly_close, dollar_volume, formation_dates, is_dates, oos_dates)


def _illiq_returns(
    data: Data, lookback: int, dates: pd.DatetimeIndex, *, cost_bps: float = COST_BPS
) -> pd.Series:
    signal = illiquidity_signal(data.dollar_volume, data.formation_dates, lookback)
    signal = signal.reindex(data.monthly_close.index)
    result = backtest_factor(
        signal, data.monthly_close, dates,
        direction="low_long", n_per_leg=N_PER_LEG, cost_bps=cost_bps,
    )
    return result.returns


def _evaluate_is_trial(
    stage: str, config: dict[str, int], returns: pd.Series, *, compute_p: bool
) -> tuple[float, float | None]:
    annual_sharpe = sharpe_ratio(returns, periods_per_year=PERIODS_PER_YEAR)
    p_value: float | None = None
    if compute_p:
        sig = significance_test(
            returns, n_simulations=N_SIMULATIONS, periods_per_year=PERIODS_PER_YEAR,
            rng=np.random.default_rng(0),
        )
        p_value = sig.p_value
    _append_ledger(TrialRecord(stage, config, annual_sharpe, p_value, len(returns)))
    return annual_sharpe, p_value


def main() -> None:
    # Captured BEFORE this cycle writes any ledger rows -- see module
    # docstring. project_wide_n() likewise captured pre-cycle for the
    # "prior" figure reported alongside the post-cycle total.
    prior_mechanism_n = mechanism_n(HYPOTHESIS)
    prior_project_wide_n = project_wide_n()

    data = _load_data()
    print(f"Loaded {len(data.monthly_close.columns)} tickers, "
          f"{len(data.formation_dates)} formation dates "
          f"({data.formation_dates[0].date()} .. {data.formation_dates[-1].date()})")
    print(f"IS: {len(data.is_dates)} dates, OOS: {len(data.oos_dates)} dates")
    print(f"Prior mechanism-scoped N (illiquidity): {prior_mechanism_n}")
    print(f"Prior project-wide N: {prior_project_wide_n}\n")

    trial_sharpes: list[float] = []

    # --- Phase A1 --------------------------------------------------------
    is_returns_a1 = _illiq_returns(data, ILLIQ_DEFAULT, data.is_dates)
    sharpe_a1, p_a1 = _evaluate_is_trial(
        "phase_a1", {"lookback": ILLIQ_DEFAULT}, is_returns_a1, compute_p=True
    )
    trial_sharpes.append(sharpe_a1)
    assert p_a1 is not None
    print(f"[Phase A1] H-ILLIQ default(lookback={ILLIQ_DEFAULT}): "
          f"IS Sharpe={sharpe_a1:+.3f}  p={p_a1:.4f}  (n={len(is_returns_a1)})")

    mechanism_n_now = prior_mechanism_n + len(trial_sharpes)
    if p_a1 >= SIGNIFICANCE_ALPHA:
        print(f"\n-> FALSIFIED at Phase A1 (p={p_a1:.4f} >= {SIGNIFICANCE_ALPHA}). "
              "Grid not run. No lock, no OOS access.")
        _write_report(
            falsified_at_a1=True, mechanism_n_final=mechanism_n_now,
            project_wide_n_final=prior_project_wide_n + len(trial_sharpes),
            prior_mechanism_n=prior_mechanism_n, prior_project_wide_n=prior_project_wide_n,
            a1_sharpe=sharpe_a1, a1_p=p_a1,
        )
        print(f"Report: {REPORT_PATH}")
        return

    print("-> passes Phase A1, running remaining grid points.")
    best_sharpe = sharpe_a1
    best_lookback = ILLIQ_DEFAULT
    grid_configs_run = [{"lookback": ILLIQ_DEFAULT}]
    for lookback in ILLIQ_GRID:
        if lookback == ILLIQ_DEFAULT:
            continue
        returns = _illiq_returns(data, lookback, data.is_dates)
        sharpe, _ = _evaluate_is_trial("grid", {"lookback": lookback}, returns, compute_p=False)
        trial_sharpes.append(sharpe)
        grid_configs_run.append({"lookback": lookback})
        print(f"    grid lookback={lookback:2d}: IS Sharpe={sharpe:+.3f}")
        if sharpe > best_sharpe:
            best_sharpe, best_lookback = sharpe, lookback

    mechanism_n_final = prior_mechanism_n + len(trial_sharpes)
    project_wide_n_final = prior_project_wide_n + len(trial_sharpes)
    print(f"\nThis cycle ran {len(trial_sharpes)} trials.")
    print(f"Mechanism-scoped N (illiquidity) after this cycle: {mechanism_n_final}")
    print(f"Project-wide N after this cycle: {project_wide_n_final}")

    lock_config(
        LOCK_PATH,
        cycle_name="cycle3_illiquidity",
        config={"lookback": best_lookback},
        is_metric_name="is_sharpe",
        is_metric_value=best_sharpe,
    )
    print(f"LOCKED: lookback={best_lookback}  IS Sharpe={best_sharpe:+.3f}")

    # --- Reveal OOS, gated by the lock ------------------------------------
    require_locked_config(LOCK_PATH, cycle_name="cycle3_illiquidity")
    oos_returns = _illiq_returns(data, best_lookback, data.oos_dates)
    oos_sharpe = sharpe_ratio(oos_returns, periods_per_year=PERIODS_PER_YEAR)
    print(f"\nOOS Sharpe (full period, annualized): {oos_sharpe:+.3f}  (n={len(oos_returns)})")

    wf = walk_forward_analysis(
        oos_returns, is_window=12, oos_window=6, periods_per_year=PERIODS_PER_YEAR
    )
    print(f"Walk-forward: {len(wf.windows)} windows, mean IS={wf.mean_is_sharpe:+.3f}, "
          f"mean OOS={wf.mean_oos_sharpe:+.3f}")

    def _metric_at_lookback(lookback_value: float) -> float:
        lookback_int = max(1, round(lookback_value))
        returns = _illiq_returns(data, lookback_int, data.oos_dates)
        return sharpe_ratio(returns, periods_per_year=PERIODS_PER_YEAR)

    sens = parameter_sensitivity("lookback", float(best_lookback), _metric_at_lookback)
    print(f"Sensitivity: {'PLATEAU' if sens.is_plateau else 'SPIKE'}")

    mc_block = monte_carlo_block_bootstrap(
        oos_returns, n_simulations=N_SIMULATIONS, periods_per_year=PERIODS_PER_YEAR,
        rng=np.random.default_rng(1),
    )
    print(f"Monte Carlo (block bootstrap): percentile_rank={mc_block.percentile_rank:.1f}")

    trial_sharpe_variance = float(np.var(trial_sharpes, ddof=1))
    dsr = deflated_sharpe_ratio(
        oos_returns, n_trials=mechanism_n_final, trial_sharpe_variance=trial_sharpe_variance,
        periods_per_year=PERIODS_PER_YEAR,
    )
    print(f"DSR (mechanism-scoped N={mechanism_n_final}): {dsr.dsr:.6f}  "
          f"(E[max SR|H0]={dsr.expected_max_sharpe_null:+.3f})")

    trial_returns_by_config = {}
    for cfg in grid_configs_run:
        label = f"lb{cfg['lookback']}"
        all_dates = data.formation_dates
        trial_returns_by_config[label] = _illiq_returns(data, cfg["lookback"], all_dates)
    trial_returns = pd.DataFrame(trial_returns_by_config).dropna(how="any")
    n_blocks = 16 if len(trial_returns) >= 32 else (8 if len(trial_returns) >= 16 else 4)
    pbo = probability_of_backtest_overfitting(
        trial_returns, n_blocks=n_blocks, periods_per_year=PERIODS_PER_YEAR
    )
    print(f"PBO ({len(grid_configs_run)} trials, {n_blocks} blocks): {pbo.pbo:.3f}")

    reasons: list[str] = []
    if not (wf.mean_oos_sharpe > 0.0):
        reasons.append(f"walk-forward: mean OOS Sharpe {wf.mean_oos_sharpe:+.3f} <= 0")
    if not sens.is_plateau:
        reasons.append("sensitivity: SPIKE, not plateau")
    if mc_block.percentile_rank > 95.0:
        reasons.append(f"Monte Carlo: percentile_rank {mc_block.percentile_rank:.1f} > 95")
    if dsr.dsr <= 0.95:
        reasons.append(f"DSR {dsr.dsr:.4f} <= 0.95 (mechanism-scoped N={mechanism_n_final})")
    if pbo.pbo >= 0.5:
        reasons.append(f"PBO {pbo.pbo:.3f} >= 0.5")
    passes = len(reasons) == 0
    verdict_suffix = "" if passes else f" ({'; '.join(reasons)})"
    print(f"\nVERDICT: {'PASS' if passes else 'FAIL'}{verdict_suffix}")

    # --- Post-lock robustness: cost sensitivity (does not count toward N) --
    print("\nPost-lock cost-sensitivity robustness check (not counted toward N):")
    cost_sensitivity: list[tuple[float, float]] = [(COST_BPS, oos_sharpe)]
    for cost in ROBUSTNESS_COST_POINTS_BPS:
        returns = _illiq_returns(data, best_lookback, data.oos_dates, cost_bps=cost)
        sharpe = sharpe_ratio(returns, periods_per_year=PERIODS_PER_YEAR)
        cost_sensitivity.append((cost, sharpe))
        _append_diagnostic(cost, sharpe, len(returns))
        print(f"  cost={cost:5.1f}bps: OOS Sharpe={sharpe:+.3f}")

    _write_report(
        falsified_at_a1=False,
        mechanism_n_final=mechanism_n_final, project_wide_n_final=project_wide_n_final,
        prior_mechanism_n=prior_mechanism_n, prior_project_wide_n=prior_project_wide_n,
        a1_sharpe=sharpe_a1, a1_p=p_a1,
        best_lookback=best_lookback, best_sharpe=best_sharpe,
        grid_configs_run=grid_configs_run, oos_sharpe=oos_sharpe, wf=wf, sens=sens,
        mc_block=mc_block, dsr=dsr, pbo=pbo, passes=passes, reasons=reasons,
        cost_sensitivity=cost_sensitivity,
    )
    print(f"\nReport: {REPORT_PATH}")


def _write_report(
    *,
    falsified_at_a1: bool,
    mechanism_n_final: int,
    project_wide_n_final: int,
    prior_mechanism_n: int,
    prior_project_wide_n: int,
    a1_sharpe: float,
    a1_p: float,
    best_lookback: int | None = None,
    best_sharpe: float | None = None,
    grid_configs_run: list[dict[str, int]] | None = None,
    oos_sharpe: float | None = None,
    wf: Any = None,
    sens: Any = None,
    mc_block: Any = None,
    dsr: Any = None,
    pbo: Any = None,
    passes: bool | None = None,
    reasons: list[str] | None = None,
    cost_sensitivity: list[tuple[float, float]] | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# Cycle 3 report -- H-ILLIQ (cross-sectional illiquidity premium)")
    lines.append("")
    lines.append(f"Run: {datetime.now(UTC).isoformat()}")
    lines.append(
        f"IS period: {IS_START.date()} .. {IS_END.date()}; "
        f"OOS period: {OOS_START.date()} .. {OOS_END.date()}"
    )
    lines.append(
        "Mechanism: illiquidity (compensation for trading friction in a "
        "retail-dominated order book) -- new mechanism, registered in "
        "`trial_ledger.MECHANISM_BY_HYPOTHESIS` before this cycle ran. "
        "See `PREREGISTRATION_cycle3_illiquidity.md`."
    )
    lines.append("")
    lines.append(
        f"**Mechanism-scoped N (illiquidity)**: {prior_mechanism_n} prior -> "
        f"**{mechanism_n_final}** after this cycle (feeds `deflated_sharpe_ratio`)."
    )
    lines.append(
        f"**Project-wide N**: {prior_project_wide_n} prior -> **{project_wide_n_final}** "
        "after this cycle (context only, per `N_CONVENTION.md`, never substituted "
        "into the DSR formula)."
    )
    lines.append("")
    lines.append(
        f"Phase A1 (default lookback={ILLIQ_DEFAULT}): IS Sharpe={a1_sharpe:+.3f}, p={a1_p:.4f}"
    )
    lines.append("")

    if falsified_at_a1:
        lines.append(f"## Result: FALSIFIED at Phase A1 (p={a1_p:.4f} >= {SIGNIFICANCE_ALPHA})")
        lines.append("")
        lines.append(
            "Grid not run. No lock, no OOS data touched. Complete, informative negative "
            "result -- mechanism-scoped N cost is exactly 1."
        )
    else:
        assert best_lookback is not None and best_sharpe is not None
        assert grid_configs_run is not None and oos_sharpe is not None and wf is not None
        assert sens is not None and mc_block is not None and dsr is not None and pbo is not None
        assert passes is not None and reasons is not None and cost_sensitivity is not None

        lines.append(f"## Grid ({len(grid_configs_run)} trials)")
        lines.append("")
        lines.append(f"Locked config: lookback={best_lookback}  (IS Sharpe={best_sharpe:+.3f})")
        lines.append("")
        lines.append(f"OOS Sharpe (annualized, 15bps): {oos_sharpe:+.3f}")
        wf_efficiency = (
            "N/A" if wf.efficiency is None else f"{wf.efficiency:.1%}"
        )
        lines.append(
            f"Walk-forward: {len(wf.windows)} windows, mean IS={wf.mean_is_sharpe:+.3f}, "
            f"mean OOS={wf.mean_oos_sharpe:+.3f}, efficiency={wf_efficiency}"
        )
        lines.append(f"Sensitivity: {'PLATEAU' if sens.is_plateau else 'SPIKE'}")
        lines.append(f"Monte Carlo (block bootstrap) percentile: {mc_block.percentile_rank:.1f}")
        lines.append(
            f"DSR (mechanism-scoped N={mechanism_n_final}): {dsr.dsr:.4f} "
            f"(E[max SR|H0]={dsr.expected_max_sharpe_null:+.3f})"
        )
        lines.append(f"PBO ({len(grid_configs_run)} trials, {pbo.n_blocks} blocks): {pbo.pbo:.3f}")
        lines.append("")
        suffix = "" if passes else f" -- {'; '.join(reasons)}"
        lines.append(f"**VERDICT: {'PASS' if passes else 'FAIL'}**{suffix}")
        lines.append("")
        lines.append(
            "## Post-lock cost-sensitivity robustness (not counted toward N)"
        )
        lines.append("")
        lines.append(
            "Pre-registered concern: 15bps is very likely an understatement for the "
            "illiquid (long) leg specifically -- the mechanism itself is that thinly-traded "
            "names cost more to trade. Does not change the verdict above; shows how much of "
            "any edge is cost-assumption-dependent."
        )
        lines.append("")
        lines.append("| cost (bps, one-way) | OOS Sharpe |")
        lines.append("|---|---|")
        for cost, sharpe in cost_sensitivity:
            label = f"{cost:.0f}" + (" (main verdict)" if cost == COST_BPS else "")
            lines.append(f"| {label} | {sharpe:+.3f} |")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
