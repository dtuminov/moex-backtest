"""Cycle 2 of the MOEX cross-sectional-factors research track:
H-MOM-ENSEMBLE, a multi-horizon composite momentum signal.

Full protocol, rationale (including the circularity caveat re: the
diagnostic that motivated this cycle), and falsification criteria are fixed
in `PREREGISTRATION_cycle2_momentum_ensemble.md` -- this script implements
that document exactly.

Unlike cycle 1 (`run_cycle.py`), there is **no grid**: the composite
signal's horizon set {3, 6, 9, 12} months is fixed by an external
literature convention, not searched. One trial: Phase A1 (raw IS
significance) doubles as the only candidate evaluation -- if it passes,
that same evaluation is what gets locked, no further optimization.

Run with: uv run python strategies/CrossSectionalFactors/run_cycle2_momentum_ensemble.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from factors import backtest_factor, composite_momentum_signal
from panel import load_close_panel, monthly_close_panel

from moex_backtest.metrics.performance import sharpe_ratio
from moex_backtest.validation import (
    deflated_sharpe_ratio,
    lock_config,
    monte_carlo_block_bootstrap,
    require_locked_config,
    significance_test,
    walk_forward_analysis,
)

HERE = Path(__file__).parent
LEDGER_PATH = HERE / "experiments" / "grid_search_log.jsonl"
LOCK_PATH = HERE / "experiments" / "cycle2_momentum_ensemble.lock.json"
REPORT_PATH = HERE / "reports" / "cycle2_momentum_ensemble.md"

IS_START = pd.Timestamp("2018-01-01")
IS_END = pd.Timestamp("2021-12-31")
OOS_START = pd.Timestamp("2022-01-01")
OOS_END = pd.Timestamp("2026-09-09")

N_PER_LEG = 9
COST_BPS = 15.0
N_SIMULATIONS = 2000
PERIODS_PER_YEAR = 12
PRIOR_CUMULATIVE_N = 9  # from reports/cycle1.md
SIGNIFICANCE_ALPHA = 0.10

HORIZONS: tuple[int, ...] = (3, 6, 9, 12)
SKIP = 1
SHIFT_POINTS = (-2, -1, 1, 2)


def _append_trial(is_sharpe: float, p_value: float, n_obs: int) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "hypothesis": "momentum_ensemble",
        "stage": "phase_a1",
        "counts_toward_n": True,
        "config": {"horizons": list(HORIZONS), "skip": SKIP},
        "is_sharpe": is_sharpe,
        "is_p_value": p_value,
        "n_observations": n_obs,
    }
    with LEDGER_PATH.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def _append_diagnostic(shift: int, oos_sharpe: float, n_obs: int) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "hypothesis": "momentum_ensemble",
        "stage": "diagnostic_curve",
        "counts_toward_n": False,
        "note": "horizon-set-shift robustness check on the already-locked composite; not a trial",
        "config": {"horizons": [j + shift for j in HORIZONS], "skip": SKIP, "shift": shift},
        "oos_sharpe": oos_sharpe,
        "n_observations": n_obs,
    }
    with LEDGER_PATH.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def _prior_trial_sharpes() -> list[float]:
    """Reads every counted (stage in {phase_a1, grid}) trial's IS Sharpe
    already in the ledger -- cycle 1's 7 trials -- for the DSR variance
    proxy (same convention cycle 1 itself used: variance from whatever
    actual trial Sharpes are on hand, not the full historical N).
    """
    if not LEDGER_PATH.exists():
        return []
    sharpes: list[float] = []
    for line in LEDGER_PATH.read_text().splitlines():
        row = json.loads(line)
        is_counted_stage = row.get("stage") in ("phase_a1", "grid")
        is_other_hypothesis = row.get("hypothesis") != "momentum_ensemble"
        if is_counted_stage and is_other_hypothesis:
            sharpes.append(float(row["is_sharpe"]))
    return sharpes


@dataclass
class Data:
    monthly_close: pd.DataFrame
    formation_dates: pd.DatetimeIndex
    is_dates: pd.DatetimeIndex
    oos_dates: pd.DatetimeIndex


def _load_data() -> Data:
    close_panel = load_close_panel(cache_dir=HERE.parent.parent / "data" / "raw")
    monthly_close = monthly_close_panel(close_panel)
    formation_dates = monthly_close.index
    is_dates = formation_dates[(formation_dates >= IS_START) & (formation_dates <= IS_END)]
    oos_dates = formation_dates[(formation_dates >= OOS_START) & (formation_dates <= OOS_END)]
    return Data(monthly_close, formation_dates, is_dates, oos_dates)


def _ensemble_returns(
    data: Data, horizons: tuple[int, ...], dates: pd.DatetimeIndex
) -> pd.Series:
    signal = composite_momentum_signal(data.monthly_close, horizons, SKIP)
    result = backtest_factor(
        signal, data.monthly_close, dates,
        direction="high_long", n_per_leg=N_PER_LEG, cost_bps=COST_BPS,
    )
    return result.returns


def main() -> None:
    data = _load_data()
    print(f"Loaded {len(data.monthly_close.columns)} tickers, "
          f"{len(data.formation_dates)} formation dates")
    print(f"IS: {len(data.is_dates)} dates, OOS: {len(data.oos_dates)} dates")
    print(f"Composite horizons: {HORIZONS}, skip={SKIP}\n")

    # --- Phase A1 (this cycle's only trial) --------------------------------
    is_returns = _ensemble_returns(data, HORIZONS, data.is_dates)
    is_sharpe = sharpe_ratio(is_returns, periods_per_year=PERIODS_PER_YEAR)
    sig = significance_test(
        is_returns, n_simulations=N_SIMULATIONS, periods_per_year=PERIODS_PER_YEAR,
        rng=np.random.default_rng(0),
    )
    _append_trial(is_sharpe, sig.p_value, len(is_returns))
    cumulative_n = PRIOR_CUMULATIVE_N + 1
    print(f"[Phase A1] H-MOM-ENSEMBLE: IS Sharpe={is_sharpe:+.3f}  p={sig.p_value:.4f}  "
          f"(n={len(is_returns)})")
    print(f"Cumulative MOEX-track N = {cumulative_n} (this cycle's only trial).")

    if sig.p_value >= SIGNIFICANCE_ALPHA:
        print(f"\n-> FALSIFIED at Phase A1 (p={sig.p_value:.4f} >= {SIGNIFICANCE_ALPHA}). "
              "Stopping. No lock, no OOS access.")
        _write_report(cumulative_n, is_sharpe, sig.p_value, falsified=True)
        return

    print("-> passes Phase A1. No grid to run (one-trial design). Locking.")
    lock_config(
        LOCK_PATH,
        cycle_name="cycle2_momentum_ensemble",
        config={"horizons": list(HORIZONS), "skip": SKIP},
        is_metric_name="is_sharpe",
        is_metric_value=is_sharpe,
    )

    # --- Reveal OOS, gated by the lock --------------------------------------
    require_locked_config(LOCK_PATH, cycle_name="cycle2_momentum_ensemble")
    oos_returns = _ensemble_returns(data, HORIZONS, data.oos_dates)
    oos_sharpe = sharpe_ratio(oos_returns, periods_per_year=PERIODS_PER_YEAR)
    print(f"\nOOS Sharpe (full period, annualized): {oos_sharpe:+.3f}  (n={len(oos_returns)})")

    wf = walk_forward_analysis(
        oos_returns, is_window=12, oos_window=6, periods_per_year=PERIODS_PER_YEAR
    )
    print(f"Walk-forward: {len(wf.windows)} windows, mean IS={wf.mean_is_sharpe:+.3f}, "
          f"mean OOS={wf.mean_oos_sharpe:+.3f}")

    shift_results: list[tuple[int, float, int]] = []
    skipped_shifts: list[int] = []
    for shift in SHIFT_POINTS:
        shifted_horizons = tuple(j + shift for j in HORIZONS)
        if min(shifted_horizons) <= SKIP:
            # Structurally infeasible (not a data-driven choice): shifting
            # the shortest horizon (3 months) down by 2 collides with
            # skip=1 (momentum_signal requires lookback > skip). Documented
            # in the report rather than silently dropped or crashing.
            skipped_shifts.append(shift)
            print(f"  shift={shift:+d} -> horizons={shifted_horizons}: "
                  f"SKIPPED (shortest horizon <= skip={SKIP})")
            continue
        returns = _ensemble_returns(data, shifted_horizons, data.oos_dates)
        shifted_sharpe = sharpe_ratio(returns, periods_per_year=PERIODS_PER_YEAR)
        shift_results.append((shift, shifted_sharpe, len(returns)))
        _append_diagnostic(shift, shifted_sharpe, len(returns))
        print(
            f"  shift={shift:+d} -> horizons={shifted_horizons}: "
            f"OOS Sharpe={shifted_sharpe:+.3f}"
        )
    shift_robust = all(s > 0.0 for _, s, _ in shift_results) and oos_sharpe > 0.0

    mc_block = monte_carlo_block_bootstrap(
        oos_returns, n_simulations=N_SIMULATIONS, periods_per_year=PERIODS_PER_YEAR,
        rng=np.random.default_rng(1),
    )
    print(f"Monte Carlo (block bootstrap): percentile_rank={mc_block.percentile_rank:.1f}")

    prior_sharpes = _prior_trial_sharpes()
    all_trial_sharpes = [*prior_sharpes, is_sharpe]
    trial_sharpe_variance = float(np.var(all_trial_sharpes, ddof=1))
    dsr = deflated_sharpe_ratio(
        oos_returns, n_trials=cumulative_n, trial_sharpe_variance=trial_sharpe_variance,
        periods_per_year=PERIODS_PER_YEAR,
    )
    print(
        f"DSR (N={cumulative_n}): {dsr.dsr:.6f}  "
        f"(E[max SR|H0]={dsr.expected_max_sharpe_null:+.3f})"
    )

    reasons: list[str] = []
    if not (wf.mean_oos_sharpe > 0.0):
        reasons.append(f"walk-forward: mean OOS Sharpe {wf.mean_oos_sharpe:+.3f} <= 0")
    if not shift_robust:
        reasons.append("horizon-shift robustness: at least one shift (or the base) had Sharpe <= 0")
    if mc_block.percentile_rank > 95.0:
        reasons.append(f"Monte Carlo: percentile_rank {mc_block.percentile_rank:.1f} > 95")
    if dsr.dsr <= 0.95:
        reasons.append(f"DSR {dsr.dsr:.4f} <= 0.95 (cumulative N={cumulative_n})")
    passes = len(reasons) == 0

    verdict_suffix = "" if passes else f" ({'; '.join(reasons)})"
    print(f"\nVERDICT: {'PASS' if passes else 'FAIL'}{verdict_suffix}")
    _write_report(
        cumulative_n, is_sharpe, sig.p_value, falsified=False,
        oos_sharpe=oos_sharpe, wf=wf, shift_results=shift_results,
        skipped_shifts=skipped_shifts, mc_block=mc_block, dsr=dsr,
        passes=passes, reasons=reasons,
    )
    print(f"\nReport: {REPORT_PATH}")


def _write_report(
    cumulative_n: int,
    is_sharpe: float,
    p_value: float,
    *,
    falsified: bool,
    oos_sharpe: float | None = None,
    wf: object | None = None,
    shift_results: list[tuple[int, float, int]] | None = None,
    skipped_shifts: list[int] | None = None,
    mc_block: object | None = None,
    dsr: object | None = None,
    passes: bool | None = None,
    reasons: list[str] | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# Cycle 2 report -- H-MOM-ENSEMBLE (multi-horizon composite momentum)")
    lines.append("")
    lines.append(f"Run: {datetime.now(UTC).isoformat()}")
    lines.append(
        f"Horizons: {HORIZONS}, skip={SKIP} (fixed, no grid -- "
        "see PREREGISTRATION_cycle2_momentum_ensemble.md)"
    )
    lines.append(f"Cumulative MOEX-track DSR-N after this cycle: **{cumulative_n}**")
    lines.append("")
    lines.append(f"Phase A1 / only trial: IS Sharpe={is_sharpe:+.3f}, p={p_value:.4f}")
    lines.append("")

    if falsified:
        lines.append(f"## Result: FALSIFIED at Phase A1 (p={p_value:.4f} >= {SIGNIFICANCE_ALPHA})")
        lines.append("")
        lines.append("No lock, no OOS data touched. Complete, informative negative result.")
    else:
        assert oos_sharpe is not None and wf is not None and shift_results is not None
        assert mc_block is not None and dsr is not None and passes is not None
        assert reasons is not None
        lines.append(f"OOS Sharpe (full period, annualized): {oos_sharpe:+.3f}")
        lines.append(
            f"Walk-forward: {len(wf.windows)} windows, mean IS={wf.mean_is_sharpe:+.3f}, "  # type: ignore[attr-defined]
            f"mean OOS={wf.mean_oos_sharpe:+.3f}"  # type: ignore[attr-defined]
        )
        lines.append("")
        lines.append("Horizon-shift robustness (post-lock, does not count toward N):")
        lines.append("")
        lines.append("| shift | horizons | OOS Sharpe |")
        lines.append("|---|---|---|")
        lines.append(f"| 0 (locked) | {HORIZONS} | {oos_sharpe:+.3f} |")
        for shift, sharpe, _ in shift_results:
            shifted = tuple(j + shift for j in HORIZONS)
            lines.append(f"| {shift:+d} | {shifted} | {sharpe:+.3f} |")
        if skipped_shifts:
            lines.append("")
            lines.append(
                f"Shifts {skipped_shifts} not attempted: shortest horizon would fall "
                f"to <= skip={SKIP} (structurally infeasible, not a data-driven exclusion)."
            )
        lines.append("")
        lines.append(f"Monte Carlo (block bootstrap) percentile: {mc_block.percentile_rank:.1f}")  # type: ignore[attr-defined]
        lines.append(
            f"DSR (N={cumulative_n}): {dsr.dsr:.4f} "  # type: ignore[attr-defined]
            f"(E[max SR|H0]={dsr.expected_max_sharpe_null:+.3f})"  # type: ignore[attr-defined]
        )
        lines.append("PBO: out of scope this cycle (single-trial design, no competing candidates)")
        lines.append("")
        suffix = "" if passes else f" -- {'; '.join(reasons)}"
        lines.append(f"**VERDICT: {'PASS' if passes else 'FAIL'}**{suffix}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
