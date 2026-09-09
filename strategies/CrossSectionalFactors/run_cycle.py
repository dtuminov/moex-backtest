"""Cycle 1 of the MOEX cross-sectional-factors research track: cross-sectional
momentum (H-MOM) and low-volatility (H-LOWVOL), tested separately.

Full protocol, trial budget, economic rationale, and falsification criteria
are fixed in `PREREGISTRATION.md` -- this script implements that document
exactly; if you're trying to understand *why* a threshold or a grid point is
what it is, read that file, not this one.

Pipeline (see PREREGISTRATION.md's "IS / OOS split" and "Falsification
criteria" sections for the full rationale):

1. Load the 47-ticker universe, build the return/signal panels.
2. Phase A1: raw significance test of each hypothesis's *default* config on
   the IS period (2018-2021) only. A hypothesis with p >= 0.10 is falsified
   here and its grid is never run.
3. For a hypothesis that passes: run the rest of its pre-registered IS grid,
   pick the argmax-IS-Sharpe config, and `lock_config` it -- no code past
   this point can see OOS data before the lock file exists on disk.
4. Reveal OOS (2022-2026) once, for the locked config only: walk-forward,
   parameter sensitivity, both Monte Carlo variants, Deflated Sharpe Ratio
   (cumulative MOEX-track N), and CSCV/PBO over that hypothesis's own grid
   trials (recomputed full-period, post-lock -- see PBO's docstring for why
   this is legitimate to do only now, not during selection).
5. Apply the fixed verdict criteria and report.

Run with: uv run python strategies/CrossSectionalFactors/run_cycle.py
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

from factors import (
    Direction,
    backtest_factor,
    low_vol_signal,
    mask_daily_returns_for_vol,
    momentum_signal,
)
from panel import load_close_panel, monthly_close_panel

from moex_backtest.metrics.performance import sharpe_ratio
from moex_backtest.validation import (
    deflated_sharpe_ratio,
    lock_config,
    monte_carlo_block_bootstrap,
    monte_carlo_iid,
    parameter_sensitivity,
    probability_of_backtest_overfitting,
    require_locked_config,
    significance_test,
    walk_forward_analysis,
)

HERE = Path(__file__).parent
LEDGER_PATH = HERE / "experiments" / "grid_search_log.jsonl"
REPORT_PATH = HERE / "reports" / "cycle1.md"

IS_START = pd.Timestamp("2018-01-01")
IS_END = pd.Timestamp("2021-12-31")
OOS_START = pd.Timestamp("2022-01-01")
OOS_END = pd.Timestamp("2026-09-09")

N_PER_LEG = 9  # floor(47 / 5), fixed a priori -- see PREREGISTRATION.md
COST_BPS = 15.0
N_SIMULATIONS = 2000
PERIODS_PER_YEAR = 12  # monthly returns
PRIOR_CUMULATIVE_N = 2  # moex-quant-track.md calibration run
SIGNIFICANCE_ALPHA = 0.10  # Phase A1 gate

MOMENTUM_GRID: list[tuple[int, int]] = [(3, 0), (3, 1), (6, 0), (6, 1), (12, 0), (12, 1)]
MOMENTUM_DEFAULT: tuple[int, int] = (12, 1)
LOWVOL_GRID: list[int] = [3, 6, 12]
LOWVOL_DEFAULT: int = 12


@dataclass
class TrialRecord:
    hypothesis: str
    stage: str  # "phase_a1" | "grid"
    config: dict[str, int]
    is_sharpe: float
    is_p_value: float | None
    n_observations: int


def _append_ledger(record: TrialRecord) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "hypothesis": record.hypothesis,
        "stage": record.stage,
        "config": record.config,
        "is_sharpe": record.is_sharpe,
        "is_p_value": record.is_p_value,
        "n_observations": record.n_observations,
    }
    with LEDGER_PATH.open("a") as f:
        f.write(json.dumps(payload) + "\n")


@dataclass
class Data:
    monthly_close: pd.DataFrame
    masked_daily_returns: pd.DataFrame
    formation_dates: pd.DatetimeIndex
    is_dates: pd.DatetimeIndex
    oos_dates: pd.DatetimeIndex


def _load_data() -> Data:
    close_panel = load_close_panel(cache_dir=HERE.parent.parent / "data" / "raw")
    monthly_close = monthly_close_panel(close_panel)
    daily_returns = close_panel.pct_change()
    masked_daily_returns = mask_daily_returns_for_vol(daily_returns)

    formation_dates = monthly_close.index
    is_dates = formation_dates[(formation_dates >= IS_START) & (formation_dates <= IS_END)]
    oos_dates = formation_dates[(formation_dates >= OOS_START) & (formation_dates <= OOS_END)]
    return Data(monthly_close, masked_daily_returns, formation_dates, is_dates, oos_dates)


def _momentum_returns(data: Data, lookback: int, skip: int, dates: pd.DatetimeIndex) -> pd.Series:
    signal = momentum_signal(data.monthly_close, lookback, skip)
    result = backtest_factor(
        signal, data.monthly_close, dates,
        direction="high_long", n_per_leg=N_PER_LEG, cost_bps=COST_BPS,
    )
    return result.returns


def _lowvol_returns(data: Data, lookback: int, dates: pd.DatetimeIndex) -> pd.Series:
    signal = low_vol_signal(data.masked_daily_returns, data.formation_dates, lookback)
    signal = signal.reindex(data.monthly_close.index)
    result = backtest_factor(
        signal, data.monthly_close, dates,
        direction="low_long", n_per_leg=N_PER_LEG, cost_bps=COST_BPS,
    )
    return result.returns


def _evaluate_is_trial(
    hypothesis: str, stage: str, config: dict[str, int], returns: pd.Series, *, compute_p: bool
) -> tuple[float, float | None]:
    annual_sharpe = sharpe_ratio(returns, periods_per_year=PERIODS_PER_YEAR)
    p_value: float | None = None
    if compute_p:
        sig = significance_test(
            returns,
            n_simulations=N_SIMULATIONS,
            periods_per_year=PERIODS_PER_YEAR,
            rng=np.random.default_rng(0),
        )
        p_value = sig.p_value
    _append_ledger(
        TrialRecord(hypothesis, stage, config, annual_sharpe, p_value, len(returns))
    )
    return annual_sharpe, p_value


@dataclass
class LockedHypothesis:
    name: str  # "momentum" | "lowvol"
    direction: Direction
    lock_path: Path
    lookback: int
    skip: int | None  # None for lowvol
    is_sharpe: float
    grid_configs: list[dict[str, int]]  # every grid config actually run, for PBO


def _run_phase_a1_and_grid(data: Data, trial_sharpes: list[float]) -> list[LockedHypothesis]:
    locked: list[LockedHypothesis] = []

    # --- H-MOM ---------------------------------------------------------
    mom_lookback, mom_skip = MOMENTUM_DEFAULT
    mom_returns_a1 = _momentum_returns(data, mom_lookback, mom_skip, data.is_dates)
    mom_sharpe_a1, mom_p_a1 = _evaluate_is_trial(
        "momentum",
        "phase_a1",
        {"lookback": mom_lookback, "skip": mom_skip},
        mom_returns_a1,
        compute_p=True,
    )
    trial_sharpes.append(mom_sharpe_a1)
    print(
        f"[Phase A1] H-MOM  default(lookback=12,skip=1): "
        f"IS Sharpe={mom_sharpe_a1:+.3f}  p={mom_p_a1:.4f}"
    )

    assert mom_p_a1 is not None
    if mom_p_a1 >= SIGNIFICANCE_ALPHA:
        print(f"  -> FALSIFIED at Phase A1 (p={mom_p_a1:.4f} >= {SIGNIFICANCE_ALPHA}). "
              "Grid not run.")
    else:
        print("  -> passes Phase A1, running remaining grid points.")
        best_sharpe = mom_sharpe_a1
        best_config = (mom_lookback, mom_skip)
        grid_configs_run = [{"lookback": mom_lookback, "skip": mom_skip}]
        for lookback, skip in MOMENTUM_GRID:
            if (lookback, skip) == MOMENTUM_DEFAULT:
                continue
            returns = _momentum_returns(data, lookback, skip, data.is_dates)
            sharpe, _ = _evaluate_is_trial(
                "momentum", "grid", {"lookback": lookback, "skip": skip}, returns, compute_p=False
            )
            trial_sharpes.append(sharpe)
            grid_configs_run.append({"lookback": lookback, "skip": skip})
            print(f"    grid lookback={lookback:2d} skip={skip}: IS Sharpe={sharpe:+.3f}")
            if sharpe > best_sharpe:
                best_sharpe, best_config = sharpe, (lookback, skip)

        lock_path = HERE / "experiments" / "cycle1_momentum.lock.json"
        lock_config(
            lock_path,
            cycle_name="cycle1_momentum",
            config={"lookback": best_config[0], "skip": best_config[1]},
            is_metric_name="is_sharpe",
            is_metric_value=best_sharpe,
        )
        print(
            f"  LOCKED: lookback={best_config[0]} skip={best_config[1]}  "
            f"IS Sharpe={best_sharpe:+.3f}"
        )
        locked.append(
            LockedHypothesis(
                "momentum", "high_long", lock_path, best_config[0], best_config[1], best_sharpe,
                grid_configs_run,
            )
        )

    # --- H-LOWVOL --------------------------------------------------------
    lv_lookback = LOWVOL_DEFAULT
    lv_returns_a1 = _lowvol_returns(data, lv_lookback, data.is_dates)
    lv_sharpe_a1, lv_p_a1 = _evaluate_is_trial(
        "lowvol", "phase_a1", {"lookback": lv_lookback}, lv_returns_a1, compute_p=True
    )
    trial_sharpes.append(lv_sharpe_a1)
    print(
        f"[Phase A1] H-LOWVOL default(lookback=12): "
        f"IS Sharpe={lv_sharpe_a1:+.3f}  p={lv_p_a1:.4f}"
    )

    assert lv_p_a1 is not None
    if lv_p_a1 >= SIGNIFICANCE_ALPHA:
        print(f"  -> FALSIFIED at Phase A1 (p={lv_p_a1:.4f} >= {SIGNIFICANCE_ALPHA}). "
              "Grid not run.")
    else:
        print("  -> passes Phase A1, running remaining grid points.")
        best_sharpe = lv_sharpe_a1
        best_lookback = lv_lookback
        grid_configs_run = [{"lookback": lv_lookback}]
        for lookback in LOWVOL_GRID:
            if lookback == LOWVOL_DEFAULT:
                continue
            returns = _lowvol_returns(data, lookback, data.is_dates)
            sharpe, _ = _evaluate_is_trial(
                "lowvol", "grid", {"lookback": lookback}, returns, compute_p=False
            )
            trial_sharpes.append(sharpe)
            grid_configs_run.append({"lookback": lookback})
            print(f"    grid lookback={lookback:2d}: IS Sharpe={sharpe:+.3f}")
            if sharpe > best_sharpe:
                best_sharpe, best_lookback = sharpe, lookback

        lock_path = HERE / "experiments" / "cycle1_lowvol.lock.json"
        lock_config(
            lock_path,
            cycle_name="cycle1_lowvol",
            config={"lookback": best_lookback},
            is_metric_name="is_sharpe",
            is_metric_value=best_sharpe,
        )
        print(f"  LOCKED: lookback={best_lookback}  IS Sharpe={best_sharpe:+.3f}")
        locked.append(
            LockedHypothesis(
                "lowvol", "low_long", lock_path, best_lookback, None, best_sharpe, grid_configs_run
            )
        )

    return locked


@dataclass
class FullVerdict:
    hypothesis: LockedHypothesis
    oos_returns: pd.Series
    walk_forward: Any
    sensitivity: Any
    mc_iid: Any
    mc_block: Any
    dsr: Any
    pbo: Any
    passes: bool
    reasons: list[str]


def _oos_returns_for(data: Data, h: LockedHypothesis) -> pd.Series:
    """Physically gated: raises unless this hypothesis's lock file exists."""
    require_locked_config(h.lock_path, cycle_name=f"cycle1_{h.name}")
    if h.name == "momentum":
        assert h.skip is not None
        return _momentum_returns(data, h.lookback, h.skip, data.oos_dates)
    return _lowvol_returns(data, h.lookback, data.oos_dates)


def _full_period_returns_for(data: Data, hypothesis_name: str, config: dict[str, int]) -> pd.Series:
    all_dates = data.formation_dates
    if hypothesis_name == "momentum":
        return _momentum_returns(data, config["lookback"], config["skip"], all_dates)
    return _lowvol_returns(data, config["lookback"], all_dates)


def _full_verdict(
    data: Data, h: LockedHypothesis, cumulative_n: int, trial_sharpe_variance: float
) -> FullVerdict:
    oos_returns = _oos_returns_for(data, h)
    reasons: list[str] = []

    wf = walk_forward_analysis(
        oos_returns, is_window=12, oos_window=6, periods_per_year=PERIODS_PER_YEAR
    )
    wf_ok = wf.mean_oos_sharpe > 0.0
    if not wf_ok:
        reasons.append(f"walk-forward: mean OOS Sharpe {wf.mean_oos_sharpe:+.3f} <= 0")

    def _metric_at_lookback(lookback_value: float) -> float:
        lookback_int = max(1, round(lookback_value))
        if h.name == "momentum":
            assert h.skip is not None
            returns = _momentum_returns(data, lookback_int, h.skip, data.oos_dates)
        else:
            returns = _lowvol_returns(data, lookback_int, data.oos_dates)
        return sharpe_ratio(returns, periods_per_year=PERIODS_PER_YEAR)

    sens = parameter_sensitivity("lookback", float(h.lookback), _metric_at_lookback)
    if not sens.is_plateau:
        reasons.append("sensitivity: SPIKE, not plateau")

    mc_iid = monte_carlo_iid(
        oos_returns, n_simulations=N_SIMULATIONS, periods_per_year=PERIODS_PER_YEAR,
        rng=np.random.default_rng(1),
    )
    mc_block = monte_carlo_block_bootstrap(
        oos_returns, n_simulations=N_SIMULATIONS, periods_per_year=PERIODS_PER_YEAR,
        rng=np.random.default_rng(1),
    )
    if mc_block.percentile_rank > 95.0:
        reasons.append(
            f"Monte Carlo: percentile_rank {mc_block.percentile_rank:.1f} > 95 (lucky tail)"
        )

    dsr = deflated_sharpe_ratio(
        oos_returns,
        n_trials=cumulative_n,
        trial_sharpe_variance=trial_sharpe_variance,
        periods_per_year=PERIODS_PER_YEAR,
    )
    if dsr.dsr <= 0.95:
        reasons.append(f"DSR {dsr.dsr:.4f} <= 0.95 (cumulative N={cumulative_n})")

    trial_returns_by_config = {}
    for cfg in h.grid_configs:
        label = "_".join(f"{k}{v}" for k, v in cfg.items())
        trial_returns_by_config[label] = _full_period_returns_for(data, h.name, cfg)
    trial_returns = pd.DataFrame(trial_returns_by_config).dropna(how="any")
    n_blocks = 16 if len(trial_returns) >= 32 else (8 if len(trial_returns) >= 16 else 4)
    pbo = probability_of_backtest_overfitting(
        trial_returns, n_blocks=n_blocks, periods_per_year=PERIODS_PER_YEAR
    )
    if pbo.pbo >= 0.5:
        reasons.append(f"PBO {pbo.pbo:.3f} >= 0.5")

    passes = len(reasons) == 0
    return FullVerdict(h, oos_returns, wf, sens, mc_iid, mc_block, dsr, pbo, passes, reasons)


def _write_report(
    data: Data,
    locked: list[LockedHypothesis],
    verdicts: list[FullVerdict],
    both_falsified_at_a1: bool,
    cumulative_n: int,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Cycle 1 report -- CrossSectionalFactors (momentum + low-vol)")
    lines.append("")
    lines.append(f"Run: {datetime.now(UTC).isoformat()}")
    lines.append(
        f"IS period: {IS_START.date()} .. {IS_END.date()} ({len(data.is_dates)} formation dates)"
    )
    lines.append(
        f"OOS period: {OOS_START.date()} .. {OOS_END.date()} "
        f"({len(data.oos_dates)} formation dates)"
    )
    lines.append(f"Cumulative MOEX-track DSR-N after this cycle: **{cumulative_n}**")
    lines.append("")

    if both_falsified_at_a1:
        lines.append("## Result: both hypotheses falsified at Phase A1")
        lines.append("")
        lines.append(
            "Neither H-MOM nor H-LOWVOL passed the raw significance gate "
            f"(p < {SIGNIFICANCE_ALPHA}) on their pre-registered default config, IS period only. "
            "Per PREREGISTRATION.md this is a complete, informative negative result -- no grid, "
            "lock, or OOS data was touched."
        )
    else:
        for v in verdicts:
            lines.append(f"## {v.hypothesis.name}")
            lines.append("")
            lines.append(
                f"Locked config: lookback={v.hypothesis.lookback}"
                + (f" skip={v.hypothesis.skip}" if v.hypothesis.skip is not None else "")
                + f"  (IS Sharpe={v.hypothesis.is_sharpe:+.3f})"
            )
            oos_sharpe = sharpe_ratio(v.oos_returns, periods_per_year=PERIODS_PER_YEAR)
            lines.append(f"OOS Sharpe (annualized): {oos_sharpe:+.3f}")
            wf_efficiency = (
                "N/A" if v.walk_forward.efficiency is None else f"{v.walk_forward.efficiency:.1%}"
            )
            lines.append(
                f"Walk-forward: {len(v.walk_forward.windows)} windows, "
                f"mean IS={v.walk_forward.mean_is_sharpe:+.3f}, "
                f"mean OOS={v.walk_forward.mean_oos_sharpe:+.3f}, "
                f"efficiency={wf_efficiency}"
            )
            lines.append(f"Sensitivity: {'PLATEAU' if v.sensitivity.is_plateau else 'SPIKE'}")
            lines.append(
                f"Monte Carlo: iid percentile={v.mc_iid.percentile_rank:.1f}, "
                f"block percentile={v.mc_block.percentile_rank:.1f}"
            )
            lines.append(
                f"DSR (N={cumulative_n}): {v.dsr.dsr:.4f} "
                f"(E[max SR|H0]={v.dsr.expected_max_sharpe_null:+.3f})"
            )
            lines.append(f"PBO ({v.pbo.n_trials} trials, {v.pbo.n_blocks} blocks): {v.pbo.pbo:.3f}")
            lines.append("")
            verdict_suffix = "" if v.passes else f" -- {'; '.join(v.reasons)}"
            lines.append(f"**VERDICT: {'PASS' if v.passes else 'FAIL'}**{verdict_suffix}")
            lines.append("")

    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    data = _load_data()
    print(
        f"Loaded {len(data.monthly_close.columns)} tickers, "
        f"{len(data.formation_dates)} formation dates "
        f"({data.formation_dates[0].date()} .. {data.formation_dates[-1].date()})"
    )
    print(f"IS: {len(data.is_dates)} dates, OOS: {len(data.oos_dates)} dates")
    print()

    trial_sharpes: list[float] = []
    locked = _run_phase_a1_and_grid(data, trial_sharpes)

    cumulative_n = PRIOR_CUMULATIVE_N + len(trial_sharpes)
    print(
        f"\nThis cycle ran {len(trial_sharpes)} trials. "
        f"Cumulative MOEX-track N = {cumulative_n}."
    )

    if not locked:
        print("\nBoth hypotheses falsified at Phase A1. Stopping (no OOS data touched).")
        _write_report(data, locked, [], True, cumulative_n)
        return

    trial_sharpe_variance = float(np.var(trial_sharpes, ddof=1))
    verdicts: list[FullVerdict] = []
    for h in locked:
        print(f"\n=== Full verdict pipeline: {h.name} ===")
        v = _full_verdict(data, h, cumulative_n, trial_sharpe_variance)
        verdicts.append(v)
        print(f"  walk-forward mean OOS Sharpe: {v.walk_forward.mean_oos_sharpe:+.3f}")
        print(f"  sensitivity: {'PLATEAU' if v.sensitivity.is_plateau else 'SPIKE'}")
        print(f"  Monte Carlo block percentile: {v.mc_block.percentile_rank:.1f}")
        print(f"  DSR: {v.dsr.dsr:.6f}")
        print(f"  PBO: {v.pbo.pbo:.3f}")
        verdict_suffix = "" if v.passes else f" ({'; '.join(v.reasons)})"
        print(f"  VERDICT: {'PASS' if v.passes else 'FAIL'}{verdict_suffix}")

    _write_report(data, locked, verdicts, False, cumulative_n)
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
