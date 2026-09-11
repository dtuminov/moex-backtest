"""Statistical power audit of the full MOEX-track validation gate.

**Why this exists**: after 8+ crypto/Jesse cycles and 2 MOEX cycles, every
single candidate has failed the gate (significance -> walk-forward ->
sensitivity -> Monte Carlo -> DSR -> PBO, all-or-nothing). That is
consistent with "there is no edge to find" -- but it is equally consistent
with "the gate rejects real edges almost always, and we have never measured
its Type II error to tell the two apart". This script measures both error
rates honestly, by simulation, using the gate exactly as configured in
`strategies/CrossSectionalFactors/run_cycle.py` (cycle 1).

**Design**: rather than assume a distribution for monthly factor returns,
this reuses the REAL empirical residual shape, autocorrelation and
cross-config correlation of our own momentum grid (full-period 2018-2026,
demeaned -- see `_load_real_templates`, and note the demeaning was MISSING
until 10.09.2026, which invalidated this report's first release) as a
multivariate noise template, and adds a *known* drift on top
to control the true annualized Sharpe precisely. Many independent
realizations are drawn via a multivariate stationary block bootstrap (the
same Politis & Romano 1994 algorithm `validation._bootstrap` already uses,
extended here to resample multiple correlated columns with one shared
resampling path per replicate, so their real cross-correlation survives
into every synthetic draw) -- this keeps the simulation honest about T,
serial correlation, fat tails, and inter-config correlation, all of which
are real properties of this specific dataset/construction, not assumptions.

Every replicate runs the *entire* real gate (via the real, already-fixed
`moex_backtest.validation` functions -- the same code cycle 1 and cycle 2
used, not a reimplementation) on the locked (lookback=12, skip=1) config,
using:
  - the same 7-point sensitivity grid cycle 1 actually evaluated
    (lookback in {8,10,11,12,13,14,16}, skip=1, the +-10/20/30% grid around
    base=12 -- rounds to exactly this integer set),
  - the same 6-point PBO grid cycle 1 actually evaluated
    ((lookback, skip) in {(3,0),(3,1),(6,0),(6,1),(12,0),(12,1)}),
  - the real N=9, real trial_sharpe_variance from
    `strategies/CrossSectionalFactors/experiments/grid_search_log.jsonl`
    (cycle 1's actual 6 momentum-grid trial Sharpes) for DSR.

Run with: uv run python scripts/gate_power_analysis.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_STRATEGY_DIR = Path(__file__).parent.parent / "strategies" / "CrossSectionalFactors"
sys.path.insert(0, str(_STRATEGY_DIR))

from factors import backtest_factor, momentum_signal  # noqa: E402
from panel import load_close_panel, monthly_close_panel  # noqa: E402

from moex_backtest.metrics.performance import sharpe_ratio  # noqa: E402
from moex_backtest.validation import (  # noqa: E402
    deflated_sharpe_ratio,
    monte_carlo_block_bootstrap,
    parameter_sensitivity,
    probability_of_backtest_overfitting,
    significance_test,
    walk_forward_analysis,
)
from moex_backtest.validation._bootstrap import default_expected_block_length  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent
_LEDGER_DIR = REPO_ROOT / "strategies" / "CrossSectionalFactors" / "experiments"
LEDGER_PATH = _LEDGER_DIR / "grid_search_log.jsonl"
REPORT_PATH = REPO_ROOT / "reports" / "gate_power_analysis.md"

IS_START = pd.Timestamp("2018-01-01")
IS_END = pd.Timestamp("2021-12-31")
OOS_START = pd.Timestamp("2022-01-01")
OOS_END = pd.Timestamp("2026-09-09")

N_PER_LEG = 9
COST_BPS = 15.0
PERIODS_PER_YEAR = 12
SIGNIFICANCE_ALPHA = 0.10

SENSITIVITY_LOOKBACKS = (8, 10, 11, 12, 13, 14, 16)  # cycle 1's actual +-10/20/30% grid, skip=1
PBO_CONFIGS: tuple[tuple[int, int], ...] = ((3, 0), (3, 1), (6, 0), (6, 1), (12, 0), (12, 1))
LOCKED = (12, 1)

N_TRIALS_PRODUCTION = 9
N_REPLICATES = 250
N_BOOT_SIMS = 500  # inner bootstrap sims for significance_test/monte_carlo_block_bootstrap
PBO_N_BLOCKS = 8  # reduced from production's 16 for simulation speed (C(8,4)=70 vs C(16,8)=12870)
SR_GRID = (0.0, 0.5, 0.75, 1.0, 1.5, 2.0)
SEED = 42


def _config_label(lookback: int, skip: int) -> str:
    return f"{lookback}_{skip}"


def _load_real_templates() -> pd.DataFrame:
    """Real, full-period (2018-2026) momentum returns for every config
    needed by the sensitivity and PBO grids, deduplicated, common index.
    """
    close_panel = load_close_panel(cache_dir=REPO_ROOT / "data" / "raw")
    monthly_close = monthly_close_panel(close_panel)
    all_dates = monthly_close.index

    needed: set[tuple[int, int]] = {(lb, 1) for lb in SENSITIVITY_LOOKBACKS} | set(PBO_CONFIGS)
    series: dict[str, pd.Series] = {}
    for lookback, skip in sorted(needed):
        signal = momentum_signal(monthly_close, lookback, skip)
        result = backtest_factor(
            signal, monthly_close, all_dates,
            direction="high_long", n_per_leg=N_PER_LEG, cost_bps=COST_BPS,
        )
        series[_config_label(lookback, skip)] = result.returns

    frame = pd.DataFrame(series).dropna(how="any")  # common valid index across every config
    # Demean each column by its own full-period mean, so the template carries
    # only the empirical *shape* (autocorrelation, fat tails, cross-config
    # correlation) and the drift added in `_run_one_replicate` is the ONLY
    # source of true edge. Missing until 10.09.2026: without it every "true
    # SR = X" label silently meant X + that column's real historical Sharpe
    # (locked 12_1: +0.92), so the "SR = 0" row measured power at an
    # effective Sharpe near 0.9 rather than Type I error. See
    # `reports/why_no_alpha.md` A0 for the corrected numbers this restores.
    return frame - frame.mean()


def _real_trial_sharpe_variance() -> float:
    """Reads the ledger for the SAME `trial_sharpe_variance` cycle 1's
    `run_cycle.py` actually fed into `deflated_sharpe_ratio` -- which was
    computed from ALL 7 trials run that cycle (6 momentum-grid + 1 lowvol
    Phase A1), not momentum's 6 alone (`run_cycle.py: main()`:
    `trial_sharpes` accumulates across both hypotheses before either
    hypothesis's `_full_verdict` is called). Reproduces `reports/cycle1.md`'s
    reported ``E[max SR|H0]=+1.138`` at N=9 exactly -- verified below by the
    caller. Fixed across all replicates (this audit measures the gate AS
    CONFIGURED, not a re-simulated variance).
    """
    # Restricted to cycle 1's own hypotheses (momentum, lowvol) -- excludes
    # cycle 2's later momentum_ensemble trial, which postdates and is not
    # part of the N=9 cycle 1 actually computed DSR against.
    sharpes = []
    for line in LEDGER_PATH.read_text().splitlines():
        row = json.loads(line)
        if row.get("stage") in ("phase_a1", "grid") and row.get("hypothesis") in (
            "momentum", "lowvol",
        ):
            sharpes.append(float(row["is_sharpe"]))
    if len(sharpes) != 7:
        raise RuntimeError(
            f"expected exactly 7 cycle-1 trial Sharpes in ledger, found {len(sharpes)}"
        )
    return float(np.var(sharpes, ddof=1))


def _stationary_bootstrap_index(
    t: int, expected_block_length: float, rng: np.random.Generator
) -> np.ndarray:
    """Single-realization index generator, same recurrence as
    `validation._bootstrap.stationary_bootstrap_resample`'s per-simulation
    body -- extracted so the SAME index can be applied to every column of a
    multi-column frame at once, preserving real cross-column correlation
    (that function only resamples a single 1-D array).
    """
    p = 1.0 / expected_block_length
    new_block = rng.random(t) < p
    new_block[0] = True
    segment_id = np.cumsum(new_block) - 1
    block_starts_output_pos = np.flatnonzero(new_block)
    segment_start_output_pos = block_starts_output_pos[segment_id]
    offset_within_segment = np.arange(t) - segment_start_output_pos
    random_block_origin = rng.integers(0, t, size=len(block_starts_output_pos))
    return (random_block_origin[segment_id] + offset_within_segment) % t


@dataclass
class ReplicateResult:
    pass_significance: bool
    pass_walk_forward: bool
    pass_sensitivity: bool
    pass_monte_carlo: bool
    pass_dsr_n9: bool
    pass_pbo: bool
    oos_sharpe: float
    is_sharpe: float
    oos_returns: pd.Series
    """Kept so DSR can be exactly recomputed at alternative N post-hoc
    (skew/kurtosis/T all matter, not just the scalar Sharpe)."""


def _run_one_replicate(
    templates: pd.DataFrame,
    is_dates: pd.DatetimeIndex,
    oos_dates: pd.DatetimeIndex,
    true_sr: float,
    trial_sharpe_variance: float,
    rng: np.random.Generator,
) -> ReplicateResult:
    t = len(templates)
    block_len = default_expected_block_length(t)
    idx = _stationary_bootstrap_index(t, block_len, rng)
    resampled = pd.DataFrame(
        templates.to_numpy()[idx], index=templates.index, columns=templates.columns
    )

    drifted = resampled.copy()
    for col in drifted.columns:
        std = float(templates[col].std(ddof=1))
        mu = true_sr * std / np.sqrt(PERIODS_PER_YEAR)
        drifted[col] = drifted[col] + mu

    locked_label = _config_label(*LOCKED)
    is_returns = drifted.loc[drifted.index.isin(is_dates), locked_label]
    oos_returns = drifted.loc[drifted.index.isin(oos_dates), locked_label]

    is_sharpe = sharpe_ratio(is_returns, periods_per_year=PERIODS_PER_YEAR)
    sig = significance_test(
        is_returns, n_simulations=N_BOOT_SIMS, periods_per_year=PERIODS_PER_YEAR, rng=rng
    )
    pass_sig = sig.p_value < SIGNIFICANCE_ALPHA

    wf = walk_forward_analysis(
        oos_returns, is_window=12, oos_window=6, periods_per_year=PERIODS_PER_YEAR
    )
    pass_wf = wf.mean_oos_sharpe > 0.0

    def _metric_at_lookback(lookback_value: float) -> float:
        lb = max(1, round(lookback_value))
        label = _config_label(lb, 1)
        col = drifted.loc[drifted.index.isin(oos_dates), label]
        return sharpe_ratio(col, periods_per_year=PERIODS_PER_YEAR)

    sens = parameter_sensitivity("lookback", 12.0, _metric_at_lookback)
    pass_sens = sens.is_plateau

    mc = monte_carlo_block_bootstrap(
        oos_returns, n_simulations=N_BOOT_SIMS, periods_per_year=PERIODS_PER_YEAR, rng=rng
    )
    pass_mc = mc.percentile_rank <= 95.0

    oos_sharpe = sharpe_ratio(oos_returns, periods_per_year=PERIODS_PER_YEAR)
    dsr = deflated_sharpe_ratio(
        oos_returns, n_trials=N_TRIALS_PRODUCTION, trial_sharpe_variance=trial_sharpe_variance,
        periods_per_year=PERIODS_PER_YEAR,
    )
    pass_dsr = dsr.dsr > 0.95

    pbo_cols = [_config_label(lb, sk) for lb, sk in PBO_CONFIGS]
    pbo_frame = drifted.loc[drifted.index.isin(templates.index), pbo_cols]
    pbo_result = probability_of_backtest_overfitting(
        pbo_frame, n_blocks=PBO_N_BLOCKS, periods_per_year=PERIODS_PER_YEAR
    )
    pass_pbo = pbo_result.pbo < 0.5

    return ReplicateResult(
        pass_sig, pass_wf, pass_sens, pass_mc, pass_dsr, pass_pbo,
        oos_sharpe, is_sharpe, oos_returns,
    )


def main() -> None:
    print("Loading real MOEX momentum templates (11 configs, full period 2018-2026)...")
    templates = _load_real_templates()
    print(f"Common valid index: {len(templates)} months "
          f"({templates.index[0].date()} .. {templates.index[-1].date()})")

    is_dates = templates.index[(templates.index >= IS_START) & (templates.index <= IS_END)]
    oos_dates = templates.index[(templates.index >= OOS_START) & (templates.index <= OOS_END)]
    print(f"IS: {len(is_dates)} months, OOS: {len(oos_dates)} months\n")

    trial_sharpe_variance = _real_trial_sharpe_variance()
    print(
        "Real trial_sharpe_variance (cycle 1's 7 trials, both hypotheses): "
        f"{trial_sharpe_variance:.4f}"
    )
    # Sanity check against reports/cycle1.md's own printed number, computed
    # independently here via the real deflated_sharpe_ratio function --
    # confirms this script's variance source matches production exactly.
    sanity = deflated_sharpe_ratio(
        pd.Series([0.01, -0.01, 0.02, 0.0, 0.015]),  # dummy candidate, only expected_max matters
        n_trials=N_TRIALS_PRODUCTION, trial_sharpe_variance=trial_sharpe_variance,
        periods_per_year=PERIODS_PER_YEAR,
    )
    print(f"Implied E[max SR|H0] at N=9: {sanity.expected_max_sharpe_null:+.3f} "
          "(cycle1.md reports +1.138 -- should match)\n")

    rng = np.random.default_rng(SEED)
    all_results: dict[float, list[ReplicateResult]] = {}
    for true_sr in SR_GRID:
        print(f"=== true annualized Sharpe = {true_sr:.2f} ({N_REPLICATES} replicates) ===")
        results = [
            _run_one_replicate(templates, is_dates, oos_dates, true_sr, trial_sharpe_variance, rng)
            for _ in range(N_REPLICATES)
        ]
        all_results[true_sr] = results

        n = len(results)
        rate_sig = sum(r.pass_significance for r in results) / n
        rate_wf = sum(r.pass_walk_forward for r in results) / n
        rate_sens = sum(r.pass_sensitivity for r in results) / n
        rate_mc = sum(r.pass_monte_carlo for r in results) / n
        rate_dsr = sum(r.pass_dsr_n9 for r in results) / n
        rate_pbo = sum(r.pass_pbo for r in results) / n
        rate_all = sum(
            r.pass_significance and r.pass_walk_forward and r.pass_sensitivity
            and r.pass_monte_carlo and r.pass_dsr_n9 and r.pass_pbo
            for r in results
        ) / n
        print(
            f"  significance={rate_sig:.1%}  walk_forward={rate_wf:.1%}  "
            f"sensitivity={rate_sens:.1%}  monte_carlo={rate_mc:.1%}  "
            f"dsr={rate_dsr:.1%}  pbo={rate_pbo:.1%}"
        )
        print(f"  ALL-6 (production rule): {rate_all:.1%}\n")

    _write_report(templates, is_dates, oos_dates, trial_sharpe_variance, all_results)
    print(f"Report: {REPORT_PATH}")


def _write_report(
    templates: pd.DataFrame,
    is_dates: pd.DatetimeIndex,
    oos_dates: pd.DatetimeIndex,
    trial_sharpe_variance: float,
    all_results: dict[float, list[ReplicateResult]],
) -> None:
    lines: list[str] = []
    lines.append("# Gate power analysis")
    lines.append("")
    lines.append(
        "> **Supersedes the 10.09.2026 first release of this file.** That run resampled a "
        "noise template that was never demeaned, so every `true SR = X` row below actually "
        "measured behavior at `X` plus that column's own real historical Sharpe (locked "
        "`12_1`: +0.92) -- the `SR = 0` row was not a null at all. Fixed in "
        "`_load_real_templates`; see `reports/why_no_alpha.md` A0. Headline change: power at "
        "a true Sharpe of 1.0 is 1.6%, not the 38.4% first reported."
    )
    lines.append("")
    lines.append(
        f"N_REPLICATES={N_REPLICATES} per true-Sharpe value, N_BOOT_SIMS={N_BOOT_SIMS} "
        f"(vs. production 2000), PBO n_blocks={PBO_N_BLOCKS} (vs. production 16, for speed)."
    )
    lines.append(
        f"Monte Carlo SE of an estimated proportion at N={N_REPLICATES}: "
        f"~{(0.5 * 0.5 / N_REPLICATES) ** 0.5:.1%} at p=0.5, less at extreme p."
    )
    lines.append(
        f"Common template index: {len(templates)} months, "
        f"IS={len(is_dates)}, OOS={len(oos_dates)}"
    )
    lines.append(
        f"trial_sharpe_variance (real, cycle 1 momentum grid): {trial_sharpe_variance:.4f}"
    )
    lines.append("")
    lines.append("## Per-criterion pass rate and full-gate power, by true annualized Sharpe")
    lines.append("")
    lines.append(
        "| true SR | significance | walk_forward | sensitivity | monte_carlo | "
        "DSR (N=9) | PBO | **ALL-6** |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for true_sr in SR_GRID:
        results = all_results[true_sr]
        n = len(results)
        rate_sig = sum(r.pass_significance for r in results) / n
        rate_wf = sum(r.pass_walk_forward for r in results) / n
        rate_sens = sum(r.pass_sensitivity for r in results) / n
        rate_mc = sum(r.pass_monte_carlo for r in results) / n
        rate_dsr = sum(r.pass_dsr_n9 for r in results) / n
        rate_pbo = sum(r.pass_pbo for r in results) / n
        rate_all = sum(
            r.pass_significance and r.pass_walk_forward and r.pass_sensitivity
            and r.pass_monte_carlo and r.pass_dsr_n9 and r.pass_pbo
            for r in results
        ) / n
        label = "Type I error" if true_sr == 0.0 else f"power @ SR={true_sr}"
        lines.append(
            f"| {true_sr:.2f} ({label}) | {rate_sig:.1%} | {rate_wf:.1%} | {rate_sens:.1%} | "
            f"{rate_mc:.1%} | {rate_dsr:.1%} | {rate_pbo:.1%} | **{rate_all:.1%}** |"
        )
    lines.append("")

    lines.append("## Alternative combination rules (reusing the same per-criterion outcomes)")
    lines.append("")
    lines.append("| true SR | ALL-6 (AND) | >=5 of 6 | >=4 of 6 |")
    lines.append("|---|---|---|---|")
    for true_sr in SR_GRID:
        results = all_results[true_sr]
        n = len(results)
        counts = [
            sum([
                r.pass_significance, r.pass_walk_forward, r.pass_sensitivity,
                r.pass_monte_carlo, r.pass_dsr_n9, r.pass_pbo,
            ])
            for r in results
        ]
        rate_all6 = sum(c == 6 for c in counts) / n
        rate_5 = sum(c >= 5 for c in counts) / n
        rate_4 = sum(c >= 4 for c in counts) / n
        lines.append(f"| {true_sr:.2f} | {rate_all6:.1%} | {rate_5:.1%} | {rate_4:.1%} |")
    lines.append("")

    lines.append(
        "## DSR pass rate at alternative cumulative N "
        "(same simulated OOS return series, DSR exactly recomputed per replicate)"
    )
    lines.append("")
    lines.append(
        "trial_sharpe_variance held fixed at the real value above throughout -- only "
        "`n_trials` changes, isolating exactly what the cumulative-N convention costs."
    )
    lines.append("")
    lines.append(
        "| true SR | N=2 | N=3 | N=9 (production) | N=20 | N=50 | N=100 | N=500 | "
        "N=2444 (crypto-track scale) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    alt_ns = (2, 3, 9, 20, 50, 100, 500, 2444)
    for true_sr in SR_GRID:
        results = all_results[true_sr]
        n = len(results)
        cells = []
        for n_trials in alt_ns:
            passes = 0
            for r in results:
                dsr = deflated_sharpe_ratio(
                    r.oos_returns, n_trials=n_trials, trial_sharpe_variance=trial_sharpe_variance,
                    periods_per_year=PERIODS_PER_YEAR,
                )
                passes += dsr.dsr > 0.95
            cells.append(f"{passes / n:.1%}")
        lines.append(f"| {true_sr:.2f} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append(
        "## Full ALL-6-gate power at alternative cumulative N "
        "(the other 5 criteria held exactly as observed per replicate, only DSR's N varies)"
    )
    lines.append("")
    lines.append(
        "| true SR | N=2 | N=3 | N=9 (production) | N=20 | N=50 | N=100 | N=500 | "
        "N=2444 (crypto-track scale) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for true_sr in SR_GRID:
        results = all_results[true_sr]
        n = len(results)
        cells = []
        for n_trials in alt_ns:
            passes = 0
            for r in results:
                if not (
                    r.pass_significance and r.pass_walk_forward and r.pass_sensitivity
                    and r.pass_monte_carlo and r.pass_pbo
                ):
                    continue
                dsr = deflated_sharpe_ratio(
                    r.oos_returns, n_trials=n_trials, trial_sharpe_variance=trial_sharpe_variance,
                    periods_per_year=PERIODS_PER_YEAR,
                )
                passes += dsr.dsr > 0.95
            cells.append(f"{passes / n:.1%}")
        lines.append(f"| {true_sr:.2f} | " + " | ".join(cells) + " |")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
