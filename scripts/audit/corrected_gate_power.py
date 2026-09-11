"""BUG FOUND in scripts/gate_power_analysis.py (the already-committed,
memory-cited power-audit stand, eb90ff8): its own docstring says the noise
template is "the REAL empirical residual shape ... (demeaned)" but the code
never actually subtracts each column's mean before block-bootstrap
resampling -- `_load_real_templates()` returns raw real momentum returns,
and `_run_one_replicate` resamples `templates.to_numpy()` directly (grep
confirms: the string "demeaned" appears exactly once, in a comment, at
line 15; no `.mean()` subtraction exists anywhere in the file).

Consequence: every "true SR" grid label in reports/gate_power_analysis.md
is silently offset by that config's own REAL historical Sharpe (e.g. the
locked "12_1" column's real full-period annualized Sharpe is ~0.92 -- the
label "true SR=0.00 (Type I error)" is therefore actually measuring the
gate's behavior at an EFFECTIVE true Sharpe of about 0.92, not a genuine
null; "true SR=1.00" is actually effective ~1.9).

This script is a minimal, surgical fix -- demean every column of the real
templates by its own full-period mean before doing anything else -- and
reruns the exact same pipeline (same functions, same seed, same N=9,
same grids) to get the HONEST numbers. Does not modify
scripts/gate_power_analysis.py itself (flagged here, not silently patched
in place) and does not touch experiments/grid_search_log.jsonl or any
lock file -- calibration, not a new trial.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Resolved from this file's location rather than hardcoded: the only edit
# made to these scripts after the run that produced the reported tables.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

import gate_power_analysis as gpa  # noqa: E402
from moex_backtest.validation import deflated_sharpe_ratio  # noqa: E402

OUT_PATH = Path(__file__).parent / "corrected_gate_power.md"


def main() -> None:
    print("Loading real MOEX momentum templates...")
    raw_templates = gpa._load_real_templates()

    print("\nBUG CHECK -- real (undemeaned) full-period annualized Sharpe per column:")
    for col in raw_templates.columns:
        s = raw_templates[col]
        sr = float(s.mean() / s.std(ddof=1) * np.sqrt(gpa.PERIODS_PER_YEAR))
        print(f"  {col}: mean={s.mean():+.5f} sharpe={sr:+.3f}")

    templates = raw_templates - raw_templates.mean()  # THE FIX
    print(f"\nAfter fix, per-column mean (should all be ~0): "
          f"{templates.mean().abs().max():.2e} max abs")

    is_dates = templates.index[(templates.index >= gpa.IS_START) & (templates.index <= gpa.IS_END)]
    oos_dates = templates.index[(templates.index >= gpa.OOS_START) & (templates.index <= gpa.OOS_END)]
    trial_sharpe_variance = gpa._real_trial_sharpe_variance()

    rng = np.random.default_rng(gpa.SEED)
    all_results: dict[float, list] = {}
    for true_sr in gpa.SR_GRID:
        print(f"=== true SR = {true_sr:.2f} (CORRECTED, demeaned) ===", flush=True)
        results = [
            gpa._run_one_replicate(templates, is_dates, oos_dates, true_sr, trial_sharpe_variance, rng)
            for _ in range(gpa.N_REPLICATES)
        ]
        all_results[true_sr] = results

    lines = ["# CORRECTED gate power analysis (demeaning bug fixed)", ""]
    lines.append(
        "scripts/gate_power_analysis.py's docstring claims the noise template is "
        "demeaned; the code never actually does it. This rerun fixes that (subtracts "
        "each column's own full-period mean before block-bootstrap resampling) and "
        "reruns the identical pipeline, same seed=42, same N=9, same 250 replicates."
    )
    lines.append("")
    lines.append("## Per-criterion pass rate and full-gate power, CORRECTED")
    lines.append("")
    lines.append(
        "| true SR | significance | walk_forward | sensitivity | monte_carlo | "
        "DSR (N=9) | PBO | **ALL-6** |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for true_sr in gpa.SR_GRID:
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
        lines.append(
            f"| {true_sr:.2f} | {rate_sig:.1%} | {rate_wf:.1%} | {rate_sens:.1%} | "
            f"{rate_mc:.1%} | {rate_dsr:.1%} | {rate_pbo:.1%} | **{rate_all:.1%}** |"
        )
    lines.append("")

    lines.append("## CORRECTED: full-gate power at alternative N (2..10), fine grid")
    lines.append("")
    alt_ns = tuple(range(2, 11))
    header = "| true SR | " + " | ".join(f"N={n}" for n in alt_ns) + " |"
    sep = "|---|" + "---|" * len(alt_ns)
    lines.append(header)
    lines.append(sep)
    full_table: dict[float, list[float]] = {}
    for true_sr in gpa.SR_GRID:
        results = all_results[true_sr]
        n = len(results)
        cells = []
        row = []
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
                    periods_per_year=gpa.PERIODS_PER_YEAR,
                )
                passes += dsr.dsr > 0.95
            rate = passes / n
            row.append(rate)
            cells.append(f"{rate:.1%}")
        full_table[true_sr] = row
        lines.append(f"| {true_sr:.2f} | " + " | ".join(cells) + " |")
    lines.append("")

    alpha_row = full_table[0.0]
    crossover_n = next((n_ for n_, r in zip(alt_ns, alpha_row) if r <= 0.05), None)
    lines.append(
        f"**Smallest N with corrected full-gate alpha <= 5%: N={crossover_n}** "
        f"(alpha={alpha_row[alt_ns.index(crossover_n)]:.1%})" if crossover_n
        else "No N in 2..10 achieves alpha<=5% after correction"
    )
    if crossover_n:
        idx = alt_ns.index(crossover_n)
        lines.append("Power at that N, by true SR:")
        for true_sr in gpa.SR_GRID:
            lines.append(f"- SR={true_sr}: {full_table[true_sr][idx]:.1%}")

    out_text = "\n".join(lines)
    OUT_PATH.write_text(out_text + "\n")
    print("\n" + out_text)


if __name__ == "__main__":
    main()
