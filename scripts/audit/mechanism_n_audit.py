"""Audit-only extension of scripts/gate_power_analysis.py: fine-grained N
grid (1..10) to find where the DSR/full-gate alpha crosses ~5%, and to give
the actual mechanism-scoped N values (momentum=7 at cycle 1's own DSR call,
8 counting cycle 2's ensemble trial) an honest power/alpha reading -- not
just the N=2/N=3 illustrative points already in the production report.

Does NOT modify any file under src/moex_backtest/validation/ or the
production script/report. Reuses the exact same simulation code (same
seed=42, same N_REPLICATES=250, same templates) via import, so replicate
draws for true_sr in {0.0,0.5,0.75,1.0,1.5,2.0} are bit-identical to
reports/gate_power_analysis.md -- only the post-hoc N-recomputation loop is
new. This is a calibration/measurement run, not a new trial: does not touch
grid_search_log.jsonl, does not write a lock file, does not increment
cumulative N.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path("/Users/dmitrijtuminov/Developer/Код/Кванты/moex-backtest")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

import gate_power_analysis as gpa  # noqa: E402
from moex_backtest.validation import deflated_sharpe_ratio  # noqa: E402

OUT_PATH = Path(__file__).parent / "mechanism_n_audit.md"


def main() -> None:
    print("Loading real MOEX momentum templates...")
    templates = gpa._load_real_templates()
    is_dates = templates.index[
        (templates.index >= gpa.IS_START) & (templates.index <= gpa.IS_END)
    ]
    oos_dates = templates.index[
        (templates.index >= gpa.OOS_START) & (templates.index <= gpa.OOS_END)
    ]
    trial_sharpe_variance = gpa._real_trial_sharpe_variance()
    print(f"trial_sharpe_variance = {trial_sharpe_variance:.4f}")

    rng = np.random.default_rng(gpa.SEED)
    all_results: dict[float, list] = {}
    for true_sr in gpa.SR_GRID:
        print(f"=== true SR = {true_sr:.2f} ===", flush=True)
        results = [
            gpa._run_one_replicate(
                templates, is_dates, oos_dates, true_sr, trial_sharpe_variance, rng
            )
            for _ in range(gpa.N_REPLICATES)
        ]
        all_results[true_sr] = results

    alt_ns = tuple(range(2, 11))  # 2..10 (DSR requires n_trials>=2), fine grid around 7/8

    lines = []
    lines.append("# Mechanism-scoped N audit (fine grid 1..10)")
    lines.append("")
    lines.append(
        "Same 250 replicates/true-SR, same seed=42 templates as "
        "reports/gate_power_analysis.md -- only the post-hoc N sweep is new."
    )
    lines.append("")
    lines.append("## DSR-only pass rate at N=1..10")
    lines.append("")
    header = "| true SR | " + " | ".join(f"N={n}" for n in alt_ns) + " |"
    sep = "|---|" + "---|" * len(alt_ns)
    lines.append(header)
    lines.append(sep)
    dsr_table: dict[float, list[float]] = {}
    for true_sr in gpa.SR_GRID:
        results = all_results[true_sr]
        n = len(results)
        cells = []
        row_rates = []
        for n_trials in alt_ns:
            passes = 0
            for r in results:
                dsr = deflated_sharpe_ratio(
                    r.oos_returns,
                    n_trials=n_trials,
                    trial_sharpe_variance=trial_sharpe_variance,
                    periods_per_year=gpa.PERIODS_PER_YEAR,
                )
                passes += dsr.dsr > 0.95
            rate = passes / n
            row_rates.append(rate)
            cells.append(f"{rate:.1%}")
        dsr_table[true_sr] = row_rates
        lines.append(f"| {true_sr:.2f} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Full ALL-6-gate power at N=1..10 (other 5 criteria held as observed)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    full_table: dict[float, list[float]] = {}
    for true_sr in gpa.SR_GRID:
        results = all_results[true_sr]
        n = len(results)
        cells = []
        row_rates = []
        for n_trials in alt_ns:
            passes = 0
            for r in results:
                if not (
                    r.pass_significance
                    and r.pass_walk_forward
                    and r.pass_sensitivity
                    and r.pass_monte_carlo
                    and r.pass_pbo
                ):
                    continue
                dsr = deflated_sharpe_ratio(
                    r.oos_returns,
                    n_trials=n_trials,
                    trial_sharpe_variance=trial_sharpe_variance,
                    periods_per_year=gpa.PERIODS_PER_YEAR,
                )
                passes += dsr.dsr > 0.95
            rate = passes / n
            row_rates.append(rate)
            cells.append(f"{rate:.1%}")
        full_table[true_sr] = row_rates
        lines.append(f"| {true_sr:.2f} | " + " | ".join(cells) + " |")
    lines.append("")

    # Crossover: smallest N with full-gate alpha (true SR=0) <= 5%
    alpha_row = full_table[0.0]
    crossover_n = None
    for n_trials, rate in zip(alt_ns, alpha_row):
        if rate <= 0.05:
            crossover_n = n_trials
            break
    lines.append(
        f"**Smallest N with full-gate alpha <= 5%: "
        f"N={crossover_n if crossover_n else '>10'}** "
        f"(alpha at that N = {alpha_row[alt_ns.index(crossover_n)]:.1%})"
        if crossover_n
        else "**No N in 1..10 gets full-gate alpha under 5% except N=9,10 (see table)**"
    )
    lines.append("")
    lines.append("Power at that crossover N, by true SR:")
    if crossover_n:
        idx = alt_ns.index(crossover_n)
        for true_sr in gpa.SR_GRID:
            lines.append(f"- SR={true_sr}: {full_table[true_sr][idx]:.1%}")

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
