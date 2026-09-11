"""Step 2 of the T-vs-costs audit: does the REAL power gain from weekly
rebalance (T=234 vs T=56) actually materialize, once measured the same
way as scripts/gate_power_analysis.py (same gate functions, same DSR N,
same SR_GRID, same block-bootstrap noise-template method)?

Uses the REAL weekly momentum residual series (demeaned, from
weekly_rebalance_real.py's actual OOS backtest -- real autocorrelation,
real skew/kurtosis at weekly frequency) as the noise template, single
column (only DSR/significance/walk-forward/Monte Carlo apply to one
series; sensitivity and PBO need a multi-config grid this diagnostic does
not build -- building one would itself be a new trial, out of scope here).

Diagnostic only: does not touch grid_search_log.jsonl or any lock file,
does not increment cumulative N.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Resolved from this file's location rather than hardcoded: the only edit
# made to these scripts after the run that produced the reported tables.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from moex_backtest.metrics.performance import sharpe_ratio  # noqa: E402
from moex_backtest.validation import (  # noqa: E402
    deflated_sharpe_ratio,
    monte_carlo_block_bootstrap,
    significance_test,
    walk_forward_analysis,
)
from moex_backtest.validation._bootstrap import (  # noqa: E402
    default_expected_block_length,
    stationary_bootstrap_resample,
)

SCRATCH = Path("/private/tmp/claude-501/-Users-dmitrijtuminov-Documents-personal-os/"
                "d0451c5e-47fe-478f-a6c3-5f648bda65be/scratchpad")

PERIODS_PER_YEAR_WEEKLY = 52.18
PERIODS_PER_YEAR_MONTHLY = 12
N_REPLICATES = 250
N_BOOT_SIMS = 500
SIGNIFICANCE_ALPHA = 0.10
SR_GRID = (0.0, 0.5, 0.75, 1.0, 1.5, 2.0)
# real mechanism-scoped momentum N (cycle1 6 + cycle2 ensemble 1 = 7) and
# production cumulative N=9, for comparison
N_VALUES = (7, 9)
TRIAL_SHARPE_VARIANCE = 0.5596  # same real value as gate_power_analysis.py
SEED = 43  # different from monthly audit's 42 -- independent draw, not reused numbers


def _run_replicates(
    template: pd.Series, is_dates: pd.DatetimeIndex, oos_dates: pd.DatetimeIndex,
    periods_per_year: float, true_sr: float, rng: np.random.Generator,
) -> list[dict]:
    t = len(template)
    block_len = default_expected_block_length(t)
    template_arr = template.to_numpy()
    results = []
    for _ in range(N_REPLICATES):
        resampled_arr = stationary_bootstrap_resample(template_arr, 1, block_len, rng)[0]
        resampled = pd.Series(resampled_arr, index=template.index)
        std = float(template.std(ddof=1))
        mu = true_sr * std / np.sqrt(periods_per_year)
        drifted = resampled + mu

        is_ret = drifted.loc[drifted.index.isin(is_dates)]
        oos_ret = drifted.loc[drifted.index.isin(oos_dates)]

        sig = significance_test(
            is_ret, n_simulations=N_BOOT_SIMS, periods_per_year=periods_per_year, rng=rng
        )
        pass_sig = sig.p_value < SIGNIFICANCE_ALPHA

        wf_is = max(4, len(is_ret) // 4)
        wf_oos = max(2, len(oos_ret) // 8)
        try:
            wf = walk_forward_analysis(
                oos_ret, is_window=wf_is, oos_window=wf_oos, periods_per_year=periods_per_year
            )
            pass_wf = wf.mean_oos_sharpe > 0.0
        except Exception:
            pass_wf = False

        mc = monte_carlo_block_bootstrap(
            oos_ret, n_simulations=N_BOOT_SIMS, periods_per_year=periods_per_year, rng=rng
        )
        pass_mc = mc.percentile_rank <= 95.0

        dsrs = {}
        for n_trials in N_VALUES:
            dsr = deflated_sharpe_ratio(
                oos_ret, n_trials=n_trials, trial_sharpe_variance=TRIAL_SHARPE_VARIANCE,
                periods_per_year=periods_per_year,
            )
            dsrs[n_trials] = dsr.dsr > 0.95

        results.append({
            "sig": pass_sig, "wf": pass_wf, "mc": pass_mc, "dsr": dsrs,
            "oos_sharpe": sharpe_ratio(oos_ret, periods_per_year=periods_per_year),
        })
    return results


def main() -> None:
    weekly_oos = pd.read_csv(SCRATCH / "weekly_oos_net_returns.csv", index_col=0, parse_dates=True)[
        "weekly_momentum_net"
    ]
    weekly_is = pd.read_csv(SCRATCH / "weekly_is_net_returns.csv", index_col=0, parse_dates=True)[
        "weekly_momentum_net"
    ]
    weekly_full = pd.concat([weekly_is, weekly_oos]).sort_index()
    weekly_template = weekly_full - weekly_full.mean()  # demeaned residual noise template
    weekly_is_dates = weekly_is.index
    weekly_oos_dates = weekly_oos.index

    print(f"Weekly template: T={len(weekly_template)} (IS={len(weekly_is)}, OOS={len(weekly_oos)})")
    print(f"Weekly real (raw) skew={weekly_template.skew():.3f} kurt={weekly_template.kurt()+3:.3f}")

    rng = np.random.default_rng(SEED)
    all_weekly: dict[float, list[dict]] = {}
    for true_sr in SR_GRID:
        print(f"weekly true_sr={true_sr}", flush=True)
        all_weekly[true_sr] = _run_replicates(
            weekly_template, weekly_is_dates, weekly_oos_dates,
            PERIODS_PER_YEAR_WEEKLY, true_sr, rng,
        )

    lines = ["# Weekly-rebalance power audit (T=234 OOS weeks vs T=56 OOS months)", ""]
    lines.append(
        "Real weekly momentum residual template (demeaned real OOS+IS returns from "
        "weekly_rebalance_real.py), same significance/walk-forward/Monte-Carlo/DSR "
        "functions as scripts/gate_power_analysis.py. Sensitivity and PBO excluded "
        "(need a multi-config grid this diagnostic does not build)."
    )
    lines.append("")
    lines.append("## DSR pass rate at weekly T=234, mechanism-scoped N=7 and production N=9")
    lines.append("")
    lines.append("| true SR | sig | walk_forward | monte_carlo | DSR N=7 | DSR N=9 | 4-of-4 (sig+wf+mc+dsrN9) |")
    lines.append("|---|---|---|---|---|---|---|")
    for true_sr in SR_GRID:
        res = all_weekly[true_sr]
        n = len(res)
        r_sig = sum(r["sig"] for r in res) / n
        r_wf = sum(r["wf"] for r in res) / n
        r_mc = sum(r["mc"] for r in res) / n
        r_dsr7 = sum(r["dsr"][7] for r in res) / n
        r_dsr9 = sum(r["dsr"][9] for r in res) / n
        r_all4 = sum(r["sig"] and r["wf"] and r["mc"] and r["dsr"][9] for r in res) / n
        lines.append(
            f"| {true_sr:.2f} | {r_sig:.1%} | {r_wf:.1%} | {r_mc:.1%} | {r_dsr7:.1%} | "
            f"{r_dsr9:.1%} | {r_all4:.1%} |"
        )
    lines.append("")

    out_text = "\n".join(lines)
    (SCRATCH / "weekly_power_audit.md").write_text(out_text + "\n")
    print(out_text)


if __name__ == "__main__":
    main()
