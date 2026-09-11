"""How much T (OOS months, monthly frequency, same locked 12-1 config's
real residual shape, properly demeaned) would be needed to get DSR power
to ~50%/80% at the true Sharpe cycle 1 actually observed (OOS +0.562),
holding N=9 (production) and N=2 (best-case mechanism-scoped) fixed?

Synthetic-length OOS windows (T not tied to real calendar dates) drawn by
block-bootstrapping the REAL, PROPERLY DEMEANED "12_1" residual template
(same real autocorrelation/skew/kurtosis shape, arbitrary length) --
answers "how many years of data would this specific edge need" directly
from the real residual's own statistical shape, not a generic assumption.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path("/Users/dmitrijtuminov/Developer/Код/Кванты/moex-backtest")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import gate_power_analysis as gpa  # noqa: E402
from moex_backtest.validation import deflated_sharpe_ratio  # noqa: E402
from moex_backtest.validation._bootstrap import (  # noqa: E402
    default_expected_block_length,
    stationary_bootstrap_resample,
)

SEED = 44
N_REPLICATES = 400
TRUE_SR = 0.562  # cycle 1's actual real observed OOS Sharpe
TRIAL_SHARPE_VARIANCE = 0.5596
N_VALUES = (9, 2)
T_GRID = (56, 80, 100, 150, 200, 300, 450, 600, 900, 1200)


def main() -> None:
    raw_templates = gpa._load_real_templates()
    col = raw_templates["12_1"] - raw_templates["12_1"].mean()  # demeaned, fix applied
    base_arr = col.to_numpy()
    base_std = float(col.std(ddof=1))
    base_t = len(base_arr)
    print(f"base template T={base_t}, std={base_std:.5f}")

    rng = np.random.default_rng(SEED)
    lines = ["# T needed for the real observed effect (OOS Sharpe +0.562) to be detectable", ""]
    lines.append(
        f"Block-bootstrap of the real, DEMEANED locked '12_1' residual (T_base={base_t}), "
        f"synthetic OOS length T, target true annualized Sharpe={TRUE_SR} "
        "(cycle 1's actual observed OOS Sharpe -- not a hypothetical)."
    )
    lines.append("")
    header = "| T (months) | T (years) | " + " | ".join(f"DSR N={n}" for n in N_VALUES) + " |"
    lines.append(header)
    lines.append("|---|---|" + "---|" * len(N_VALUES))

    for t in T_GRID:
        block_len = default_expected_block_length(t)
        # resample by tiling the base array's bootstrap indices to length t
        # (reuse stationary_bootstrap_resample against a *repeated* base
        # array so the origin pool for blocks spans the full base_arr
        # regardless of t > base_t)
        pass_counts = {n: 0 for n in N_VALUES}
        for _ in range(N_REPLICATES):
            resampled = stationary_bootstrap_resample(base_arr, 1, block_len, rng)[0]
            if t <= base_t:
                draw = resampled[:t]
            else:
                # need more than base_t points: concatenate independent
                # block-bootstrap draws of the SAME base template until
                # length t is reached (still respects base autocorrelation
                # within each block; boundary effects at concat points are
                # a known, minor block-bootstrap approximation)
                parts = [resampled]
                while sum(len(p) for p in parts) < t:
                    parts.append(stationary_bootstrap_resample(base_arr, 1, block_len, rng)[0])
                draw = np.concatenate(parts)[:t]
            mu = TRUE_SR * base_std / np.sqrt(12)
            series = pd.Series(draw + mu)
            for n_trials in N_VALUES:
                dsr = deflated_sharpe_ratio(
                    series, n_trials=n_trials, trial_sharpe_variance=TRIAL_SHARPE_VARIANCE,
                    periods_per_year=12,
                )
                pass_counts[n_trials] += dsr.dsr > 0.95
        row = [f"{t}", f"{t/12:.1f}"] + [f"{pass_counts[n]/N_REPLICATES:.1%}" for n in N_VALUES]
        lines.append("| " + " | ".join(row) + " |")
        print(row)

    out = "\n".join(lines)
    Path(__file__).parent.joinpath("t_needed.md").write_text(out + "\n")
    print("\n" + out)


if __name__ == "__main__":
    main()
