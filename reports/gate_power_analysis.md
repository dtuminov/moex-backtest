# Gate power analysis

> **Supersedes the 10.09.2026 first release of this file.** That run resampled a noise template that was never demeaned, so every `true SR = X` row below actually measured behavior at `X` plus that column's own real historical Sharpe (locked `12_1`: +0.92) -- the `SR = 0` row was not a null at all. Fixed in `_load_real_templates`; see `reports/why_no_alpha.md` A0. Headline change: power at a true Sharpe of 1.0 is 1.6%, not the 38.4% first reported.

N_REPLICATES=250 per true-Sharpe value, N_BOOT_SIMS=500 (vs. production 2000), PBO n_blocks=8 (vs. production 16, for speed).
Monte Carlo SE of an estimated proportion at N=250: ~3.2% at p=0.5, less at extreme p.
Common template index: 88 months, IS=32, OOS=56
trial_sharpe_variance (real, cycle 1 momentum grid): 0.5596

## Per-criterion pass rate and full-gate power, by true annualized Sharpe

| true SR | significance | walk_forward | sensitivity | monte_carlo | DSR (N=9) | PBO | **ALL-6** |
|---|---|---|---|---|---|---|---|
| 0.00 (Type I error) | 17.2% | 60.8% | 92.0% | 100.0% | 0.0% | 44.0% | **0.0%** |
| 0.50 (power @ SR=0.5) | 40.0% | 87.6% | 94.8% | 100.0% | 0.8% | 47.2% | **0.0%** |
| 0.75 (power @ SR=0.75) | 60.0% | 97.2% | 93.6% | 100.0% | 1.6% | 45.2% | **0.8%** |
| 1.00 (power @ SR=1.0) | 64.8% | 96.0% | 92.4% | 100.0% | 5.2% | 48.8% | **1.6%** |
| 1.50 (power @ SR=1.5) | 79.6% | 100.0% | 90.4% | 100.0% | 32.4% | 45.6% | **12.0%** |
| 2.00 (power @ SR=2.0) | 96.8% | 100.0% | 96.4% | 100.0% | 50.4% | 48.8% | **26.8%** |

## Alternative combination rules (reusing the same per-criterion outcomes)

| true SR | ALL-6 (AND) | >=5 of 6 | >=4 of 6 |
|---|---|---|---|
| 0.00 | 0.0% | 4.8% | 32.8% |
| 0.50 | 0.0% | 18.4% | 57.6% |
| 0.75 | 0.8% | 23.6% | 73.2% |
| 1.00 | 1.6% | 33.2% | 74.8% |
| 1.50 | 12.0% | 46.4% | 90.0% |
| 2.00 | 26.8% | 68.0% | 97.6% |

## DSR pass rate at alternative cumulative N (same simulated OOS return series, DSR exactly recomputed per replicate)

trial_sharpe_variance held fixed at the real value above throughout -- only `n_trials` changes, isolating exactly what the cumulative-N convention costs.

| true SR | N=2 | N=3 | N=9 (production) | N=20 | N=50 | N=100 | N=500 | N=2444 (crypto-track scale) |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.8% | 0.4% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.50 | 12.0% | 6.4% | 0.8% | 0.4% | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.75 | 21.6% | 11.2% | 1.6% | 0.4% | 0.0% | 0.0% | 0.0% | 0.0% |
| 1.00 | 40.8% | 26.0% | 5.2% | 1.2% | 0.0% | 0.0% | 0.0% | 0.0% |
| 1.50 | 69.2% | 56.0% | 32.4% | 20.4% | 10.0% | 5.2% | 0.8% | 0.0% |
| 2.00 | 85.2% | 74.0% | 50.4% | 36.8% | 26.0% | 19.2% | 6.8% | 1.6% |

## Full ALL-6-gate power at alternative cumulative N (the other 5 criteria held exactly as observed per replicate, only DSR's N varies)

| true SR | N=2 | N=3 | N=9 (production) | N=20 | N=50 | N=100 | N=500 | N=2444 (crypto-track scale) |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.50 | 1.6% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.75 | 6.4% | 4.0% | 0.8% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 1.00 | 15.6% | 9.2% | 1.6% | 0.4% | 0.0% | 0.0% | 0.0% | 0.0% |
| 1.50 | 23.6% | 20.4% | 12.0% | 8.0% | 4.8% | 3.2% | 0.4% | 0.0% |
| 2.00 | 39.6% | 34.0% | 26.8% | 20.8% | 15.2% | 10.8% | 4.0% | 1.2% |
