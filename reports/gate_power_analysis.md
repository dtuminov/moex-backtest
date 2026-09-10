# Gate power analysis

N_REPLICATES=250 per true-Sharpe value, N_BOOT_SIMS=500 (vs. production 2000), PBO n_blocks=8 (vs. production 16, for speed).
Monte Carlo SE of an estimated proportion at N=250: ~3.2% at p=0.5, less at extreme p.
Common template index: 88 months, IS=32, OOS=56
trial_sharpe_variance (real, cycle 1 momentum grid): 0.5596

## Per-criterion pass rate and full-gate power, by true annualized Sharpe

| true SR | significance | walk_forward | sensitivity | monte_carlo | DSR (N=9) | PBO | **ALL-6** |
|---|---|---|---|---|---|---|---|
| 0.00 (Type I error) | 62.8% | 98.0% | 95.2% | 100.0% | 3.6% | 93.6% | **1.6%** |
| 0.50 (power @ SR=0.5) | 78.0% | 99.2% | 90.8% | 100.0% | 21.6% | 92.8% | **15.2%** |
| 0.75 (power @ SR=0.75) | 91.2% | 100.0% | 92.0% | 100.0% | 34.4% | 95.2% | **27.2%** |
| 1.00 (power @ SR=1.0) | 94.8% | 100.0% | 90.8% | 100.0% | 48.8% | 90.8% | **38.4%** |
| 1.50 (power @ SR=1.5) | 98.4% | 100.0% | 93.2% | 100.0% | 65.2% | 94.8% | **56.0%** |
| 2.00 (power @ SR=2.0) | 100.0% | 100.0% | 88.4% | 100.0% | 82.8% | 91.6% | **68.8%** |

## Alternative combination rules (reusing the same per-criterion outcomes)

| true SR | ALL-6 (AND) | >=5 of 6 | >=4 of 6 |
|---|---|---|---|
| 0.00 | 1.6% | 57.6% | 94.0% |
| 0.50 | 15.2% | 72.0% | 95.2% |
| 0.75 | 27.2% | 86.4% | 99.2% |
| 1.00 | 38.4% | 87.2% | 99.6% |
| 1.50 | 56.0% | 95.6% | 100.0% |
| 2.00 | 68.8% | 94.4% | 99.6% |

## DSR pass rate at alternative cumulative N (same simulated OOS return series, DSR exactly recomputed per replicate)

trial_sharpe_variance held fixed at the real value above throughout -- only `n_trials` changes, isolating exactly what the cumulative-N convention costs.

| true SR | N=2 | N=3 | N=9 (production) | N=20 | N=50 | N=100 | N=500 | N=2444 (crypto-track scale) |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 36.8% | 22.8% | 3.6% | 1.2% | 0.4% | 0.0% | 0.0% | 0.0% |
| 0.50 | 62.4% | 46.4% | 21.6% | 13.2% | 4.0% | 2.0% | 0.8% | 0.4% |
| 0.75 | 79.6% | 62.0% | 34.4% | 21.2% | 8.4% | 5.6% | 1.2% | 0.4% |
| 1.00 | 86.0% | 74.4% | 48.8% | 38.8% | 21.6% | 15.6% | 2.4% | 0.4% |
| 1.50 | 96.8% | 87.6% | 65.2% | 54.8% | 48.0% | 40.8% | 27.2% | 14.0% |
| 2.00 | 99.2% | 97.2% | 82.8% | 70.4% | 58.8% | 53.2% | 39.2% | 29.2% |

## Full ALL-6-gate power at alternative cumulative N (the other 5 criteria held exactly as observed per replicate, only DSR's N varies)

| true SR | N=2 | N=3 | N=9 (production) | N=20 | N=50 | N=100 | N=500 | N=2444 (crypto-track scale) |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 20.8% | 14.8% | 1.6% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.50 | 44.0% | 32.8% | 15.2% | 10.0% | 2.8% | 1.2% | 0.4% | 0.4% |
| 0.75 | 64.0% | 49.2% | 27.2% | 15.2% | 6.0% | 3.6% | 1.2% | 0.4% |
| 1.00 | 68.0% | 59.2% | 38.4% | 29.6% | 14.4% | 9.2% | 1.2% | 0.4% |
| 1.50 | 85.2% | 77.2% | 56.0% | 48.0% | 42.0% | 35.6% | 22.4% | 10.8% |
| 2.00 | 80.8% | 79.2% | 68.8% | 59.2% | 48.8% | 44.0% | 31.6% | 22.4% |
