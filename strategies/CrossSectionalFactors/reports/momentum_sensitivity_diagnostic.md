# Diagnostic: is H-MOM's lookback=12 sensitivity spike structural or noise?

Run: 2026-09-10T05:03:22.595242+00:00
**Not a new alpha-search cycle.** Every point below is logged to `experiments/grid_search_log.jsonl` with `stage=diagnostic_curve`, `counts_toward_n=false`. Cumulative MOEX-track DSR-N stays at **9** (unchanged from `reports/cycle1.md`). `experiments/cycle1_momentum.lock.json` was read, never rewritten.

## Curves (skip=1 fixed, lookback swept 2..20 months)

### IS_2018_2021

| lookback | Sharpe | SE(Sharpe) | n_obs |
|---|---|---|---|
| 2 | -0.367 | 0.508 | 46 |
| 3 | +0.550 | 0.553 | 45 |
| 4 | +0.775 | 0.523 | 44 |
| 5 | +1.052 | 0.600 | 43 |
| 6 | +1.138 | 0.536 | 42 |
| 7 | +1.181 | 0.569 | 41 |
| 8 | +0.995 | 0.575 | 40 |
| 9 | +1.072 | 0.592 | 39 |
| 10 | +1.162 | 0.612 | 38 |
| 11 | +1.475 | 0.664 | 37 |
| 12 | +1.624 | 0.596 | 36 | **<- locked**
| 13 | +1.376 | 0.685 | 35 |
| 14 | +0.902 | 0.635 | 34 |
| 15 | +0.843 | 0.724 | 33 |
| 16 | +0.481 | 0.676 | 32 |
| 17 | +0.485 | 0.690 | 31 |
| 18 | +0.099 | 0.649 | 30 |
| 19 | +0.311 | 0.673 | 29 |
| 20 | +0.348 | 0.683 | 28 |

Peak: lookback=12 (Sharpe=+1.624). Shape: single isolated peak (spike).

### OOS_full_2022_2026

| lookback | Sharpe | SE(Sharpe) | n_obs |
|---|---|---|---|
| 2 | -0.465 | 0.461 | 56 |
| 3 | -0.025 | 0.468 | 56 |
| 4 | +0.137 | 0.455 | 56 |
| 5 | -0.316 | 0.486 | 56 |
| 6 | -0.149 | 0.462 | 56 |
| 7 | -0.437 | 0.451 | 56 |
| 8 | -0.406 | 0.456 | 56 |
| 9 | +0.031 | 0.469 | 56 |
| 10 | +0.465 | 0.499 | 56 |
| 11 | +0.628 | 0.504 | 56 |
| 12 | +0.562 | 0.499 | 56 | **<- locked**
| 13 | +0.263 | 0.471 | 56 |
| 14 | +0.119 | 0.470 | 56 |
| 15 | -0.086 | 0.463 | 56 |
| 16 | -0.088 | 0.464 | 56 |
| 17 | -0.049 | 0.465 | 56 |
| 18 | -0.080 | 0.462 | 56 |
| 19 | -0.028 | 0.465 | 56 |
| 20 | +0.112 | 0.467 | 56 |

Peak: lookback=11 (Sharpe=+0.628). Shape: smooth interior peak (plateau-like).

### OOS_H1

| lookback | Sharpe | SE(Sharpe) | n_obs |
|---|---|---|---|
| 2 | -0.359 | 0.658 | 28 |
| 3 | +0.203 | 0.657 | 28 |
| 4 | +0.227 | 0.639 | 28 |
| 5 | -0.155 | 0.682 | 28 |
| 6 | -0.164 | 0.658 | 28 |
| 7 | -0.565 | 0.638 | 28 |
| 8 | -0.340 | 0.650 | 28 |
| 9 | +0.185 | 0.692 | 28 |
| 10 | +0.644 | 0.754 | 28 |
| 11 | +0.646 | 0.745 | 28 |
| 12 | +0.652 | 0.741 | 28 | **<- locked**
| 13 | +0.175 | 0.674 | 28 |
| 14 | +0.095 | 0.674 | 28 |
| 15 | +0.036 | 0.669 | 28 |
| 16 | +0.064 | 0.672 | 28 |
| 17 | -0.048 | 0.663 | 28 |
| 18 | -0.152 | 0.649 | 28 |
| 19 | +0.062 | 0.674 | 28 |
| 20 | +0.448 | 0.672 | 28 |

Peak: lookback=12 (Sharpe=+0.652). Shape: single isolated peak (spike).

### OOS_H2

| lookback | Sharpe | SE(Sharpe) | n_obs |
|---|---|---|---|
| 2 | -0.635 | 0.660 | 28 |
| 3 | -0.435 | 0.657 | 28 |
| 4 | -0.014 | 0.667 | 28 |
| 5 | -0.575 | 0.658 | 28 |
| 6 | -0.139 | 0.664 | 28 |
| 7 | -0.267 | 0.661 | 28 |
| 8 | -0.530 | 0.664 | 28 |
| 9 | -0.179 | 0.662 | 28 |
| 10 | +0.230 | 0.671 | 28 |
| 11 | +0.600 | 0.662 | 28 |
| 12 | +0.440 | 0.656 | 28 | **<- locked**
| 13 | +0.410 | 0.633 | 28 |
| 14 | +0.144 | 0.662 | 28 |
| 15 | -0.247 | 0.650 | 28 |
| 16 | -0.283 | 0.659 | 28 |
| 17 | -0.052 | 0.663 | 28 |
| 18 | -0.001 | 0.667 | 28 |
| 19 | -0.137 | 0.663 | 28 |
| 20 | -0.290 | 0.665 | 28 |

Peak: lookback=11 (Sharpe=+0.600). Shape: smooth interior peak (plateau-like).

## Autocorrelation of the locked (12, 1) return series

IS: lag-1 autocorr=+0.072, T=36 months, rough N_eff~31.2
OOS: lag-1 autocorr=+0.108, T=56 months, rough N_eff~45.1

N_eff = T(1-rho)/(1+rho), the standard AR(1) effective-sample-size approximation -- indicative, not exact (monthly momentum returns are not a clean AR(1) process).
