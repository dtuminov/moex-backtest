# Cycle 2 report -- H-MOM-ENSEMBLE (multi-horizon composite momentum)

Run: 2026-09-10T05:20:46.852596+00:00
Horizons: (3, 6, 9, 12), skip=1 (fixed, no grid -- see PREREGISTRATION_cycle2_momentum_ensemble.md)
Cumulative MOEX-track DSR-N after this cycle: **10**

Phase A1 / only trial: IS Sharpe=+1.149, p=0.0330

OOS Sharpe (full period, annualized): +0.178
Walk-forward: 7 windows, mean IS=+0.639, mean OOS=+0.938

Horizon-shift robustness (post-lock, does not count toward N):

| shift | horizons | OOS Sharpe |
|---|---|---|
| 0 (locked) | (3, 6, 9, 12) | +0.178 |
| -1 | (2, 5, 8, 11) | -0.082 |
| +1 | (4, 7, 10, 13) | -0.050 |
| +2 | (5, 8, 11, 14) | -0.078 |

Shifts [-2] not attempted: shortest horizon would fall to <= skip=1 (structurally infeasible, not a data-driven exclusion).

Monte Carlo (block bootstrap) percentile: 47.7
DSR (N=10): 0.0248 (E[max SR|H0]=+1.100)
PBO: out of scope this cycle (single-trial design, no competing candidates)

**VERDICT: FAIL** -- horizon-shift robustness: at least one shift (or the base) had Sharpe <= 0; DSR 0.0248 <= 0.95 (cumulative N=10)
