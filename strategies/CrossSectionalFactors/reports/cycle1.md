# Cycle 1 report -- CrossSectionalFactors (momentum + low-vol)

Run: 2026-09-09T21:31:44.397382+00:00
IS period: 2018-01-01 .. 2021-12-31 (48 formation dates)
OOS period: 2022-01-01 .. 2026-09-09 (56 formation dates)
Cumulative MOEX-track DSR-N after this cycle: **9**

## momentum

Locked config: lookback=12 skip=1  (IS Sharpe=+1.624)
OOS Sharpe (annualized): +0.562
Walk-forward: 7 windows, mean IS=+0.965, mean OOS=+1.276, efficiency=132.2%
Sensitivity: SPIKE
Monte Carlo: iid percentile=49.5, block percentile=46.7
DSR (N=9): 0.1242 (E[max SR|H0]=+1.138)
PBO (6 trials, 16 blocks): 0.032

**VERDICT: FAIL** -- sensitivity: SPIKE, not plateau; DSR 0.1242 <= 0.95 (cumulative N=9)

