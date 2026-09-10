# PREREGISTRATION — CrossSectionalFactors, cycle 2 (H-MOM-ENSEMBLE)

Written 2026-09-10, before computing the composite signal below against
returns. This is a genuinely new hypothesis and a separately-budgeted
trial, **not** a re-lock of cycle 1's candidate.

## Context

Cycle 1 (`PREREGISTRATION.md`, `reports/cycle1.md`) locked single-lookback
momentum at lookback=12/skip=1 and failed the full verdict on two grounds:
`parameter_sensitivity` flagged SPIKE, and DSR at cumulative N=9 was 0.124
(cumulative N was already too expensive for the observed OOS Sharpe). A
follow-up diagnostic (`reports/momentum_sensitivity_diagnostic.md`) found
the SPIKE label was largely an artifact of `validation/sensitivity.py`
comparing the grid's *global* max jump (which fell between the unrelated
lookback=8/10 points, a real regime boundary unrelated to 12's own
stability) against 12's own *local* neighborhood (which was smooth). That
tool bug is now fixed (commit `03e05df`) — under the corrected tool, cycle
1's own sensitivity check would have read PLATEAU.

**This does not retroactively change cycle 1's recorded FAIL** — DSR failed
independently, on its own terms, and that reason still stands untouched.
Per prior instruction, past cycles are not replayed.

## Important methodological caveat — stated honestly, not glossed over

The diagnostic that motivated this cycle characterized OOS performance
(2022-2026) across a dense lookback grid and found a "good zone" of roughly
10-14 months. If this cycle's ensemble were built by selecting that
empirically-best-performing zone, the test would be circular: OOS 2022-2026
would no longer be a genuine held-out period for this new hypothesis — it
would just be reporting the result of a search that already happened.

**To avoid this, the ensemble below uses a construction that is NOT
calibrated to the empirically observed 10-14 zone.** It uses the standard
academic multi-horizon momentum convention — J ∈ {3, 6, 9, 12} months, the
four formation horizons most commonly tested since Jegadeesh & Titman
(1993) — chosen for being a generic literature convention, not because
these particular four scored best in cycle 1 or the diagnostic (they
didn't: J=3 was cycle 1's *worst* grid point, IS Sharpe +0.550).

**Honest residual limitation**: OOS 2022-2026 was already examined
extensively during cycle 1's verdict pipeline and the diagnostic
(walk-forward OOS Sharpes, Monte Carlo on OOS returns, a dense lookback
scan on OOS). No genuinely fresh data exists beyond 2026-09-09 to hold out
instead. What is preserved is narrower than a true blind test: the specific
ensemble construction (which horizons, how combined) is fixed by an
external convention rather than reverse-engineered from this data's own
performance curve. This cycle's result should be read with that caveat, not
as a fully independent confirmation.

## Hypothesis

**H-MOM-ENSEMBLE**: cross-sectional composite momentum combining 4
formation horizons (J ∈ {3, 6, 9, 12} months, skip=1 month uniformly —
matching cycle 1's locked skip convention) via cross-sectional percentile
rank averaging (not raw-return averaging, which would let the longest
horizon's larger-magnitude returns dominate the composite). This is the
standard construction for multi-horizon composite momentum scores (cf.
`quant-validation-methodology.md`'s CTREND reference; the crypto/Jesse
track's own cycle 5 composite-momentum design, "3 lookback-горизонта ROC").
Composite score is only defined where all 4 horizons have valid data
(effectively gated by J=12's own history requirement — no loss of usable
history relative to cycle 1). Long top quintile / short bottom quintile of
the composite rank, monthly rebalance, `n_per_leg=9`, 15bps one-way cost —
all identical to cycle 1's portfolio construction; only the signal changes.

**Economic rationale**: unchanged from cycle 1 (retail dominance in MOEX
trading volume, asymmetric short-selling frictions, post-2022 capital
segmentation — see `PREREGISTRATION.md`). This cycle does not add a new
economic story; it tests whether the same hypothesis is more robustly
captured by a multi-horizon construction than by a single point estimate —
itself a standard robustness practice in the momentum literature (averaging
across horizons reduces the estimation noise inherent in picking one
lookback, which is exactly what cycle 1's dense scan showed has SE ~0.5-0.7
per point at this sample size).

## Trial budget — fixed, will not be exceeded

**One trial.** No grid. This is a deliberate departure from cycle 1: the
entire point of moving to an ensemble is to stop searching over which
single lookback to use. Phase A1 (raw significance test, IS period,
2018-2021) evaluates the composite signal exactly as specified above — if
it fails (p >= 0.10), the hypothesis is falsified at a cost of exactly 1 to
cumulative N, no further steps. If it passes, that same evaluation is what
gets locked — no further optimization, so the cost stays 1 either way.

**Cumulative MOEX-track DSR-N after this cycle: 10** (9 + 1), regardless of
outcome.

**Post-lock robustness check (does not count toward N)**: matching cycle
1's own precedent — its post-lock `parameter_sensitivity` perturbations
were computed but never logged as trials, since they measure the
already-selected candidate's stability rather than compete to be selected.
Here: shift the whole 4-horizon set uniformly by {-2, -1, +1, +2} months
(e.g. +2 tests J ∈ {5, 8, 11, 14}) and report OOS Sharpe at each shift — a
composite-signal analog of `parameter_sensitivity`, since that tool's
single-scalar multiplicative perturbation model doesn't apply to a 4-value
set. Logged to the ledger as `stage=diagnostic_curve`,
`counts_toward_n=false`, exactly like the earlier momentum diagnostic.

## Falsification criteria — fixed now, will not be reinterpreted after seeing results

1. **Phase A1 gate**: raw significance test (stationary bootstrap, IS
   2018-2021 only) on the composite signal. p >= 0.10 -> falsified, stop —
   no lock, no OOS access, no further steps.
2. **No sign-flipping**: long top / short bottom of the composite rank is
   fixed for the whole cycle.
3. **Final verdict** (only if Phase A1 passes) requires **all** of:
   - OOS walk-forward: mean OOS Sharpe > 0.
   - Horizon-shift robustness: OOS Sharpe stays positive across all 4
     shift points (the ensemble-appropriate stand-in for
     `parameter_sensitivity`'s plateau check).
   - Monte Carlo (block bootstrap): `percentile_rank <= 95`.
   - DSR at cumulative N=10: `dsr > 0.95`.
   - **PBO is explicitly out of scope this cycle** (it needs >= 2
     competing trial columns; there is exactly one trial here by design) —
     stated as a limitation, not silently skipped.

Missing any applicable criterion is a fail — no partial credit.
