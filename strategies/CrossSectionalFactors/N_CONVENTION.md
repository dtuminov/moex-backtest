# DSR trial-counting convention for the MOEX track

Decided 10.09.2026, after `reports/why_no_alpha.md` showed the previous
convention could not pass a real effect at any sample size. Implemented in
`trial_ledger.py`. This file is the rule; the module is the arithmetic.

## The rule

Two counters, answering two different questions. Only the first one is ever
substituted into a formula.

**1. Mechanism-scoped N — goes into `deflated_sharpe_ratio(n_trials=...)`.**
Counts only trials of the same economic mechanism as the candidate being
judged. Current state (`uv run python strategies/CrossSectionalFactors/trial_ledger.py`):

| mechanism | N | what it covers |
|---|---:|---|
| momentum | 7 | cycle 1 grid + Phase A1 (6), cycle 2 ensemble (1) |
| pairs_cointegration | 2 | `moex-pairs-trading` validation-layer calibration |
| lowvol | 1 | cycle 1 Phase A1 |
| *a new mechanism* | 0 | starts at its own cycle's trials, inherits nothing |

**2. Project-wide count — reported, never substituted.** Currently **10**.
Carried in every cycle report next to the DSR line, as the standing
Harvey/Liu/Zhu (RFS 2016) caveat: a researcher who has worked through many
distinct anomalies should be more skeptical overall, even though no single
DSR computation sees that history.

## Why this changed

The previous convention fed the project-wide count into DSR. With N=9 that
puts `E[max SR|H0]` at **1.138** annualized, while the largest true effect
this track has actually observed is **+0.562** (cycle 1 momentum, real OOS).

That threshold is **invariant to sample size**. DSR deflates by how many
trials were searched, not by how much data was collected: as T grows, only
the estimator's variance around the true Sharpe shrinks, while `E[max SR|H0]`
stays put. So a true effect below the threshold converges, with more data,
toward *certainly failing* — `scripts/audit/t_needed.py` measures power flat
at ~0% out to 1200 months (100 years) of synthetic history. The gate was not
underpowered in a way more data could fix; it was measuring the wrong N.

Mechanism-scoping is not a loosening for convenience. Bailey & López de Prado
define `N` as the number of trials in the **one selection process** that
produced the candidate. A momentum grid tells you nothing about how likely an
untested dividend-gap hypothesis is to be overfit, so including it in that
hypothesis's `N` inflates the threshold with irrelevant history. Restoring
the formula's own quantity is the fix; the project-wide count keeps the
legitimate cross-hypothesis concern visible instead of silently dropping it.

Honest cost: mechanism-scoping is a **secondary** lever, not a rescue. On the
corrected stand, moving from N=9 to N=2 raises full-gate power at a true
Sharpe of 1.0 only from **1.6% to 15.6%**. The gate stays conservative.

## How a cycle applies it

1. Register the hypothesis's mechanism in `MECHANISM_BY_HYPOTHESIS`
   (`trial_ledger.py`) **before** the cycle runs. An unregistered name
   raises — deliberately, so that "momentum, but rebuilt differently" cannot
   quietly reset N to 1. Reusing an existing mechanism inherits its N; a
   genuinely new mechanism starts at 0 and counts only its own trials.
2. `n_trials = mechanism_n(mechanism) + <trials this cycle>`.
3. Report both numbers: the mechanism-scoped N that DSR used, and
   `project_wide_n()` beside it.
4. `counts_toward_n: false` still marks post-lock diagnostics and robustness
   checks, which are not selection and never enter either counter.

## Explicitly not changed

- **The gate stays AND-of-6.** Relaxing to ≥5-of-6 was measured on the
  corrected stand (α≈4.8%, power 33.2% vs 1.6% at SR=1.0) and **rejected**:
  DSR is the one criterion that encodes how many trials were searched, and a
  rule that lets precisely that criterion fail dilutes the only defense
  against selection bias the gate has.
- **No a-priori IS-Sharpe screening bar.** The measured IS→OOS shrinkage
  (0.346) and the N=2–3 thresholds (0.39–0.64) imply a hypothesis needs a
  plausible IS Sharpe near 2.0–3.0 to have workable power. That is recorded
  as context for judging a pre-registration, not as a gate that blocks a
  cycle from starting.
