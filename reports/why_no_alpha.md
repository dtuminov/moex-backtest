# Why 10+ research cycles, two tracks, zero proven edges — diagnosis and recommendation

Scope: audits the MOEX track's own power-analysis stand (`scripts/gate_power_analysis.py`,
commit `eb90ff8`), extends it, and cross-checks the "no edge found" outcome against what is
actually required for [[finam-quant-transition]]. Everything below was computed on real data
in this repo; every number is either a direct quote from an existing report/ledger or a fresh
computation from a script listed in the relevant section. Nothing here increments cumulative
N or touches `experiments/grid_search_log.jsonl` / any lock file — this is calibration and
diagnosis, not a new trial. No file under `src/moex_backtest/validation/` was modified.

**Headline, ahead of the detail**: the audit stand that produced the project's own headline
numbers (memory `moex-quant-track.md` §15: "α=1.6%, power=38.4% at SR=1.0, DSR is the
constraint") has a real bug — its noise template is not actually demeaned, despite the
docstring's claim. Fixing it makes the diagnosis *more* pessimistic, not less: power at a
realistic true Sharpe (0.5–1.0) drops to **0–5%**, and a dedicated experiment shows this
cannot be fixed by collecting more data at N=9 — a true Sharpe of 0.562 (cycle 1's actual
observed OOS Sharpe) provably never clears the N=9 threshold, at *any* sample size, including
100 years of data. Separately, the actual, stated requirement for the goal this track serves
is not a certified backtested edge at all.

---

## (a) Diagnosis, with numbers

### A0. A bug in the audit tool itself, found while extending it

`scripts/gate_power_analysis.py`'s docstring (line 15) claims the noise template is "the REAL
empirical residual shape ... (subtracted its own mean, so shocks are shocks)". Grepping the
file for `mean`/`demean` finds exactly one match — that comment. There is no `.mean()`
subtraction anywhere in `_load_real_templates()` or `_run_one_replicate()`. The block-bootstrap
resamples the **raw, non-demeaned** real momentum returns and adds the target drift on top of
whatever real historical mean the column already had.

Real full-period (2018–2026) annualized Sharpe baked into each of the 12 grid columns used by
the audit (computed via `corrected_gate_power.py`, reused from the real templates):

| config | 3_0 | 3_1 | 6_0 | 6_1 | 8_1 | 10_1 | 11_1 | 12_0 | **12_1 (locked)** | 13_1 | 14_1 | 16_1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real Sharpe | 0.42 | 0.14 | 0.56 | 0.34 | 0.19 | 0.76 | 0.98 | 1.10 | **0.93** | 0.57 | 0.38 | 0.11 |

So every "true SR = X" row in `reports/gate_power_analysis.md` was actually testing an
*effective* true SR of roughly `X + 0.93` for the DSR/significance/walk-forward/Monte-Carlo
criteria (which use the locked `12_1` column) — the row labeled "0.00 (Type I error)" was
really testing behavior at an effective Sharpe near 0.9, not a genuine null. This directly
explains a detail memory §15 flagged without diagnosing it: "significance проходит 62.8% уже
при SR=0 — цена дешёвого гейта" — the gate wasn't dishonest; the "null" it was being tested
against wasn't actually null.

**Fix applied (in a new scratch script, not in the checked-in `scripts/gate_power_analysis.py`
— flagged here per the task's rule against silently changing validation code)**: subtract each
column's own full-period mean before resampling. Same seed (42), same 250 replicates/point,
same real `trial_sharpe_variance=0.5596`, same production N=9. Full corrected 6-criterion
table:

| true SR | significance | walk_forward | sensitivity | monte_carlo | DSR (N=9) | PBO | **ALL-6** |
|---|---|---|---|---|---|---|---|
| 0.00 (α) | 17.2% | 60.8% | 92.0% | 100.0% | 0.0% | 44.0% | **0.0%** |
| 0.50 | 40.0% | 87.6% | 94.8% | 100.0% | 0.8% | 47.2% | **0.0%** |
| 0.75 | 60.0% | 97.2% | 93.6% | 100.0% | 1.6% | 45.2% | **0.8%** |
| 1.00 | 64.8% | 96.0% | 92.4% | 100.0% | 5.2% | 48.8% | **1.6%** |
| 1.50 | 79.6% | 100.0% | 90.4% | 100.0% | 32.4% | 45.6% | **12.0%** |
| 2.00 | 96.8% | 100.0% | 96.4% | 100.0% | 50.4% | 48.8% | **26.8%** |

Corrected α is *lower* than reported (0.0% vs 1.6% — makes sense, the fake drift is gone), but
corrected power at a realistic effect (SR=1.0) collapses from the reported 38.4% to **1.6%**.
Even at SR=2.0 — a spectacular, rarely-seen edge — corrected power is only 26.8%, not 68.8%.
The qualitative conclusion memory §15 drew ("DSR is the connecting constraint") **still
holds** (DSR's own marginal pass rate, 0–50%, is still far below the other five criteria's
44–100%) — but the quantitative picture the project has been operating on since 10.09.2026 was
substantially too optimistic. This is worth fixing in `scripts/gate_power_analysis.py` itself
(one-line change: demean in `_load_real_templates`) and rerunning as a tracked update to
`reports/gate_power_analysis.md` — not done here, since that file is production output the
task asked not to silently alter.

**A second, related correction**: memory §15 tested and rejected relaxing the vote from "all 6"
to "≥5 of 6" because it measured α=57.6% at (mislabeled) SR=0. Recomputed with the same
demeaning fix:

| true SR | ALL-6 (AND) | ≥5 of 6 | ≥4 of 6 |
|---|---|---|---|
| 0.00 | 0.0% | **4.8%** | 32.8% |
| 0.50 | 0.0% | 18.4% | 57.6% |
| 0.75 | 0.8% | 23.6% | 73.2% |
| 1.00 | 1.6% | 33.2% | 74.8% |
| 1.50 | 12.0% | 46.4% | 90.0% |
| 2.00 | 26.8% | 68.0% | 97.6% |

Corrected, ≥5-of-6 gives α≈4.8% (essentially the nominal 5% target) with a real, large power
gain (e.g. 33.2% vs 1.6% at SR=1.0) — the opposite of what §15 concluded, because §15's
rejection was itself built on the buggy numbers. ≥4-of-6 is still clearly too loose (33%
false-positive rate). This is a legitimate lever, with one honest caveat: DSR is the one
criterion that specifically encodes "how many trials were searched" (the multiple-testing
correction); a voting rule that lets DSR be the one criterion that fails is, philosophically,
diluting the part of the gate that exists *specifically* to prevent p-hacking-by-search. It
buys power at a real methodological cost, not a free lunch.

### A1. Does mechanism-scoped N fix the power problem? (task item 1)

Mechanism-scoped N, retroactively applied to the real ledger
(`strategies/CrossSectionalFactors/experiments/grid_search_log.jsonl`): momentum mechanism =
6 (cycle 1 grid+Phase A1) + 1 (cycle 2 ensemble) = **7**; low-vol = 1; the pairs-calibration
N=2 is a different mechanism entirely and is correctly excluded. This matches the open proposal
already on record in memory §15.

Fine N grid (2..10), **corrected (demeaned) simulation**, full ALL-6 gate:

| true SR | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 | N=9 | N=10 |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.50 | 1.6% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.75 | 6.4% | 4.0% | 2.8% | 2.4% | 1.6% | 1.2% | 1.2% | 0.8% | 0.8% |
| 1.00 | 15.6% | 9.2% | 5.2% | 4.8% | 4.0% | 2.4% | 2.4% | 1.6% | 1.6% |
| 1.50 | 23.6% | 20.4% | 16.8% | 14.4% | 12.8% | 12.4% | 12.4% | 12.0% | 11.2% |
| 2.00 | 39.6% | 34.0% | 32.4% | 32.0% | 30.0% | 29.2% | 27.2% | 26.8% | 26.0% |

**Answer to "is there a convention that gives acceptable power without pushing α above ~5%":**
α is a non-issue after the bug fix — it stays at 0.0% from N=2 all the way to N=10 (the other
five criteria already filter noise hard enough on their own; DSR's N barely matters for α once
the null is genuinely null). The real constraint is power, and **N barely moves it**: going
from the production N=9 all the way down to the smallest legally defensible N=2 (Bailey/López
de Prado's formula requires N≥2) raises power at a realistic SR=1.0 from 1.6% to 15.6% — a real
but modest gain, and nowhere near "solved." At the true OOS Sharpe cycle 1 actually observed
(+0.562), even N=2 barely helps (see A3 below). **N-convention is a secondary lever, not the
fix** — this is the opposite of what the pre-correction numbers implied.

### A2. Is T too small — and does a rebalance-frequency change help? (task item 2)

Real weekly-rebalanced momentum was built on the daily bars already on disk (same 12-1
economic window, re-expressed as 52-week lookback / 4-week skip; same locked construction:
`high_long`, `n_per_leg=9`, `cost_bps=15`), backtested for real (`weekly_rebalance_real.py`),
not simulated:

| | monthly (cycle 1, real) | weekly (real, this audit) |
|---|---|---|
| T (OOS) | 56 months | 234 weeks (4.2×) |
| mean turnover | 23.2%/month | 11.61%/**week** → 50.5%/month-equivalent (2.2×) |
| OOS Sharpe, gross | ≈0.60 | **+0.233** |
| OOS Sharpe, net | +0.562 | **+0.189** |
| annualized cost drag | ≈42 bps/yr | **90.9 bps/yr** |

Two independent problems, both real, both measured: (1) weekly rebalance **more than doubles**
turnover-driven cost drag, as expected; (2) the **gross** signal itself collapses at weekly
resolution (0.233 vs ≈0.60) — cross-sectional momentum ranks computed weekly are simply
noisier/more reversal-contaminated than monthly ranks, a real, not hypothetical, finding.

Isolating whether the T gain alone (holding this real weekly residual's own autocorrelation/
skew/kurtosis shape fixed) helps DSR power: `weekly_power_audit.py` block-bootstraps the real,
**properly demeaned** weekly residual and reruns significance/walk-forward/Monte-Carlo/DSR
(sensitivity and PBO excluded — they need a multi-config grid this diagnostic doesn't build):

| true SR | sig | walk_forward | monte_carlo | DSR N=7 | DSR N=9 | 4-of-4 |
|---|---|---|---|---|---|---|
| 0.00 | 12.8% | 55.6% | 100.0% | 0.0% | 0.0% | 0.0% |
| 0.50 | 40.8% | 86.0% | 100.0% | 0.4% | 0.4% | 0.0% |
| 0.75 | 46.8% | 93.6% | 100.0% | 1.6% | 0.8% | 0.8% |
| 1.00 | 68.0% | 96.8% | 100.0% | 6.0% | 4.4% | 3.2% |
| 1.50 | 86.8% | 100.0% | 100.0% | 27.2% | 21.2% | 16.8% |
| 2.00 | 96.8% | 100.0% | 100.0% | 50.0% | 44.4% | 42.4% |

Compared apples-to-apples against the *corrected* monthly ALL-6 numbers in A0 (1.6%/12.0%/26.8%
at SR=1.0/1.5/2.0, on a stricter 6-criterion basis that also includes sensitivity and PBO),
weekly's 4-of-6-criteria numbers (3.2%/16.8%/42.4%) are roughly comparable, not a clear win —
once the missing sensitivity+PBO criteria are accounted for (PBO alone passes only ~45–49% of
the time in the monthly audit, an attrition weekly's 4-of-4 number doesn't yet include), the
honest weekly full-gate estimate is likely similar to or worse than monthly, not better.
**The 4.2× growth in T is real, but it is fully absorbed by a fatter-tailed, more strongly
negatively-skewed return distribution at weekly frequency** (raw weekly residual skew=-1.19,
kurtosis=10.3, vs monthly's much milder shape) — DSR's step-2 variance correction
(`(kurtosis-1)/4 · SR²` term) penalizes exactly this, offsetting most of the `1/√T` gain. Part
of this (not all — even excluding the single reopening-halt week of 2022-03-25, skew/kurtosis
drop to -0.31/3.97 but power is still ≈5.6% at SR=1.0, N=9) traces to the same known 2022
market-halt data discontinuity documented in memory §10, landing undiluted in one weekly bar
instead of averaged across a full month.

**A cleaner, decisive test of "is T the real bottleneck"**: `t_needed.py` block-bootstraps the
real, demeaned locked `12_1` monthly residual to synthetic OOS lengths, holding the *actual
observed* true Sharpe (+0.562, cycle 1's real number) fixed, and asks how much T would be
needed:

| T (months) | T (years) | DSR power, N=9 | DSR power, N=2 |
|---|---|---|---|
| 56 | 4.7 | 1.0% | 14.8% |
| 100 | 8.3 | 0.0% | 14.5% |
| 300 | 25.0 | 0.0% | 23.0% |
| 600 | 50.0 | 0.0% | 31.5% |
| 900 | 75.0 | 0.0% | 43.8% |
| 1200 | 100.0 | **0.0%** | 54.8% |

At N=9, power is **flat at ~0% out to 100 years of data** — not a typo, and not noise: the
production DSR threshold at N=9 (E[max SR|H0]=1.138 annualized, computed in A3 below) exceeds
the true Sharpe (0.562) itself, and this threshold is **provably T-invariant** (the formula's
`E[max SR_N]` term doesn't shrink with T; only the *estimator's* variance around the true value
shrinks). A true effect smaller than the threshold converges, with more data, toward certainly
failing DSR, not toward passing it — this is DSR working exactly as designed, not a data
shortage. Even at the best-case N=2 (threshold ≈0.389, which 0.562 *does* exceed), reaching
even 55% power needs on the order of 100 years of OOS history, because the true effect sits
close to the threshold and the residual distribution is fat-tailed. **Conclusion: T is
genuinely a real constraint, but it is not a fixable one on this dataset/instrument/horizon —
no realistic rebalance-frequency change or additional years of MOEX history (2018–2026 is
already the full available depth) closes this gap.**

### A3. What does this mean for screening future hypotheses?

`E[max SR|H0]` (annualized, using momentum's real trial-Sharpe variance=0.5596 as an
illustrative proxy — a genuinely new mechanism would have its own, unknown variance) by N:

| N | 2 | 3 | 4 | 5 | 6 | 7 | 9 | 20 | 50 |
|---|---|---|---|---|---|---|---|---|---|
| threshold | 0.389 | 0.638 | 0.787 | 0.892 | 0.973 | 1.037 | 1.138 | 1.422 | 1.703 |

And the realized IS→OOS shrinkage this track has actually observed twice now (cycle 1:
momentum IS Sharpe +1.624 → OOS +0.562, a ratio of **0.346**) is the best available calibration
for "how much of an exciting in-sample Sharpe survives contact with held-out data" on this
specific dataset. Combined: a new hypothesis needs an a priori plausible **in-sample** Sharpe
of roughly 2.0–3.0 to have a realistic shot at leaving enough OOS Sharpe (≈0.7–1.0) to clear
even a generous mechanism-scoped N=2–3 threshold with workable (not just theoretical) power —
this is a concrete, computed screening bar, not a round number.

### A4. Is the hypothesis/universe space wrong? (task item 3)

Mechanism-based case for MOEX-specific niches (already researched, memory §7 — restated here
with the screening bar from A3 applied, not re-litigated from scratch):

- **Retail dominance (70.7% of 2025 equity volume, MOEX data)** is the one structural fact that
  actually differentiates MOEX from arbitraged developed/crypto markets — it licenses
  behavioral mechanisms (underreaction/momentum, overreaction/reversal) at a scale unlikely to
  already be arbitraged away by the kind of capital that dominates flow elsewhere.
- **Asymmetric, discretionary short-selling constraints** (exchange can halt new shorts on a
  name at its own discretion — documented case, ЕвроТранс, July 2026) is a *specific,
  mechanical* friction, not a vague "market inefficiency" claim — it predicts a testable,
  directional asymmetry (drift stronger/longer on the long side after bad news than mean-
  reversion trades can arbitrage away), which is a sharper hypothesis than generic momentum.
- **Post-2022 capital segmentation** (foreign arbitrage capital largely cut off) plausibly
  raises the *ceiling* on how much of a real anomaly survives before being arbitraged flat —
  this is a reason to expect a higher true SR on a genuine niche than the same construction
  would have on, say, S&P 500 names, not a reason any specific factor here works.
- **What this track has NOT yet tested, and which the retail/friction argument specifically
  supports**: dividend-gap predictability (documented pattern: 113 days-to-close in 2021 vs 33
  in 2023 — a large regime swing, itself informative about time-varying retail attention, but
  a genuinely different, shorter-holding-period mechanism than momentum, worth revisiting now
  that fundamental-data-quality concerns can be scoped rather than assumed disqualifying);
  second-echelon/lower-liquidity name illiquidity premium (a distinct mechanism from momentum:
  compensation for retail-dominated order books being harder to fill, not price
  under/overreaction); and momentum specifically *conditioned* on the short-constraint
  asymmetry (long legs only, or long-short with a wider band on the short side) rather than
  symmetric long-short momentum, which is what was actually tested and failed twice.
- **What should stay deprioritized**: value (fundamental-data-quality risk, unchanged from
  memory §7's original reasoning) and generic dividend-gap-as-momentum-clone (crowds the same
  retail flow the rest of the case rests on).

None of this is guaranteed to clear the A3 bar — it hasn't been tested — but it is a
mechanism-first case, not a factor-name list, and it directly targets frictions plausible only
on this specific market, which is the actual argument for why MOEX might have room that
picked-over developed-market factors don't.

### A5. Is the problem framing itself wrong? (task item 4)

This is the one place the diagnosis goes beyond statistics. `memory/finam-quant-transition.md`
records the actual, stated requirement from the head of Finam's quant department (call,
11.08.2026): (1) register for the Finam hackathon, (2) participate in **Finam Arena**
(₽3,000,000 on a contest account, **2 months on real live quotes** via Finam Trade API,
evaluated by realized return — not a backtested DSR/PBO certificate), (3) a quant-oriented
resume built around **pet projects + demonstrated areas of knowledge**. Nowhere in that
requirement is "a statistically certified backtested edge." The actual gate this track is
ultimately serving does not require what this track has spent 10+ cycles failing to produce.

Given A2's finding — a true, real Sharpe of 0.562 on this exact dataset/construction
mathematically cannot pass the production gate at *any* sample size — continuing to search for
a DSR-certifiable edge on 47 MOEX names, daily bars, monthly-scale cross-sectional factors, is
close to provably not the fastest path to the actual goal. What this track *has* produced that
directly serves the stated requirement: a real typed data client that broke a real API
limitation (`FinamClient`), a 47-name cross-sectional research pipeline, and — the most
differentiated asset — a from-scratch, textbook-correct DSR/PBO/walk-forward/sensitivity
validation stack, exercised honestly enough that it just caught a bug in its own power-audit
tool mid-task. A hiring manager in a real quant department is far more likely to be impressed
by "here is my validation methodology, here is how I measured its own statistical power, here
is a bug I found and fixed in my own audit code" than by an LLM-cycle "found alpha" claim,
which any competent reviewer would (correctly) distrust on priors alone.

---

## (b) Ranked options

**1. Fix the compass — cheap, do regardless of what else is chosen.**
Effect: corrects the project's own understanding of its gate (this report *is* that
correction); formally adopt mechanism-scoped N (momentum=7, low-vol=1, future mechanisms
start at 1) as the number that actually goes into the DSR formula, with the cumulative
project-wide count (currently 10) kept and reported separately per memory §15's original
Harvey/Liu/Zhu framing; decide, with eyes open about the methodological cost noted in A0,
whether ≥5-of-6 voting is acceptable (α≈4.8%, real power gain) or the full AND-of-6 stays.
Cost: a few hours — patch `scripts/gate_power_analysis.py`'s demeaning bug, rerun and replace
`reports/gate_power_analysis.md`, update memory §15's cited numbers, write the N-convention
decision down once. Verify: rerunning this report's scripts reproduces the corrected tables
above; no new trial, no N increment.

**2. Reframe the deliverable to what is actually being asked for — recommended primary step.**
Effect: converts 10+ cycles of already-completed, genuinely rigorous work (data
infrastructure, a from-scratch DSR/PBO/CSCV stack, an honest self-diagnosis that caught its own
tooling bug) into the actual pet-project + "areas of knowledge" artifact the department head
asked for, and unblocks Finam Arena participation (a fundamentally different, feasible
validation regime — 2 months, real quotes, evaluated by return, not by a statistical
certificate this specific setup has been shown mathematically unlikely to ever produce). Cost:
mostly psychological (letting go of "must find a certified edge" as the definition of success)
plus real but bounded packaging/integration work (resume, Arena registration + live/paper
execution wiring against `FinamClient`). Verify: hackathon registered, Arena participation
live, resume drafted and reviewed — outcomes external to this repo, checkable directly against
the three items memory records the department head actually asked for.

**3. One more disciplined, pre-registered cycle on a higher-plausible-SR, structurally
different mechanism — worth trying in parallel, not instead of #2.**
Candidates from A4: short-constraint-conditioned asymmetric momentum, dividend-gap timing,
second-echelon illiquidity premium — chosen because each rests on a friction specific to MOEX's
retail-dominated, capital-segmented structure, not a factor name already priced out elsewhere.
Go in pre-committed to the A3 screening bar (only proceed if the pre-registered economic
argument plausibly implies IS Sharpe ≳2.0–3.0, given the measured 0.346 IS→OOS shrinkage and
the N=2–3 threshold of 0.39–0.64) and to mechanism-scoped N (starts at 1, not inheriting 10).
Cost: one full cycle, budget 10-20 IS-trials per memory §6's rule, real risk it fails a third
time — A2's finding that T cannot be fixed on this dataset applies to any new hypothesis on the
same instrument/horizon just as much as to momentum. Verify: same pipeline, same pre-
registration discipline, this time with the corrected gate and honestly-scoped N.

**4. Reject: further rebalance-frequency tuning (weekly/daily) as a "more data" fix.**
A2 measured this directly, not on priors: turnover-driven cost drag more than doubles, the
gross signal itself weakens (0.233 vs ≈0.60 annualized Sharpe), and the T gain is absorbed by a
fatter-tailed, more negatively-skewed return distribution — net DSR power is a wash at best,
not an improvement. Do not spend a cycle on this again without a new, specific reason to expect
otherwise.

---

## (c) Recommendation

Do #1 immediately (it is cheap, corrects a real bug, and everything else should be built on the
corrected numbers, not the ones in memory §15), then pursue #2 as the primary next step and #3
as a bounded, pre-registered, parallel bet — not because the search-for-alpha framing was
wrong in principle, but because A2's `t_needed.py` result is decisive: on this exact
dataset/instrument/horizon, a true Sharpe of 0.562 (the actual, real, twice-independently-
observed OOS momentum result) mathematically cannot pass the production gate at any sample
size, and even the most generous defensible N-convention only gets there with on the order of
a century of data that does not exist and will not exist before this decision matters. That is
not "keep testing hypotheses" territory — it is a proof that this specific search, as
currently scoped, is very unlikely to ever produce what it has been asked to produce, while the
actual, stated requirement for [[finam-quant-transition]] (hackathon + Arena + a
petproject-and-knowledge-areas resume) does not require it at all, and the infrastructure and
methodology already built — including the fact that this very audit caught a real bug in its
own tooling — is a stronger, more honest artifact to show a quant hiring manager than a
backtested Sharpe number would have been anyway.

---

## Appendix: scripts and where the numbers came from

All scripts below live only in the session scratchpad (not committed), reuse the real,
already-reviewed code in `scripts/gate_power_analysis.py`,
`strategies/CrossSectionalFactors/{panel,factors}.py`, and `src/moex_backtest/validation/*`
unmodified, and read only real data already on disk (`data/raw/`,
`strategies/CrossSectionalFactors/experiments/grid_search_log.jsonl`). None write to the
ledger or a lock file; none increment cumulative N.

- `corrected_gate_power.py` — A0/A1: demeaning-bug fix, corrected 6-criterion table, corrected
  fine-N grid (2–10).
- `weekly_rebalance_real.py` — A2: real weekly-rebalanced momentum backtest (turnover, gross/
  net Sharpe) on real daily bars.
- `weekly_power_audit.py` — A2: DSR/significance/walk-forward/Monte-Carlo power at weekly T,
  using the real (properly demeaned) weekly residual as the noise template.
- `t_needed.py` — A2: DSR power vs synthetic OOS length at the real observed effect size
  (SR=0.562), N=9 and N=2.
- `mechanism_n_audit.py` — an earlier, pre-bugfix version of the N=2..10 sweep; superseded by
  `corrected_gate_power.py`'s fine-N table above but left as a record of the pre-correction
  numbers this report's A0 corrects.
