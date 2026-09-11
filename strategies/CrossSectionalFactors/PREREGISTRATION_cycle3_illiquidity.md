# PREREGISTRATION — CrossSectionalFactors, cycle 3 (H-ILLIQ)

Written 2026-09-10, before computing any illiquidity signal against
returns — no cross-sectional liquidity/return relationship has been looked
at on this dataset before (see "Circularity check" below). This is a
genuinely new economic mechanism, not a rebuild of momentum or low-vol, and
is registered as such in `trial_ledger.py`'s `MECHANISM_BY_HYPOTHESIS`
before this file was written.

## Why this mechanism, chosen from `reports/why_no_alpha.md` A4

A4 named three candidates: short-constraint-conditioned asymmetric
momentum, dividend-gap predictability, and second-echelon illiquidity
premium. Illiquidity is the one chosen for this cycle, for a data reason
and a mechanism reason:

- **Data reason**: neither of the other two is implementable without new
  data collection. Short-ban asymmetry needs a systematic record of which
  names had exchange-imposed short-selling restrictions and when — this
  project has exactly one documented anecdote (EuroTrans, 2026-07), not a
  dataset. Dividend-gap timing needs ex-dividend dates and dividend
  amounts, which are not in the cached `data/raw/*.parquet` files (OHLCV
  only, confirmed by inspection: `SECID, TRADEDATE, OPEN, HIGH, LOW, CLOSE,
  VOLUME`) and would require a new `FinamClient` data path, a new
  data-quality review, and a new cycle just to establish trust in the
  input — out of scope for a single cycle. Illiquidity needs only `VOLUME`,
  which is already in every cached file, already reviewed for data quality
  (`known_events.py`), for the same 47 names and dates cycles 1-2 already
  used.
- **Mechanism reason, from a cause, not a factor name**: `VOLUME × CLOSE`
  (daily traded value in rubles) is a direct proxy for how costly a name is
  to trade in size. In a market where 70.7% of 2025 equity volume is retail
  (moex.com/n76900) — participants who trade in small size, are not
  continuously providing two-sided liquidity, and are not arbitraging price
  impact across the liquidity spectrum the way professional market-makers
  do — the compensation an investor needs to hold a thinly-traded name
  (harder to enter/exit without moving the price) is a *different* causal
  story from momentum (speed of information diffusion) or low-vol
  (risk-seeking retail bidding up high-beta "lottery" names): it is a
  friction/compensation-for-immediacy story (Amihud & Mendelson 1986;
  Amihud 2002), not a behavioral-misreaction story. The same post-2022
  capital-segmentation argument used for momentum/low-vol
  (`PREREGISTRATION.md` point 3) applies here with, if anything, more
  force: the institutional/arbitrage capital that would normally price
  illiquidity risk efficiently across names is exactly the capital that has
  been reduced.

## Hypothesis

**H-ILLIQ**: cross-sectional trailing mean daily traded value (`VOLUME ×
CLOSE`, rubles) predicts next-month relative return, inverted — low-
liquidity (low mean traded value) names outperform high-liquidity names on
a risk-adjusted basis, compensating holders for the cost of trading them.
Long bottom (least-liquid) quintile, short top (most-liquid) quintile,
monthly rebalance.

**Deliberately not log-transformed**: `_quintile_legs` selects the top/
bottom `n_per_leg` names by rank, and rank order is invariant to any
monotonic transform (log or otherwise) of the signal. Log-transforming
raw mean traded value would change nothing about which names enter which
leg — stated here so the choice not to bother is visibly a reasoned one,
not an oversight.

**Deliberately not masking known-event days from the volume lookback**
(unlike `low_vol_signal`, which masks the 2022-03-24 halt-reopening return
and CBOM's 2026-04-13 event from realized-*volatility* windows via
`known_events.mask_returns_for_vol_estimation`): that function masks
*returns* because an extreme return distorts a variance estimate
quadratically. A volume spike distorts a 12-month rolling *mean* linearly
and boundedly — CBOM's 764.8M-ruble spike day contributes at most 1/252nd
of a 12-month window's average, a real but modest effect, not the kind of
estimator-breaking distortion the return-volatility case has. Kept
unmasked as a considered decision.

## Universe, data, and an honest power caveat specific to this hypothesis

Same 47-ticker universe, same cached daily OHLCV (2018-01-03 to
2026-09-09), same monthly rebalance convention as cycles 1-2. No new data,
no network calls.

**Caveat stated up front, not after seeing results**: `scripts/
import_universe.py`'s universe is explicitly curated as MOEX's *liquid*
names (memory `moex-quant-track.md` §2: "47 ликвидных акций"). Even this
universe's least-liquid decile is still substantially more liquid than
genuine second-echelon/thinly-traded MOEX names excluded from the universe
entirely. If the illiquidity-premium mechanism is real, restricting the
test to an already-liquidity-filtered universe likely **compresses** the
available cross-sectional spread and understates the effect relative to a
test that could include the true illiquid tail — this is a real limitation
on this cycle's power to detect the mechanism, independent of
`reports/why_no_alpha.md`'s T-driven power problem, and is not grounds to
discount a FAIL as merely "the universe was wrong" after the fact; it is
recorded now as a reason a null result here would be less informative
about the mechanism itself than about this specific implementation of it.

## Circularity check

Has any prior cycle on this project looked at the relationship between
liquidity/volume and forward returns on this data? No. The only prior
contact with per-name volume levels: the CBOM data-quality investigation
(`known_events.py`'s docstring, memory §2) established that CBOM's
*typical* daily volume is low (0.6-15M rubles) relative to its spike day
(764.8M) — this is a statement about one name's baseline trading activity,
used to confirm a data anomaly was a real corporate event, not a data
error. It carries no information about whether illiquid names as a group
earned higher or lower *forward returns* historically, so it does not
compromise this cycle's OOS as a genuine held-out test. Checked, judged
immaterial.

## Trial budget — fixed, will not be exceeded

One free parameter: the trailing-average lookback window. Portfolio depth
(`n_per_leg=9`, quintile-style — floor(47/5)) and holding period (1 month)
are fixed a priori for the same reason cycle 1 fixed them (memory §6.2:
sweeping a parameter that isn't the hypothesis itself invites a spike, not
a plateau) and for direct comparability with cycles 1-2's construction.

| Stage | Configs | Count |
|---|---|---|
| Phase A1 (cheap gate) | lookback=12mo (default) | 1 |
| IS grid (only if Phase A1 passes) | lookback ∈ {3, 6, 12} months | 3 (incl. the A1 point) |
| **Total ceiling** | | **3** |

Grid size matches H-LOWVOL's cycle-1 precedent exactly (same one free
dimension, same three points) — not picked smaller than that precedent for
this cycle specifically, and not picked larger. If Phase A1 fails, the
grid is not run and this cycle costs exactly 1 trial to mechanism-scoped N.

**Mechanism-scoped N this cycle feeds into DSR**: `mechanism_n("illiquidity")
+ trials this cycle` = `0 + (1 or 3)` = **1 or 3**, computed via
`trial_ledger.mechanism_n`, not a hardcoded constant. Project-wide count
(currently 10) is reported alongside per `N_CONVENTION.md`, never
substituted into the formula.

## Transaction costs

15 bps one-way on leg-membership turnover — identical convention to cycles
1-2, for comparability. **Stated limitation, not corrected here**: 15bps is
very likely an *understatement* for the bottom (illiquid) leg specifically
— the entire premise of an illiquidity premium is that thinly-traded names
cost more to trade, and a uniform cost assumption across both legs cannot
reflect that asymmetry. The main verdict below uses 15bps for direct
comparability with cycles 1-2's gate criteria; a **post-lock robustness
check** (not counted toward N — a stress test of the one already-selected
candidate, exactly like cycle 1's post-lock sensitivity perturbations and
cycle 2's horizon-shift check) reruns the locked config's OOS Sharpe at
30/50/100bps one-way to show how much of any edge is cost-assumption-
dependent.

## IS / OOS split

Identical to cycles 1-2: IS = 2018-01-01 to 2021-12-31 (Phase A1 + grid
argmax-IS-Sharpe selection only); OOS = 2022-01-01 to 2026-09-09, physically
gated behind `moex_backtest.validation.lock.require_locked_config` — no
code in this cycle computes an OOS return before the winning config (if
any) is written to `experiments/cycle3_illiquidity.lock.json`.

## Falsification criteria — fixed now, will not be reinterpreted after seeing results

1. **Phase A1 gate**: raw significance test (stationary bootstrap, IS
   2018-2021 only) on the default (lookback=12) config. `p >= 0.10` ->
   falsified at Phase A1, grid not run, cost to mechanism-scoped N is 1.
2. **No sign-flipping**: long-illiquid/short-liquid is fixed for the whole
   cycle. A negative Sharpe is reported as a negative Sharpe, never
   reinterpreted as "trade it the other way."
3. **Final verdict** (only if Phase A1 passes and the grid runs) requires
   **all** of:
   - OOS walk-forward: mean OOS Sharpe > 0.
   - `parameter_sensitivity` on the locked lookback: `is_plateau=True`.
   - `monte_carlo_block_bootstrap` on OOS returns: `percentile_rank <= 95`.
   - `deflated_sharpe_ratio` at mechanism-scoped N (`mechanism_n("illiquidity")
     + this cycle's trials`, i.e. 3 if the grid ran): `dsr > 0.95`.
   - `probability_of_backtest_overfitting` on the 3-config IS grid trial
     matrix: `pbo < 0.5`.
   Missing any is a FAIL for the cycle — no partial credit. A FAIL is a
   complete, informative, reportable result, not a reason to try a fourth
   grid point or a different cost assumption inside this cycle's N budget.

## What happens after this cycle

Reported in `reports/cycle3_illiquidity.md` regardless of outcome, with
both mechanism-scoped N (feeds DSR) and project-wide N (context only,
per `N_CONVENTION.md`) shown explicitly. The post-lock cost-sensitivity
check (if the cycle reaches a lock) is reported alongside the verdict, not
used to alter it. What to try next is left to the user, consistent with
`memory/feedback-github-only-proven-alpha.md` and this track's established
practice — not pre-committed here.
