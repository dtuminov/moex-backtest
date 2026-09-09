# PREREGISTRATION — CrossSectionalFactors, cycle 1

Written 2026-09-10, **before** any momentum or low-volatility signal has
been computed against MOEX returns. This is the first alpha-search cycle
of the MOEX track (`memory/moex-quant-track.md`) — the prior N=2 was a
calibration run against an already-known result, not a search.

## Hypotheses

**H-MOM (momentum)**: cross-sectional 12-1-month momentum (rank stocks by
trailing return, skipping the most recent month) predicts next-month
relative return on the MOEX 47-stock universe. Long top quintile, short
bottom quintile.

**H-LOWVOL (low-volatility)**: cross-sectional trailing realized volatility
predicts next-month relative return, inverted — low realized vol stocks
outperform high realized vol stocks on a risk-adjusted basis. Long bottom
(lowest-vol) quintile, short top (highest-vol) quintile.

Tested **separately**, not combined — a combined signal is out of scope for
this cycle and, if attempted at all, is a separate cycle 2 gated on both
factors individually passing this cycle's full verdict criteria (section
"Verdict criteria" below).

## Economic rationale (written before looking at any return data)

1. **Retail dominance**: private investors were 70.7% of MOEX equity
   trading volume in 2025 (74% in 2024; moex.com/n76900,
   smart-lab.ru/blog/news/1321376.php), dominant in the morning (84.3%) and
   evening (78.4%) sessions. A market this retail-heavy is the textbook
   precondition for underreaction-driven momentum and limits-to-arbitrage
   -driven low-vol mispricing — both anomalies' standard behavioral
   explanations assume a less-sophisticated, attention- and
   leverage-constrained participant base, which is a much better
   description of MOEX today than of a market dominated by institutional
   arbitrage capital.
2. **Asymmetric, discretionary short-selling frictions**: the exchange can
   and does suspend new short positions on a specific name when it judges
   volatility/imbalance risk elevated (2026-07 EuroTrans precedent,
   tbank.ru/invest/social/profile/T-Investments/pochemu-birzhi-zapreshchayut-korotkiye-pozitsii/).
   This is not a uniform, symmetric friction — negative information is
   structurally slower to arbitrage into price than positive information,
   which is a direct mechanism for momentum specifically (not just "frictions
   exist" in general).
3. **Post-2022 capital segmentation**: foreign institutional arbitrage
   capital has been largely locked out of MOEX since 2022. Both anomalies
   are well-documented to have compressed or disappeared in fully-arbitraged
   developed markets after publication/discovery (the standard
   limits-to-arbitrage story) — segmentation is a direct, structural reason
   the arbitrage capacity that erodes these effects elsewhere is unusually
   scarce here.
4. **Prior literature on MOEX specifically**: a 2019-2021 study of 16
   momentum strategies on Russian equities found positive momentum returns
   net of transaction costs, with lower volatility/tail risk than the index
   (ResearchGate 368823069) — this is not a purely imported-prior hypothesis,
   there is a direct domestic precedent.

**Explicitly not tested this cycle, and why**: value (fundamental data
quality is compromised for several universe members post-2022 —
redomiciliations, sanctioned entities, incomplete disclosure — testing value
now risks a noisy, low-power test that would burn DSR-N for a low-quality
signal); dividend-gap trading (the anomaly is extensively discussed among
MOEX retail traders themselves — smart-lab.ru, T-Bank community posts — a
crowded trade among exactly the participant base whose slow reaction is
supposed to be the edge's source is a weaker prior than momentum/low-vol).

## Universe and data

47 MOEX tickers per `scripts/import_universe.py` (as actually cached in
`data/raw/finam_daily_*.parquet`, verified 2026-09-10): AFLT, AKRN, ALRS,
ASTR, BANE, BSPB, CBOM, CHMF, DSKY, FEES, FLOT, GAZP, GMKN, HYDR, IRAO,
KZOS, LENT, LKOH, LSRG, MAGN, MGNT, MOEX, MTLR, MTSS, NLMK, NMTP, NVTK,
OZON, PHOR, PIKK, PLZL, POSI, ROSN, RTKM, RUAL, SBER, SELG, SIBN, SNGS, T,
TATN, TRNFP, UPRO, VKCO, VTBR, X5, YDEX.

**Acknowledged limitation, stated upfront**: this is today's liquid
universe applied retroactively across 2018-2026, not a real point-in-time
membership list — a form of survivorship bias (delisted/renamed names are
represented by their current successor ticker or absent entirely). Not
fixed this cycle; noted so the final verdict isn't read as stronger than it
is.

Daily OHLCV, 2018-01-03 to 2026-09-09, `ParquetCache`-backed, no network
calls needed for this cycle. Monthly rebalance at each month's last
available trading day.

**Data-quality handling, fixed by the already-implemented
`moex_backtest.data.known_events` (see `git log`, prior commit)**: the
2022-03-24 halt-reopening return and CBOM's 2026-04-13 +54% move are
excluded from rolling volatility-estimation windows only (they remain in
the price series and in the momentum return calculation — a momentum
signal is supposed to see the real return over its lookback, including a
month that happens to contain the halt or the CBOM event; only a
volatility estimate is distorted by treating either as an ordinary single
day).

## Trial budget — fixed, will not be exceeded

| Stage | Configs | Count |
|---|---|---|
| Phase A1 (cheap gate) | H-MOM: lookback=12mo, skip=1mo. H-LOWVOL: lookback=12mo | 2 |
| IS grid, H-MOM (only if H-MOM passes A1) | lookback ∈ {3,6,12} × skip ∈ {0,1} months | 6 (incl. the A1 point) |
| IS grid, H-LOWVOL (only if H-LOWVOL passes A1) | lookback ∈ {3,6,12} months | 3 (incl. the A1 point) |
| **Total ceiling** | | **9** |

`n_legs` (portfolio depth) is fixed a priori at **quintiles** (9 stocks per
leg, floor(47/5)) and is **not** swept — cycle 5 of the crypto/Jesse track
found sweeping portfolio depth produces a spike, not a plateau, and is not
even guaranteed to be a local optimum at the "obvious" choice. Holding
period is fixed at 1 month (matches the monthly rebalance, no overlapping
tranches — kept simple deliberately to hold trial count down).

If a hypothesis fails Phase A1, its grid points are **not** run — the
DSR-N cost is only the 1 Phase A1 trial for that hypothesis, not the full
6 or 3.

Combined MOEX-track cumulative DSR-N after this cycle: **2 (prior
calibration) + however many of the 9 configs above actually execute**
(reported exactly, not estimated, in the final report — see
`memory/moex-quant-track.md` for the running total).

## Transaction costs

15 bps one-way, applied to each name's monthly leg-membership turnover
(a name entering or leaving a leg pays the cost; a name that stays in the
same leg across a rebalance does not). Applied to every reported net
return series in this cycle — there is no "gross" headline number.

## IS / OOS split

- **Parameter-selection IS period**: 2018-01-01 to 2021-12-31 (Phase A1 +
  grid argmax IS Sharpe selection happens on this window only).
- **Locked, OOS period**: 2022-01-01 to 2026-09-09. Physically gated behind
  `moex_backtest.validation.lock.require_locked_config` — no code in this
  cycle computes a return on this window before the winning config (if any)
  is written to `experiments/cycle1.lock.json` via `lock_config`.
- Within the OOS period, `walk_forward_analysis` slides is_window=12mo /
  oos_window=6mo (non-overlapping OOS across windows) for a rolling
  stability check — this reuses the same tool and convention already
  established in `moex-pairs-trading/scripts/validate_pair_selection.py`
  for the sibling project; its "IS" label here means "earlier chunk of the
  already-locked, already-out-of-original-sample OOS period", not a second
  round of parameter fitting.

## Falsification criteria — fixed now, will not be reinterpreted after seeing results

1. **Phase A1 gate**: if the raw (unoptimized, default-config) stationary-
   bootstrap significance test on the IS period gives p >= 0.10 for a
   hypothesis, that hypothesis is **falsified at Phase A1** — its grid is
   not run, full stop. If both fail, the cycle ends here and is reported as
   a complete, informative negative result, not "inconclusive".
2. **No sign-flipping**: a negative Sharpe is never reinterpreted as
   "the opposite direction works" — that is the exact data-snooping move
   `FundingShockReversal` (crypto/Jesse cycle 8) pre-registered against.
   Long/short legs as defined above are fixed for the whole cycle.
3. **Final verdict** (only reached if Phase A1 passes and the IS grid runs)
   requires **all** of:
   - OOS walk-forward: reported mean OOS Sharpe > 0 (raw Sharpes reported
     directly, per the codebase's existing convention, if mean IS-of-
     walk-forward Sharpe <= 0 — efficiency % is not interpretable in that
     case).
   - `parameter_sensitivity` on the locked lookback: `is_plateau=True`.
   - `monte_carlo_block_bootstrap` on the OOS returns: observed Sharpe's
     `percentile_rank <= 95` (not in the lucky tail).
   - `deflated_sharpe_ratio` on the cumulative MOEX-track N: `dsr > 0.95`.
   - `probability_of_backtest_overfitting` on the IS grid trial matrix:
     `pbo < 0.5` (a value >= 0.5 is treated as a fail; < 0.2 is the
     comfortable zone, but not a separate hard requirement beyond < 0.5).
   Missing **any** of these is a fail for that hypothesis — no partial
   credit, no "close enough".

## What happens after this cycle

If both hypotheses fail Phase A1 (a legitimate possible outcome): report the
falsification, update the MOEX-track N ledger by +2, and return the
decision of what to try next (new hypothesis vs. a different universe/data
source vs. pausing the track) to the user rather than pre-committing to it
here.

If one or both pass the full verdict: report the passing configuration(s)
in full, and explicitly recommend NOT combining into a blended factor or
moving to paper-trading discussion in the same turn this cycle's pipeline
finishes — that is a separate decision for the user, consistent with
`memory/feedback-github-only-proven-alpha.md`.
