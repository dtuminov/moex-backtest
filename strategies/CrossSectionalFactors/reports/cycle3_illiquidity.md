# Cycle 3 report -- H-ILLIQ (cross-sectional illiquidity premium)

Run: 2026-09-10T08:00:53.353103+00:00
IS period: 2018-01-01 .. 2021-12-31; OOS period: 2022-01-01 .. 2026-09-09
Mechanism: illiquidity (compensation for trading friction in a retail-dominated order book) -- new mechanism, registered in `trial_ledger.MECHANISM_BY_HYPOTHESIS` before this cycle ran. See `PREREGISTRATION_cycle3_illiquidity.md`.

**Mechanism-scoped N (illiquidity)**: 0 prior -> **3** after this cycle (feeds `deflated_sharpe_ratio`).
**Project-wide N**: 10 prior -> **13** after this cycle (context only, per `N_CONVENTION.md`, never substituted into the DSR formula).

Phase A1 (default lookback=12): IS Sharpe=+0.674, p=0.0760

## Grid (3 trials)

Locked config: lookback=12  (IS Sharpe=+0.674)

OOS Sharpe (annualized, 15bps): +1.087
Walk-forward: 7 windows, mean IS=+0.960, mean OOS=+0.895, efficiency=93.2%
Sensitivity: PLATEAU
Monte Carlo (block bootstrap) percentile: 50.4
DSR (mechanism-scoped N=3): 0.9833 (E[max SR|H0]=+0.165)
PBO (3 trials, 16 blocks): 0.300

**VERDICT: PASS**

**Margins, stated honestly rather than left implicit**: this passes every
required criterion, but not all by the same comfortable distance momentum
cycle 1 passed its dead-on-arrival criteria by. Phase A1 p=0.076 is close
to the 0.10 cutoff (contrast momentum's p=0.001) -- a somewhat weaker raw
signal than momentum's original, before any of the corrections/mechanism
scoping in this cycle's favor. PBO=0.300 clears the <0.5 bar but sits
outside the <0.2 "comfortable zone" this project has used as an informal
reference (momentum cycle 1: PBO=0.032). What is *not* marginal: DSR=0.983
at mechanism-scoped N=3 has real headroom above the 0.95 threshold (E[max
SR|H0]=+0.165 vs. observed +1.087, a >6x margin), sensitivity is a genuine
wide plateau (not a corrected-after-the-fact borderline call), and Monte
Carlo's percentile=50.4 is as close to "exactly the middle of the null
distribution" as this project has ever recorded (momentum cycle 1: 46.7).
This is a real, multi-criteria PASS -- not force-fit -- but a soft Phase A1
p-value and a mid-range PBO are the parts of it worth remembering, not
glossing into a uniformly clean story.

**One more thing worth naming plainly**: `trial_sharpe_variance` here comes
from only 3 trial Sharpes (2 degrees of freedom) -- the first real exercise
of mechanism-scoped N at the small end of what the formula allows. `E[max
SR|H0]` is itself an estimate, and at this few trials that estimate carries
real sampling uncertainty of its own, on top of the uncertainty DSR already
prices in. This is not a flaw in this cycle's arithmetic -- it is the
literal, inherent cost of the mechanism-scoping decision buying low-N
power (`N_CONVENTION.md`): a brand-new mechanism's first cycle will always
face this until enough of its own trial history accumulates to estimate
the variance term more precisely. Worth knowing as the texture of what the
convention actually feels like in practice, not just its power numbers on
paper.

## Post-lock cost-sensitivity robustness (not counted toward N)

Pre-registered concern: 15bps is very likely an understatement for the illiquid (long) leg specifically -- the mechanism itself is that thinly-traded names cost more to trade. Does not change the verdict above; shows how much of any edge is cost-assumption-dependent.

| cost (bps, one-way) | OOS Sharpe |
|---|---|
| 15 (main verdict) | +1.087 |
| 30 | +1.082 |
| 50 | +1.076 |
| 100 | +1.059 |

**Why so flat**: mean OOS turnover is only **6.1% of positions/month**
(vs. momentum's 23.2%) -- illiquidity tier is a slow-moving structural
property, so the leg composition barely rotates. This cuts both ways: it
makes the result genuinely cost-robust, but it also means the backtest's
cost model (turnover x bps) may still understate real-world entry cost for
*initially building* a meaningful position in thin names, which a
per-rebalance turnover cost does not capture. Flagged, not corrected here.

## Additional post-lock robustness diagnostics (not counted toward N, `counts_toward_n=false` in the ledger, 23 rows)

Run because the low turnover above raised a specific, checkable concern
before trusting the PASS at face value: with such slow leg rotation, is
this actually a genuine cross-sectional effect, or a near-static bet on a
handful of names that happened to do well?

**Leg persistence**: two names (KZOS, BANE) sit in the illiquid (long) leg
in **100% of the 56 OOS months**; three more (LENT 50/56, LSRG 49/56, AKRN
48/56) are in it 85-89% of the time. The portfolio is much closer to a
mostly-static basket than a genuinely rotating monthly re-sort.

**Concentration check -- exclude the two 100%-persistent names entirely**
(45-name universe, `n_per_leg` still 9, otherwise identical construction):
OOS Sharpe **+1.026** (vs. +1.087 with them). The result is **not** a
2-name story -- excluding the most persistent names barely moves it.
Reinforced by individual name performance over the OOS window: KZOS itself
was actually a **loser** (annualized mean return -8.5%) and HYDR (41/56
months in the leg) **-10.8%** -- the long leg's positive Sharpe survives
*despite* two of its most persistent members losing money, which is hard
to reconcile with "this is just 2-3 lucky picks."

**Long/short decomposition** (gross, no cost): long (illiquid) leg alone,
OOS Sharpe +0.373, mean annualized return **+10.7%**; short (liquid) leg
alone, OOS Sharpe -0.243, mean annualized return **-7.3%**; universe-wide
equal-weight mean over the same window, **-1.3%**/yr. Both legs contribute
in the pre-registered direction (illiquid outperforms, liquid
underperforms, both relative to a roughly flat/slightly negative market) --
not one leg doing all the work.

**Walk-forward, per window** (is_window=12mo, oos_window=6mo, matches the
gate's own criterion of mean OOS Sharpe > 0, not "every window positive" --
shown here for the spread, per the project's own standing lesson that an
aggregate can look fine while masking window-to-window noise):

| window | IS Sharpe | OOS Sharpe |
|---|---|---|
| 0 | +1.842 | +1.718 |
| 1 | +1.534 | -0.206 |
| 2 | +0.818 | +2.546 |
| 3 | +0.854 | +1.504 |
| 4 | +2.109 | **-0.878** |
| 5 | +0.060 | -0.033 |
| 6 | -0.495 | +1.615 |

5 of 7 windows OOS-positive; window 4's -0.878 is the one clear outlier
(no obvious data-quality flag found on inspection -- reported, not
explained away).

**Dense sensitivity curve** (lookback 2..24mo, 23 additional points, full
OOS Sharpe at each, cost 15bps): not a spike riding one lucky grid point.
Sharpe dips to 0.67-0.78 for lookback 5-9mo, then rises to a **wide, flat
plateau of roughly 0.97-1.13 across the entire lookback=10..24 range**
(locked 12 sits inside it: 13->1.133, 16->1.088, 22->1.084, 23->1.10).
This is a materially cleaner plateau than momentum cycle 1 ever showed
(that one needed a tool bug fixed first and still only found a narrower
10-14 hump) -- and the plateau sits exactly where the pre-registration's a
priori reasoning said to expect it (liquidity tier is structural, use a
long lookback), not in a region reverse-engineered from this result.

**OOS Sharpe (+1.087) exceeding IS Sharpe (+0.674) is new for this
project** -- momentum cycle 1 (IS +1.624 -> OOS +0.562, ratio 0.35) and the
cycle 2 ensemble both decayed sharply IS-to-OOS, the usual and expected
pattern. Here it went the other way. Not treated as a red flag by itself:
the pre-registration's mechanism argument specifically predicts this
*could* happen -- "post-2022 capital segmentation" as a reason to expect a
*larger* effect applies with more force in exactly the OOS window
(2022-2026), which starts precisely at the sanctions/segmentation break.
Recorded as a pattern consistent with the pre-registered economic story,
not as independent confirmation of it -- a single cycle can't distinguish
"the mechanism genuinely strengthened after 2022" from "OOS happened to be
a favorable draw," and this report does not claim to have resolved that.

**Net read on these diagnostics**: none of them overturn the gate's PASS,
and none were used to select or adjust the locked config (all run
post-lock, on the already-fixed lookback=12, or as a fixed ablation).
Together they make the PASS considerably more credible than the bare
DSR/PBO numbers alone would suggest -- but the low-turnover,
concentrated-membership character of this factor is a real, structural
feature worth carrying into any decision about what to do with this
result, not an artifact to wave away.

## Round 2: two objections to "proven alpha," addressed with data

The PASS above is accepted as real (independently reconfirmed: DSR still
clears 0.95 at N=9 and even N=13, so the mechanism-scoped N convention is
not what is carrying this result -- see below). It is not yet treated as
*proven alpha* because of two specific, checkable objections. Both are
addressed here with real computation; three new candidate constructions
were run to do it, each logged as a genuine trial (any construction that
could plausibly become the live rule counts, per the standing rule --
descriptive-only checks, like "which names sit in which leg," do not).
Capacity was also checked (not reproduced here): at Arena account size
(RUB 3M, 9 names/leg, ~RUB 167k/name), the thinnest long-leg name's OOS
mean daily traded value (KZOS, RUB 20.5M/day) still leaves the position at
~0.8% of that day's turnover -- capacity is not a binding constraint at
this size.

**Mechanism-scoped N (illiquidity) after round 2: 3 -> 6. Project-wide:
13 -> 16.** Updated DSR on the original (as-locked, 47-name) OOS returns,
same `trial_sharpe_variance` recomputed from all 6 trials now on record
(0.6737, 0.3030, 0.5815, 0.7119, 0.6737, 0.6737 -- var=0.02346, notably
*tighter* than at N=3, since two of the three new trials left the IS
Sharpe unchanged): **DSR(N=6) = 0.9797** (E[max SR|H0]=+0.199), still
comfortably above 0.95. The mechanism-scoped N convention is not buying
this PASS -- the low `trial_sharpe_variance` (0.023-0.037 across every
recomputation so far, vs. momentum's 0.560) is what keeps `E[max SR|H0]`
low, and that is a property of how tightly clustered this mechanism's own
trials have been, not of N itself. Worth remembering as a standing
methodological note, not something to re-litigate every cycle: a narrow
grid of similar trials is, by the formula's own construction, cheaper to
clear than a wide one -- an honest incentive to keep grids narrow for the
right reason (a genuinely well-specified, a-priori-narrowed hypothesis),
not to game DSR.

### Objection A (the one that matters for whether/when to turn this off): illiquidity premium, or a post-2022 sanctions-segmentation trade?

**The short (liquid) leg is heavily concentrated in the same handful of
sanctioned megacaps and systemic banks** (Нефтегаз+Металлурги export
names, plus SBER/VTBR -- the two explicitly, directly sanctioned banks):
SBER, GAZP, LKOH are in the short leg in **100% of the 56 OOS months**;
VTBR in 54/56. Across all OOS short-leg name-months, **71.2%** belong to
this 15-name set (GAZP, LKOH, ROSN, NVTK, TATN, SNGS, SIBN, GMKN, NLMK,
CHMF, MAGN, RUAL, PLZL, SBER, VTBR).

**But this is not new to OOS.** The same check on the IS period
(2018-2021, before any sanctions existed) finds the short leg was **80.7%**
these same names -- slightly *higher* than OOS, not lower. GAZP, SBER, LKOH
have simply always been MOEX's most liquid names, sanctions or not; "short
the most-liquid quintile" and "short the systemic megacaps" are nearly the
same statement on this specific market, mechanically, in both periods.
This weakens the strong form of the objection (the *construction* isn't
newly capturing sanctioned names post-2022 -- it always has).

**What did change is the return behavior of roughly the same basket**: IS
Sharpe on this construction was weak (+0.674, p=0.076, barely inside the
Phase A1 gate) and OOS Sharpe is much stronger (+1.087) -- with near-
identical leg composition throughout. That is exactly the pattern a
segmentation-driven regime shift in the *megacaps' relative returns*
(not the construction) would produce, and is consistent with, though not
proof of, the pre-registration's own "post-2022 capital segmentation"
argument -- which specifically predicted a *larger* effect after 2022 for
this reason.

**Direct test (new trial, mechanism N 3->4): exclude the 15 sanctioned-
megacap names from the eligible universe entirely** (both legs; 32 names
remain, `n_per_leg` rescaled to floor(32/5)=6, same lookback=12/cost=15bps
otherwise identical). Run post-hoc, after already knowing the 47-name
OOS result -- not a blind test, stated plainly.

| | 47-name (locked) | 32-name (megacaps excluded) |
|---|---|---|
| IS Sharpe | +0.674 | +0.712 |
| IS p-value | 0.0760 | 0.1070 (would just miss Phase A1 as a fresh hypothesis) |
| OOS Sharpe | +1.087 | **+0.827** |

**Verdict on Objection A: partially confirmed.** The edge survives without
the sanctioned megacaps (+0.827 is still a real, economically meaningful
OOS Sharpe) -- this is not *purely* a segmentation trade. But roughly a
quarter of the measured edge (1.087 -> 0.827, a ~24% Sharpe reduction) is
attributable to the megacap short specifically, and the 32-name-only
version's own IS signal is right at the edge of the significance
threshold on its own (p=0.107) -- weaker standalone evidence than the
47-name version's already-marginal p=0.076. **Read this as a mechanism
that is real but state-dependent, not a pure structural premium.**

One reassuring, unprompted finding on the long side: the illiquid leg is
**not** sector-concentrated -- across its persistent members it spans 9 of
the universe's 10 sectors (удобрения, нефтегаз, ритейл, прочее, транспорт,
энергетика, металлурги, банки, телеком) -- a single-sector bet was a
plausible alternative confound and the data does not support it.

**Practical implication for Arena deployment, since this is a live
candidate and not just a report**: deploy this, if at all, as an explicitly
regime-conditional strategy, not a structural one. The stated reversal
condition: a durable reopening of MOEX to foreign arbitrage capital, or a
material lifting of the sanctions specifically on the 15 named megacaps --
either would remove the ~quarter of the edge this test attributes to
segmentation, and there is no data-driven basis here for claiming the
remaining ~0.83 Sharpe is itself immune to a broader regime shift, only
that it survived *this one* specific ablation.

### Objection B: survivorship in the 47-name universe

**Full-market count of MOEX delistings/permanent suspensions 2018-2026,
and whether they would have ranked in the bottom liquidity quintile: not
computed.** No network access in this environment to query MOEX ISS's
historical securities list (`curl`/`WebFetch` to `iss.moex.com` both
failed to connect, confirmed before writing this section), and no
point-in-time historical listing snapshot is cached anywhere in this
repo -- `scripts/import_universe.py`'s 47-name list is a single,
hand-curated, present-day ("as of Sept 2026") liquid universe applied
retroactively across the whole window, a limitation already flagged (not
newly discovered) in `PREREGISTRATION.md`'s cycle 1. Do not read anything
below as a substitute for that missing count.

**What is checked, on real data**: within the *current* 47-name universe,
exactly **one** ticker's real trading history ends materially before the
window's end -- **DSKY** (real data 2018-01-03 .. 2024-10-14, vs. every
other name running to 2026-09-09). This is a direct, data-visible instance
of exactly the risk Objection B describes: DSKY spent **20 of 56 OOS
months (36%) in the illiquid long leg** before dropping out of
consideration, and its last visible monthly closes show a real decline
(68-71 in Aug-Oct 2023 to 43-52 in the year before data stops, roughly
-30% off its recent high) -- a deteriorating, increasingly thin name, not
a stable one. What ultimately happened to DSKY after 2024-10-14
(suspension, delisting terms, buyout price) is **not** in this dataset and
is not asserted here.

**A related but distinct issue, found and fixed while investigating this
(two more trials, mechanism N 4->6)**: the production `illiquidity_signal`
let DSKY's stale rolling estimate keep it *selected* into the long leg for
roughly 6 more months (through formation date 2025-03-31) after its real
data stopped. First attempt at a fix (cap the `reindex(..., method="ffill")`
step to `limit=5`, matching `panel.load_close_panel`'s own convention) made
**zero** difference -- wrong diagnosis: the real mechanism is
`rolling(window=252, min_periods=126)` itself legitimately still finding
>=126 real historical DSKY observations inside its trailing window for
months after the stock stopped trading, not a reindex artifact. Second
attempt -- an explicit per-date eligibility mask off `close_panel`'s own
already-ffilled NaN status, independent of the rolling window's
`min_periods` logic -- correctly trims DSKY's eligibility to 2024-09-30
and its long-leg months from 20 to 14. Result: **OOS Sharpe +1.104**
(vs. the locked +1.087) -- this construction bug was not inflating the
PASS; if anything the opposite. Flagged as a real, worth-fixing issue in
`strategies/CrossSectionalFactors/factors.py`'s `illiquidity_signal` (and,
by the identical rolling+reindex pattern, potentially `low_vol_signal` --
not checked, that hypothesis is already closed) for anyone deploying this
live, since a real deployment cannot silently hold a position in a name
that has stopped trading -- not applied to `factors.py` itself here, per
the standing rule against changing already-reviewed code without a
separate, explicit decision to do so.

**Direction-of-bias reasoning (not a magnitude estimate -- explicitly not
attempted without real delisting data)**: the 47-name universe was
assembled by looking backward from Sept 2026 at what counts as "liquid
today." Any name that failed or was permanently delisted at any point in
2018-2026 and is not among today's 47 is invisible to this study by
construction -- a strictly stronger version of DSKY's mid-series exit,
which at least appears for part of the window. If failing companies tend
to become illiquid before they fail (economically plausible, and the one
example on hand fits that pattern), and this backtest can only ever
observe illiquid names that *survived* to be worth including in 2026, then
the long leg's measured return is a survivor-conditional estimate of the
true illiquidity-premium return, biased toward **overstating** it. The
direction is unambiguous; the size is not computed and is not
guessable from what is on disk. One partial, also-unmeasured mitigant: the
universe's actual illiquid tail (KZOS, BANE, LENT, LSRG, AKRN -- all
established, dividend-paying mid-caps) is not the small/shell-company
profile most exposed to outright failure, which plausibly bounds this
effect somewhat without eliminating it.

**Verdict on Objection B: real, correctly directional, not quantifiable
here.** Treat the reported OOS Sharpe (+1.087, or +1.104 with the
eligibility fix) as an upper bound on the true achievable illiquidity
premium, not a point estimate, until a proper point-in-time historical
universe exists -- a real data-engineering project (a historical MOEX
listing snapshot, likely requiring a paid data source or manual archival
work), explicitly not attempted here given the Arena timeline.

### Net read after round 2

Not overturned, not fully vindicated. The gate's PASS stands and survives
three more honestly-counted trials (DSR still 0.98 at N=6). Objection A is
partially confirmed: roughly a quarter of the edge is a segmentation bet
with a nameable reversal condition, not a pure structural premium --
deploy, if at all, with that condition stated up front, not as a
set-and-forget position. Objection B is real and directionally understood
(overstatement, from survivorship) but not sized, and cannot be sized with
data available in this environment before the Arena start date -- the
honest response is to size any live allocation conservatively against the
stated Sharpe, not to treat +1.087 as a reliable point forecast.

*(Arena has since been cancelled -- Finam's selection now runs through a
separate ZipLime championship instead, so the urgency behind that framing
no longer applies; the finding itself stands and round 3 below builds on
it.)*

## Round 3: Objection B, resolved further via the Finam Trade API

MOEX ISS is confirmed unreachable from this environment (`curl`/`WebFetch`
both fail against `iss.moex.com`) -- that part of round 2 stands. But the
network is not fully blocked: GitHub and `api.finam.ru` are both reachable,
and `FinamClient`'s own credentials are already in `.env`. This section
uses the live Finam Trade API, not MOEX ISS, to get real (not reasoned-
about) numbers on survivorship scale.

**Environment note, for the record**: `httpx` and stdlib `urllib` both
time out mid-TLS-handshake against `api.finam.ru` from this sandbox
(confirmed: DNS resolves IPv4-only, so not an IPv6 routing issue); `curl`
against the identical host succeeds immediately. All calls below are
routed through `curl` via `subprocess` for exactly this reason. A real,
reproducible quirk of this environment, not of the API.

### Path 1: `AllAssets(only_disabled=true)` -- does the API expose archived instruments with a status flag?

Yes. `GET /v1/assets/all?cursor=...&only_disabled=true` (documented in
`FinamWeb/finam-trade-api`'s `assets_service.proto`, fetched live from
GitHub) returns `Asset` records carrying `is_archived: bool` -- exactly
the kind of status flag Objection B's write-up said would be needed.
Paginated this fully: **10,115 unique archived/disabled instruments**
across all exchanges/types (deduplicated from 14,890 raw rows -- the
`next_cursor` sequence eventually looped back to an already-seen value,
confirmed by checking that new-unique-symbol discovery had already
plateaued at row ~10,114 before duplicates started at ~10,115; the
fetch was killed at that point rather than left to cycle indefinitely).
Filtered to `mic=MISX` (Moscow Exchange) and `type=EQUITIES`: **267
unique archived MOEX equity instruments.**

Classified those 267 by hand (ticker-suffix and name-pattern rules, not
memory of individual company histories):

| category | count | why excluded / kept |
|---|---:|---|
| `-RM` suffix (foreign-company mirror instruments -- Boeing, Meta, GE, Exxon, etc.) | 131 | not Russian companies, irrelevant to this universe's survivorship question |
| GDR/ADR/depositary-receipt forms of a company whose primary MOEX ticker is already in the current 47-name universe (`SBER-ME`->SBER, `VTBR-ME`->VTBR, `OGZD`("Газпром др")->GAZP, `YNDX`->YDEX, `TCSG`->T, `FIVE`->X5, `MAIL`("VK-гдр")->VKCO, `RUALR`->RUAL, `LNTA`("Лента др")->LENT, etc.) | 36 | the *listing vehicle* was retired (largely the 2022 forced delisting of Russian GDRs from London), the *company* did not disappear -- already-known redomiciliation pattern, not new survivorship risk |
| **remaining, no identified successor in the current 47-name universe** | **100** | **the real candidate pool** |

**DSKY is directly present in this list with `is_archived: true`** --
independent, live cross-confirmation of the cached-data finding from round
2, not a coincidence of this project's own data pipeline.

### Path 2: does `bars` return history for a name that stopped trading?

Yes, checked live, not just inferred from cached data. `GET
/v1/instruments/DSKY@MISX/bars` for 2025-01-01..2025-02-01 returns
**0 bars** (matches the cached parquet exactly: no rows after
2024-10-14); the same call for 2024-09-01..2024-11-01 returns 31 bars
that match the cached closes exactly, day for day, ending at the same
2024-10-14 / 51.02 close already on disk. The API does not error on a
delisted symbol -- it returns an empty result -- which is itself useful:
it means bar-gap detection could in principle screen the 100-name
candidate list at scale, if that became worth doing.

### Sampling the 100-name candidate pool for real volume levels

Spot-checked bars (mid-June of 2019/2021/2023, one month each -- a
deliberately small, fast sample, not a full reconstruction) for 6 of the
100 candidates, picked because their names were legible enough to have
*some* prior context for interpretation (not because of any expected
result): MFON (MegaFon), POGR (Petropavlovsk), DIXY, AVAZ (AvtoVAZ),
URKA (Uralkali), QIWI. This sandbox's connectivity to `api.finam.ru` was
noticeably flaky under repeated calls (a nontrivial fraction of requests
timed out, `curl` rc=28) -- timeouts are reported honestly as
**inconclusive** below, not silently treated as "0 bars i.e. not
trading":

| ticker | 2019-06 | 2021-06 | 2023-06 |
|---|---|---|---|
| MFON | timeout (inconclusive) | timeout (inconclusive) | 0 bars (confirmed not trading) |
| POGR | 0 bars (confirmed not trading) | timeout (inconclusive) | 0 bars (confirmed not trading) |
| DIXY | 0 bars (confirmed not trading) | timeout (inconclusive) | 0 bars (confirmed not trading) |
| AVAZ | timeout (inconclusive) | 0 bars (confirmed not trading) | 0 bars (confirmed not trading) |
| **URKA** | **19 bars, median RUB 2,353,872/day** | timeout (inconclusive) | 0 bars (confirmed not trading) |
| QIWI | 19 bars, median RUB 9,841,545/day | 22 bars, median RUB 91,160,334/day | timeout (inconclusive) |

**The single most important number in this table**: URKA (Uralkali)
traded, confirmed live from the API, in June 2019 -- squarely inside this
cycle's own IS period -- at a median daily traded value of **RUB
2.35 million**. The current 47-name universe's *own* thinnest member
(KZOS) averages **RUB 20.5 million/day** over the OOS period (round 1's
capacity check) -- almost 9x higher. This is not a hypothetical: a real,
MOEX-listed, actively-traded-in-2019 company existed at a liquidity level
well below anything in the current backtest's bottom quintile, and it is
invisible to this study because it was delisted (by 2023, confirmed) and
is therefore absent from `scripts/import_universe.py`'s present-day list.
QIWI's 2019 level (RUB 9.8M/day) is also below KZOS's, though QIWI's
liquidity clearly was not stable over time (RUB 91M/day by 2021).

### Order-of-magnitude answer, and what remains genuinely unsized

**Scale: order of 100** (not 10, not 1,000) Russian-domestic MOEX equity
listings have been archived with no continuing listing in the current
universe, per Finam's own system -- a real, API-confirmed count, not a
guess. **At least one directly confirmed instance** (URKA) establishes
that this pool does contain names that would have qualified for the
bottom liquidity quintile during this study's own IS period, not just
plausibly-similar-looking small companies. The remaining ~94 of the 100
were **not** individually checked -- no list of their individual fates or
liquidity histories is asserted here, consistent with the instruction not
to invent one.

**What is still not computed, and why not attempted here**: a Sharpe- or
return-level bias estimate. That would require fetching full-history bars
for all ~100 candidates (not 6), determining each one's actual trading
window and whether/when it would have ranked in the bottom quintile
month-by-month, and reconstructing a genuine point-in-time monthly panel
to rerun the quintile sort against -- a real data-engineering project on
the order of `scripts/import_universe.py` itself, not a same-session
diagnostic. It would also, per the standing rule, **not** be a free
diagnostic: a return series computed from a reconstructed, tradeable
universe is exactly the kind of candidate that would need to be logged as
a new illiquidity-mechanism trial (mechanism N 6->7) the moment its
Sharpe was reported as a number to evaluate, which is a separate, real
decision to make deliberately, not a byproduct of measuring scale.

**Revised verdict on Objection B**: still not sized in Sharpe terms, but
the directional argument from round 2 is no longer just reasoning about
what *should* be true structurally -- it is now anchored to a real,
API-confirmed population of ~100 candidates and at least one concrete,
below-threshold, IS-period example. This raises confidence in the
direction (overstatement) without changing it, and does not license
attaching a number to +1.087 that isn't there. Same operational
conclusion as round 2, on firmer ground: **treat the reported OOS Sharpe
as an upper bound on the true achievable illiquidity premium, not a point
estimate.**
