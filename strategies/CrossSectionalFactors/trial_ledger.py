"""Mechanism-scoped DSR trial counting for the MOEX track.

**The convention this implements** (decided 10.09.2026, see
`N_CONVENTION.md` for the full argument and `reports/why_no_alpha.md` for the
power numbers that forced it):

Two counters are kept, and they answer different questions:

- **Mechanism-scoped N** -- goes into :func:`deflated_sharpe_ratio`. Counts
  only trials of the *same economic mechanism* as the candidate being judged.
  This is literally what Bailey & Lopez de Prado's formula licenses: ``N`` is
  the number of trials in the *one selection process* that produced this
  candidate. A momentum grid does not make an untested dividend-gap
  hypothesis more likely to be overfit.
- **Project-wide count** -- reported alongside, never substituted into the
  formula. This is the Harvey/Liu/Zhu (RFS 2016) concern: an researcher who
  works through many distinct anomalies should be more skeptical overall,
  even though no single DSR computation sees that history.

**Why the split** (this is the part that cost the track two cycles): the
production convention fed the project-wide count into DSR, which drove
``E[max SR|H0]`` to 1.138 annualized -- *above* the true Sharpe the track
actually observed (+0.562). That threshold does not shrink with sample size,
so the candidate could not have passed at any T, however real it was. See
`reports/why_no_alpha.md` A2. Mechanism-scoping is not a loosening for
convenience; it restores the quantity the formula is defined over.

Run ``uv run python strategies/CrossSectionalFactors/trial_ledger.py`` to
print the current counts.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

LEDGER_PATH: Final = Path(__file__).parent / "experiments" / "grid_search_log.jsonl"

#: Which economic mechanism each logged hypothesis belongs to. A new
#: hypothesis MUST be registered here before its cycle runs -- an unmapped
#: name raises rather than silently starting its own counter, so that
#: "momentum, but rebuilt differently" cannot quietly reset N to 1.
MECHANISM_BY_HYPOTHESIS: Final[dict[str, str]] = {
    "momentum": "momentum",
    "momentum_ensemble": "momentum",  # cycle 2: same mechanism, different construction
    "lowvol": "lowvol",
    # cycle 3: genuinely new mechanism (compensation for trading-friction in a
    # retail-dominated order book), not a rebuild of momentum or lowvol --
    # starts at N=0, see PREREGISTRATION_cycle3_illiquidity.md.
    "illiquidity": "illiquidity",
}

#: Trials run in sibling repositories on this track, not present in this
#: ledger. Kept explicit so the project-wide count stays honest.
EXTERNAL_TRIALS: Final[tuple[tuple[str, str, int], ...]] = (
    ("pairs_cointegration", "moex-pairs-trading validation-layer calibration (09.09.2026)", 2),
)


def mechanism_of(hypothesis: str) -> str:
    """Mechanism a hypothesis belongs to, or raise if it was never registered."""
    try:
        return MECHANISM_BY_HYPOTHESIS[hypothesis]
    except KeyError:
        raise KeyError(
            f"hypothesis {hypothesis!r} is not registered in MECHANISM_BY_HYPOTHESIS. "
            "Declare its mechanism before running the cycle: reusing an existing "
            "mechanism inherits that mechanism's N, a genuinely new one starts at 1."
        ) from None


def _load_rows(ledger_path: Path = LEDGER_PATH) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        return []
    with ledger_path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def counted_rows(ledger_path: Path = LEDGER_PATH) -> list[dict[str, Any]]:
    """Ledger rows that count as DSR trials.

    ``counts_toward_n`` post-dates cycle 1, so rows written before it exist
    without the field; those are cycle 1's real trials and do count. Only an
    explicit ``False`` (diagnostics, robustness checks run after the lock)
    is excluded.
    """
    return [r for r in _load_rows(ledger_path) if r.get("counts_toward_n") is not False]


def mechanism_n(mechanism: str, ledger_path: Path = LEDGER_PATH) -> int:
    """Prior trial count for one mechanism -- the N that goes into DSR.

    A mechanism with no history returns 0; the caller adds the current
    cycle's own trials on top, so a genuinely new mechanism starts at its
    own trial count rather than inheriting the track's.
    """
    rows = counted_rows(ledger_path)
    n = sum(1 for r in rows if mechanism_of(r["hypothesis"]) == mechanism)
    n += sum(count for name, _, count in EXTERNAL_TRIALS if name == mechanism)
    return n


def project_wide_n(ledger_path: Path = LEDGER_PATH) -> int:
    """Every trial on the MOEX track -- reported, never fed into the formula."""
    return len(counted_rows(ledger_path)) + sum(count for _, _, count in EXTERNAL_TRIALS)


def mechanism_breakdown(ledger_path: Path = LEDGER_PATH) -> dict[str, int]:
    """Per-mechanism trial counts, for reporting."""
    counts: dict[str, int] = {}
    for row in counted_rows(ledger_path):
        mech = mechanism_of(row["hypothesis"])
        counts[mech] = counts.get(mech, 0) + 1
    for name, _, count in EXTERNAL_TRIALS:
        counts[name] = counts.get(name, 0) + count
    return counts


def _format(breakdown: dict[str, int], project_wide: int) -> Iterable[str]:
    yield "Mechanism-scoped N (feeds the DSR formula):"
    for mech, count in sorted(breakdown.items(), key=lambda kv: (-kv[1], kv[0])):
        yield f"  {mech:<22} {count}"
    yield ""
    yield f"Project-wide count (reported separately, Harvey/Liu/Zhu): {project_wide}"
    yield "A mechanism absent above starts at 0 and counts only its own cycle's trials."


if __name__ == "__main__":
    for line in _format(mechanism_breakdown(), project_wide_n()):
        print(line)
