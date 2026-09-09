"""Config lock: enforces the in-sample/out-of-sample boundary as an
infrastructure control, not a prompt instruction or a self-reported promise.

**Why this exists.** "Don't look at OOS metrics until after picking the
final config" is easy to state and easy to violate by accident -- across
this project's crypto/Jesse track, walk-forward efficiency quietly collapsed
more than once because a "final candidate" was picked by eye from OOS/test
results already visible during iteration (``memory/jesse-trade-algo-strategy.md``,
cycle 2's post-mortem). The fix that worked there (cycle 6,
``FundingCarrySpread``) was mechanical, not verbal: write the locked-in
config to disk *before* touching OOS data, and have the OOS-consuming code
itself refuse to run without that file present. An LLM-driven research loop
in particular cannot reliably self-police "I promise I haven't peeked" --
this makes peeking-before-locking structurally impossible instead of merely
discouraged (see ``quant-validation-methodology.md`` section 6: "avoid
brittle language instructions in favor of enforced validation at the API
boundary").

**Pattern**: after an in-sample-only grid search picks a final parameter
configuration, call :func:`lock_config` once (writes a JSON file to disk,
timestamped, one-way -- refuses to silently overwrite an existing lock).
Every function that consumes out-of-sample data starts with a call to
:func:`require_locked_config`; if the lock file isn't there yet, or is for a
different cycle, the call raises immediately instead of running.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ConfigNotLockedError(RuntimeError):
    """Raised when OOS-gated code runs before a matching lock file exists."""


@dataclass(frozen=True, slots=True)
class LockedConfig:
    cycle_name: str
    config: dict[str, Any]
    is_metric_name: str
    is_metric_value: float
    locked_at: str
    """ISO-8601 UTC timestamp of when :func:`lock_config` wrote this file."""


def lock_config(
    path: Path | str,
    *,
    cycle_name: str,
    config: dict[str, Any],
    is_metric_name: str,
    is_metric_value: float,
) -> LockedConfig:
    """Writes the final in-sample-selected configuration to `path` as the
    machine-checkable OOS gate (see module docstring).

    Raises `FileExistsError` if `path` already exists: locking is a
    one-time, one-way action per cycle. Re-locking (e.g. after peeking at
    OOS and wanting to pick a different config) must be a deliberate,
    visible act -- delete the file yourself and call this again -- never a
    silent overwrite, since silently re-locking after peeking is exactly the
    data-snooping failure mode this module exists to prevent.
    """
    p = Path(path)
    if p.exists():
        raise FileExistsError(
            f"{p} already exists -- a config lock is one-time and one-way; delete it "
            "explicitly first if you really intend to re-lock (re-locking after peeking "
            "at OOS is exactly the data-snooping failure mode this file exists to prevent)"
        )
    locked = LockedConfig(
        cycle_name=cycle_name,
        config=config,
        is_metric_name=is_metric_name,
        is_metric_value=is_metric_value,
        locked_at=datetime.now(UTC).isoformat(),
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(locked), indent=2, ensure_ascii=False, sort_keys=False) + "\n")
    return locked


def require_locked_config(path: Path | str, *, cycle_name: str) -> LockedConfig:
    """Physically gates OOS-consuming code: raises `ConfigNotLockedError`
    unless `path` exists and was locked (via :func:`lock_config`) for this
    exact `cycle_name`. Call this as the first line of any function that
    touches out-of-sample data.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigNotLockedError(
            f"{p} does not exist -- refusing to proceed: out-of-sample-gated code "
            f"requires lock_config() to have been called for cycle {cycle_name!r} "
            "before this point (see moex_backtest.validation.lock module docstring)"
        )
    raw: dict[str, Any] = json.loads(p.read_text())
    if raw.get("cycle_name") != cycle_name:
        raise ConfigNotLockedError(
            f"{p} was locked for cycle {raw.get('cycle_name')!r}, not {cycle_name!r} -- "
            "refusing to treat a different cycle's lock as authorization for this one"
        )
    return LockedConfig(**raw)
