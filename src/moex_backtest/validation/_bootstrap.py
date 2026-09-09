"""Resampling primitives shared by :mod:`significance` and :mod:`monte_carlo`.

Two resamplers, both returning a ``(n_simulations, T)`` matrix of resampled
per-period returns from an input array of length ``T``:

- :func:`iid_resample` — ordinary (i.i.d.) bootstrap: every resampled point is
  an independent draw with replacement from the original series. Destroys any
  autocorrelation in the original series.
- :func:`stationary_bootstrap_resample` — the stationary bootstrap of Politis
  & Romano (1994), "The Stationary Bootstrap", *JASA* 89(428):1303-1313. Draws
  variable-length, circularly-wrapped blocks (block length ~
  ``Geometric(1/expected_block_length)``) instead of single points, so runs of
  serially-correlated returns (volatility clustering, momentum/mean-reversion
  in the underlying spread) are preserved in each resampled path. Preferred
  over the i.i.d. bootstrap for daily financial returns, which are not
  independent draws.
"""

from __future__ import annotations

import numpy as np


def iid_resample(returns: np.ndarray, n_simulations: int, rng: np.random.Generator) -> np.ndarray:
    """I.i.d. bootstrap: for each of ``n_simulations`` rows, draw ``T`` points
    independently and with replacement from ``returns`` (``T = len(returns)``).

    Returns an ``(n_simulations, T)`` array.
    """
    if returns.ndim != 1:
        raise ValueError(f"returns must be 1-D, got shape {returns.shape}")
    t = len(returns)
    if t == 0:
        raise ValueError("returns is empty")
    indices = rng.integers(0, t, size=(n_simulations, t))
    return returns[indices]


def stationary_bootstrap_resample(
    returns: np.ndarray,
    n_simulations: int,
    expected_block_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap.

    For each output position, with probability ``p = 1 / expected_block_length``
    a *new* block starts at a uniformly random index into ``returns``;
    otherwise the previous index is continued (wrapping circularly past the
    end of ``returns`` back to index 0). Block lengths are therefore
    ``Geometric(p)``-distributed and average ``expected_block_length`` — long
    enough to keep short-range serial dependence intact, but random enough
    that the resampled series is itself stationary (unlike a fixed-length
    moving-block bootstrap, which is not).

    Implementation note: the recurrence "continue previous index + 1, unless
    a fresh block starts" is computed by grouping output positions into
    contiguous same-block segments (``segment_id = cumsum(new_block) - 1``)
    and then vectorizing the within-segment offset from each segment's own
    random start — avoids an explicit T-length Python loop per simulation.

    Returns an ``(n_simulations, T)`` array.
    """
    if returns.ndim != 1:
        raise ValueError(f"returns must be 1-D, got shape {returns.shape}")
    t = len(returns)
    if t == 0:
        raise ValueError("returns is empty")
    if expected_block_length < 1.0:
        raise ValueError(f"expected_block_length must be >= 1, got {expected_block_length}")

    p = 1.0 / expected_block_length
    out = np.empty((n_simulations, t), dtype=returns.dtype)
    for sim in range(n_simulations):
        new_block = rng.random(t) < p
        new_block[0] = True  # the first position always starts a (the first) block
        segment_id = np.cumsum(new_block) - 1
        block_starts_output_pos = np.flatnonzero(new_block)
        segment_start_output_pos = block_starts_output_pos[segment_id]
        offset_within_segment = np.arange(t) - segment_start_output_pos
        random_block_origin = rng.integers(0, t, size=len(block_starts_output_pos))
        source_index = (random_block_origin[segment_id] + offset_within_segment) % t
        out[sim] = returns[source_index]
    return out


def default_expected_block_length(n_observations: int) -> float:
    """``n_observations ** (1/3)``, rounded to the nearest integer >= 1.

    A standard simple rule of thumb for bootstrap block length as a function
    of sample size (see e.g. Politis & White (2004), "Automatic Block-Length
    Selection for the Dependent Bootstrap", *Econometric Reviews* 23(1):53-70,
    which formalizes optimal block-length selection; this is the
    order-of-magnitude cube-root heuristic commonly used as a fast default
    when the full spectral-estimation procedure isn't warranted). Callers with
    a specific reason to expect longer/shorter dependence (e.g. a known
    mean-reversion half-life) should pass their own ``expected_block_length``
    instead of relying on this default.
    """
    if n_observations < 1:
        raise ValueError(f"n_observations must be >= 1, got {n_observations}")
    return float(max(1, round(n_observations ** (1.0 / 3.0))))
