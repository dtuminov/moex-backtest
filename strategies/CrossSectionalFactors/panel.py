"""Load the 47-ticker MOEX universe from the on-disk parquet cache and build
an aligned daily-close panel for cross-sectional factor construction.

No network access -- reads only `data/raw/finam_daily_*.parquet`, already
populated by `scripts/import_universe.py`. Raises if a ticker's cache file
is missing rather than silently shrinking the universe (see
strategies/CrossSectionalFactors/PREREGISTRATION.md for the fixed 47-name
list this must match exactly).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

UNIVERSE: tuple[str, ...] = (
    "AFLT", "AKRN", "ALRS", "ASTR", "BANE", "BSPB", "CBOM", "CHMF", "DSKY", "FEES",
    "FLOT", "GAZP", "GMKN", "HYDR", "IRAO", "KZOS", "LENT", "LKOH", "LSRG", "MAGN",
    "MGNT", "MOEX", "MTLR", "MTSS", "NLMK", "NMTP", "NVTK", "OZON", "PHOR", "PIKK",
    "PLZL", "POSI", "ROSN", "RTKM", "RUAL", "SBER", "SELG", "SIBN", "SNGS", "T",
    "TATN", "TRNFP", "UPRO", "VKCO", "VTBR", "X5", "YDEX",
)  # fmt: skip
_CACHE_KEY_SUFFIX = "_2018-01-01_2026-09-09"


def load_close_panel(cache_dir: Path | str = "data/raw") -> pd.DataFrame:
    """Wide panel of daily close prices: index=date (tz-naive, normalized to
    midnight), columns=ticker (see `UNIVERSE`). Outer-joins every ticker's
    own trading-date index, then forward-fills gaps up to 5 trading days
    (isolated one-off gaps only -- a genuinely long halt/delisting is left
    as NaN rather than silently carrying a stale price for months).
    """
    cache_dir = Path(cache_dir)
    series: dict[str, pd.Series] = {}
    for ticker in UNIVERSE:
        path = cache_dir / f"finam_daily_{ticker}{_CACHE_KEY_SUFFIX}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- run scripts/import_universe.py first")
        frame = pd.read_parquet(path).sort_values("TRADEDATE")
        idx = pd.DatetimeIndex(frame["TRADEDATE"]).tz_convert(None).normalize()
        series[ticker] = pd.Series(frame["CLOSE"].to_numpy(dtype=float), index=idx)

    panel = pd.DataFrame(series).sort_index()
    return panel.ffill(limit=5)


def monthly_close_panel(close_panel: pd.DataFrame) -> pd.DataFrame:
    """`close_panel` resampled to one row per calendar month (last available
    close that month, per ticker) -- the formation-date price series
    momentum is computed from. The resulting index's calendar-month-end
    labels (not necessarily themselves a trading day) are the formation
    dates used everywhere else in this cycle.
    """
    return close_panel.resample("ME").last()
