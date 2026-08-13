"""Local parquet cache for MOEX ISS history frames.

Closed trading days never change, so a keyed on-disk cache avoids
re-fetching the same range on every run while a strategy is being
iterated on. Callers that need today's still-in-progress session should
bypass the cache for `end == today`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd


class ParquetCache:
    """Get-or-fetch cache keyed by an arbitrary string, backed by parquet files."""

    def __init__(self, cache_dir: Path | str = "data/raw") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        return self._cache_dir / f"{key}.parquet"

    def get_or_fetch(self, key: str, fetch: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        path = self.path_for(key)
        if path.exists():
            return pd.read_parquet(path)
        frame = fetch()
        if not frame.empty:
            frame.to_parquet(path)
        return frame
