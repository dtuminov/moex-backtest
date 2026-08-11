"""Tests for ParquetCache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from moex_backtest.data.cache import ParquetCache


def test_get_or_fetch_calls_fetch_once_and_reuses_cache(tmp_path: Path) -> None:
    cache = ParquetCache(cache_dir=tmp_path)
    calls = 0

    def fetch() -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return pd.DataFrame({"a": [1, 2, 3]})

    first = cache.get_or_fetch("key", fetch)
    second = cache.get_or_fetch("key", fetch)

    assert calls == 1
    pd.testing.assert_frame_equal(first, second)


def test_empty_frame_is_returned_but_not_persisted(tmp_path: Path) -> None:
    cache = ParquetCache(cache_dir=tmp_path)

    result = cache.get_or_fetch("key", pd.DataFrame)

    assert result.empty
    assert not cache.path_for("key").exists()


def test_different_keys_do_not_collide(tmp_path: Path) -> None:
    cache = ParquetCache(cache_dir=tmp_path)
    cache.get_or_fetch("a", lambda: pd.DataFrame({"v": [1]}))
    cache.get_or_fetch("b", lambda: pd.DataFrame({"v": [2]}))

    assert cache.path_for("a").exists()
    assert cache.path_for("b").exists()
