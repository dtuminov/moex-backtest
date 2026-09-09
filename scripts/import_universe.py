"""Import daily OHLCV history for a broad liquid MOEX equity universe via
`FinamClient`, cached to `data/raw/` via `ParquetCache`.

This is the data-layer prerequisite for cross-sectional research across
sectors (oil & gas, banks, metals & mining, telecom, retail, IT, chemicals,
utilities, transport) — individual equities anonymous MOEX ISS access caps
at ~35 trading days, not enough for this; `FinamClient` lifts that cap (see
`scripts/finam_history_check.py` for the standalone verification).

One bad ticker must not kill the run: each symbol is fetched independently,
failures are logged and the run continues. Tries a documented fallback
ticker for names that changed post-2022 (redomiciliations/renames), since
the "current" MOEX ticker for some of these names shifted after the
original 2022-2024 corporate restructurings.

Run with: uv run python scripts/import_universe.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd
from dotenv import load_dotenv

from moex_backtest.data import FinamAPIError, FinamClient, ParquetCache

MIC = "MISX"
START = datetime(2018, 1, 1, tzinfo=UTC)
END = datetime.now(UTC)
# Un-adjusted OHLCV (Finam's Bars API carries no split/dividend-adjustment
# field, see FinamClient docstring) — a single-day move past this is flagged
# as a likely un-adjusted corporate action rather than a genuine price shock.
JUMP_FLAG_THRESHOLD = 0.30


@dataclass(frozen=True)
class Ticker:
    sector: str
    symbol: str
    # Post-2022 redomiciliation/rename fallback, tried if `symbol` 4xx's.
    fallback: str | None = None


UNIVERSE: list[Ticker] = [
    # Oil & gas
    Ticker("Нефтегаз", "GAZP"),
    Ticker("Нефтегаз", "LKOH"),
    Ticker("Нефтегаз", "ROSN"),
    Ticker("Нефтегаз", "NVTK"),
    Ticker("Нефтегаз", "TATN"),
    Ticker("Нефтегаз", "SNGS"),
    Ticker("Нефтегаз", "TRNFP"),
    Ticker("Нефтегаз", "SIBN"),
    Ticker("Нефтегаз", "BANE"),
    # Banks
    Ticker("Банки", "SBER"),
    Ticker("Банки", "VTBR"),
    Ticker("Банки", "CBOM"),
    Ticker("Банки", "BSPB"),
    # Metals & mining
    Ticker("Металлурги", "GMKN"),
    Ticker("Металлурги", "NLMK"),
    Ticker("Металлурги", "CHMF"),
    Ticker("Металлурги", "MAGN"),
    Ticker("Металлурги", "ALRS"),
    Ticker("Металлурги", "PLZL"),
    Ticker("Металлурги", "RUAL"),
    Ticker("Металлурги", "MTLR"),
    Ticker("Металлурги", "SELG"),
    # Telecom
    Ticker("Телеком", "MTSS"),
    Ticker("Телеком", "RTKM"),
    # Retail — X5/TCS both redomiciled 2024-2025, ticker changed; try new
    # ticker first, fall back to the pre-redomiciliation one.
    Ticker("Ритейл", "MGNT"),
    Ticker("Ритейл", "LENT"),
    Ticker("Ритейл", "OZON"),
    Ticker("Ритейл", "DSKY"),
    Ticker("Ритейл", "X5", fallback="FIVE"),
    # IT
    Ticker("IT", "POSI"),
    Ticker("IT", "ASTR"),
    Ticker("IT", "VKCO"),
    Ticker("IT", "YDEX", fallback="YNDX"),
    # Chemicals / fertilizers
    Ticker("Удобрения", "PHOR"),
    Ticker("Удобрения", "KZOS"),
    Ticker("Удобрения", "AKRN"),
    # Utilities
    Ticker("Энергетика", "HYDR"),
    Ticker("Энергетика", "IRAO"),
    Ticker("Энергетика", "FEES"),
    Ticker("Энергетика", "UPRO"),
    # Transport
    Ticker("Транспорт", "AFLT"),
    Ticker("Транспорт", "NMTP"),
    Ticker("Транспорт", "FLOT"),
    # Other blue chips
    Ticker("Прочее", "MOEX"),
    Ticker("Прочее", "PIKK"),
    Ticker("Прочее", "LSRG"),
    Ticker("Прочее", "T", fallback="TCSG"),
]


@dataclass
class Result:
    sector: str
    symbol_used: str
    rows: int
    first_day: date | None
    last_day: date | None
    jump_flags: list[tuple[date, float]]
    error: str | None


def _flag_jumps(frame_close: list[float], frame_dates: list[date]) -> list[tuple[date, float]]:
    flags: list[tuple[date, float]] = []
    for i in range(1, len(frame_close)):
        prev, curr = frame_close[i - 1], frame_close[i]
        if prev <= 0:
            continue
        pct = (curr - prev) / prev
        if abs(pct) >= JUMP_FLAG_THRESHOLD:
            flags.append((frame_dates[i], pct))
    return flags


def _fetch_one(client: FinamClient, cache: ParquetCache, symbol: str) -> tuple[int, pd.DataFrame]:
    key = f"finam_daily_{symbol.replace('@', '_')}_{START.date()}_{END.date()}"
    frame = cache.get_or_fetch(key, lambda: client.daily_bars(f"{symbol}@{MIC}", START, END))
    return len(frame), frame


def main() -> None:
    load_dotenv()
    secret = os.environ.get("FINAM_SECRET_TOKEN")
    if not secret:
        print("FINAM_SECRET_TOKEN is not set (expected in a root .env) — aborting.")
        sys.exit(1)

    results: list[Result] = []
    with FinamClient(secret=secret) as client:
        client.authenticate()
        cache = ParquetCache()

        for t in UNIVERSE:
            symbol_used = t.symbol
            try:
                rows, frame = _fetch_one(client, cache, t.symbol)
                if rows == 0 and t.fallback:
                    raise FinamAPIError(f"zero rows for {t.symbol}, trying fallback")
            except FinamAPIError as exc:
                if t.fallback:
                    try:
                        symbol_used = t.fallback
                        rows, frame = _fetch_one(client, cache, t.fallback)
                    except FinamAPIError as exc2:
                        both = f"{t.symbol}/{t.fallback}"
                        results.append(Result(t.sector, both, 0, None, None, [], str(exc2)))
                        print(f"FAIL  {t.sector:12s} {both}: {exc2}")
                        continue
                else:
                    results.append(Result(t.sector, t.symbol, 0, None, None, [], str(exc)))
                    print(f"FAIL  {t.sector:12s} {t.symbol}: {exc}")
                    continue

            if rows == 0:
                results.append(Result(t.sector, symbol_used, 0, None, None, [], "zero rows"))
                print(f"EMPTY {t.sector:12s} {symbol_used}: 0 rows")
                continue

            dates = [d.date() for d in frame["TRADEDATE"]]
            closes = list(frame["CLOSE"])
            jumps = _flag_jumps(closes, dates)
            results.append(Result(t.sector, symbol_used, rows, dates[0], dates[-1], jumps, None))
            jump_note = f" [{len(jumps)} jump-flags]" if jumps else ""
            span = f"{dates[0]} .. {dates[-1]}"
            print(f"OK    {t.sector:12s} {symbol_used:8s} {rows:5d} rows  {span}{jump_note}")

    ok = [r for r in results if r.error is None and r.rows > 0]
    failed = [r for r in results if r.error is not None or r.rows == 0]

    print()
    print(f"Imported: {len(ok)}/{len(UNIVERSE)} tickers")
    print(f"Total rows: {sum(r.rows for r in ok)}")
    if ok:
        earliest = min(r.first_day for r in ok if r.first_day)
        latest = max(r.last_day for r in ok if r.last_day)
        print(f"Coverage: {earliest} .. {latest}")

    if failed:
        print()
        print("Failed/empty tickers:")
        for r in failed:
            print(f"  {r.sector:12s} {r.symbol_used}: {r.error}")

    flagged = [r for r in ok if r.jump_flags]
    if flagged:
        print()
        print("Tickers with >=30% single-day CLOSE jumps (likely un-adjusted corp. action):")
        for r in flagged:
            for d, pct in r.jump_flags:
                print(f"  {r.symbol_used:8s} {d}: {pct:+.1%}")


if __name__ == "__main__":
    main()
