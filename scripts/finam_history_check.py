"""Verify that authenticated Finam Trade API access lifts MOEX ISS's anonymous
~35-trading-day depth limit for individual equities.

Authenticates with `FINAM_SECRET_TOKEN` (loaded from a root `.env`, never
printed), fetches multi-year daily history for SBER via
`FinamClient.daily_bars`, and reports how many days/years actually came
back — the empirical answer to "does auth remove the depth cap".

This script makes real network calls to api.finam.ru and needs real
credentials; it is not run in CI or in the test suite (see tests/test_finam.py
for the mocked-HTTP unit tests).

Run with: uv run python scripts/finam_history_check.py
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta

from dotenv import load_dotenv

from moex_backtest.data import FinamAPIError, FinamClient

SYMBOL = "SBER@MISX"
YEARS_REQUESTED = 5


def main() -> None:
    load_dotenv()
    secret = os.environ.get("FINAM_SECRET_TOKEN")
    if not secret:
        print("FINAM_SECRET_TOKEN is not set (expected in a root .env) — aborting.")
        sys.exit(1)

    end = datetime.now(UTC)
    start = end - timedelta(days=365 * YEARS_REQUESTED)

    print(
        f"Requesting {SYMBOL} daily bars for {start.date()} .. {end.date()} "
        f"({YEARS_REQUESTED} years requested)..."
    )

    with FinamClient(secret=secret) as client:
        try:
            client.authenticate()
            print("Auth OK (JWT obtained, token itself not printed).")
            frame = client.daily_bars(SYMBOL, start, end)
        except FinamAPIError as exc:
            print(f"Finam API error: {exc}")
            sys.exit(1)

    if frame.empty:
        print(
            "Got zero rows back — either the symbol/permissions are wrong or "
            "the API returned no data for this range."
        )
        sys.exit(1)

    first_day = frame["TRADEDATE"].min().date()
    last_day = frame["TRADEDATE"].max().date()
    span_days = (last_day - first_day).days
    num_rows = len(frame)

    print()
    print(f"Rows returned:        {num_rows}")
    print(f"First trading day:    {first_day}")
    print(f"Last trading day:     {last_day}")
    print(f"Calendar span:        {span_days} days (~{span_days / 365.25:.2f} years)")
    print()

    anon_moex_iss_limit_days = 35
    if span_days > anon_moex_iss_limit_days:
        print(
            f"RESULT: {span_days} days of history >> the anonymous MOEX ISS "
            f"~{anon_moex_iss_limit_days}-trading-day cap for equities — the "
            "depth limit is lifted by authenticated Finam Trade API access."
        )
    else:
        print(
            f"RESULT: only got {span_days} days back — the depth limit does NOT "
            "look lifted; investigate before relying on this for research."
        )

    if span_days < 365 * YEARS_REQUESTED - anon_moex_iss_limit_days:
        print(
            f"NOTE: requested {YEARS_REQUESTED} years but got ~{span_days / 365.25:.2f} — "
            "if this recurs, TIME_FRAME_D's documented 365-day-per-request depth "
            "limit may be an absolute (not per-window) cap; check FinamClient.history's "
            "chunking assumption in src/moex_backtest/data/finam.py."
        )


if __name__ == "__main__":
    main()
