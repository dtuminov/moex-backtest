"""Typed client for the Moscow Exchange (MOEX) ISS market data API.

Reference: https://iss.moex.com/iss/reference/
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import date
from types import TracebackType
from typing import Any, Final

import httpx
import pandas as pd

_BASE_URL: Final = "https://iss.moex.com/iss"
_PAGE_SIZE: Final = 100  # rows per page, enforced server-side, not configurable
_DATE_FMT: Final = "%Y-%m-%d"


class MoexISSError(RuntimeError):
    """Raised when the ISS API returns an error or an unexpected payload."""


class MoexISSClient:
    """Thin, typed wrapper around the MOEX ISS ``history`` endpoints.

    Handles the two quirks that matter for backtesting research, found by
    probing the API directly:

    * results are paginated at a fixed page size (:data:`_PAGE_SIZE`)
      regardless of the requested date range, so a full history has to be
      fetched page by page;
    * anonymous (unauthenticated) access is depth-limited differently per
      market — equities (``stock``/``shares``) return roughly the last 35
      trading days no matter how far back ``start`` is, while indices and
      FORTS futures/options are not depth-limited at all. This client does
      not work around the equities limit; callers that need full equity
      history need an authenticated ISS session, out of scope here.
    """

    def __init__(
        self,
        base_url: str = _BASE_URL,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        request_delay: float = 0.1,
        client: httpx.Client | None = None,
        page_size: int = _PAGE_SIZE,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._request_delay = request_delay
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        # Real ISS pages at exactly `_PAGE_SIZE`; overridable so pagination
        # logic can be exercised in tests without mocking 100-row pages.
        self._page_size = page_size

    def __enter__(self) -> MoexISSClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # -- generic history ----------------------------------------------------

    def history(
        self,
        engine: str,
        market: str,
        secid: str,
        start: date,
        end: date,
        board: str | None = None,
    ) -> pd.DataFrame:
        """Fetch full daily trading history for one instrument.

        Paginates transparently and concatenates all pages. Returns an empty
        DataFrame (not an error) if the instrument has no history in range.
        """
        path = f"/history/engines/{engine}/markets/{market}"
        if board is not None:
            path += f"/boards/{board}"
        path += f"/securities/{secid}.json"
        params = {
            "iss.meta": "off",
            "from": start.strftime(_DATE_FMT),
            "till": end.strftime(_DATE_FMT),
        }

        columns: list[str] | None = None
        rows: list[list[Any]] = []
        for page_columns, page_rows in self._iter_pages(path, params):
            columns = columns or page_columns
            rows.extend(page_rows)

        if columns is None or not rows:
            return pd.DataFrame()

        frame = pd.DataFrame(rows, columns=columns)
        frame["TRADEDATE"] = pd.to_datetime(frame["TRADEDATE"])
        return frame.sort_values("TRADEDATE").drop_duplicates("TRADEDATE").reset_index(drop=True)

    # -- convenience wrappers over the instrument families we actually use --

    def shares_history(
        self, secid: str, start: date, end: date, board: str = "TQBR"
    ) -> pd.DataFrame:
        """Daily OHLCV history for an equity.

        Anonymous access is capped at roughly the last 35 trading days
        regardless of ``start`` — see the class docstring.
        """
        return self.history("stock", "shares", secid, start, end, board=board)

    def index_history(self, secid: str, start: date, end: date) -> pd.DataFrame:
        """Daily close-level history for an index (e.g. IMOEX, MCFTR, RVI). Not depth-limited."""
        return self.history("stock", "index", secid, start, end)

    def futures_history(self, secid: str, start: date, end: date) -> pd.DataFrame:
        """Daily OHLCV + settlement history for a FORTS futures contract. Not depth-limited."""
        return self.history("futures", "forts", secid, start, end)

    def options_history(self, secid: str, start: date, end: date) -> pd.DataFrame:
        """Daily OHLCV + settlement history for a FORTS option.

        Not depth-limited, but most rows away from expiry carry zero real
        trades (only the exchange's settlement/theoretical price) — check
        ``NUMTRADES`` before treating a day's price as market-observed.
        """
        return self.history("futures", "options", secid, start, end)

    # -- transport ------------------------------------------------------------

    def _iter_pages(
        self, path: str, params: dict[str, str]
    ) -> Iterator[tuple[list[str], list[list[Any]]]]:
        offset = 0
        while True:
            payload = self._get(path, {**params, "start": offset})
            block = _extract_history_block(payload)
            columns, rows = block["columns"], block["data"]
            yield columns, rows
            if len(rows) < self._page_size:
                return
            offset += len(rows)
            if self._request_delay:
                time.sleep(self._request_delay)

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.get(url, params=params)
                response.raise_for_status()
                return response.json()  # type: ignore[no-any-return]
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt < self._max_retries:
                    time.sleep(self._retry_backoff * (2**attempt))
        raise MoexISSError(
            f"GET {url} failed after {self._max_retries + 1} attempts"
        ) from last_error


def _extract_history_block(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return payload["history"]  # type: ignore[no-any-return]
    except KeyError as exc:
        raise MoexISSError(
            f"unexpected ISS payload shape: missing 'history' key ({payload!r})"
        ) from exc
