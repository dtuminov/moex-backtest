"""Typed client for the Finam Trade API REST gateway (auth + historical bars).

Reference: https://tradeapi.finam.ru/ ; REST spec: https://api.finam.ru/docs/rest/
(mirrored OpenAPI/proto definitions: https://github.com/FinamWeb/finam-trade-api)
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Any, Final

import httpx
import pandas as pd

_BASE_URL: Final = "https://api.finam.ru"

# Per-timeframe depth limit for a single `Bars` request window, per the API's
# published docs (docs/swagger/api.swagger.json in the finam-trade-api repo,
# description of MarketDataService_Bars' `timeframe` parameter). Unlike MOEX
# ISS's row-count pagination, this is a *span* limit on `interval.start_time`
# / `interval.end_time`: a single call cannot cover more than this much time,
# but nothing stops the window from being moved arbitrarily far into the past
# across repeated calls — which is what `history()` below does automatically.
_DEPTH_LIMITS: Final[dict[str, timedelta]] = {
    "TIME_FRAME_M1": timedelta(days=7),
    "TIME_FRAME_M5": timedelta(days=30),
    "TIME_FRAME_M15": timedelta(days=30),
    "TIME_FRAME_M30": timedelta(days=30),
    "TIME_FRAME_H1": timedelta(days=30),
    "TIME_FRAME_H2": timedelta(days=30),
    "TIME_FRAME_H4": timedelta(days=30),
    "TIME_FRAME_H8": timedelta(days=30),
    "TIME_FRAME_D": timedelta(days=365),
    "TIME_FRAME_W": timedelta(days=365 * 5),
    "TIME_FRAME_MN": timedelta(days=365 * 5),
    "TIME_FRAME_QR": timedelta(days=365 * 5),
}

# Documented JWT lifetime; used only as a fallback if a `TokenDetails` call
# (which returns the server's own `expires_at`) fails for some reason.
_TOKEN_TTL_FALLBACK: Final = timedelta(minutes=15)
# Refresh this long before actual expiry, so a request started just before
# expiry doesn't race a token that dies mid-flight.
_REFRESH_MARGIN: Final = timedelta(seconds=30)


class FinamAPIError(RuntimeError):
    """Raised when the Finam Trade API returns an error or an unexpected payload."""


class FinamClient:
    """Thin, typed wrapper around the Finam Trade API REST gateway.

    Handles the two things that matter for data-research use, before any
    order/account logic is involved:

    * **auth**: exchanges a long-lived secret token (``FINAM_SECRET_TOKEN``)
      for a JWT via ``POST /v1/sessions``, and re-authenticates transparently
      before any request whose token is missing, expired, or about to expire
      (see :data:`_REFRESH_MARGIN`) — callers never handle the JWT
      themselves. Token TTL is read from the server via
      ``POST /v1/sessions/details`` right after auth, not assumed, since it's
      the authoritative source; the documented 15-minute TTL
      (:data:`_TOKEN_TTL_FALLBACK`) is only a fallback if that call fails.
    * **history depth**: unlike anonymous MOEX ISS (~35 trading days for
      individual equities regardless of the requested range — see
      :class:`moex_backtest.data.moex_iss.MoexISSClient`), authenticated
      ``Bars`` access is not row-count paginated but *span*-limited per
      request (:data:`_DEPTH_LIMITS`, e.g. 365 days per call for daily bars).
      :meth:`history` chunks an arbitrary ``[start, end)`` range into windows
      that respect this limit and stitches the results back together, so
      callers can ask for years of daily history in one call, same as
      ``MoexISSClient.shares_history``.

    The secret token is never logged: it's held only in memory as
    ``self._secret``, sent once per re-auth in a POST body, and errors raised
    by this client never include it.
    """

    def __init__(
        self,
        secret: str,
        base_url: str = _BASE_URL,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        request_delay: float = 0.3,
        client: httpx.Client | None = None,
    ) -> None:
        if not secret:
            raise ValueError("secret must not be empty")
        self._secret = secret
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        # Default spacing keeps callers under the documented 200 req/min
        # limit (0.3s/request => 200/min) even without external throttling.
        self._request_delay = request_delay
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        self._token: str | None = None
        self._token_expires_at: datetime | None = None

    def __enter__(self) -> FinamClient:
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

    # -- auth ------------------------------------------------------------------

    def authenticate(self) -> str:
        """Exchange the secret token for a fresh JWT and cache its expiry.

        Called automatically by any request method before a request that
        needs a token which is missing or close to expiry; exposed directly
        for callers that just want to verify credentials up front.
        """
        payload = self._request("POST", "/v1/sessions", json={"secret": self._secret})
        token = payload.get("token")
        if not isinstance(token, str) or not token:
            raise FinamAPIError(f"unexpected auth payload shape: missing 'token' ({payload!r})")
        self._token = token
        self._token_expires_at = self._fetch_token_expiry(token)
        return token

    def _fetch_token_expiry(self, token: str) -> datetime:
        """Look up the JWT's real expiry via `TokenDetails`.

        Falls back to `_TOKEN_TTL_FALLBACK` from now if the details call
        fails or returns an unparsable timestamp — auth already succeeded at
        this point, so a degraded-but-working expiry estimate is better than
        raising.
        """
        try:
            details = self._request("POST", "/v1/sessions/details", json={"token": token})
            expires_at_raw = details.get("expires_at")
            if isinstance(expires_at_raw, str):
                return _parse_rfc3339(expires_at_raw)
        except (FinamAPIError, ValueError):
            pass
        return datetime.now(UTC) + _TOKEN_TTL_FALLBACK

    def _ensure_token(self) -> str:
        now = datetime.now(UTC)
        stale = (
            self._token is None
            or self._token_expires_at is None
            or now >= self._token_expires_at - _REFRESH_MARGIN
        )
        if stale:
            return self.authenticate()
        assert self._token is not None  # narrowed by `stale` check above
        return self._token

    # -- market data -------------------------------------------------------------

    def bars(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "TIME_FRAME_D"
    ) -> pd.DataFrame:
        """Fetch one page of aggregated bars for `symbol` in `[start, end)`.

        A single call is subject to the API's per-timeframe depth limit (see
        :data:`_DEPTH_LIMITS`, e.g. 365 days for ``TIME_FRAME_D``) — use
        :meth:`history` for transparent multi-year assembly across that
        limit. Returns an empty DataFrame if the instrument has no bars in
        range.
        """
        params = {
            "timeframe": timeframe,
            "interval.start_time": _format_rfc3339(start),
            "interval.end_time": _format_rfc3339(end),
        }
        payload = self._authed_request("GET", f"/v1/instruments/{symbol}/bars", params=params)
        raw_bars = payload.get("bars")
        if raw_bars is None:
            raise FinamAPIError(f"unexpected bars payload shape: missing 'bars' ({payload!r})")
        if not raw_bars:
            return pd.DataFrame()

        rows = [
            {
                "SECID": symbol,
                "TRADEDATE": bar["timestamp"],
                "OPEN": _decimal(bar["open"]),
                "HIGH": _decimal(bar["high"]),
                "LOW": _decimal(bar["low"]),
                "CLOSE": _decimal(bar["close"]),
                "VOLUME": _decimal(bar["volume"]),
            }
            for bar in raw_bars
        ]
        frame = pd.DataFrame(rows)
        frame["TRADEDATE"] = pd.to_datetime(frame["TRADEDATE"])
        return frame.sort_values("TRADEDATE").drop_duplicates("TRADEDATE").reset_index(drop=True)

    def history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "TIME_FRAME_D",
    ) -> pd.DataFrame:
        """Fetch full history for `symbol` across `[start, end)`, any span.

        Chunks the range into windows no wider than the timeframe's depth
        limit (:data:`_DEPTH_LIMITS`) and stitches the pages together —
        callers don't need to think about the per-request span cap, same as
        `MoexISSClient.history` hides MOEX ISS's row-count pagination.
        """
        window = _DEPTH_LIMITS.get(timeframe)
        if window is None:
            raise ValueError(f"unknown timeframe {timeframe!r}")

        frames: list[pd.DataFrame] = []
        window_start = start
        first = True
        while window_start < end:
            if not first and self._request_delay:
                time.sleep(self._request_delay)
            first = False
            window_end = min(window_start + window, end)
            frame = self.bars(symbol, window_start, window_end, timeframe=timeframe)
            if not frame.empty:
                frames.append(frame)
            window_start = window_end

        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        return combined.sort_values("TRADEDATE").drop_duplicates("TRADEDATE").reset_index(drop=True)

    def daily_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Full daily OHLCV history for `symbol` across `[start, end)`.

        Convenience wrapper over `history` with `timeframe="TIME_FRAME_D"` —
        this is what actually lifts MOEX ISS's ~35-trading-day anonymous
        depth limit for individual equities, per the 365-day-per-request
        `TIME_FRAME_D` cap chunked transparently across the full range.
        """
        return self.history(symbol, start, end, timeframe="TIME_FRAME_D")

    # -- transport ---------------------------------------------------------------

    def _authed_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        """Issue a request that requires a JWT, refreshing/retrying on auth failure.

        Proactively refreshes an expired-or-near-expiry token before the
        call (`_ensure_token`); if the server still rejects the token with
        401 (clock skew, mid-flight revocation), forces one re-auth and
        retries the request exactly once before giving up.

        ``max_retries=0`` disables the transport-level retry loop for this
        call. State-changing calls (order placement) must pass it: a retried
        POST that actually succeeded server-side but whose response was lost
        places the order twice. The 401 re-auth retry stays, because a
        request rejected for auth never reached the matching engine.
        """
        token = self._ensure_token()
        try:
            return self._request(
                method,
                path,
                params=params,
                json=json,
                headers={"Authorization": token},
                max_retries=max_retries,
            )
        except FinamAPIError as exc:
            if "401" not in str(exc):
                raise
            self._token = None
            self._token_expires_at = None
            token = self._ensure_token()
            return self._request(
                method,
                path,
                params=params,
                json=json,
                headers={"Authorization": token},
                max_retries=max_retries,
            )

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        retries = self._max_retries if max_retries is None else max_retries
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = self._client.request(
                    method, url, params=params, json=json, headers=headers
                )
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = httpx.HTTPStatusError(
                        f"{response.status_code}", request=response.request, response=response
                    )
                else:
                    response.raise_for_status()
                    return response.json()  # type: ignore[no-any-return]
            except httpx.HTTPStatusError as exc:
                # Any other 4xx (bad symbol, bad params, permanent 401/404) is
                # a permanent client error for this attempt — retrying with
                # backoff just burns the retry budget on something that will
                # never resolve on its own. Only 429 (rate limit) and 5xx
                # (transient server trouble) are worth retrying.
                if exc.response.status_code < 500 and exc.response.status_code != 429:
                    raise FinamAPIError(
                        f"{method} {url} failed with {exc.response.status_code}"
                    ) from exc
                last_error = exc
            except httpx.TransportError as exc:
                last_error = exc
            if attempt < retries:
                time.sleep(self._retry_backoff * (2**attempt))
        raise FinamAPIError(f"{method} {url} failed after {retries + 1} attempts") from last_error


def _decimal(field: dict[str, Any]) -> float:
    try:
        return float(field["value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FinamAPIError(f"unexpected decimal field shape: {field!r}") from exc


def _format_rfc3339(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_rfc3339(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
