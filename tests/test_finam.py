"""Tests for FinamClient. All HTTP is mocked via respx — no network access.

No real secret/JWT ever appears here: auth responses use an obviously-fake
token string, matching how test_moex_iss.py never touches a real endpoint.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pandas as pd
import pytest
import respx

from moex_backtest.data.finam import FinamAPIError, FinamClient

_FAKE_TOKEN = "fake.jwt.token"
_SESSIONS_URL = "https://api.finam.ru/v1/sessions"
_DETAILS_URL = "https://api.finam.ru/v1/sessions/details"


def _auth_route(expires_in: timedelta = timedelta(minutes=15)) -> None:
    respx.post(_SESSIONS_URL).mock(return_value=httpx.Response(200, json={"token": _FAKE_TOKEN}))
    expires_at = (datetime.now(UTC) + expires_in).strftime("%Y-%m-%dT%H:%M:%SZ")
    respx.post(_DETAILS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "created_at": "2026-01-01T00:00:00Z",
                "expires_at": expires_at,
                "account_ids": ["192264"],
                "readonly": False,
            },
        )
    )


def _bar(day: str, close: float) -> dict[str, Any]:
    return {
        "timestamp": day,
        "open": {"value": str(close - 1)},
        "high": {"value": str(close + 1)},
        "low": {"value": str(close - 2)},
        "close": {"value": str(close)},
        "volume": {"value": "1000"},
        "is_data_snapshot": False,
    }


def _bars_response(symbol: str, bars: list[dict[str, Any]]) -> dict[str, Any]:
    return {"symbol": symbol, "bars": bars}


def _one_bar_response() -> httpx.Response:
    return httpx.Response(200, json=_bars_response("SBER@MISX", [_bar("2026-08-10T00:00:00Z", 1)]))


@respx.mock
def test_authenticate_sets_token_and_expiry_from_token_details() -> None:
    _auth_route(expires_in=timedelta(minutes=15))

    with FinamClient(secret="s3cr3t") as client:
        token = client.authenticate()

    assert token == _FAKE_TOKEN
    assert client._token_expires_at is not None


@respx.mock
def test_bars_authenticates_transparently_and_parses_dataframe() -> None:
    _auth_route()
    bars_route = respx.get(
        "https://api.finam.ru/v1/instruments/SBER@MISX/bars",
    ).mock(
        return_value=httpx.Response(
            200, json=_bars_response("SBER@MISX", [_bar("2026-08-10T00:00:00Z", 300.5)])
        )
    )

    with FinamClient(secret="s3cr3t", request_delay=0) as client:
        frame = client.bars(
            "SBER@MISX",
            datetime(2026, 8, 1, tzinfo=UTC),
            datetime(2026, 8, 10, tzinfo=UTC),
        )

    assert bars_route.called
    auth_header = bars_route.calls[0].request.headers["authorization"]
    assert auth_header == _FAKE_TOKEN
    assert list(frame["SECID"]) == ["SBER@MISX"]
    assert frame["CLOSE"].iloc[0] == 300.5
    assert frame["TRADEDATE"].iloc[0] == pd.Timestamp("2026-08-10", tz="UTC")


@respx.mock
def test_bars_empty_range_returns_empty_frame() -> None:
    _auth_route()
    respx.get("https://api.finam.ru/v1/instruments/SBER@MISX/bars").mock(
        return_value=httpx.Response(200, json=_bars_response("SBER@MISX", []))
    )

    with FinamClient(secret="s3cr3t", request_delay=0) as client:
        frame = client.bars(
            "SBER@MISX",
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
        )

    assert frame.empty


@respx.mock
def test_history_chunks_range_past_daily_depth_limit() -> None:
    # A 900-day range for TIME_FRAME_D (365-day depth limit) must be split
    # into 3 request windows, each carrying one bar, then stitched together.
    _auth_route()
    route = respx.get("https://api.finam.ru/v1/instruments/SBER@MISX/bars")
    route.side_effect = [
        httpx.Response(200, json=_bars_response("SBER@MISX", [_bar("2023-01-01T00:00:00Z", 100)])),
        httpx.Response(200, json=_bars_response("SBER@MISX", [_bar("2024-01-01T00:00:00Z", 200)])),
        httpx.Response(200, json=_bars_response("SBER@MISX", [_bar("2025-01-01T00:00:00Z", 300)])),
    ]

    with FinamClient(secret="s3cr3t", request_delay=0) as client:
        frame = client.history(
            "SBER@MISX",
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 1, 1, tzinfo=UTC) + timedelta(days=900),
            timeframe="TIME_FRAME_D",
        )

    assert route.call_count == 3
    assert len(frame) == 3
    assert list(frame["CLOSE"]) == [100.0, 200.0, 300.0]


@respx.mock
def test_history_unknown_timeframe_raises_value_error() -> None:
    with (
        FinamClient(secret="s3cr3t", request_delay=0) as client,
        pytest.raises(ValueError, match="unknown timeframe"),
    ):
        client.history(
            "SBER@MISX",
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 1, 2, tzinfo=UTC),
            timeframe="NOT_A_TIMEFRAME",
        )


@respx.mock
def test_expired_token_is_refreshed_before_next_request() -> None:
    # First auth returns a token that's already effectively expired (0s TTL);
    # the client must re-authenticate before the second `bars` call rather
    # than reuse the stale token.
    respx.post(_SESSIONS_URL).mock(
        side_effect=[
            httpx.Response(200, json={"token": "stale.token"}),
            httpx.Response(200, json={"token": _FAKE_TOKEN}),
        ]
    )
    respx.post(_DETAILS_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                json={"expires_at": (datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")},
            ),
            httpx.Response(
                200,
                json={
                    "expires_at": (datetime.now(UTC) + timedelta(minutes=15)).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    )
                },
            ),
        ]
    )
    bars_route = respx.get("https://api.finam.ru/v1/instruments/SBER@MISX/bars").mock(
        return_value=_one_bar_response()
    )

    with FinamClient(secret="s3cr3t", request_delay=0) as client:
        client.authenticate()
        client.bars(
            "SBER@MISX",
            datetime(2026, 8, 1, tzinfo=UTC),
            datetime(2026, 8, 10, tzinfo=UTC),
        )

    assert bars_route.calls[0].request.headers["authorization"] == _FAKE_TOKEN


@respx.mock
def test_reactive_401_forces_reauth_and_retries_once() -> None:
    _auth_route()
    bars_route = respx.get("https://api.finam.ru/v1/instruments/SBER@MISX/bars")
    bars_route.side_effect = [
        httpx.Response(401),
        _one_bar_response(),
    ]

    with FinamClient(secret="s3cr3t", request_delay=0) as client:
        frame = client.bars(
            "SBER@MISX",
            datetime(2026, 8, 1, tzinfo=UTC),
            datetime(2026, 8, 10, tzinfo=UTC),
        )

    assert bars_route.call_count == 2
    assert len(frame) == 1


@respx.mock
def test_auth_missing_token_raises_finam_api_error() -> None:
    respx.post(_SESSIONS_URL).mock(return_value=httpx.Response(200, json={"unexpected": {}}))

    with FinamClient(secret="s3cr3t", request_delay=0) as client, pytest.raises(FinamAPIError):
        client.authenticate()


@respx.mock
def test_token_details_failure_falls_back_to_default_ttl() -> None:
    respx.post(_SESSIONS_URL).mock(return_value=httpx.Response(200, json={"token": _FAKE_TOKEN}))
    respx.post(_DETAILS_URL).mock(return_value=httpx.Response(500))

    with FinamClient(
        secret="s3cr3t", request_delay=0, retry_backoff=0.001, max_retries=0
    ) as client:
        token = client.authenticate()

    assert token == _FAKE_TOKEN
    assert client._token_expires_at is not None


@respx.mock
def test_transient_5xx_is_retried_then_succeeds() -> None:
    _auth_route()
    route = respx.get("https://api.finam.ru/v1/instruments/SBER@MISX/bars")
    route.side_effect = [
        httpx.Response(502),
        _one_bar_response(),
    ]

    with FinamClient(secret="s3cr3t", request_delay=0, retry_backoff=0.001) as client:
        frame = client.bars(
            "SBER@MISX",
            datetime(2026, 8, 1, tzinfo=UTC),
            datetime(2026, 8, 10, tzinfo=UTC),
        )

    assert route.call_count == 2
    assert len(frame) == 1


@respx.mock
def test_rate_limit_429_is_retried_then_succeeds() -> None:
    _auth_route()
    route = respx.get("https://api.finam.ru/v1/instruments/SBER@MISX/bars")
    route.side_effect = [
        httpx.Response(429),
        _one_bar_response(),
    ]

    with FinamClient(secret="s3cr3t", request_delay=0, retry_backoff=0.001) as client:
        frame = client.bars(
            "SBER@MISX",
            datetime(2026, 8, 1, tzinfo=UTC),
            datetime(2026, 8, 10, tzinfo=UTC),
        )

    assert route.call_count == 2
    assert len(frame) == 1


@respx.mock
def test_transient_transport_error_is_retried_then_succeeds() -> None:
    _auth_route()
    route = respx.get("https://api.finam.ru/v1/instruments/SBER@MISX/bars")
    route.side_effect = [
        httpx.ConnectError("connection refused"),
        _one_bar_response(),
    ]

    with FinamClient(secret="s3cr3t", request_delay=0, retry_backoff=0.001) as client:
        frame = client.bars(
            "SBER@MISX",
            datetime(2026, 8, 1, tzinfo=UTC),
            datetime(2026, 8, 10, tzinfo=UTC),
        )

    assert route.call_count == 2
    assert len(frame) == 1


@respx.mock
def test_404_fails_immediately_without_retrying() -> None:
    _auth_route()
    route = respx.get("https://api.finam.ru/v1/instruments/NOSUCH@MISX/bars").mock(
        return_value=httpx.Response(404)
    )

    with (
        FinamClient(secret="s3cr3t", request_delay=0, retry_backoff=0.001, max_retries=3) as client,
        pytest.raises(FinamAPIError),
    ):
        client.bars(
            "NOSUCH@MISX",
            datetime(2026, 8, 1, tzinfo=UTC),
            datetime(2026, 8, 10, tzinfo=UTC),
        )

    assert route.call_count == 1


@respx.mock
def test_exhausted_retries_raise_finam_api_error() -> None:
    _auth_route()
    respx.get("https://api.finam.ru/v1/instruments/SBER@MISX/bars").mock(
        return_value=httpx.Response(500)
    )

    with (
        FinamClient(secret="s3cr3t", request_delay=0, retry_backoff=0.001, max_retries=2) as client,
        pytest.raises(FinamAPIError),
    ):
        client.bars(
            "SBER@MISX",
            datetime(2026, 8, 1, tzinfo=UTC),
            datetime(2026, 8, 10, tzinfo=UTC),
        )


def test_empty_secret_raises_value_error() -> None:
    with pytest.raises(ValueError, match="secret"):
        FinamClient(secret="")
