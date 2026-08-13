"""Tests for MoexISSClient. All HTTP is mocked via respx — no network access."""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx
import pandas as pd
import pytest
import respx

from moex_backtest.data.moex_iss import MoexISSClient, MoexISSError

_COLUMNS = ["BOARDID", "TRADEDATE", "SECID", "CLOSE", "VOLUME", "NUMTRADES"]


def _payload(rows: list[list[Any]]) -> dict[str, Any]:
    return {"history": {"columns": _COLUMNS, "data": rows}}


def _row(day: str, close: float, numtrades: int = 1) -> list[Any]:
    return ["TQBR", day, "SBER", close, 1000, numtrades]


@respx.mock
def test_history_single_page_parses_dataframe() -> None:
    route = respx.get(
        "https://iss.moex.com/iss/history/engines/stock/markets/shares"
        "/boards/TQBR/securities/SBER.json"
    ).mock(return_value=httpx.Response(200, json=_payload([_row("2026-08-10", 300.5)])))

    with MoexISSClient(request_delay=0) as client:
        frame = client.shares_history("SBER", date(2026, 8, 1), date(2026, 8, 10))

    assert route.called
    assert list(frame["SECID"]) == ["SBER"]
    assert frame["TRADEDATE"].iloc[0] == pd.Timestamp("2026-08-10")


@respx.mock
def test_history_paginates_until_a_short_page() -> None:
    full_page = [_row(f"2026-01-{i:02d}", 100 + i) for i in range(1, 4)]
    last_page = [_row("2026-01-10", 200)]
    route = respx.get(
        "https://iss.moex.com/iss/history/engines/stock/markets/index/securities/IMOEX.json"
    )
    route.side_effect = [
        httpx.Response(200, json=_payload(full_page)),
        httpx.Response(200, json=_payload(last_page)),
    ]

    with MoexISSClient(page_size=3, request_delay=0) as client:
        frame = client.index_history("IMOEX", date(2026, 1, 1), date(2026, 1, 10))

    assert route.call_count == 2
    assert len(frame) == 4
    assert route.calls[0].request.url.params["start"] == "0"
    assert route.calls[1].request.url.params["start"] == "3"


@respx.mock
def test_history_empty_range_returns_empty_frame() -> None:
    respx.get(
        "https://iss.moex.com/iss/history/engines/futures/markets/options/securities/SR1.json"
    ).mock(return_value=httpx.Response(200, json=_payload([])))

    with MoexISSClient(request_delay=0) as client:
        frame = client.options_history("SR1", date(2026, 1, 1), date(2026, 1, 2))

    assert frame.empty


@respx.mock
def test_missing_history_key_raises_moex_iss_error() -> None:
    respx.get(
        "https://iss.moex.com/iss/history/engines/stock/markets/shares"
        "/boards/TQBR/securities/SBER.json"
    ).mock(return_value=httpx.Response(200, json={"unexpected": {}}))

    with MoexISSClient(request_delay=0) as client, pytest.raises(MoexISSError):
        client.shares_history("SBER", date(2026, 1, 1), date(2026, 1, 2))


@respx.mock
def test_transient_5xx_is_retried_then_succeeds() -> None:
    route = respx.get(
        "https://iss.moex.com/iss/history/engines/stock/markets/index/securities/IMOEX.json"
    )
    route.side_effect = [
        httpx.Response(502),
        httpx.Response(200, json=_payload([_row("2026-08-10", 3000)])),
    ]

    with MoexISSClient(request_delay=0, retry_backoff=0.001) as client:
        frame = client.index_history("IMOEX", date(2026, 8, 1), date(2026, 8, 10))

    assert route.call_count == 2
    assert len(frame) == 1


@respx.mock
def test_transient_transport_error_is_retried_then_succeeds() -> None:
    route = respx.get(
        "https://iss.moex.com/iss/history/engines/stock/markets/index/securities/IMOEX.json"
    )
    route.side_effect = [
        httpx.ConnectError("connection refused"),
        httpx.Response(200, json=_payload([_row("2026-08-10", 3000)])),
    ]

    with MoexISSClient(request_delay=0, retry_backoff=0.001) as client:
        frame = client.index_history("IMOEX", date(2026, 8, 1), date(2026, 8, 10))

    assert route.call_count == 2
    assert len(frame) == 1


@respx.mock
def test_404_fails_immediately_without_retrying() -> None:
    # A typo'd secid returns a permanent 404 — retrying it with backoff
    # would just burn the whole retry budget on an error that can never
    # resolve. Only transient 5xx/transport errors should be retried.
    route = respx.get(
        "https://iss.moex.com/iss/history/engines/stock/markets/shares"
        "/boards/TQBR/securities/NOSUCH.json"
    ).mock(return_value=httpx.Response(404))

    with (
        MoexISSClient(request_delay=0, retry_backoff=0.001, max_retries=3) as client,
        pytest.raises(MoexISSError),
    ):
        client.shares_history("NOSUCH", date(2026, 8, 1), date(2026, 8, 10))

    assert route.call_count == 1


@respx.mock
def test_exhausted_retries_raise_moex_iss_error() -> None:
    respx.get(
        "https://iss.moex.com/iss/history/engines/stock/markets/index/securities/IMOEX.json"
    ).mock(return_value=httpx.Response(500))

    with (
        MoexISSClient(request_delay=0, retry_backoff=0.001, max_retries=2) as client,
        pytest.raises(MoexISSError),
    ):
        client.index_history("IMOEX", date(2026, 8, 1), date(2026, 8, 10))
