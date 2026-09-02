import inspect
import json
from dataclasses import FrozenInstanceError

import httpx
import pandas as pd
import pytest

import nakagai.data.alpaca as alpaca_module
from nakagai.data.alpaca import (
    AlpacaBarBatchResult,
    AlpacaBarMember,
    AlpacaProvider,
    frame_from_rows,
)

START = pd.Timestamp("2026-06-01", tz="UTC")
END = pd.Timestamp("2026-06-02", tz="UTC")

PAGE1 = {
    "bars": [
        {"t": "2026-06-01T13:30:00Z", "o": 1.0, "h": 1.5, "l": 0.5, "c": 1.2, "v": 100},
        {"t": "2026-06-01T13:45:00Z", "o": 1.2, "h": 1.6, "l": 1.0, "c": 1.4, "v": 110},
    ],
    "next_page_token": "tok123",
}
PAGE2 = {
    "bars": [{"t": "2026-06-01T14:00:00Z", "o": 1.4, "h": 1.7, "l": 1.3, "c": 1.5, "v": 120}],
    "next_page_token": None,
}


def _mock_client(captured):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        page = PAGE2 if request.url.params.get("page_token") == "tok123" else PAGE1
        return httpx.Response(200, json=page)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_paginates_and_normalizes():
    captured = []
    p = AlpacaProvider(key_id="k", secret="s", client=_mock_client(captured), sleep=lambda _: None)
    df = p.fetch_bars("SPY", "15m", pd.Timestamp("2026-06-01", tz="UTC"), pd.Timestamp("2026-06-02", tz="UTC"))
    assert len(df) == 3
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index[0] == pd.Timestamp("2026-06-01 13:30", tz="UTC")
    assert len(captured) == 2


def test_public_row_converter_normalizes_provider_rows():
    frame = frame_from_rows(PAGE1["bars"])
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.index.tolist() == [
        pd.Timestamp("2026-06-01 13:30", tz="UTC"),
        pd.Timestamp("2026-06-01 13:45", tz="UTC"),
    ]
    assert frame["close"].tolist() == [1.2, 1.4]


def test_missing_credentials_fails_fast(monkeypatch):
    """No keys must produce an actionable error, not a raw 401 from the API."""
    import pytest

    monkeypatch.delenv("ALPACA_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    p = AlpacaProvider(client=_mock_client([]))
    with pytest.raises(RuntimeError, match="ALPACA_KEY_ID"):
        p.fetch_bars("SPY", "15m", pd.Timestamp("2026-06-01", tz="UTC"), pd.Timestamp("2026-06-02", tz="UTC"))


def _rate_limited_client(captured, fail_times, retry_after=None):
    """429 for the first fail_times requests, then serve PAGE2."""
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if len(captured) <= fail_times:
            headers = {"Retry-After": retry_after} if retry_after else {}
            return httpx.Response(429, headers=headers, json={"message": "rate limit"})
        return httpx.Response(200, json=PAGE2)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_429_sleeps_and_retries_honoring_retry_after():
    captured, sleeps = [], []
    p = AlpacaProvider(key_id="k", secret="s",
                       client=_rate_limited_client(captured, fail_times=2, retry_after="7"),
                       sleep=sleeps.append)
    df = p.fetch_bars("SPY", "15m", pd.Timestamp("2026-06-01", tz="UTC"), pd.Timestamp("2026-06-02", tz="UTC"))
    assert len(df) == 1  # survived the limiter instead of skipping the symbol
    assert len(captured) == 3
    assert sleeps.count(7.0) == 2  # Retry-After honored on both 429s


def test_429_backs_off_exponentially_without_retry_after():
    captured, sleeps = [], []
    p = AlpacaProvider(key_id="k", secret="s",
                       client=_rate_limited_client(captured, fail_times=3),
                       sleep=sleeps.append)
    p.fetch_bars("SPY", "15m", pd.Timestamp("2026-06-01", tz="UTC"), pd.Timestamp("2026-06-02", tz="UTC"))
    retry_sleeps = [s for s in sleeps if s >= 1.0]
    assert retry_sleeps == [1.0, 2.0, 4.0]


def test_429_exhausted_raises():
    import pytest

    captured, sleeps = [], []
    p = AlpacaProvider(key_id="k", secret="s",
                       client=_rate_limited_client(captured, fail_times=99),
                       sleep=sleeps.append)
    with pytest.raises(httpx.HTTPStatusError):
        p.fetch_bars("SPY", "15m", pd.Timestamp("2026-06-01", tz="UTC"), pd.Timestamp("2026-06-02", tz="UTC"))
    assert len(captured) == 6  # initial attempt + 5 retries


def test_pacing_spaces_out_paginated_requests():
    captured, sleeps = [], []
    p = AlpacaProvider(key_id="k", secret="s", client=_mock_client(captured),
                       min_interval=0.35, sleep=sleeps.append)
    p.fetch_bars("SPY", "15m", pd.Timestamp("2026-06-01", tz="UTC"), pd.Timestamp("2026-06-02", tz="UTC"))
    assert len(captured) == 2
    # no sleep before the first request, one pacing sleep before the second
    assert len(sleeps) == 1
    assert 0 < sleeps[0] <= 0.35


def test_request_params_and_auth():
    captured = []
    p = AlpacaProvider(key_id="k", secret="s", client=_mock_client(captured), sleep=lambda _: None)
    p.fetch_bars("SPY", "1h", pd.Timestamp("2026-06-01", tz="UTC"), pd.Timestamp("2026-06-02", tz="UTC"))
    req = captured[0]
    assert req.url.path == "/v2/stocks/SPY/bars"
    assert req.url.params["timeframe"] == "1Hour"
    assert req.url.params["adjustment"] == "split"
    assert req.url.params["feed"] == "iex"
    assert req.headers["APCA-API-KEY-ID"] == "k"
    assert req.headers["APCA-API-SECRET-KEY"] == "s"


def test_credentials_default_from_env(monkeypatch):
    monkeypatch.setenv("ALPACA_KEY_ID", "k-env")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s-env")
    p = AlpacaProvider(client=object.__new__(httpx.Client))
    assert (p.key_id, p.secret) == ("k-env", "s-env")


def test_explicit_credentials_beat_env(monkeypatch):
    monkeypatch.setenv("ALPACA_KEY_ID", "k-env")
    p = AlpacaProvider(key_id="k-arg", secret="s-arg",
                       client=object.__new__(httpx.Client))
    assert (p.key_id, p.secret) == ("k-arg", "s-arg")


# -- multi-symbol fetch -------------------------------------------------------
# The scaling seam. fetch_bars costs one request per symbol per timeframe, and at
# 0.35s pacing a 100-symbol watchlist across three timeframes cannot finish
# inside a 15-minute scan bar. Note `bars` is an OBJECT keyed by symbol here,
# unlike the single-symbol endpoint's array, and `limit` is 10,000 bars TOTAL
# across symbols, so one symbol can straddle a page boundary.

MULTI_PAGE1 = {
    "bars": {
        "AAPL": [
            {"t": "2026-06-01T13:30:00Z", "o": 1.0, "h": 1.5, "l": 0.5, "c": 1.2, "v": 100},
            {"t": "2026-06-01T13:45:00Z", "o": 1.2, "h": 1.6, "l": 1.0, "c": 1.4, "v": 110},
        ],
    },
    "next_page_token": "tok-multi",
}
MULTI_PAGE2 = {
    "bars": {
        "AAPL": [{"t": "2026-06-01T14:00:00Z", "o": 1.4, "h": 1.7, "l": 1.3, "c": 1.5, "v": 120}],
        "MSFT": [{"t": "2026-06-01T13:30:00Z", "o": 2.0, "h": 2.5, "l": 1.5, "c": 2.2, "v": 200}],
    },
    "next_page_token": None,
}


def _multi_client(captured):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        page = MULTI_PAGE2 if request.url.params.get("page_token") == "tok-multi" else MULTI_PAGE1
        return httpx.Response(200, json=page)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_bars_multi_uses_one_request_per_page_not_per_symbol():
    """The whole point: N symbols cost pages, not N requests."""
    captured = []
    p = AlpacaProvider(key_id="k", secret="s", client=_multi_client(captured), sleep=lambda _: None)
    p.fetch_bars_multi(["AAPL", "MSFT"], "15m",
                       pd.Timestamp("2026-06-01", tz="UTC"),
                       pd.Timestamp("2026-06-02", tz="UTC"))
    assert len(captured) == 2
    assert captured[0].url.path == "/v2/stocks/bars"
    assert captured[0].url.params["symbols"] == "AAPL,MSFT"
    assert captured[0].url.params["timeframe"] == "15Min"
    assert captured[0].url.params["feed"] == "iex"
    assert captured[0].url.params["adjustment"] == "split"
    assert captured[0].headers["APCA-API-KEY-ID"] == "k"


def test_fetch_bars_multi_merges_a_symbol_split_across_pages():
    """Alpaca sorts by symbol then ts, so one symbol's bars can straddle a page
    boundary. Dropping the earlier page's rows would silently truncate history."""
    p = AlpacaProvider(key_id="k", secret="s", client=_multi_client([]), sleep=lambda _: None)
    result = p.fetch_bars_multi(["AAPL", "MSFT"], "15m",
                                pd.Timestamp("2026-06-01", tz="UTC"),
                                pd.Timestamp("2026-06-02", tz="UTC"))
    assert result.requested == ("AAPL", "MSFT")
    assert tuple(member.symbol for member in result.members) == result.requested
    assert len(result.members[0].rows) == 3
    assert len(result.members[1].rows) == 1
    assert result.members[0].rows[0]["t"] == "2026-06-01T13:30:00Z"
    with pytest.raises(FrozenInstanceError):
        result.members[0].present = False
    with pytest.raises(TypeError):
        result.members[0].rows[0]["c"] = 0.0


def test_batch_results_deep_copy_and_freeze_direct_construction():
    """The public result types own an immutable snapshot even when callers
    construct them from mutable lists and nested dictionaries."""
    requested = ["SPY"]
    nested = {"venues": ["IEX"], "route": {"code": "A"}}
    row = {"t": "2026-06-01T13:30:00Z", "meta": nested}
    rows = [row]
    members = [AlpacaBarMember("SPY", True, rows)]

    result = AlpacaBarBatchResult(requested, members)

    requested.append("QQQ")
    members.clear()
    rows.clear()
    nested["venues"].append("NYSE")
    nested["route"]["code"] = "B"
    assert result.requested == ("SPY",)
    assert isinstance(result.members, tuple)
    assert isinstance(result.members[0].rows, tuple)
    assert result.members[0].rows[0]["meta"]["venues"] == ("IEX",)
    assert result.members[0].rows[0]["meta"]["route"]["code"] == "A"
    with pytest.raises(TypeError):
        result.members[0].rows[0]["meta"]["route"]["code"] = "C"
    with pytest.raises(AttributeError):
        result.members[0].rows[0]["meta"]["venues"].append("ARCA")


def test_fetch_bars_multi_deep_freezes_provider_json_without_alias(monkeypatch):
    nested = {"venues": ["IEX"], "route": {"code": "A"}}
    payload = {
        "bars": {"SPY": [{
            "t": "2026-06-01T13:30:00Z", "o": 1.0, "h": 1.5,
            "l": 0.5, "c": 1.2, "v": 100, "meta": nested,
        }]},
        "next_page_token": None,
    }

    class Response:
        def json(self):
            return payload

    p = AlpacaProvider(
        key_id="k", secret="s",
        client=object.__new__(httpx.Client), sleep=lambda _: None,
    )
    monkeypatch.setattr(p, "_get", lambda *_args, **_kwargs: Response())

    result = p.fetch_bars_multi(["SPY"], "15m", START, END)
    nested["venues"].append("NYSE")
    nested["route"]["code"] = "B"

    meta = result.members[0].rows[0]["meta"]
    assert meta["venues"] == ("IEX",)
    assert meta["route"]["code"] == "A"
    with pytest.raises(TypeError):
        meta["route"]["code"] = "C"


def test_fetch_bars_multi_distinguishes_omitted_from_explicit_empty_members():
    """Removing QQQ from the response must change present, while an explicit
    empty row list stays a successful response member."""
    payloads = iter([
        {"bars": {"SPY": []}, "next_page_token": None},
        {"bars": {"SPY": [], "QQQ": []}, "next_page_token": None},
    ])

    def handler(_request):
        return httpx.Response(200, json=next(payloads))

    p = AlpacaProvider(
        key_id="k", secret="s",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )
    omitted = p.fetch_bars_multi(["SPY", "QQQ"], "15m", START, END)
    explicit = p.fetch_bars_multi(["SPY", "QQQ"], "15m", START, END)

    assert omitted.members == (
        AlpacaBarMember("SPY", True, ()),
        AlpacaBarMember("QQQ", False, ()),
    )
    assert explicit.members == (
        AlpacaBarMember("SPY", True, ()),
        AlpacaBarMember("QQQ", True, ()),
    )


def test_fetch_bars_multi_refuses_a_request_above_the_configured_ceiling():
    """Chunking belongs to data.sync, so this boundary cannot hide more than
    one logical provider request behind one call."""
    captured = []
    p = AlpacaProvider(key_id="k", secret="s", client=_multi_client(captured),
                       sleep=lambda _: None, max_symbols_per_request=100)
    with pytest.raises(ValueError, match="at most 100 symbols"):
        p.fetch_bars_multi(
            [f"S{i:03d}" for i in range(101)], "1d", START, END)
    assert captured == []


def test_fetch_bars_multi_keeps_malformed_rows_without_constructing_a_frame(
        monkeypatch):
    """Moving frame construction back into the provider would reject QQQ and
    hide the valid sibling before pair-local consumers can classify either."""
    malformed = {
        "t": "2026-06-01T13:30:00Z", "o": 9.0, "h": 8.0,
        "l": 10.0, "c": 9.0, "v": 100,
    }
    payload = {
        "bars": {"SPY": MULTI_PAGE2["bars"]["MSFT"], "QQQ": [malformed]},
        "next_page_token": None,
    }
    client = httpx.Client(transport=httpx.MockTransport(
        lambda _request: httpx.Response(200, json=payload)))
    p = AlpacaProvider(key_id="k", secret="s", client=client,
                       sleep=lambda _: None)
    monkeypatch.setattr(
        alpaca_module.pd, "DataFrame",
        lambda *args, **kwargs: pytest.fail("provider constructed a DataFrame"),
    )

    result = p.fetch_bars_multi(["SPY", "QQQ"], "15m", START, END)

    assert result.members[0].present is True
    assert result.members[0].rows[0]["c"] == 2.2
    assert result.members[1].present is True
    assert dict(result.members[1].rows[0]) == malformed


def test_fetch_bars_multi_discards_a_staged_first_page_when_continuation_times_out():
    """Returning the first page before the continuation completes would make
    one logical provider request partially visible."""
    captured = []

    def handler(request):
        captured.append(request)
        if request.url.params.get("page_token") == "continue":
            raise httpx.ReadTimeout("continuation timed out", request=request)
        return httpx.Response(200, json={
            "bars": {"SPY": MULTI_PAGE2["bars"]["MSFT"]},
            "next_page_token": "continue",
        })

    p = AlpacaProvider(
        key_id="k", secret="s",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )
    result = None
    with pytest.raises(httpx.ReadTimeout, match="continuation timed out"):
        result = p.fetch_bars_multi(["SPY", "QQQ"], "15m", START, END)

    assert result is None
    assert len(captured) == 2


def test_fetch_bars_multi_rejects_a_repeated_continuation_token_atomically():
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(200, json={
            "bars": {"SPY": [MULTI_PAGE2["bars"]["MSFT"][0]]},
            "next_page_token": "repeat",
        })

    p = AlpacaProvider(
        key_id="k", secret="s",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )
    result = None
    with pytest.raises(ValueError, match="repeated next_page_token 'repeat'"):
        result = p.fetch_bars_multi(["SPY"], "15m", START, END)

    assert result is None
    assert len(captured) == 2


def test_fetch_bars_multi_uppercases_and_dedupes_requested_symbols():
    captured = []
    p = AlpacaProvider(key_id="k", secret="s", client=_multi_client(captured), sleep=lambda _: None)
    result = p.fetch_bars_multi(["aapl", "AAPL", "msft"], "15m",
                                pd.Timestamp("2026-06-01", tz="UTC"),
                                pd.Timestamp("2026-06-02", tz="UTC"))
    assert captured[0].url.params["symbols"] == "AAPL,MSFT"
    assert result.requested == ("AAPL", "MSFT")
    assert tuple(member.symbol for member in result.members) == result.requested


def test_fetch_bars_multi_rejects_an_unsupported_timeframe():
    p = AlpacaProvider(key_id="k", secret="s", client=_multi_client([]), sleep=lambda _: None)
    with pytest.raises(ValueError, match="5m"):
        p.fetch_bars_multi(["AAPL"], "5m",
                           pd.Timestamp("2026-06-01", tz="UTC"),
                           pd.Timestamp("2026-06-02", tz="UTC"))


def test_fetch_bars_multi_with_no_symbols_makes_no_request():
    captured = []
    p = AlpacaProvider(key_id="k", secret="s", client=_multi_client(captured), sleep=lambda _: None)
    assert p.fetch_bars_multi([], "15m",
                              pd.Timestamp("2026-06-01", tz="UTC"),
                              pd.Timestamp("2026-06-02", tz="UTC")) == (
        AlpacaBarBatchResult((), ()))
    assert captured == []


def test_fetch_bars_multi_missing_credentials_fails_fast(monkeypatch):
    monkeypatch.delenv("ALPACA_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    p = AlpacaProvider(client=_multi_client([]), sleep=lambda _: None)
    with pytest.raises(RuntimeError, match="ALPACA_KEY_ID"):
        p.fetch_bars_multi(["AAPL"], "15m",
                           pd.Timestamp("2026-06-01", tz="UTC"),
                           pd.Timestamp("2026-06-02", tz="UTC"))


def test_fetch_bars_multi_has_only_the_hard_cut_result_contract():
    """A mapping return annotation or provider-owned chunk loop would restore
    the superseded API and split one call across multiple logical requests."""
    annotation = inspect.signature(AlpacaProvider.fetch_bars_multi).return_annotation
    assert annotation is AlpacaBarBatchResult
    source = inspect.getsource(AlpacaProvider.fetch_bars_multi)
    assert "range(" not in source
    assert "dict[str, pd.DataFrame]" not in source
