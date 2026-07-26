import json

import httpx
import pandas as pd
import pytest

from nakagai.data.alpaca import AlpacaProvider

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
    frames = p.fetch_bars_multi(["AAPL", "MSFT"], "15m",
                                pd.Timestamp("2026-06-01", tz="UTC"),
                                pd.Timestamp("2026-06-02", tz="UTC"))
    assert len(frames["AAPL"]) == 3          # 2 from page 1 + 1 from page 2
    assert len(frames["MSFT"]) == 1
    assert list(frames["AAPL"].columns) == ["open", "high", "low", "close", "volume"]
    assert frames["AAPL"].index[0] == pd.Timestamp("2026-06-01 13:30", tz="UTC")
    assert frames["AAPL"].index.is_monotonic_increasing


def test_fetch_bars_multi_returns_an_empty_frame_for_a_symbol_with_no_bars():
    """A symbol absent from the response is a normal outcome (halted, delisted,
    no IEX prints). It must read as "no bars", never as a missing key, so the
    caller's per-symbol loop cannot KeyError on a bad ticker."""
    p = AlpacaProvider(key_id="k", secret="s", client=_multi_client([]), sleep=lambda _: None)
    frames = p.fetch_bars_multi(["AAPL", "MSFT", "NOSUCH"], "15m",
                                pd.Timestamp("2026-06-01", tz="UTC"),
                                pd.Timestamp("2026-06-02", tz="UTC"))
    assert set(frames) == {"AAPL", "MSFT", "NOSUCH"}
    assert frames["NOSUCH"].empty
    assert list(frames["NOSUCH"].columns) == ["open", "high", "low", "close", "volume"]
    assert str(frames["NOSUCH"].index.tz) == "UTC"


def test_fetch_bars_multi_chunks_so_one_request_cannot_be_unboundedly_large():
    """limit is 10,000 bars TOTAL across symbols, so a wide request paginates
    heavily. Chunking keeps each request's page count predictable."""
    captured = []
    p = AlpacaProvider(key_id="k", secret="s", client=_multi_client(captured),
                       sleep=lambda _: None, max_symbols_per_request=2)
    p.fetch_bars_multi(["AAPL", "MSFT", "NVDA", "AMD", "TSLA"], "1d",
                       pd.Timestamp("2026-06-01", tz="UTC"),
                       pd.Timestamp("2026-06-02", tz="UTC"))
    batched = [r.url.params["symbols"] for r in captured]
    assert "AAPL,MSFT" in batched
    assert "NVDA,AMD" in batched
    assert "TSLA" in batched


def test_fetch_bars_multi_does_not_leak_a_page_token_between_chunks():
    """Each chunk starts its own pagination. Carrying the previous chunk's
    page_token forward would ask Alpaca to resume a different query."""
    captured = []
    p = AlpacaProvider(key_id="k", secret="s", client=_multi_client(captured),
                       sleep=lambda _: None, max_symbols_per_request=2)
    p.fetch_bars_multi(["AAPL", "MSFT", "NVDA", "AMD"], "1d",
                       pd.Timestamp("2026-06-01", tz="UTC"),
                       pd.Timestamp("2026-06-02", tz="UTC"))
    firsts = [r for r in captured if r.url.params["symbols"] == "NVDA,AMD"]
    assert firsts, "the second chunk was never requested"
    assert "page_token" not in firsts[0].url.params


def test_fetch_bars_multi_uppercases_and_dedupes_requested_symbols():
    captured = []
    p = AlpacaProvider(key_id="k", secret="s", client=_multi_client(captured), sleep=lambda _: None)
    frames = p.fetch_bars_multi(["aapl", "AAPL", "msft"], "15m",
                                pd.Timestamp("2026-06-01", tz="UTC"),
                                pd.Timestamp("2026-06-02", tz="UTC"))
    assert captured[0].url.params["symbols"] == "AAPL,MSFT"
    assert set(frames) == {"AAPL", "MSFT"}


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
                              pd.Timestamp("2026-06-02", tz="UTC")) == {}
    assert captured == []


def test_fetch_bars_multi_missing_credentials_fails_fast(monkeypatch):
    monkeypatch.delenv("ALPACA_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    p = AlpacaProvider(client=_multi_client([]), sleep=lambda _: None)
    with pytest.raises(RuntimeError, match="ALPACA_KEY_ID"):
        p.fetch_bars_multi(["AAPL"], "15m",
                           pd.Timestamp("2026-06-01", tz="UTC"),
                           pd.Timestamp("2026-06-02", tz="UTC"))
