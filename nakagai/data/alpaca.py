"""Deep intraday history via Alpaca Market Data v2 (free tier, IEX feed)."""

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import httpx
import pandas as pd

from nakagai.data.base import DataProvider
from nakagai.data.schema import empty_bars, validate_bars

_TF = {"15m": "15Min", "1h": "1Hour", "1d": "1Day"}
_BASE = "https://data.alpaca.markets"
_MAX_429_RETRIES = 5


@dataclass(frozen=True, slots=True)
class AlpacaBarMember:
    """One requested symbol's lossless membership and staged raw rows."""

    symbol: str
    present: bool
    rows: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rows",
            tuple(_freeze_json(row) for row in self.rows),
        )


@dataclass(frozen=True, slots=True)
class AlpacaBarBatchResult:
    """The atomic result of one bounded, fully paginated provider request."""

    requested: tuple[str, ...]
    members: tuple[AlpacaBarMember, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested", tuple(self.requested))
        object.__setattr__(self, "members", tuple(self.members))


def _freeze_json(value: object) -> object:
    """Copy and recursively freeze one JSON-shaped provider value."""
    if isinstance(value, Mapping):
        return MappingProxyType({
            key: _freeze_json(member) for key, member in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(member) for member in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_json(member) for member in value)
    if isinstance(value, bytearray):
        return bytes(value)
    return value


def _frame_from_rows(rows: list[Mapping[str, object]]) -> pd.DataFrame:
    """Alpaca's raw bar dicts to the canonical schema. One implementation,
    shared by the single-symbol and multi-symbol fetches."""
    if not rows:
        return empty_bars()
    df = pd.DataFrame(rows).rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("t"), utc=True), name="ts")
    return validate_bars(df)


class AlpacaProvider(DataProvider):
    def __init__(self, key_id: str | None = None, secret: str | None = None, client: httpx.Client | None = None,
                 min_interval: float = 0.35, sleep=time.sleep,
                 max_symbols_per_request: int = 100):
        # Credentials come from explicit args or the environment; this module
        # reads no platform settings so it can ship in the standalone core.
        self.key_id = key_id or os.environ.get("ALPACA_KEY_ID", "")
        self.secret = secret or os.environ.get("ALPACA_SECRET_KEY", "")
        self.client = client or httpx.Client(base_url=_BASE, timeout=30)
        # 200 requests/min per account on the free tier; 0.35s spacing keeps a
        # single process comfortably under it even across a full backfill.
        self.min_interval = min_interval
        self._sleep = sleep
        self._last_request: float | None = None
        # `limit` is 10,000 bars TOTAL across symbols, so a wide request
        # paginates rather than failing. 100 keeps each request to a handful of
        # pages at 15m over a 40-day window (roughly 740 bars per symbol), which
        # is what the scan loop asks for. Alpaca documents no symbol cap.
        self.max_symbols_per_request = max(1, int(max_symbols_per_request))

    def _get(self, url: str, params: dict, headers: dict) -> httpx.Response:
        """Paced GET that sleeps through 429s instead of failing the symbol.
        A skipped symbol means the grid silently runs on stale or empty bars,
        so waiting out the limiter is always the better trade."""
        for attempt in range(_MAX_429_RETRIES + 1):
            if self._last_request is not None:
                wait = self.min_interval - (time.monotonic() - self._last_request)
                if wait > 0:
                    self._sleep(wait)
            r = self.client.get(url, params=params, headers=headers)
            self._last_request = time.monotonic()
            if r.status_code != 429:
                r.raise_for_status()
                return r
            if attempt == _MAX_429_RETRIES:
                r.raise_for_status()
            try:
                delay = max(float(r.headers.get("Retry-After")), 1.0)
            except (TypeError, ValueError):
                delay = min(2.0 ** attempt, 30.0)
            self._sleep(delay)
        raise AssertionError("unreachable")

    def fetch_bars(self, symbol, timeframe, start, end):
        if timeframe not in _TF:
            raise ValueError(f"unsupported timeframe {timeframe}")
        if not self.key_id or not self.secret:
            raise RuntimeError(
                "Alpaca credentials not set: export ALPACA_KEY_ID and ALPACA_SECRET_KEY "
                "(e.g. `set -a; source .env.local; set +a`) before running this command")
        headers = {"APCA-API-KEY-ID": self.key_id, "APCA-API-SECRET-KEY": self.secret}
        params = {
            "timeframe": _TF[timeframe],
            "start": start.isoformat(),
            "end": end.isoformat(),
            "adjustment": "split",
            "feed": "iex",
            "limit": 10_000,
        }
        rows = []
        while True:
            r = self._get(f"{_BASE}/v2/stocks/{symbol}/bars", params, headers)
            payload = r.json()
            rows.extend(payload.get("bars") or [])
            token = payload.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        return _frame_from_rows(rows)

    def fetch_bars_multi(self, symbols: list[str], timeframe: str,
                         start: pd.Timestamp,
                         end: pd.Timestamp) -> AlpacaBarBatchResult:
        """Return one bounded logical request after all pages have succeeded.

        This is the scaling seam. fetch_bars costs one request per symbol per
        timeframe, and at 0.35s pacing a 100-symbol watchlist across three
        timeframes cannot finish inside a 15-minute scan bar. The multi-symbol
        endpoint collapses one bounded symbol chunk to pages. Callers own
        chunking so each completed logical request can commit independently.

        Raw member rows remain unclassified here. This preserves the difference
        between an omitted requested member and an explicitly empty member, and
        lets downstream pair-local validation retain valid siblings.
        """
        if timeframe not in _TF:
            raise ValueError(f"unsupported timeframe {timeframe}")
        wanted = list(dict.fromkeys(s.upper() for s in symbols if s))
        if not wanted:
            return AlpacaBarBatchResult((), ())
        if len(wanted) > self.max_symbols_per_request:
            raise ValueError(
                "fetch_bars_multi accepts at most "
                f"{self.max_symbols_per_request} symbols per logical request")
        if not self.key_id or not self.secret:
            raise RuntimeError(
                "Alpaca credentials not set: export ALPACA_KEY_ID and ALPACA_SECRET_KEY "
                "(e.g. `set -a; source .env.local; set +a`) before running this command")
        headers = {"APCA-API-KEY-ID": self.key_id, "APCA-API-SECRET-KEY": self.secret}
        rows: dict[str, list[Mapping[str, object]]] = {
            symbol: [] for symbol in wanted}
        present = dict.fromkeys(wanted, False)
        seen_tokens: set[str] = set()
        params = {
            "symbols": ",".join(wanted),
            "timeframe": _TF[timeframe],
            "start": start.isoformat(),
            "end": end.isoformat(),
            "adjustment": "split",
            "feed": "iex",
            "limit": 10_000,
        }
        while True:
            response = self._get(f"{_BASE}/v2/stocks/bars", params, headers)
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise ValueError("Alpaca multi-bar response is not an object")
            page = payload.get("bars")
            if not isinstance(page, Mapping):
                raise ValueError("Alpaca multi-bar response bars is not an object")
            for raw_symbol, raw_rows in page.items():
                if not isinstance(raw_symbol, str):
                    raise ValueError("Alpaca multi-bar member symbol is not text")
                symbol = raw_symbol.upper()
                if symbol not in rows:
                    continue
                if not isinstance(raw_rows, list):
                    raise ValueError(
                        f"Alpaca multi-bar member {symbol} rows is not a list")
                if any(not isinstance(row, Mapping) for row in raw_rows):
                    raise ValueError(
                        f"Alpaca multi-bar member {symbol} contains a non-object row")
                present[symbol] = True
                rows[symbol].extend(raw_rows)
            token = payload.get("next_page_token")
            if token is None or token == "":
                break
            if not isinstance(token, str):
                raise ValueError("Alpaca multi-bar next_page_token is not text")
            if token in seen_tokens:
                raise ValueError(
                    f"Alpaca multi-bar repeated next_page_token {token!r}")
            seen_tokens.add(token)
            params["page_token"] = token
        return AlpacaBarBatchResult(
            requested=tuple(wanted),
            members=tuple(
                AlpacaBarMember(
                    symbol=symbol,
                    present=present[symbol],
                    rows=rows[symbol],
                )
                for symbol in wanted
            ),
        )
