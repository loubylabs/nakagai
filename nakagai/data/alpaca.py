"""Deep intraday history via Alpaca Market Data v2 (free tier, IEX feed)."""

import os
import time

import httpx
import pandas as pd

from nakagai.data.base import DataProvider
from nakagai.data.schema import validate_bars

_TF = {"15m": "15Min", "1h": "1Hour", "1d": "1Day"}
_BASE = "https://data.alpaca.markets"
_MAX_429_RETRIES = 5


class AlpacaProvider(DataProvider):
    def __init__(self, key_id: str | None = None, secret: str | None = None, client: httpx.Client | None = None,
                 min_interval: float = 0.35, sleep=time.sleep):
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
        if not rows:
            idx = pd.DatetimeIndex([], tz="UTC", name="ts")
            return pd.DataFrame({c: pd.Series(dtype="float64") for c in ["open", "high", "low", "close", "volume"]}, index=idx)
        df = pd.DataFrame(rows).rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("t"), utc=True), name="ts")
        return validate_bars(df)
