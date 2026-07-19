"""Robinhood MCP historicals -> canonical bars. Agent-mediated by design:
the connector calls get_equity_historicals and saves the JSON; this module
normalizes it for `nakagai ingest`. No network code lives here."""

import pandas as pd

_COLS = {"open_price": "open", "high_price": "high", "low_price": "low", "close_price": "close"}
_BAR_COLS = ["open", "high", "low", "close", "volume"]


def _empty() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in _BAR_COLS}, index=idx)
    df["interpolated"] = pd.Series(dtype="bool")
    return df


def parse_robinhood_bars(payload: dict, symbol: str) -> pd.DataFrame:
    """Extract one symbol's bars. Keeps `interpolated` so the cache can drop them."""
    results = (payload.get("data") or {}).get("results") or []
    bars = next((r.get("bars") or [] for r in results if r.get("symbol") == symbol), [])
    if not bars:
        return _empty()
    df = pd.DataFrame(bars)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("begins_at"), utc=True), name="ts")
    df = df.rename(columns=_COLS)
    interpolated = df["interpolated"].fillna(False).astype(bool) if "interpolated" in df.columns else False
    out = df[_BAR_COLS].astype("float64").sort_index()
    out["interpolated"] = interpolated
    return out


def resample_5m_to_15m(df: pd.DataFrame) -> pd.DataFrame:
    """Left-edge 15m bars from 5m bars; a 15m bar is interpolated if ANY constituent was."""
    if df.empty:
        return df
    agg = df.resample("15min").agg({"open": "first", "high": "max", "low": "min",
                                    "close": "last", "volume": "sum", "interpolated": "max"})
    agg["interpolated"] = agg["interpolated"].astype(bool)
    return agg.dropna(subset=["open"])


def payload_interval(payload: dict, symbol: str) -> str | None:
    """The `interval` field of the matched symbol's result, or None if absent/no match."""
    results = (payload.get("data") or {}).get("results") or []
    match = next((r for r in results if r.get("symbol") == symbol), None)
    return match.get("interval") if match else None


def drift_report(new: pd.DataFrame, existing: pd.DataFrame, threshold: float = 0.001) -> pd.DataFrame:
    """Same-ts closes differing by more than `threshold` (fraction). Flagged, never averaged."""
    cols = ["cached_close", "incoming_close", "rel_diff"]
    overlap = new.index.intersection(existing.index)
    if overlap.empty:
        return pd.DataFrame(columns=cols)
    incoming, cached = new.loc[overlap, "close"], existing.loc[overlap, "close"]
    rel = (incoming - cached).abs() / cached.abs()
    bad = rel > threshold
    return pd.DataFrame({"cached_close": cached[bad], "incoming_close": incoming[bad], "rel_diff": rel[bad]})
