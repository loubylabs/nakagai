"""Provider-neutral current facts admitted by ScreenSpec."""

from collections.abc import Mapping
from types import MappingProxyType

DISCOVERY_FACTS: tuple[str, ...] = (
    "float_shares",
    "shares_outstanding",
    "market_cap",
    "price",
    "change_pct",
    "gap_pct",
    "session_volume",
)

FACT_LABELS: Mapping[str, str] = MappingProxyType({
    "float_shares": "float shares",
    "shares_outstanding": "shares outstanding",
    "market_cap": "market capitalization",
    "price": "price",
    "change_pct": "change percent",
    "gap_pct": "gap percent",
    "session_volume": "session volume",
})
