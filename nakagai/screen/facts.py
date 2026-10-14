"""Provider-neutral current facts admitted by ScreenSpec."""

from collections.abc import Mapping
from types import MappingProxyType

DISCOVERY_FACTS: tuple[str, ...] = (
    "shares_outstanding",
    "market_cap",
    "price",
    "change_pct",
    "gap_pct",
    "session_volume",
    "shelf_filings_365d",
    "offering_filings_365d",
    "atm_offerings_365d",
    "short_pct_outstanding",
    "days_to_cover",
    "short_interest_change_pct",
)

FACT_LABELS: Mapping[str, str] = MappingProxyType({
    "shares_outstanding": "shares outstanding",
    "market_cap": "market capitalization",
    "price": "price",
    "change_pct": "change percent",
    "gap_pct": "gap percent",
    "session_volume": "session volume",
    "shelf_filings_365d": "share registrations, 365 days",
    "offering_filings_365d": "offering prospectuses, 365 days",
    "atm_offerings_365d": "at-the-market offerings, 365 days",
    "short_pct_outstanding": "short interest, % of shares outstanding",
    "days_to_cover": "days to cover",
    "short_interest_change_pct": "short interest change percent",
})
