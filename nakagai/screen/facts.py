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
    "price_to_ncav",
    "insider_buyers_30d",
    "insider_buy_usd_30d",
    "congress_buyers_30d",
    "fund_holders",
    "fund_holders_added",
    "days_to_earnings",
    "news_new_stories",
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
    "price_to_ncav": "price to NCAV",
    "insider_buyers_30d": "insiders buying, 30 days",
    "insider_buy_usd_30d": "insider buying, USD, 30 days",
    "congress_buyers_30d": "members of Congress buying, 30 days",
    "fund_holders": "tracked funds holding",
    "fund_holders_added": "tracked funds that opened a position",
    "days_to_earnings": "days to next earnings",
    "news_new_stories": "new news stories, last 7 days",
})
