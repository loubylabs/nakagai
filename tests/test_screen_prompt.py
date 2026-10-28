"""The ScreenSpec prompt and public capabilities share one projection."""

import json

from nakagai.screen.facts import DISCOVERY_FACTS, FACT_LABELS
from nakagai.screen.prompt import render_screen_prompt, screen_capabilities
from nakagai.screen.sectors import SECTOR_NAMES
from nakagai.screen.spec import validate_screen_spec

EXPECTED_FACTS = (
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
)


def test_discovery_fact_names_and_label_keys_are_exact():
    assert DISCOVERY_FACTS == EXPECTED_FACTS
    assert tuple(FACT_LABELS) == EXPECTED_FACTS


def test_capability_fact_groups_partition_the_exact_vocabulary():
    grouped = [
        fact
        for names in screen_capabilities()["fact_groups"].values()
        for fact in names
    ]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(EXPECTED_FACTS)


def test_screen_capabilities_publish_the_full_market_contract():
    capabilities = screen_capabilities()
    assert capabilities["base_universe"] == "All eligible US common stocks"
    assert capabilities["fact_groups"] == {
        "fundamentals": [
            "shares_outstanding", "market_cap", "price_to_ncav",
        ],
        "market_activity": [
            "price", "change_pct", "gap_pct", "session_volume",
        ],
        "dilution": [
            "shelf_filings_365d", "offering_filings_365d",
            "atm_offerings_365d",
        ],
        "short_interest": [
            "short_pct_outstanding", "days_to_cover",
            "short_interest_change_pct",
        ],
        "insiders": ["insider_buyers_30d", "insider_buy_usd_30d"],
        "congress": ["congress_buyers_30d"],
        "funds": ["fund_holders", "fund_holders_added"],
        "earnings": ["days_to_earnings"],
    }
    assert capabilities["fact_labels"] == dict(FACT_LABELS)
    assert capabilities["technical_scope"] == "daily"
    assert capabilities["logic"] == ["all", "any", "not"]
    assert capabilities["examples"] == [
        "Market cap under $300 million with price under $10",
        "RSI under 30 with volume above twice its 20-day average.",
    ]


def test_prompt_advertises_every_discovery_fact_from_the_closed_vocabulary():
    prompt = render_screen_prompt()
    for fact in DISCOVERY_FACTS:
        assert f"- {fact}: {FACT_LABELS[fact]}" in prompt
    assert prompt.count("# Current discovery facts") == 1


def test_prompt_states_each_new_fact_window():
    flat = " ".join(render_screen_prompt().split())
    for sentence in (
        "price_to_ncav below 1 means the market values the company under its net current assets",
        "insider facts count open-market purchases (Form 4 code P) traded in the last 30 days",
        "congress_buyers_30d counts buy disclosures FILED in the last 30 days",
        "fund facts count the named 13F funds we track, from each fund's newest filed quarter",
        "days_to_earnings is whole days from today (US Eastern) to the next scheduled report; 0 is today",
    ):
        assert flat.count(sentence) == 1, sentence


def test_prompt_teaches_internet_platforms_span_two_sectors():
    flat = " ".join(render_screen_prompt().split())
    assert ('list both "Communication Services" and "Technology" '
            "and add a clarification") in flat
    assert ("fund_holders_added counts tracked funds with no position "
            "in their previous quarter") in flat


def test_prompt_teaches_the_universe():
    prompt = render_screen_prompt()
    assert "# Universe" in prompt
    for name in SECTOR_NAMES:
        assert name in prompt
    assert "four-digit SEC SIC codes" in " ".join(prompt.split())
    marker = 'Description: "oversold chip stocks"\n'
    assert marker in prompt
    reply = json.loads(prompt.split(marker)[1].split("\n\n")[0])
    spec = reply["spec"]
    assert spec["universe"] == {"industries": ["3674"]}
    assert validate_screen_spec(spec) == []


def test_prompt_refuses_float_as_not_supported():
    prompt = render_screen_prompt()
    assert 'Description: "Any low float bangers?"' in prompt
    assert '{"not_expressible": "Float is not supported."}' in prompt
    assert "never substitute shares outstanding" in prompt
    assert "float_shares" not in prompt
