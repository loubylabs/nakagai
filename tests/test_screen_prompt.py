"""The ScreenSpec prompt and public capabilities share one projection."""

from nakagai.screen.facts import DISCOVERY_FACTS, FACT_LABELS
from nakagai.screen.prompt import render_screen_prompt, screen_capabilities

EXPECTED_FACTS = (
    "float_shares",
    "shares_outstanding",
    "market_cap",
    "price",
    "change_pct",
    "gap_pct",
    "session_volume",
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
            "float_shares", "shares_outstanding", "market_cap",
        ],
        "market_activity": [
            "price", "change_pct", "gap_pct", "session_volume",
        ],
    }
    assert capabilities["fact_labels"] == dict(FACT_LABELS)
    assert capabilities["technical_scope"] == "daily"
    assert capabilities["logic"] == ["all", "any", "not"]
    assert capabilities["examples"] == [
        "Float under 20 million with price under $10",
        "RSI under 30 with volume above twice its 20-day average.",
    ]


def test_prompt_advertises_every_discovery_fact_from_the_closed_vocabulary():
    prompt = render_screen_prompt()
    for fact in DISCOVERY_FACTS:
        assert f"- {fact}: {FACT_LABELS[fact]}" in prompt
    assert prompt.count("# Current discovery facts") == 1


def test_prompt_compiles_low_float_instead_of_refusing_it():
    prompt = render_screen_prompt()
    assert 'Description: "Any low float bangers?"' in prompt
    assert '"fact": "float_shares"' in prompt
    assert '"rhs": 20000000' in prompt
    assert "fewer than 20 million float shares" in prompt
    assert "bangers adds no financial condition" in prompt
    assert "grammar has no fundamentals" not in prompt
