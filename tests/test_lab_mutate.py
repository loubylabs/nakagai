"""Trial generation: spec mutation, validation, and stable hashing."""

from nakagai.lab.mutate import spec_hash


def test_spec_hash_ignores_key_order():
    a = {"version": 2, "name": "x", "timeframe": "1h"}
    b = {"timeframe": "1h", "name": "x", "version": 2}
    assert spec_hash(a) == spec_hash(b)


def test_spec_hash_changes_with_content():
    a = {"version": 2, "name": "x", "timeframe": "1h"}
    b = {"version": 2, "name": "x", "timeframe": "15m"}
    assert spec_hash(a) != spec_hash(b)


def test_spec_hash_is_short_hex():
    h = spec_hash({"name": "x"})
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


from nakagai.lab.mutate import Site, mutable_sites

# The shipped rsi_reversion play, which is the shape every literal mutation
# has to handle: periods nested under indicators, bare numeric comparands,
# and the risk block's own numbers.
BASE = {
    "version": 2,
    "name": "rsi_reversion",
    "timeframe": "1h",
    "long": {"all": [
        {"lhs": {"ind": "rsi", "n": 14}, "op": "crosses_above", "rhs": 30},
        {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 200}},
    ]},
    "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
             "target": {"kind": "rr", "rr": 1.5}},
}


def test_mutable_sites_finds_every_number_worth_moving():
    kinds = sorted(s.kind for s in mutable_sites(BASE))
    assert kinds == ["mult", "period", "period", "period", "rr", "threshold"]


def test_mutable_sites_reaches_into_a_dict_rhs():
    # The sma(200) trend filter lives inside an rhs object, not beside it.
    paths = {s.path for s in mutable_sites(BASE)}
    assert ("long", "all", 1, "rhs", "n") in paths


def test_mutable_sites_finds_the_bare_threshold():
    paths = {s.path for s in mutable_sites(BASE)}
    assert ("long", "all", 0, "rhs") in paths


def test_mutable_sites_never_touches_version_or_name():
    paths = {s.path for s in mutable_sites(BASE)}
    assert ("version",) not in paths
    assert ("name",) not in paths


def test_mutable_sites_on_a_spec_with_no_numbers_is_empty():
    assert mutable_sites({"version": 2, "name": "x", "timeframe": "1h"}) == []


def test_mutable_sites_excludes_booleans_under_site_keys():
    # bool is an int subclass in Python, so a flag placed under a site key
    # must still be excluded; if it were not, this would report both paths
    # as mutable numeric literals.
    spec = {"lhs": {"ind": "rsi", "n": True}, "op": ">", "rhs": False}
    paths = {s.path for s in mutable_sites(spec)}
    assert ("lhs", "n") not in paths
    assert ("rhs",) not in paths


import pytest

from nakagai.lab.mutate import Trial, literal_trials
from nakagai.strategies.rules.spec import validate_spec


def test_literal_trials_are_all_valid_specs():
    trials = literal_trials(BASE, n=12, seed=7)
    assert len(trials) == 12
    for t in trials:
        assert validate_spec(t.spec) == [], t.spec


def test_literal_trials_are_distinct():
    trials = literal_trials(BASE, n=12, seed=7)
    assert len({t.id for t in trials}) == 12


def test_literal_trials_are_deterministic_for_a_seed():
    a = literal_trials(BASE, n=8, seed=3)
    b = literal_trials(BASE, n=8, seed=3)
    assert [t.spec_hash for t in a] == [t.spec_hash for t in b]


def test_literal_trials_differ_across_seeds():
    a = literal_trials(BASE, n=8, seed=3)
    b = literal_trials(BASE, n=8, seed=4)
    assert [t.spec_hash for t in a] != [t.spec_hash for t in b]


def test_literal_trials_do_not_mutate_the_base():
    before = spec_hash(BASE)
    literal_trials(BASE, n=6, seed=1)
    assert spec_hash(BASE) == before


def test_literal_trials_carry_the_rules_strategy_name():
    assert all(t.strategy == "rules" for t in literal_trials(BASE, n=3, seed=1))


def test_literal_trials_keep_periods_at_two_or_more():
    # A period of 0 or 1 is meaningless to every indicator in the grammar.
    for t in literal_trials(BASE, n=25, seed=11):
        for site in mutable_sites(t.spec):
            if site.kind == "period":
                node = t.spec
                for key in site.path[:-1]:
                    node = node[key]
                assert node[site.path[-1]] >= 2


def test_literal_trials_refuses_a_spec_with_nothing_to_move():
    with pytest.raises(ValueError, match="no mutable"):
        literal_trials({"version": 2, "name": "x", "timeframe": "1h"},
                       n=3, seed=1)


def test_literal_trials_raises_when_the_mutant_space_is_exhausted():
    # One mutable site: a period pinned at PERIOD_MIN. Its reachable,
    # post-clamp, post-no-op-guard values are only {3, 4}, a two-element
    # space, so a third distinct mutant can never be produced. The rhs is an
    # expression (not a bare number) and risk is left out entirely (so
    # validate_risk falls back to its own in-range defaults), which keeps
    # this the *only* mutable literal in the spec.
    tiny = {
        "version": 2,
        "name": "tiny",
        "timeframe": "1h",
        "long": {"all": [
            {"lhs": {"ind": "rsi", "n": 2}, "op": ">", "rhs": {"src": "close"}},
        ]},
    }
    assert validate_spec(tiny) == []
    assert len(mutable_sites(tiny)) == 1
    with pytest.raises(ValueError, match="could only build"):
        literal_trials(tiny, n=3, seed=1)


from nakagai.lab.mutate import composite_trials
from nakagai.strategies.composite.spec import validate_composite_spec
from nakagai.strategies.catalog import load_catalog
from pathlib import Path

SPECS = Path(__file__).resolve().parents[1] / "nakagai/strategies/catalog/specs"


def _members():
    return load_catalog(SPECS)


def test_composite_trials_are_all_valid():
    members = _members()
    trials = composite_trials(sorted(members), n=10, seed=5, members=members)
    assert len(trials) == 10
    for t in trials:
        assert validate_composite_spec(t.spec, members, allow_refs=False) == [], t.spec


def test_composite_trials_carry_the_composite_strategy_name():
    members = _members()
    trials = composite_trials(sorted(members), n=4, seed=5, members=members)
    assert all(t.strategy == "composite" for t in trials)


def test_composite_trials_only_reference_catalog_members():
    members = _members()
    for t in composite_trials(sorted(members), n=10, seed=2, members=members):
        for block in t.spec["blocks"].values():
            assert block["strategy"] in members
            assert block["params"] == {}


def test_composite_trials_are_deterministic_for_a_seed():
    members = _members()
    a = composite_trials(sorted(members), n=6, seed=9, members=members)
    b = composite_trials(sorted(members), n=6, seed=9, members=members)
    assert [t.spec_hash for t in a] == [t.spec_hash for t in b]


def test_composite_trials_are_distinct():
    members = _members()
    trials = composite_trials(sorted(members), n=8, seed=4, members=members)
    assert len({t.id for t in trials}) == 8


def test_composite_trials_need_two_members():
    with pytest.raises(ValueError, match="at least 2"):
        composite_trials(["sma_cross"], n=3, seed=1, members=_members())


from nakagai.lab.mutate import _composite_key

# The two specs below describe the identical strategy, "macd_trend OR
# rsi_reversion", but with the plays sitting at swapped block ids. A
# syntactic dedupe key (spec_hash of the spec minus its name) treats these
# as two different trials, since the "blocks" dict content differs; the
# semantic key must treat them as one.
_COMMON = {
    "version": 1,
    "window_bars": 4,
    "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
             "target": {"kind": "rr", "rr": 1.5}},
}


def test_composite_key_collapses_a_block_id_swap():
    a = {**_COMMON, "name": "x",
         "blocks": {"a": {"strategy": "macd_trend", "params": {}},
                    "b": {"strategy": "rsi_reversion", "params": {}}},
         "long": {"any": ["a", "b"]}}
    b = {**_COMMON, "name": "y",
         "blocks": {"a": {"strategy": "rsi_reversion", "params": {}},
                    "b": {"strategy": "macd_trend", "params": {}}},
         "long": {"any": ["a", "b"]}}
    assert spec_hash(a) != spec_hash(b)          # syntactically different specs
    assert _composite_key(a) == _composite_key(b)  # but the same strategy


def test_composite_key_keeps_the_confluence_position_distinguished():
    # {"all": [x, {"any": [y, z]}]} means "x AND (y OR z)". Swapping which
    # play is x (the distinguished, non-voted leg) versus which play sits in
    # the "any" group changes the strategy, even though the same three plays
    # and the same tree shape are used both times. Over-canonicalizing (e.g.
    # sorting across the nesting boundary) would wrongly collapse these.
    a = {**_COMMON, "name": "x",
         "blocks": {"a": {"strategy": "macd_trend", "params": {}},
                    "b": {"strategy": "rsi_reversion", "params": {}},
                    "c": {"strategy": "sma_cross", "params": {}}},
         "long": {"all": ["a", {"any": ["b", "c"]}]}}
    b = {**_COMMON, "name": "y",
         "blocks": {"a": {"strategy": "rsi_reversion", "params": {}},
                    "b": {"strategy": "macd_trend", "params": {}},
                    "c": {"strategy": "sma_cross", "params": {}}},
         "long": {"all": ["a", {"any": ["b", "c"]}]}}
    assert _composite_key(a) != _composite_key(b)


def test_composite_key_is_order_independent_within_the_nested_any():
    # Within the confluence shape's own "any" group, member order is still
    # symmetric: swapping y and z there (leaving the distinguished leg alone)
    # must still collide.
    a = {**_COMMON, "name": "x",
         "blocks": {"a": {"strategy": "macd_trend", "params": {}},
                    "b": {"strategy": "rsi_reversion", "params": {}},
                    "c": {"strategy": "sma_cross", "params": {}}},
         "long": {"all": ["a", {"any": ["b", "c"]}]}}
    b = {**_COMMON, "name": "y",
         "blocks": {"a": {"strategy": "macd_trend", "params": {}},
                    "b": {"strategy": "sma_cross", "params": {}},
                    "c": {"strategy": "rsi_reversion", "params": {}}},
         "long": {"all": ["a", {"any": ["b", "c"]}]}}
    assert _composite_key(a) == _composite_key(b)


def test_composite_trials_are_semantically_distinct():
    # The regression the review caught: t.id (derived from the full,
    # block-id-sensitive spec_hash) was always unique even when two trials
    # described the same strategy with swapped block ids. This checks the
    # dedupe key the module itself uses for identity, so a reintroduced
    # syntactic-only dedupe would fail it even though test_composite_trials
    # _are_distinct (which only checks t.id) would still pass.
    members = _members()
    trials = composite_trials(sorted(members), n=10, seed=5, members=members)
    keys = {_composite_key(t.spec) for t in trials}
    assert len(keys) == 10
