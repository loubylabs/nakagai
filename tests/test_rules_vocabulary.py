import json
from pathlib import Path

import pytest

from nakagai.strategies.catalog import load_entries
from nakagai.strategies.rules import spec_hash, validate_spec
from nakagai.strategies.rules.frame_eval import FrameEval
from nakagai.strategies.rules.vocabulary import Term, core_vocabulary


SPECS = (Path(__file__).resolve().parents[1]
         / "nakagai" / "strategies" / "catalog" / "specs")
CATALOG_HASHES = {
    "macd_trend": "f0ae773d5e7dc97927179deaa5e5251ceef22f3415f355cbc9d4ed44aade036b",
    "rsi_reversion": "fff25b183270e100746bb808c711d0f671b41cd376e9258dfd80dd1807a75549",
    "sma_cross": "54fae37f27bd44e495cef6b9822cc804b8a3f8a50685c1073294885432bbfc2d",
}


def _spec(node):
    return {"version": 2, "name": "injected", "timeframe": "15m",
            "long": {"all": [{"lhs": node, "op": ">", "rhs": 0}]}}


def test_one_term_owns_its_complete_contract():
    term = core_vocabulary().indicators["sma"]
    assert term.kind == "series"
    assert term.args == {"n": (2, 500)}
    assert term.defaults == {"n": 20}
    assert callable(term.fn)
    assert term.pine is None


def test_injected_term_is_used_by_validation_and_frame_eval(make_bars):
    vocab = core_vocabulary().with_terms(
        Term("double_close", "series", {}, {}, lambda s, _args: s * 2,
             doc="twice the input series")
    )
    bars = make_bars(n=30)
    spec = _spec({"ind": "double_close"})
    assert validate_spec(spec, vocab) == []
    assert FrameEval({"15m": bars}, vocabulary=vocab).series(
        {"ind": "double_close"}, "15m"
    ).equals(bars.close * 2)


def test_vocabulary_is_immutable_and_rejects_cross_namespace_duplicates():
    vocab = core_vocabulary()
    with pytest.raises(TypeError):
        vocab.indicators["new"] = vocab.indicators["sma"]
    with pytest.raises(ValueError, match="duplicate vocabulary term 'sma'"):
        vocab.with_terms(Term("sma", "primitive", {}, {}, lambda *_: None))


def test_existing_catalog_hashes_do_not_move():
    got = {name: spec_hash(entry["spec"])
           for name, entry in load_entries(SPECS).items()}
    assert json.dumps(got, sort_keys=True) == json.dumps(CATALOG_HASHES, sort_keys=True)


def test_superseded_registry_names_are_not_exported():
    from nakagai.strategies.rules import exprs, primitives, spec

    legacy = {
        spec: ("SERIES_INDICATORS", "BAR_INDICATORS", "INDICATORS", "ARG_DEFAULTS"),
        exprs: ("_SERIES_FNS", "_FRAME_FNS", "_BAR_FNS"),
        primitives: ("PRIMITIVES", "ARG_DEFAULTS", "END_ANCHORED",
                     "SESSION_SCOPED_PRIMS"),
    }
    for module, names in legacy.items():
        assert all(not hasattr(module, name) for name in names)
