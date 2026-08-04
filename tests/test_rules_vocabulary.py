import importlib
import json
from functools import cache
from pathlib import Path

import pandas as pd
import pytest

from nakagai.data.schema import DEFAULT_TIMEFRAMES
from nakagai.engine.context import PreloadedBars, build_context
from nakagai.icir import window_icir
from nakagai.nlbuilder.prompt import render_system_prompt
from nakagai.screen.compiler import compile_screen
from nakagai.screen.runner import run_screen
from nakagai.screen.spec import describe_screen, validate_screen_spec
from nakagai.strategies.catalog import load_entries
from nakagai.strategies.rules import spec_hash, validate_spec
from nakagai.strategies.rules.frame_eval import FrameEval
from nakagai.strategies.rules.spec import validate_condition_group
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
    # The Pine slot is part of the contract, on both sides of the vocabulary: a
    # primitive carries the helpers its lowering leans on, an indicator that
    # needs none carries an empty tuple.
    assert callable(term.pine.emit)
    assert term.pine.helpers == ()
    gap = core_vocabulary().primitives["gap_pct"]
    assert callable(gap.pine.emit)
    assert gap.pine.helpers == ("nk_gap_pct",)


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


def _double_close() -> Term:
    return Term("double_close", "series", {}, {}, lambda s, _args: s * 2,
                doc="twice the input series")


def _raises_type_error(call) -> bool:
    """True when binding the call's arguments fails, which is what a
    keyword-only `vocabulary` is for. Any other exception means the arguments
    bound, so the Vocabulary landed in some earlier parameter."""
    try:
        call()
    except TypeError:
        return True
    except Exception:
        return False
    return False


def test_a_positional_vocabulary_cannot_bind_to_an_earlier_parameter():
    """The hazard these keyword-only markers close.

    validate_condition_group(group, "conditions", vocab) reads naturally and
    is the shape the plan documents, but `tf` sits third. Bound there, nothing
    raises: `tf in SESSION_ALIGNED` and `tf if tf in TIMEFRAMES else "1h"` both
    just read False for a non-string, so the call returns a clean-looking error
    list validated against the CORE vocabulary while the caller believes it
    injected one. Every seam below has the same shape, an optional positional
    sitting in front of `vocabulary`, so every one of them must refuse it.
    """
    vocab = core_vocabulary().with_terms(_double_close())
    group = {"all": [{"lhs": {"ind": "double_close"}, "op": ">", "rhs": 0}]}
    now = pd.Timestamp("2026-07-17 20:05", tz="UTC")
    traps = {
        "validate_condition_group":
            lambda: validate_condition_group(group, "conditions", vocab),
        "FrameEval":
            lambda: FrameEval({}, DEFAULT_TIMEFRAMES, vocab),
        "PreloadedBars":
            lambda: PreloadedBars(None, "SPY", DEFAULT_TIMEFRAMES, vocab),
        "build_context":
            lambda: build_context(None, "SPY", now, DEFAULT_TIMEFRAMES, vocab),
        "window_icir":
            lambda: window_icir({}, None, "SPY", None, DEFAULT_TIMEFRAMES, vocab),
        "render_system_prompt":
            lambda: render_system_prompt(None, vocab),
        "validate_screen_spec":
            lambda: validate_screen_spec({}, vocab),
        "describe_screen":
            lambda: describe_screen({}, vocab),
        "run_screen":
            lambda: run_screen({}, [], None, None, None, 60, vocab),
        "compile_screen":
            lambda: compile_screen("x", None, "model", 2, vocab),
    }
    bound_anyway = [name for name, call in traps.items()
                    if not _raises_type_error(call)]
    assert bound_anyway == []


def test_the_keyword_form_reaches_the_validator_and_the_default_does_not():
    """The other half: keyword injection works, and its absence is visible.

    Without the injected vocabulary the same group is refused by name rather
    than accepted silently, which is what makes a missed injection findable.
    """
    vocab = core_vocabulary().with_terms(_double_close())
    group = {"all": [{"lhs": {"ind": "double_close"}, "op": ">", "rhs": 0}]}
    assert validate_condition_group(group, "conditions", vocabulary=vocab) == []
    errs = validate_condition_group(group, "conditions")
    assert errs and "unknown indicator 'double_close'" in errs[0]


def test_term_refuses_an_unknown_kind():
    # kind is a Literal annotation, which is documentation and not a check.
    # Unrefused, "seires" routes to the indicator namespace and evaluates as a
    # series indicator, with the typo never surfacing.
    with pytest.raises(ValueError, match="unknown kind 'seires'"):
        Term("typo", "seires", {}, {}, lambda *_: None)


def test_term_refuses_a_non_callable_fn():
    with pytest.raises(TypeError, match="term 'broken' needs a callable fn"):
        Term("broken", "series", {}, {}, None)


def test_term_refuses_a_default_its_arg_schema_does_not_declare():
    with pytest.raises(ValueError, match=r"defaults \['m'\] are not in its arg"):
        Term("drifted", "series", {"n": (2, 500)}, {"n": 20, "m": 5},
             lambda *_: None)


def test_every_core_term_satisfies_the_term_checks():
    # core_vocabulary() builds every Term through __post_init__, so this asserts
    # the checks above are satisfied by the shipped declarations rather than
    # only by the hand-written ones in this file.
    vocab = core_vocabulary()
    assert len(vocab.indicators) == 22 and len(vocab.primitives) == 15
    for term in vocab.all_terms():
        assert set(term.defaults) <= set(term.args)


def test_vocabulary_is_immutable_and_rejects_cross_namespace_duplicates():
    vocab = core_vocabulary()
    with pytest.raises(TypeError):
        vocab.indicators["new"] = vocab.indicators["sma"]
    with pytest.raises(ValueError, match="duplicate vocabulary term 'sma'"):
        vocab.with_terms(Term("sma", "primitive", {}, {}, lambda *_: None))


def test_a_vocabulary_cannot_be_a_cache_key():
    # Its docstring says so, and a docstring alone rots. Frozen dataclass, so
    # __hash__ exists and the failure is a TypeError from inside the CALL, not
    # a decorator error at import: a @cache keyed on one looks fine until the
    # day it runs. Cache on the factory instead.
    cached = cache(lambda vocab: len(vocab.indicators))
    with pytest.raises(TypeError, match="unhashable"):
        cached(core_vocabulary())
    assert cache(lambda factory: len(factory().indicators))(core_vocabulary) == 22


def test_existing_catalog_hashes_do_not_move():
    got = {name: spec_hash(entry["spec"])
           for name, entry in load_entries(SPECS, core_vocabulary).items()}
    assert json.dumps(got, sort_keys=True) == json.dumps(CATALOG_HASHES, sort_keys=True)


def test_superseded_registry_names_are_not_exported():
    from nakagai.strategies.rules import primitives, spec

    legacy = {
        spec: ("SERIES_INDICATORS", "BAR_INDICATORS", "INDICATORS", "ARG_DEFAULTS"),
        primitives: ("PRIMITIVES", "ARG_DEFAULTS", "END_ANCHORED",
                     "SESSION_SCOPED_PRIMS", "DRIVING_FRAME_INTRADAY_PRIMS"),
    }
    # One assert per name, not `all(...)` over the lot: collapsed, a regression
    # reports a bare `assert False` and names no symbol, so whoever hits it has
    # to bisect the tuple by hand to learn which registry came back.
    for module, names in legacy.items():
        for name in names:
            assert not hasattr(module, name), \
                f"{module.__name__}.{name} is back; it was retired by the vocabulary"


def test_the_retired_exprs_module_is_gone():
    # exprs.py held one private helper with one importer once its documented
    # reason to exist (the _SERIES_FNS/_FRAME_FNS/_BAR_FNS tables) moved into
    # the vocabulary. The helper lives in frame_eval.py now.
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("nakagai.strategies.rules.exprs")
