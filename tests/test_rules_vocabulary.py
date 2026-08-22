import importlib
import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nakagai.data.schema import DEFAULT_TIMEFRAMES
from nakagai.engine.context import build_context
from nakagai.nlbuilder.prompt import render_system_prompt
from nakagai.screen.compiler import compile_screen
from nakagai.screen.runner import run_screen
from nakagai.screen.spec import describe_screen, validate_screen_spec
from nakagai.strategies.catalog import load_entries
from nakagai.strategies.rules import spec_hash, validate_spec
from nakagai.strategies.rules.canon import canonical_expr, canonical_spec
from nakagai.strategies.rules.frame_eval import FrameEval
from nakagai.strategies.rules.spec import validate_condition_group
from nakagai.strategies.rules.vocabulary import (
    CONDITION_ARG, Term, core_vocabulary, is_condition_rule, is_range_rule)


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
        "build_context":
            lambda: build_context(None, "SPY", now, DEFAULT_TIMEFRAMES, vocab),
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


def test_condition_arg_is_recognized_and_is_not_a_choice_rule():
    assert is_condition_rule(CONDITION_ARG)
    assert is_condition_rule(core_vocabulary().primitives["bars_since"].args["cond"])
    assert not is_condition_rule(("safe", "fast"))
    assert not is_condition_rule((2.0, 500.0))


def test_range_rule_accepts_real_numpy_bounds_but_not_choices_or_booleans():
    assert is_range_rule((np.int64(5), np.float64(120.0)))
    assert is_range_rule((2, 500))
    assert not is_range_rule(("low", "high"))
    assert not is_range_rule((True, False))
    assert not is_range_rule((1,))


def test_a_condition_typed_arg_may_not_declare_a_default():
    with pytest.raises(ValueError, match="condition-typed arg"):
        Term("bad_default", "primitive", {"cond": CONDITION_ARG},
             {"cond": {"lhs": 1, "op": ">", "rhs": 2}}, lambda *a: None)


def test_a_condition_typed_arg_is_refused_outside_primitive_kind():
    with pytest.raises(ValueError, match="condition-typed"):
        Term("bad_kind", "series", {"cond": CONDITION_ARG}, {}, lambda *a: None)
    # primitive kind is fine, and must stay fine
    Term("ok_kind", "primitive", {"cond": CONDITION_ARG}, {}, lambda *a: None)


def test_a_term_may_declare_at_most_one_condition_typed_arg():
    """The third refusal in the constructor, and Pine's rather than N3-D13's.

    `TermCall.source` is one slot, so the Pine walk lowers one condition and
    hands it to an emit taking one operand. A term declaring two validated
    clean and lowered to `nk_bars_since(close > open)` with the second
    condition absent from the chart entirely, so the engine computed
    `up AND down` and the chart computed `up`.

    Refused where the term is built rather than where it is charted: a
    vocabulary that cannot be charted is wrong when it is constructed, not
    when someone asks for a chart. The message names both args, because the
    author of a two-condition term needs to know which one Pine could not
    take, and lifting this is a real piece of work rather than an oversight.
    """
    with pytest.raises(ValueError, match="at most one"):
        Term("two_conds", "primitive",
             {"up": CONDITION_ARG, "down": CONDITION_ARG}, {},
             lambda *a: None)
    # one is fine under any name, and must stay fine
    Term("one_cond", "primitive", {"when": CONDITION_ARG}, {}, lambda *a: None)


def test_a_condition_typed_arg_gets_the_evaluator_injected_by_type(
        make_bars, count_where_vocab):
    """N3-D9. The injection is keyed on the arg's declared type, so a term
    registered outside core evaluates with no new evaluator code. Keyed on the
    name "bars_since" instead, this validates clean and then raises."""
    vocab = count_where_vocab
    bars = make_bars(n=30)  # close is above open on every bar
    node = {"prim": "count_where",
            "when": {"lhs": {"src": "close"}, "op": ">", "rhs": {"src": "open"}}}
    assert validate_spec(_spec(node), vocab) == []
    out = FrameEval({"15m": bars}, vocabulary=vocab).series(node, "15m")
    assert out.iloc[-1] == 5.0


def test_an_end_anchored_primitive_gets_the_evaluator_injected_too(make_bars):
    """The injection has to sit ABOVE frame_eval's end_anchored branch, and
    the callback it injects has to answer for the frame it is handed.

    N3-D13 refuses a condition-typed arg outside kind 'primitive' and nowhere
    else, so an end-anchored primitive that declares one is constructible and
    validates clean. It then dispatches through end_anchored_series, which
    returns before the injection block, so the term is called without the
    evaluator it declared it needs and raises at evaluation. The term needs
    the callback regardless of which dispatch it takes.

    `count_end` deliberately does NOT re-slice what the callback returns. An
    earlier draft wrote `.loc[bars.index]` inside the term, which clipped what
    the callback had failed to clip, so the assertion below passed against a
    callback that answered over the whole frame: the guard compensated for the
    defect it existed to catch. A term written the natural way is the only one
    that can tell the two apart, and nothing obliges a term registered outside
    core to write the defensive version.
    """

    def count_end(ctx, bars, when, eval_fn=None):
        # One float off the tail of the frame handed in, which is what
        # end_anchored means: the count of qualifying bars in the prefix.
        if eval_fn is None:
            raise RuntimeError("missing eval_fn")
        return float(eval_fn(when, bars).sum())

    vocab = core_vocabulary().with_terms(
        Term("count_end", "primitive", {"when": CONDITION_ARG}, {}, count_end,
             end_anchored=True))
    bars = make_bars(n=30)  # close is above open on every bar
    node = {"prim": "count_end",
            "when": {"lhs": {"src": "close"}, "op": ">", "rhs": {"src": "open"}}}
    assert validate_spec(_spec(node), vocab) == []
    out = FrameEval({"15m": bars}, vocabulary=vocab).series(node, "15m")
    # row i is the count over bars[:i+1], so the last row counts every bar and
    # the first counts one: a broadcast tail value would read 30.0 on both.
    assert out.iloc[-1] == 30.0 and out.iloc[0] == 1.0


def test_canon_rekeys_a_second_condition_taking_terms_condition_arg(
        count_where_vocab):
    """Case 4, the subtle one. Keyed on the literal key "cond", a condition
    arg named anything else falls into the generic args merge, where _num
    passes a dict straight through: nothing INSIDE the condition is
    canonicalized, so two nodes whose conditions differ only in an omitted
    default or an int-versus-float literal canonicalize apart and hash apart.
    """
    vocab = count_where_vocab
    implicit = {"prim": "count_where",
                "when": {"lhs": {"ind": "sma", "n": 20}, "op": ">", "rhs": 1}}
    explicit = {"prim": "count_where", "n": 5,
                "when": {"lhs": {"ind": "sma", "n": 20, "of": {"src": "close"}},
                         "op": ">", "rhs": 1.0}}
    assert canonical_expr(implicit, vocab) == canonical_expr(explicit, vocab)
    # and the interior really was canonicalized, rather than both sides being
    # passed through untouched and comparing equal by accident
    assert canonical_expr(implicit, vocab)["when"]["lhs"] == {
        "ind": "sma", "n": 20.0, "of": {"src": "close"}}


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


def _section(prompt: str, header: str) -> str:
    """One "#" section of a rendered prompt, its header line included.

    Scoped rather than searched whole, because a bare `x in prompt` over a
    prompt this large passes for the wrong reason routinely: a term name
    appears in an example, a word appears in unrelated prose, and the
    assertion says nothing about the paragraph it claims to be about.
    """
    return prompt.split(header, 1)[1].split("\n#", 1)[0]


def test_prompt_renders_a_condition_typed_arg_readably():
    prompt = render_system_prompt()
    # The rendered LIST line, whole: prose elsewhere mentioning the shape must
    # not be able to satisfy this.
    assert "- bars_since(cond={lhs,op,rhs})" in prompt.splitlines()
    assert "cond=condition" not in prompt


def test_prompt_describes_a_second_condition_taking_term_with_no_new_prose(
        count_where_vocab):
    """Why _bounds reads the arg's declared type rather than a name: a term
    registered outside core is described correctly with nobody editing the
    prompt's text, which is what node 02's generated terms need."""
    prompt = render_system_prompt(None, vocabulary=count_where_vocab)
    assert "- count_where(when={lhs,op,rhs}, n=(1, 50))" in prompt.splitlines()


def test_the_primitives_header_states_the_cross_prohibition_generically():
    # The header block only, up to the first rendered term line.
    header = _section(render_system_prompt(), "# Primitives").split("\n- ", 1)[0]
    assert "bars_since" not in header  # not hand-written for one term
    assert "cross" in header  # and the prohibition it carried is not lost


def test_prompt_documents_not_in_the_grammar_paragraph():
    grammar = _section(render_system_prompt(), "# Grammar")
    assert '{"not":' in grammar
    # and the worked example shows its argument as a GROUP, the only shape
    # _check_group accepts.
    assert '{"not": {"all":' in grammar


def test_canon_group_handles_not_without_iterating_its_dict_as_a_list():
    """canon.py's _canon_group: `for i in g[key]` over a dict (not's value)
    iterates the dict's KEY STRINGS rather than raising a shape error, and
    _canon_cond then indexes a str."""
    spec = {"version": 2, "timeframe": "1h",
            "long": {"not": {"any": [
                {"lhs": {"src": "close"}, "op": ">", "rhs": 1}]}},
            "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
                     "target": {"kind": "rr", "rr": 2.0}}}
    canon = canonical_spec(spec)
    assert canon["long"] == {"not": {"any": [
        {"lhs": {"src": "close"}, "op": ">", "rhs": 1.0}]}}
