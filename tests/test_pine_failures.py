"""What the Pine compiler refuses, and how precisely it says so.

Every refusal here has the same shape: generation stops and names the exact
RuleSpec path and term, rather than emitting Pine that looks plausible and
trades differently from the engine.
"""

import ast
from pathlib import Path

import pytest

import nakagai
from nakagai.strategies.rules import (
    PineCompileError, PineExpr, PineHelper, PineLowering, lower_pine,
    validate_spec,
)
from nakagai.strategies.rules.pine.lowerings import HELPERS
from nakagai.strategies.rules.vocabulary import Term, core_vocabulary


def _spec_using(name, **node):
    return {"version": 2, "name": "probe", "timeframe": "15m",
            "long": {"all": [{"lhs": {"ind": name, **node}, "op": ">",
                              "rhs": 0}]}}


def _vocabulary_with_unsupported_indicator(name):
    return core_vocabulary().with_terms(
        Term(name, "series", {}, {}, lambda series, _args: series,
             doc="an injected term with no Pine form"))


def _emit_close(ctx, call):
    return PineExpr(ctx.calc(call, "close"))


def test_missing_lowering_names_term_and_path():
    vocab = _vocabulary_with_unsupported_indicator("moon_phase")
    with pytest.raises(PineCompileError) as exc:
        lower_pine(_spec_using("moon_phase"), vocab)
    assert exc.value.code == "pine_unsupported"
    assert exc.value.path == "long.all[0].lhs"
    assert exc.value.term == "moon_phase"


def test_a_lowering_that_names_an_unknown_helper_is_refused():
    vocab = core_vocabulary().with_terms(
        Term("borrowed", "series", {}, {}, lambda series, _args: series,
             pine=PineLowering(_emit_close, helpers=("nk_moon",))))
    with pytest.raises(PineCompileError) as exc:
        lower_pine(_spec_using("borrowed"), vocab)
    assert exc.value.code == "pine_unknown_helper"
    assert exc.value.term == "borrowed"
    assert "nk_moon" in str(exc.value)


def _borrowing(name: str, helper_id: str):
    return core_vocabulary().with_terms(
        Term(name, "series", {}, {}, lambda series, _args: series,
             pine=PineLowering(_emit_close, helpers=(helper_id,))))


def test_a_helper_dependency_nothing_answers_stops_generation(monkeypatch):
    # A helper's dependencies decide what is defined above it. One naming a
    # helper the registry does not hold would emit a call to an undefined Pine
    # function, which TradingView reports from the chart rather than here.
    monkeypatch.setitem(HELPERS, "nk_borrowed",
                        PineHelper("nk_borrowed", "nk_borrowed() => nk_moon()",
                                   ("nk_moon",)))
    with pytest.raises(PineCompileError) as exc:
        lower_pine(_spec_using("borrowed"), _borrowing("borrowed", "nk_borrowed"))
    assert exc.value.code == "pine_generation_failed"
    assert "nk_moon" in str(exc.value)


def test_helpers_that_call_each_other_in_a_cycle_stop_generation(monkeypatch):
    # Pine defines a function above its callers, so a cycle has no order at
    # all. Emitting one of the two first would leave the other undefined.
    for one, other in (("nk_ping", "nk_pong"), ("nk_pong", "nk_ping")):
        monkeypatch.setitem(HELPERS, one,
                            PineHelper(one, f"{one}() => {other}()", (other,)))
    with pytest.raises(PineCompileError) as exc:
        lower_pine(_spec_using("looped"), _borrowing("looped", "nk_ping"))
    assert exc.value.code == "pine_generation_failed"
    assert "nk_ping" in str(exc.value) and "nk_pong" in str(exc.value)


def test_a_choice_argument_whose_default_drifted_is_refused():
    # The validator checks omitted choice defaults before the compiler can
    # receive an unknown literal.
    def emit(ctx, call):
        return PineExpr(ctx.calc(call, f"nk_side_{ctx.choice(call, 'side')}"))

    vocab = core_vocabulary().with_terms(
        Term("tilted", "series", {"side": ("long", "short")},
             {"side": "sideways"}, lambda series, _args: series,
             pine=PineLowering(emit)))
    with pytest.raises(PineCompileError) as exc:
        lower_pine(_spec_using("tilted"), vocab)
    assert exc.value.code == "invalid_spec"
    assert "tilted.side" in str(exc.value)
    assert "sideways" in str(exc.value)


def test_two_arguments_that_sanitize_alike_collide_rather_than_merge():
    def emit(ctx, call):
        return PineExpr(ctx.calc(
            call, f"ta.sma(close, {ctx.arg(call, 'n.x')} + {ctx.arg(call, 'n_x')})"))

    vocab = core_vocabulary().with_terms(
        Term("twin", "series", {"n.x": (1, 10), "n_x": (1, 10)},
             {"n.x": 2, "n_x": 3}, lambda series, _args: series,
             pine=PineLowering(emit)))
    with pytest.raises(PineCompileError) as exc:
        lower_pine(_spec_using("twin"), vocab)
    assert exc.value.code == "pine_identifier_collision"
    assert "nk_long_all_0_lhs_twin_n_x" in str(exc.value)


def test_a_default_outside_its_own_bounds_is_refused():
    # The validator checks omitted defaults against the term's own bounds before
    # the compiler can receive an unsavable call.
    vocab = core_vocabulary().with_terms(
        Term("drifted", "series", {"n": (2, 500)}, {"n": 1000},
             lambda series, _args: series,
             pine=PineLowering(lambda ctx, call: PineExpr(
                 ctx.calc(call, f"ta.sma(close, {ctx.arg(call, 'n')})")))))
    with pytest.raises(PineCompileError) as exc:
        lower_pine(_spec_using("drifted"), vocab)
    assert exc.value.code == "invalid_spec"
    assert "drifted.n" in str(exc.value)


def test_an_argument_the_schema_never_declared_is_refused():
    vocab = core_vocabulary().with_terms(
        Term("mismatched", "series", {"n": (2, 500)}, {"n": 20},
             lambda series, _args: series,
             pine=PineLowering(lambda ctx, call: PineExpr(
                 ctx.calc(call, f"ta.sma(close, {ctx.arg(call, 'length')})")))))
    with pytest.raises(PineCompileError) as exc:
        lower_pine(_spec_using("mismatched"), vocab)
    assert exc.value.code == "pine_bad_input"
    assert "length" in str(exc.value)


def test_a_field_mapping_that_misses_a_declared_choice_is_refused():
    # The grammar admits every field a term declares, so a Pine form covering
    # only some of them is a hole in the compiler, not in the spec. It is
    # caught inside ctx.fields rather than after emit returns, because every
    # real lowering ends in `ctx.fields(...)[call.field]`: the missing key
    # would otherwise be a bare KeyError naming neither term nor path.
    def emit(ctx, call):
        return PineExpr(ctx.fields(call, {"a": ctx.local(call, "a", "close")})["a"])

    vocab = core_vocabulary().with_terms(
        Term("partial", "series", {"field": ("a", "b")}, {"field": "a"},
             lambda series, _args: series, pine=PineLowering(emit)))
    with pytest.raises(PineCompileError) as exc:
        lower_pine(_spec_using("partial", field="b"), vocab)
    assert exc.value.code == "pine_bad_input"
    assert exc.value.term == "partial"
    assert exc.value.path == "long.all[0].lhs"
    assert "'b'" in str(exc.value)
    # The refusal is the mapping's, not the node's: it stands even for the one
    # field this lowering does answer.
    with pytest.raises(PineCompileError) as exc:
        lower_pine(_spec_using("partial", field="a"), vocab)
    assert exc.value.code == "pine_bad_input"


def test_a_lowering_that_names_no_field_at_all_is_refused_too():
    # The other half of the same hole: a term declaring fields whose Pine form
    # never calls ctx.fields answers one unnamed value, which no field selects.
    vocab = core_vocabulary().with_terms(
        Term("unnamed", "series", {"field": ("a", "b")}, {"field": "a"},
             lambda series, _args: series, pine=PineLowering(_emit_close)))
    with pytest.raises(PineCompileError) as exc:
        lower_pine(_spec_using("unnamed", field="a"), vocab)
    assert exc.value.code == "pine_bad_input"
    assert exc.value.term == "unnamed"
    assert exc.value.path == "long.all[0].lhs"


def test_a_spec_that_does_not_validate_is_refused_before_any_pine_is_built():
    with pytest.raises(PineCompileError) as exc:
        lower_pine({"version": 2, "name": "probe", "timeframe": "15m"})
    assert exc.value.code == "invalid_spec"
    assert "long/short" in str(exc.value)


def test_an_invalid_spec_carries_the_validator_strings_it_was_refused_by():
    # A caller rendering this as a 422 owes the user the validator's own
    # wording, per error. Recovering it by joining and re-splitting the
    # message would break on any error containing the separator, and
    # re-running validate_spec would be a second pass that could disagree.
    spec = {"version": 2, "name": "probe", "timeframe": "15m",
            "long": {"all": [{"lhs": {"ind": "sma", "n": 9000}, "op": ">",
                              "rhs": 0}]}}
    with pytest.raises(PineCompileError) as exc:
        lower_pine(spec)
    assert exc.value.code == "invalid_spec"
    assert exc.value.errors == tuple(validate_spec(spec, core_vocabulary()))
    assert exc.value.errors
    for err in exc.value.errors:
        assert err in str(exc.value)


def test_every_other_refusal_carries_no_error_list():
    # `errors` is the invalid_spec refusal's alone. A caller reading it on any
    # other code would render an empty list as if the validator had spoken.
    vocab = _vocabulary_with_unsupported_indicator("moon_phase")
    with pytest.raises(PineCompileError) as exc:
        lower_pine(_spec_using("moon_phase"), vocab)
    assert exc.value.errors == ()


def test_a_timeframe_inside_another_timeframe_is_refused():
    # request.security cannot be nested, and the engine's own evaluator answers
    # this shape happily, so it has to be refused loudly rather than flattened.
    spec = {"version": 2, "name": "probe", "timeframe": "15m",
            "long": {"all": [{"lhs": {"ind": "sma", "tf": "1h",
                                      "of": {"src": "close", "tf": "4h"}},
                              "op": ">", "rhs": 0}]}}
    with pytest.raises(PineCompileError) as exc:
        lower_pine(spec)
    # pine_unsupported, because this reaches the compiler from a spec the
    # validator accepts: it is the language's limit, not a broken compiler,
    # and a caller maps it to 422 alongside every other thing Pine cannot say.
    assert exc.value.code == "pine_unsupported"
    assert "does not nest" in str(exc.value)
    assert exc.value.path == "long.all[0].lhs.of"
    # A source has no term to name; the next test covers the one that has.
    assert exc.value.term == ""


def test_a_nested_timeframe_names_the_term_it_refuses():
    spec = {"version": 2, "name": "probe", "timeframe": "15m",
            "long": {"all": [{"lhs": {"ind": "sma", "tf": "1h",
                                      "of": {"ind": "ema", "n": 5,
                                             "tf": "4h"}},
                              "op": ">", "rhs": 0}]}}
    with pytest.raises(PineCompileError) as exc:
        lower_pine(spec)
    assert exc.value.code == "pine_unsupported"
    assert "does not nest" in str(exc.value)
    assert exc.value.path == "long.all[0].lhs.of"
    assert exc.value.term == "ema"


def _codes_the_compiler_can_raise() -> set[str]:
    """Every code literal nakagai/ hands PineCompileError, off its AST.

    Source rather than behaviour on purpose: a code only some unreached branch
    can raise is still part of the contract, and a test that could only see the
    ones this suite happens to trigger would miss exactly the new one. The
    WHOLE package, walked recursively rather than globbed one directory deep,
    for the same reason: today only compiler.py and lower.py refuse, and a
    first refusal added in render.py, or anywhere else under nakagai/, has to
    land here too. The call is matched on either a bare `PineCompileError(...)`
    or a qualified `model.PineCompileError(...)`, since a qualified call is the
    same refusal and an id-only match would let it through unseen.
    """
    out, sites = set(), 0
    root = Path(nakagai.__file__).parent
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call)
                    and (getattr(node.func, "id", None) == "PineCompileError"
                         or getattr(node.func, "attr", None)
                         == "PineCompileError")):
                sites += 1
                assert isinstance(node.args[0], ast.Constant), \
                    f"{path.name}:{node.lineno} raises a code that is not a literal"
                out.add(node.args[0].value)
    assert sites, "found no refusal at all, so this proves nothing"
    return out


def test_the_whole_refusal_vocabulary_is_exactly_these_six_codes():
    # The platform pins this core by SHA and maps the codes to HTTP status, so
    # the set is a published contract rather than an implementation detail. A
    # seventh code added without a caller taught about it lands in that
    # caller's 500 bucket, whatever it actually means, and the caller cannot
    # fix a name from its side. Only invalid_spec and pine_unsupported are
    # things a user did; the other four are the compiler admitting a hole.
    assert _codes_the_compiler_can_raise() == {
        "invalid_spec", "pine_unsupported", "pine_bad_input",
        "pine_identifier_collision", "pine_unknown_helper",
        "pine_generation_failed"}


def test_a_term_whose_pine_slot_is_not_a_lowering_is_refused_where_it_is_built():
    with pytest.raises(TypeError, match="term 'raw' needs a PineLowering"):
        Term("raw", "series", {}, {}, lambda series, _args: series,
             pine=_emit_close)


_NOT_RISK = {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
             "target": {"kind": "rr", "rr": 2.0}}


def test_not_is_refused_with_a_named_message_top_level():
    spec = {"version": 2, "name": "x", "timeframe": "15m",
            "long": {"not": {"all": [{"lhs": {"src": "close"}, "op": ">",
                                      "rhs": 1}]}},
            "risk": _NOT_RISK}
    assert validate_spec(spec) == [], "the spec must be VALID before Pine sees it"
    with pytest.raises(PineCompileError) as exc:
        lower_pine(spec)
    assert exc.value.code == "pine_unsupported"
    assert "`not`" in str(exc.value)


def test_not_is_refused_with_a_named_message_nested():
    spec = {"version": 2, "name": "y", "timeframe": "15m",
            "long": {"all": [
                {"not": {"any": [{"lhs": {"src": "close"}, "op": ">",
                                  "rhs": 1}]}},
                {"lhs": {"src": "close"}, "op": ">", "rhs": 0}]},
            "risk": _NOT_RISK}
    assert validate_spec(spec) == [], "the spec must be VALID before Pine sees it"
    with pytest.raises(PineCompileError) as exc:
        lower_pine(spec)
    assert exc.value.code == "pine_unsupported"
    assert "`not`" in str(exc.value)


def test_not_is_refused_with_a_named_message_on_the_chart_composition_path():
    """The SECOND refusal site. _decide splits a tree that reads a foreign
    timeframe and composes it on the chart through _tree, which unpacks a
    group key of its own; a play whose own frame is not the driving frame
    never reaches _group at all. Without a refusal there too, `not` reaches
    the leaf lowerer, which reads `lhs` off a key string and dies with a
    TypeError instead of a named refusal.
    """
    spec = {"version": 2, "name": "z", "timeframe": "1h",
            "long": {"not": {"all": [{"lhs": {"src": "close", "tf": "1d"},
                                      "op": ">", "rhs": 1}]}},
            "risk": _NOT_RISK}
    assert validate_spec(spec) == [], "the spec must be VALID before Pine sees it"
    with pytest.raises(PineCompileError) as exc:
        lower_pine(spec)
    assert exc.value.code == "pine_unsupported"
    assert "`not`" in str(exc.value)
