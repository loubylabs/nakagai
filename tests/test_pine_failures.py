"""What the Pine compiler refuses, and how precisely it says so.

Every refusal here has the same shape: generation stops and names the exact
RuleSpec path and term, rather than emitting Pine that looks plausible and
trades differently from the engine.
"""

import pytest

from nakagai.strategies.rules import (
    PineCompileError, PineExpr, PineLowering, lower_pine,
)
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


def test_a_stateful_primitive_is_named_as_the_gap_it_is():
    # The primitives are Task 3's, and until then a spec using one is refused by
    # name rather than compiled into something that only resembles it.
    spec = {"version": 2, "name": "probe", "timeframe": "15m",
            "long": {"all": [{"lhs": {"prim": "gap_pct"}, "op": ">", "rhs": 1}]}}
    with pytest.raises(PineCompileError) as exc:
        lower_pine(spec)
    assert exc.value.code == "pine_unsupported"
    assert exc.value.term == "gap_pct"
    assert exc.value.path == "long.all[0].lhs"


def test_a_lowering_that_names_an_unknown_helper_is_refused():
    vocab = core_vocabulary().with_terms(
        Term("borrowed", "series", {}, {}, lambda series, _args: series,
             pine=PineLowering(_emit_close, helpers=("nk_moon",))))
    with pytest.raises(PineCompileError) as exc:
        lower_pine(_spec_using("borrowed"), vocab)
    assert exc.value.code == "pine_unknown_helper"
    assert exc.value.term == "borrowed"
    assert "nk_moon" in str(exc.value)


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
    # validate_spec only ever checks the values a SPEC supplies, so a term whose
    # own default sits outside its declared bounds reaches the compiler intact.
    vocab = core_vocabulary().with_terms(
        Term("drifted", "series", {"n": (2, 500)}, {"n": 1000},
             lambda series, _args: series,
             pine=PineLowering(lambda ctx, call: PineExpr(
                 ctx.calc(call, f"ta.sma(close, {ctx.arg(call, 'n')})")))))
    with pytest.raises(PineCompileError) as exc:
        lower_pine(_spec_using("drifted"), vocab)
    assert exc.value.code == "pine_bad_input"
    assert exc.value.path == "long.all[0].lhs.drifted.n"


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
    assert exc.value.code == "pine_invalid_spec"
    assert "long/short" in str(exc.value)


def test_a_timeframe_inside_another_timeframe_is_refused():
    # request.security cannot be nested, and the engine's own evaluator answers
    # this shape happily, so it has to be refused loudly rather than flattened.
    spec = {"version": 2, "name": "probe", "timeframe": "15m",
            "long": {"all": [{"lhs": {"ind": "sma", "tf": "1h",
                                      "of": {"src": "close", "tf": "4h"}},
                              "op": ">", "rhs": 0}]}}
    with pytest.raises(PineCompileError) as exc:
        lower_pine(spec)
    assert exc.value.code == "pine_nested_timeframe"
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
    assert exc.value.code == "pine_nested_timeframe"
    assert exc.value.path == "long.all[0].lhs.of"
    assert exc.value.term == "ema"


def test_a_term_whose_pine_slot_is_not_a_lowering_is_refused_where_it_is_built():
    with pytest.raises(TypeError, match="term 'raw' needs a PineLowering"):
        Term("raw", "series", {}, {}, lambda series, _args: series,
             pine=_emit_close)
