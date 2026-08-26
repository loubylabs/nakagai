"""The vocabulary: one Term per name the RuleSpec grammar admits.

A Term owns a name's WHOLE contract in one place: its argument schema and
bounds, its defaults, the executable function, the doc line the NL prompts
render, the three causal flags described below, its window reducer contract,
and the slot a Pine lowering attaches to. Nothing about a name lives anywhere
else, so declaring one is one statement and injecting one is one `with_terms`
call.

A Vocabulary holds indicator, primitive, and window namespaces because the
grammar spells them under different keys. Term names are unique across the two
term namespaces, so no term is ever reachable under two readings. Window names
are unique within their separate scope namespace.

The three causal flags, and what each one refuses:

- `end_anchored`: the term reads ONE level off the end of the frame it is
  handed, not a causal series. A whole-frame pass may not broadcast such a
  value across history (that would be lookahead), so it is evaluated row by row
  over the replay's bounded span instead, and the grammar refuses the shapes
  that would read it outside that span.
- `session_scoped`: the term reads its own frame's session or calendar
  structure rather than plain OHLCV structure, so it is refused a foreign `tf`
  outright.
- `driving_frame_intraday`: the term cannot be answered at all on a
  session-aligned driving frame, where one bar IS the whole session.

The last two are two different rules over two deliberately different sets, not
one set read twice. The per-term comments in core_vocabulary() say why each
term carries the flags it carries; read them before moving a flag.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache
from numbers import Real
from types import MappingProxyType
from typing import Literal, TypeAlias

from nakagai.strategies import indicators as ind
from nakagai.strategies.rules import primitives as prim
from nakagai.strategies.rules.pine import lowerings as pine
from nakagai.strategies.rules.pine.model import PineLowering
from nakagai.strategies.rules.windows import WindowSpec

# The arg rule marking a condition-typed argument: its value is a condition
# node, {"lhs": <expr>, "op": ..., "rhs": <expr>}, never a bare value. A string
# rather than a tuple, so is_choice_rule reads False for it and a generic reader
# must branch on this BEFORE the choice and range branches rather than adding a
# third elif after them: spec.py's non-choice branch does `lo, hi = rule`, which
# raises ValueError on any bare string.
CONDITION_ARG: Literal["condition"] = "condition"
ArgRule: TypeAlias = tuple[Real, Real] | tuple[str, ...] | Literal["condition"]
KINDS = ("series", "frame", "bar", "primitive")
WINDOW_REDUCERS = ("max", "min", "first", "last")
WindowReducer: TypeAlias = Literal["max", "min", "first", "last"]


def is_condition_rule(rule: ArgRule) -> bool:
    """True for an arg whose value is a condition node, never a bare value."""
    return rule == CONDITION_ARG


def is_json_number(value) -> bool:
    """True for exact built-in JSON numeric values, excluding booleans."""
    return type(value) in (int, float)


def is_choice_rule(rule: ArgRule) -> bool:
    """True for an arg that names one of a fixed set (field, direction, kind).

    The two arg shapes are told apart by their contents in two places, the
    validator and the Pine input collector, so the predicate is one function
    rather than a copy each: a rule read as a range in one and as a choice in
    the other would validate a spec the compiler then refuses.
    """
    return isinstance(rule, tuple) and all(isinstance(r, str) for r in rule)


def is_range_rule(rule: ArgRule) -> bool:
    """True for a finite, ordered two-item range, including NumPy reals."""
    if not (isinstance(rule, tuple) and len(rule) == 2
            and all(isinstance(value, Real) and not isinstance(value, bool)
                    for value in rule)):
        return False
    try:
        low, high = (float(value) for value in rule)
    except (OverflowError, TypeError, ValueError):
        return False
    return math.isfinite(low) and math.isfinite(high) and low <= high


@dataclass(frozen=True)
class Term:
    name: str
    kind: Literal["series", "frame", "bar", "primitive"]
    args: Mapping[str, ArgRule]
    defaults: Mapping[str, object]
    fn: Callable
    doc: str = ""
    end_anchored: bool = False
    session_scoped: bool = False
    driving_frame_intraday: bool = False
    pine: PineLowering | None = None
    window_reduce: WindowReducer | None = None
    window_required: bool = False

    def __post_init__(self) -> None:
        # The Literal annotation is documentation, not a runtime check, and
        # every reader routes on `kind` by asking whether it equals one known
        # value. An injected Term(kind="seires") would therefore land in the
        # indicator namespace and be evaluated as a series indicator, quietly,
        # with the typo never surfacing. Same for a non-callable fn (a
        # TypeError from deep inside the evaluator, one frame short of naming
        # the term) and for a default whose arg the schema does not declare
        # (it merges into the call and _check_args never sees it, because
        # _check_args only walks the keys the SPEC supplied). Refuse all three
        # where the term is built, so the message names the term.
        if self.kind not in KINDS:
            raise ValueError(f"term {self.name!r} has unknown kind "
                             f"{self.kind!r} (valid: {KINDS})")
        if not callable(self.fn):
            raise TypeError(f"term {self.name!r} needs a callable fn, got "
                            f"{type(self.fn).__name__}")
        # Same reasoning one slot along: a bare emit function passed as `pine=`
        # is callable and would sail through every isinstance-free reader until
        # the compiler asked it for `.emit` and reported an AttributeError from
        # inside the walk, naming no term.
        if self.pine is not None and not isinstance(self.pine, PineLowering):
            raise TypeError(f"term {self.name!r} needs a PineLowering in its "
                            f"pine slot, got {type(self.pine).__name__}")
        if (self.window_reduce is not None
                and self.window_reduce not in WINDOW_REDUCERS):
            raise ValueError(f"term {self.name!r} has unknown window reducer "
                             f"{self.window_reduce!r} "
                             f"(valid: {WINDOW_REDUCERS})")
        if self.window_required and self.window_reduce is None:
            raise ValueError(f"term {self.name!r} requires a window reducer")
        if self.window_reduce is not None and self.kind != "series":
            raise ValueError(f"only a series term may declare a window "
                             f"reducer; term {self.name!r} has kind "
                             f"{self.kind!r}")
        args = MappingProxyType(dict(self.args))
        defaults = MappingProxyType(dict(self.defaults))
        undeclared = set(defaults) - set(args)
        if undeclared:
            raise ValueError(f"term {self.name!r} defaults "
                             f"{sorted(undeclared)} are not in its arg schema "
                             f"{sorted(args)}")
        # N3-D13, both halves. A condition-typed arg is refused a default,
        # because _check_args only ever walks the keys a SPEC supplies, so a
        # default condition would be validated by nothing (a cross op inside
        # one is forbidden, and no guard would ever see it). It is also refused
        # outside `primitive` kind, because only the primitive branch of
        # frame_eval hands a term the frame and **kwargs the injected evaluator
        # rides in on; a `series` term is called fn(series, args) and could
        # never evaluate a condition even with a callback. Both refuse at
        # construction, so a mis-declared term fails when the vocabulary is
        # built rather than when a spec happens to reach it.
        condition_args = sorted(a for a, r in args.items()
                                if is_condition_rule(r))
        defaulted = [a for a in condition_args if a in defaults]
        if defaulted:
            raise ValueError(f"term {self.name!r} declares a default for "
                             f"condition-typed arg(s) {defaulted}, which is "
                             "not allowed: a default condition would be "
                             "validated by nothing")
        if condition_args and self.kind != "primitive":
            raise ValueError(f"term {self.name!r} declares condition-typed "
                             f"arg(s) {condition_args} on kind {self.kind!r}; "
                             "a condition-typed arg is only allowed on kind "
                             "'primitive'")
        # At most one, and the constraint belongs to Pine rather than to the
        # grammar. `TermCall.source` is a single slot: the walk lowers one
        # condition into it and hands it to an emit function that takes one
        # operand. With two, pine/lower.py selected one out of a set and
        # dropped the other from `args` as well, so the chart computed
        # different logic from the engine and said nothing. Measured: a term
        # declaring `up` and `down` validated clean and lowered to
        # `nk_bars_since(close > open)` with `down` absent entirely.
        #
        # The other five walkers are already plural-safe (canon re-keys every
        # condition arg; the evaluator injects one callback they all share), so
        # lifting this means giving TermCall a source per condition and
        # teaching every emit which one it wants. Refused here rather than in
        # the Pine walk because a vocabulary that cannot be charted is wrong
        # when it is built, not when someone asks for a chart.
        if len(condition_args) > 1:
            raise ValueError(f"term {self.name!r} declares condition-typed "
                             f"args {condition_args}; a term may declare at "
                             "most one, because Pine lowers a condition into a "
                             "single call.source slot and would silently drop "
                             "the rest")
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "defaults", defaults)


@dataclass(frozen=True)
class Vocabulary:
    """Immutable Term namespaces and window rows for one grammar.

    UNHASHABLE, despite being a frozen dataclass: both fields are mappings, so
    the generated __hash__ raises TypeError on the dicts it hashes. That makes
    @cache or @lru_cache keyed on a Vocabulary fail at CALL time rather than at
    import, which is the trap. Cache on the FACTORY instead (VocabularyFactory
    is a plain callable and hashes by identity), which is what the catalog
    loaders take and why they take it.
    """

    indicators: Mapping[str, Term]
    primitives: Mapping[str, Term]
    windows: Mapping[str, WindowSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        indicators = dict(self.indicators)
        primitives = dict(self.primitives)
        windows = dict(self.windows)
        overlap = indicators.keys() & primitives.keys()
        if overlap:
            name = sorted(overlap)[0]
            raise ValueError(f"duplicate vocabulary term {name!r}")
        object.__setattr__(self, "indicators", MappingProxyType(indicators))
        object.__setattr__(self, "primitives", MappingProxyType(primitives))
        object.__setattr__(self, "windows", MappingProxyType(windows))

    def with_terms(self, *terms: Term) -> "Vocabulary":
        indicators, primitives = dict(self.indicators), dict(self.primitives)
        for term in terms:
            if term.name in indicators or term.name in primitives:
                raise ValueError(f"duplicate vocabulary term {term.name!r}")
            target = primitives if term.kind == "primitive" else indicators
            target[term.name] = term
        return Vocabulary(indicators, primitives, self.windows)

    def with_windows(self, *rows: WindowSpec) -> "Vocabulary":
        windows = dict(self.windows)
        for row in rows:
            if row.name in windows:
                raise ValueError(f"duplicate vocabulary window {row.name!r}")
            windows[row.name] = row
        return Vocabulary(self.indicators, self.primitives, windows)

    def resolve(self, kind: str, name: str) -> Term:
        terms = self.primitives if kind == "primitive" else self.indicators
        return terms[name]

    def all_terms(self) -> tuple[Term, ...]:
        return tuple(self.indicators.values()) + tuple(self.primitives.values())


VocabularyFactory: TypeAlias = Callable[[], Vocabulary]


def _series(name, args, defaults, fn, pine, *, window_reduce=None,
            window_required=False) -> Term:
    return Term(name, "series", args, defaults, fn, pine=pine,
                window_reduce=window_reduce, window_required=window_required)


def _frame(name, args, defaults, fn, pine) -> Term:
    return Term(name, "frame", args, defaults, fn, pine=pine)


def _bar(name, args, defaults, fn, pine) -> Term:
    return Term(name, "bar", args, defaults, fn, pine=pine)


def _primitive(name, args, defaults, fn, pine, *, end_anchored=False,
               session_scoped=False, driving_frame_intraday=False) -> Term:
    return Term(name, "primitive", args, defaults, fn, doc=fn.__doc__ or "",
                end_anchored=end_anchored, session_scoped=session_scoped,
                driving_frame_intraday=driving_frame_intraday, pine=pine)


def _window_required(*_args):
    raise RuntimeError("window-required term cannot run without its window")


@cache
def core_vocabulary() -> Vocabulary:
    indicators = (
        _series("sma", {"n": (2, 500)}, {"n": 20},
                lambda s, a: ind.sma(s, a["n"]),
                PineLowering(pine.emit_series_call("ta.sma", "n"))),
        _series("ema", {"n": (2, 500)}, {"n": 20},
                lambda s, a: ind.ema(s, a["n"]),
                PineLowering(pine.emit_series_call("ta.ema", "n"))),
        _series("rsi", {"n": (2, 100)}, {"n": 14},
                lambda s, a: ind.rsi(s, a["n"]),
                PineLowering(pine.emit_series_call("ta.rsi", "n"))),
        _series("roc", {"n": (1, 500)}, {"n": 20},
                lambda s, a: ind.roc(s, a["n"]),
                PineLowering(pine.emit_series_call("ta.roc", "n"))),
        _series("zscore", {"n": (2, 500)}, {"n": 20},
                lambda s, a: ind.zscore(s, a["n"]),
                PineLowering(pine.emit_zscore, helpers=(pine.DIV,))),
        _series("highest", {"n": (2, 500)}, {"n": 20},
                lambda s, a: ind.highest(s, a["n"]),
                PineLowering(pine.emit_series_call("ta.highest", "n")),
                window_reduce="max"),
        _series("lowest", {"n": (2, 500)}, {"n": 20},
                lambda s, a: ind.lowest(s, a["n"]),
                PineLowering(pine.emit_series_call("ta.lowest", "n")),
                window_reduce="min"),
        _series("first", {}, {}, _window_required, None,
                window_reduce="first", window_required=True),
        _series("last", {}, {}, _window_required, None,
                window_reduce="last", window_required=True),
        _series("stdev", {"n": (2, 500)}, {"n": 20},
                lambda s, a: ind.stdev(s, a["n"]),
                PineLowering(pine.emit_series_call("ta.stdev", "n"))),
        _frame("macd", {"fast": (2, 100), "slow": (3, 200),
                         "signal": (2, 100),
                         "field": ("macd", "signal", "hist")},
               {"fast": 12, "slow": 26, "signal": 9, "field": "macd"},
               lambda s, a: ind.macd(s, a["fast"], a["slow"], a["signal"]),
               PineLowering(pine.emit_macd)),
        _frame("bb", {"n": (2, 200), "k": (0.5, 5.0),
                       "field": ("upper", "mid", "lower")},
               {"n": 20, "k": 2.0, "field": "mid"},
               lambda s, a: ind.bollinger(s, a["n"], a["k"]),
               PineLowering(pine.emit_bb)),
        _bar("atr", {"n": (2, 100)}, {"n": 14},
             lambda b, a: ind.atr(b, a["n"]),
             PineLowering(pine.emit_bar_call("ta.atr", "n"))),
        _bar("donchian", {"n": (2, 300),
                           "field": ("upper", "lower", "mid")},
             {"n": 20, "field": "upper"},
             lambda b, a: ind.donchian(b, a["n"]),
             PineLowering(pine.emit_donchian)),
        _bar("supertrend", {"n": (2, 100), "mult": (0.5, 10.0),
                             "field": ("line", "direction")},
             {"n": 10, "mult": 3.0, "field": "line"},
             lambda b, a: ind.supertrend(b, a["n"], a["mult"]),
             PineLowering(pine.emit_supertrend)),
        _bar("vwap", {}, {}, lambda b, _a: ind.session_vwap(b),
             PineLowering(pine.emit_vwap)),
        _bar("stoch", {"n": (2, 100), "d": (1, 50),
                        "field": ("k", "d")},
             {"n": 14, "d": 3, "field": "k"},
             lambda b, a: ind.stoch(b, a["n"], a["d"]),
             PineLowering(pine.emit_stoch)),
        _bar("adx", {"n": (2, 100)}, {"n": 14},
             lambda b, a: ind.adx(b, a["n"]),
             PineLowering(pine.emit_adx)),
        _bar("obv", {}, {}, lambda b, _a: ind.obv(b),
             PineLowering(pine.emit_obv)),
        _bar("ichimoku", {"tenkan_n": (2, 100), "kijun_n": (2, 200),
                           "senkou_n": (2, 300), "disp": (1, 100),
                           "field": ("tenkan", "kijun", "senkou_a", "senkou_b")},
             {"tenkan_n": 9, "kijun_n": 26, "senkou_n": 52, "disp": 26,
              "field": "tenkan"},
             lambda b, a: ind.ichimoku(b, a["tenkan_n"], a["kijun_n"],
                                        a["senkou_n"], a["disp"]),
             PineLowering(pine.emit_ichimoku)),
        _bar("keltner", {"n": (2, 200), "mult": (0.5, 10.0),
                          "field": ("upper", "mid", "lower")},
             {"n": 20, "mult": 2.0, "field": "mid"},
             lambda b, a: ind.keltner(b, a["n"], a["mult"]),
             PineLowering(pine.emit_keltner)),
        _bar("cci", {"n": (2, 200)}, {"n": 20},
             lambda b, a: ind.cci(b, a["n"]),
             PineLowering(pine.emit_bar_call("ta.cci", "n", source="hlc3"))),
        _bar("mfi", {"n": (2, 100)}, {"n": 14},
             lambda b, a: ind.mfi(b, a["n"]),
             PineLowering(pine.emit_bar_call("ta.mfi", "n", source="hlc3"))),
        _bar("wpr", {"n": (2, 100)}, {"n": 14},
             lambda b, a: ind.wpr(b, a["n"]),
             PineLowering(pine.emit_bar_call("ta.wpr", "n"))),
    )
    # session_scoped, on the terms below that carry it: these read the driving
    # frame's own session or calendar structure (elapsed
    # session minutes, a bar's place in the session's volume shape, the bar's
    # calendar weekday) rather than plain OHLCV structure. Feeding them a `tf`
    # swaps in a different frame's bars, which silently degenerates:
    # minutes_into_session on "1d" bars is 0 everywhere (one row is the whole
    # session); rvol on "1d" bars has one bar
    # per session, so its same-clock-time bucket is the whole series and the
    # primitive quietly becomes a plain trailing-median volume ratio, a
    # different measurement wearing the same name, while on "1h" bars the
    # buckets are whole hours, so a 15m spec's 09:30 bar is answered from a
    # 09:30-to-10:30 aggregate; day_of_week reads calendar identity that
    # belongs to the spec's own session, and it reads the weekday off whichever
    # clock ITS OWN frame's cadence calls for, so answering it from a foreign
    # frame is a category error twice over. These must always run on the spec's
    # own driving bars, so `tf` is rejected outright.
    #
    # driving_frame_intraday is the SECOND rule and a second set: what a
    # session-aligned driving frame cannot answer at all, because there one bar
    # IS the whole session. minutes_into_session is 0 on every bar, and rvol's
    # same-clock-time bucket becomes the entire series.
    # session_scoped is the right set for the foreign-`tf` rule and the WRONG
    # set for this one, which is why the two flags do not track each other; see
    # day_of_week and rvol.
    primitives = (
        _primitive("gap_pct", {}, {}, prim.gap_pct,
                   PineLowering(pine.emit_primitive(pine.GAP_PCT),
                                helpers=(pine.GAP_PCT,))),
        _primitive("swing_high", {"k": (1, 10)}, {"k": 3}, prim.swing_high,
                   PineLowering(pine.emit_swing(pine.SWING_HIGH),
                                helpers=(pine.SWING_HIGH,))),
        _primitive("swing_low", {"k": (1, 10)}, {"k": 3}, prim.swing_low,
                   PineLowering(pine.emit_swing(pine.SWING_LOW),
                                helpers=(pine.SWING_LOW,))),
        # Session-scoped, and deliberately NOT driving_frame_intraday. Reading
        # a weekday off a 1h frame inside a 15m spec is a category error, which
        # is why it takes no tf; reading a weekday off your own daily bars is
        # not. A daily bar is one session, so its weekday is exactly the
        # calendar identity the primitive promises, and day_of_week reads a
        # session frame on the clock that reading needs (see its docstring).
        # turnaround_tuesday is a shipped 1d catalog play whose entire premise
        # is day_of_week; refusing it would break a shipped play over a reading
        # that is right.
        _primitive("day_of_week", {}, {}, prim.day_of_week,
                   PineLowering(pine.emit_primitive(pine.DAY_OF_WEEK),
                                helpers=(pine.DAY_OF_WEEK,)),
                   session_scoped=True),
        _primitive("minutes_into_session", {}, {}, prim.minutes_into_session,
                   PineLowering(pine.emit_primitive(pine.MINUTES_INTO_SESSION),
                                helpers=(pine.MINUTES_INTO_SESSION,)),
                   session_scoped=True, driving_frame_intraday=True),
        # driving_frame_intraday, and that is a decision rather than an
        # accident of grouping. On daily bars rvol does not go NaN, it
        # collapses to today's volume over the trailing median daily volume:
        # not meaningless, but a different measurement wearing the same name,
        # and a spec author reading "relative volume" gets the session-shape
        # answer the primitive's docstring promises on no bar of it. Nothing
        # shipped depends on it (capitulation_snap, the only catalog user of
        # rvol, drives off 1h), so refusing costs nothing today. If a daily
        # relative-volume reading is wanted later it is a separate term with
        # its own name, never an overload of this one, and the refusal message
        # in spec.py says so.
        _primitive("rvol", {"sessions": (5, 60)}, {"sessions": 20}, prim.rvol,
                   PineLowering(pine.emit_primitive(pine.RVOL, "sessions"),
                                helpers=(pine.RVOL,)),
                   session_scoped=True, driving_frame_intraday=True),
        _primitive("bars_since", {"cond": CONDITION_ARG}, {}, prim.bars_since,
                   PineLowering(pine.emit_bars_since,
                                helpers=(pine.BARS_SINCE,))),
        # end_anchored, here and on order_block: the value is anchored to the
        # END of the frame handed in, one float off the tail rather than a
        # causal series. A whole-frame pass may not broadcast that across
        # history (it would be lookahead), so these are evaluated row by row
        # over a bounded span instead. Bounded by `lookback`, so this costs
        # what the per-bar path always cost.
        _primitive("fvg_nearest", {"direction": ("long", "short"),
                                    "field": ("top", "bottom", "mid"),
                                    "state": ("open", "inverted"),
                                    "min_size_atr": (0.05, 2.0),
                                    "lookback": (10, 200)},
                   {"direction": "long", "field": "top", "state": "open",
                    "min_size_atr": 0.25, "lookback": 40},
                   prim.fvg_nearest,
                   PineLowering(pine.emit_fvg_nearest,
                                helpers=(pine.FVG_NEAREST,)),
                   end_anchored=True),
        _primitive("leg_retrace", {"direction": ("long", "short"),
                                    "k": (1, 10)},
                   {"direction": "long", "k": 3}, prim.leg_retrace,
                   PineLowering(pine.emit_leg_retrace,
                                helpers=(pine.LEG_RETRACE,))),
        _primitive("order_block", {"direction": ("long", "short"),
                                    "field": ("top", "bottom", "mid"),
                                    "body_atr": (0.5, 5.0),
                                    "lookback": (10, 200)},
                   {"direction": "long", "field": "top", "body_atr": 1.5,
                    "lookback": 40}, prim.order_block,
                   PineLowering(pine.emit_order_block,
                                helpers=(pine.ORDER_BLOCK,)),
                   end_anchored=True),
    )
    return Vocabulary({term.name: term for term in indicators},
                      {term.name: term for term in primitives})


def resolve_vocabulary(vocabulary: Vocabulary | None) -> Vocabulary:
    return core_vocabulary() if vocabulary is None else vocabulary
