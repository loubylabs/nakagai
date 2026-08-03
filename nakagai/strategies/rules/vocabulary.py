from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Literal, TypeAlias

from nakagai.strategies import indicators as ind
from nakagai.strategies.rules import primitives as prim
from nakagai.strategies.rules.pine.model import PineLowering

ArgRule: TypeAlias = tuple[float, float] | tuple[str, ...]


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", MappingProxyType(dict(self.args)))
        object.__setattr__(self, "defaults", MappingProxyType(dict(self.defaults)))


@dataclass(frozen=True)
class Vocabulary:
    indicators: Mapping[str, Term]
    primitives: Mapping[str, Term]

    def __post_init__(self) -> None:
        indicators = dict(self.indicators)
        primitives = dict(self.primitives)
        overlap = indicators.keys() & primitives.keys()
        if overlap:
            name = sorted(overlap)[0]
            raise ValueError(f"duplicate vocabulary term {name!r}")
        object.__setattr__(self, "indicators", MappingProxyType(indicators))
        object.__setattr__(self, "primitives", MappingProxyType(primitives))

    def with_terms(self, *terms: Term) -> "Vocabulary":
        indicators, primitives = dict(self.indicators), dict(self.primitives)
        for term in terms:
            if term.name in indicators or term.name in primitives:
                raise ValueError(f"duplicate vocabulary term {term.name!r}")
            target = primitives if term.kind == "primitive" else indicators
            target[term.name] = term
        return Vocabulary(indicators, primitives)

    def resolve(self, kind: str, name: str) -> Term:
        terms = self.primitives if kind == "primitive" else self.indicators
        return terms[name]

    def all_terms(self) -> tuple[Term, ...]:
        return tuple(self.indicators.values()) + tuple(self.primitives.values())


VocabularyFactory: TypeAlias = Callable[[], Vocabulary]


def _series(name, args, defaults, fn) -> Term:
    return Term(name, "series", args, defaults, fn)


def _frame(name, args, defaults, fn) -> Term:
    return Term(name, "frame", args, defaults, fn)


def _bar(name, args, defaults, fn) -> Term:
    return Term(name, "bar", args, defaults, fn)


def _primitive(name, args, defaults, fn, *, end_anchored=False,
               session_scoped=False, driving_frame_intraday=False) -> Term:
    return Term(name, "primitive", args, defaults, fn, doc=fn.__doc__ or "",
                end_anchored=end_anchored, session_scoped=session_scoped,
                driving_frame_intraday=driving_frame_intraday)


@cache
def core_vocabulary() -> Vocabulary:
    indicators = (
        _series("sma", {"n": (2, 500)}, {"n": 20},
                lambda s, a: ind.sma(s, a["n"])),
        _series("ema", {"n": (2, 500)}, {"n": 20},
                lambda s, a: ind.ema(s, a["n"])),
        _series("rsi", {"n": (2, 100)}, {"n": 14},
                lambda s, a: ind.rsi(s, a["n"])),
        _series("roc", {"n": (1, 500)}, {"n": 20},
                lambda s, a: ind.roc(s, a["n"])),
        _series("zscore", {"n": (2, 500)}, {"n": 20},
                lambda s, a: ind.zscore(s, a["n"])),
        _series("highest", {"n": (2, 500)}, {"n": 20},
                lambda s, a: ind.highest(s, a["n"])),
        _series("lowest", {"n": (2, 500)}, {"n": 20},
                lambda s, a: ind.lowest(s, a["n"])),
        _series("stdev", {"n": (2, 500)}, {"n": 20},
                lambda s, a: ind.stdev(s, a["n"])),
        _frame("macd", {"fast": (2, 100), "slow": (3, 200),
                         "signal": (2, 100),
                         "field": ("macd", "signal", "hist")},
               {"fast": 12, "slow": 26, "signal": 9, "field": "macd"},
               lambda s, a: ind.macd(s, a["fast"], a["slow"], a["signal"])),
        _frame("bb", {"n": (2, 200), "k": (0.5, 5.0),
                       "field": ("upper", "mid", "lower")},
               {"n": 20, "k": 2.0, "field": "mid"},
               lambda s, a: ind.bollinger(s, a["n"], a["k"])),
        _bar("atr", {"n": (2, 100)}, {"n": 14},
             lambda b, a: ind.atr(b, a["n"])),
        _bar("donchian", {"n": (2, 300),
                           "field": ("upper", "lower", "mid")},
             {"n": 20, "field": "upper"},
             lambda b, a: ind.donchian(b, a["n"])),
        _bar("supertrend", {"n": (2, 100), "mult": (0.5, 10.0),
                             "field": ("line", "direction")},
             {"n": 10, "mult": 3.0, "field": "line"},
             lambda b, a: ind.supertrend(b, a["n"], a["mult"])),
        _bar("vwap", {}, {}, lambda b, _a: ind.session_vwap(b)),
        _bar("stoch", {"n": (2, 100), "d": (1, 50),
                        "field": ("k", "d")},
             {"n": 14, "d": 3, "field": "k"},
             lambda b, a: ind.stoch(b, a["n"], a["d"])),
        _bar("adx", {"n": (2, 100)}, {"n": 14},
             lambda b, a: ind.adx(b, a["n"])),
        _bar("obv", {}, {}, lambda b, _a: ind.obv(b)),
        _bar("ichimoku", {"tenkan_n": (2, 100), "kijun_n": (2, 200),
                           "senkou_n": (2, 300), "disp": (1, 100),
                           "field": ("tenkan", "kijun", "senkou_a", "senkou_b")},
             {"tenkan_n": 9, "kijun_n": 26, "senkou_n": 52, "disp": 26,
              "field": "tenkan"},
             lambda b, a: ind.ichimoku(b, a["tenkan_n"], a["kijun_n"],
                                        a["senkou_n"], a["disp"])),
        _bar("keltner", {"n": (2, 200), "mult": (0.5, 10.0),
                          "field": ("upper", "mid", "lower")},
             {"n": 20, "mult": 2.0, "field": "mid"},
             lambda b, a: ind.keltner(b, a["n"], a["mult"])),
        _bar("cci", {"n": (2, 200)}, {"n": 20},
             lambda b, a: ind.cci(b, a["n"])),
        _bar("mfi", {"n": (2, 100)}, {"n": 14},
             lambda b, a: ind.mfi(b, a["n"])),
        _bar("wpr", {"n": (2, 100)}, {"n": 14},
             lambda b, a: ind.wpr(b, a["n"])),
    )
    primitives = (
        _primitive("opening_range_high", {"minutes": (5, 120)}, {"minutes": 30},
                   prim.opening_range_high, session_scoped=True,
                   driving_frame_intraday=True),
        _primitive("opening_range_low", {"minutes": (5, 120)}, {"minutes": 30},
                   prim.opening_range_low, session_scoped=True,
                   driving_frame_intraday=True),
        _primitive("prev_session_high", {}, {}, prim.prev_session_high),
        _primitive("prev_session_low", {}, {}, prim.prev_session_low),
        _primitive("prev_session_close", {}, {}, prim.prev_session_close),
        _primitive("gap_pct", {}, {}, prim.gap_pct),
        _primitive("swing_high", {"k": (1, 10)}, {"k": 3}, prim.swing_high),
        _primitive("swing_low", {"k": (1, 10)}, {"k": 3}, prim.swing_low),
        _primitive("day_of_week", {}, {}, prim.day_of_week, session_scoped=True),
        _primitive("minutes_into_session", {}, {}, prim.minutes_into_session,
                   session_scoped=True, driving_frame_intraday=True),
        _primitive("rvol", {"sessions": (5, 60)}, {"sessions": 20}, prim.rvol,
                   session_scoped=True, driving_frame_intraday=True),
        _primitive("bars_since", {"cond": "condition"}, {}, prim.bars_since),
        _primitive("fvg_nearest", {"direction": ("long", "short"),
                                    "field": ("top", "bottom", "mid"),
                                    "state": ("open", "inverted"),
                                    "min_size_atr": (0.05, 2.0),
                                    "lookback": (10, 200)},
                   {"direction": "long", "field": "top", "state": "open",
                    "min_size_atr": 0.25, "lookback": 40},
                   prim.fvg_nearest, end_anchored=True),
        _primitive("leg_retrace", {"direction": ("long", "short"),
                                    "k": (1, 10)},
                   {"direction": "long", "k": 3}, prim.leg_retrace),
        _primitive("order_block", {"direction": ("long", "short"),
                                    "field": ("top", "bottom", "mid"),
                                    "body_atr": (0.5, 5.0),
                                    "lookback": (10, 200)},
                   {"direction": "long", "field": "top", "body_atr": 1.5,
                    "lookback": 40}, prim.order_block, end_anchored=True),
    )
    return Vocabulary({term.name: term for term in indicators},
                      {term.name: term for term in primitives})


def resolve_vocabulary(vocabulary: Vocabulary | None) -> Vocabulary:
    return core_vocabulary() if vocabulary is None else vocabulary
