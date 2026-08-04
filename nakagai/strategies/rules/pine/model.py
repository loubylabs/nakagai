"""What a lowered RuleSpec is made of, before any target renders it.

Three shapes live here and nothing else, so that vocabulary.py can import the
lowering slot without dragging the compiler in behind it:

- the pieces a lowering builds (PineExpr, PineInput, PineHelper, TermCall),
- the program those pieces add up to (PineProgram) and the artifact pair a
  renderer turns it into (PineBundle),
- the one refusal type (PineCompileError).

PineProgram is deliberately target-neutral: ordered inputs, ordered
calculations, helper sources, the identifiers that carry the long and short
decisions, and the risk and exit expressions. It holds no `indicator()` or
`strategy()` statement, because those are the two renderers' entire difference
and a program that already picked one could only render the other by editing
text it had already committed to.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

# Bumped when the same spec would lower to different Pine. Part of a program's
# identity, so a saved artifact can say which compiler wrote it.
GENERATOR_VERSION = "1"


class PineCompileError(ValueError):
    """A refusal to generate, named precisely enough to act on.

    `code` is the machine-readable reason, `path` the RuleSpec path that
    carries it (spelled exactly as validate_spec spells it), and `term` the
    vocabulary term involved when one is.
    """

    def __init__(self, code: str, message: str, *, path: str = "",
                 term: str = ""):
        super().__init__(message)
        self.code, self.path, self.term = code, path, term


@dataclass(frozen=True)
class RulePath:
    """Where in a spec a value came from, in both spellings it needs.

    `text` is the validator's spelling (long.all[0].lhs), which is what an
    error has to say; the parts are what an identifier is built from.
    """

    parts: tuple[str, ...] = ()

    def child(self, *parts) -> "RulePath":
        return RulePath(self.parts + tuple(str(part) for part in parts))

    @property
    def text(self) -> str:
        out = ""
        for part in self.parts:
            if part.isdigit():
                out += f"[{part}]"
            else:
                out = f"{out}.{part}" if out else part
        return out


@dataclass(frozen=True)
class PineExpr:
    """One Pine expression: an identifier, a call, or a parenthesized tree."""

    text: str


@dataclass(frozen=True)
class PineInput:
    """One number the chart's user can move, named by the path that fixed it.

    An int input with float bounds does not compile on TradingView, so `kind`
    and `bounds` are decided together and stay type-exact.
    """

    name: str
    label: str
    kind: Literal["int", "float"]
    default: int | float
    bounds: tuple[int | float, int | float] | None = None


@dataclass(frozen=True)
class PineHelper:
    """A Pine function a lowering leans on, declared once per program."""

    id: str
    source: str
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class TermCall:
    """One term node being lowered: everything its emit function may read.

    `slot` is the identifier stem reserved for this node (`nk_macd_1`), so a
    multi-field term names its members `nk_macd_1_signal` and friends without
    inventing a second numbering. `content` is the node's canonical form with
    the field stripped, which is what makes two spellings of the same
    calculation share one set of inputs.

    `source` is the operand the term is applied to, already lowered: the `of`
    series for an indicator that measures one, and the condition for a
    primitive that takes one. Both arrive here rather than as spec shapes,
    because walking a spec belongs to the walk and an emit function only ever
    composes Pine out of pieces it is handed.
    """

    # `Term` and `PineContext` stay forward references on purpose: vocabulary.py
    # imports this module for the lowering slot, and lower.py imports
    # vocabulary.py, so a real import either way would close the cycle.
    term: "Term"
    args: Mapping
    path: RulePath
    slot: str
    source: str = ""
    content: str = ""

    @property
    def field(self) -> str:
        return str(self.args.get("field", ""))


@dataclass(frozen=True)
class PineLowering:
    """A term's Pine form: how to emit it, and what it leans on to run."""

    emit: Callable[["PineContext", TermCall], PineExpr]
    helpers: tuple[str, ...] = ()


@dataclass(frozen=True)
class PineRisk:
    """The risk block, lowered: what each distance means, and what carries it.

    A `kind` is a literal the renderer branches on; the field beside it is
    always a Pine identifier. Keeping the two apart in the type is the point:
    a renderer that had to tell them apart by key name would eventually read
    one as the other and emit `strategy.exit(stop="atr")`.

    `stop` names an absolute price distance when stop_kind is "atr", and a
    percent input when it is "percent". `target` names a multiple of the risked
    distance when target_kind is "rr", and a percent input when it is
    "percent".
    """

    stop_kind: Literal["atr", "percent"]
    stop: str
    target_kind: Literal["rr", "percent"]
    target: str


@dataclass(frozen=True)
class PineExits:
    """The exit block, lowered. Every member is empty when the spec omits it.

    Same split as PineRisk: trailing_kind is a literal, and every other member
    is a Pine identifier.
    """

    signal: str = ""
    trailing_kind: Literal["", "atr", "percent"] = ""
    trailing: str = ""
    time_stop_bars: str = ""
    breakeven_rr: str = ""


@dataclass(frozen=True)
class PineProgram:
    """One spec, lowered. Target-neutral by contract; see the module docstring."""

    title: str
    spec_hash: str
    generator_version: str
    # The timeframe the calculations below are written for. The spec's own,
    # because a node without a `tf` is emitted on the chart's bars rather than
    # requested, so this is the one chart the program's arithmetic is true on
    # and the renderer owes a runtime guard against every other one.
    chart: str
    inputs: tuple[PineInput, ...]
    helpers: tuple[PineHelper, ...]
    calculations: tuple[str, ...]
    long_decision: str
    short_decision: str
    risk: PineRisk
    exits: PineExits
    # How far back a non-constant offset reaches, 0 when every offset the
    # program emits is a constant. TradingView infers a series' historical
    # buffer from the offsets it can see, and answers "Pine cannot determine
    # the referencing length of series" for one it cannot, so a renderer owes
    # a max_bars_back declaration whenever this is set.
    max_bars_back: int
    warnings: tuple[str, ...]
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class PineBundle:
    """The pair of artifacts a program renders to, with its provenance."""

    indicator: str
    strategy: str
    spec_hash: str
    generator_version: str
    warnings: tuple[str, ...]
    assumptions: tuple[str, ...]
