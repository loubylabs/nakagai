"""One lowered program, written out as the two artifacts a user pastes.

The indicator and the strategy are not two translations of a spec. They are one
translation with two endings: the header, the runtime guards, the inputs, the
helpers, the calculations and the two decision identifiers are rendered once,
by _body, and embedded verbatim in both. Only the ending differs, and each one
is small: markers, frozen levels and one alert on the indicator; sizing, orders
and the exit ledger on the strategy. A user who marks a bar with the indicator
and trades it with the strategy is looking at the same bar by construction,
rather than because two renderers were kept in step by hand.

Generation is atomic for the same reason. render() builds both before it builds
a bundle, so a failure in either half leaves a caller with neither rather than
with one artifact it could ship beside a missing one.

Three of the strategy's decisions mirror engine conventions exactly, and each
is the kind that looks arbitrary until it is wrong:

- THE FILL BAR IS HELD BAR ONE. RuleStrategy.manage runs in the same loop pass
  as the fill, so a position is one bar old on the bar it opened. A time stop
  counting from zero would hold every trade one bar too long.
- A STOP ONLY EVER RATCHETS TOWARD PRICE. Long takes the max, short the min,
  and a NaN candidate moves nothing.
- THE LEVELS FREEZE AT THE SIGNAL. The engine measures its stop and target from
  the close of the bar that decided, then fills at the next bar's open. A
  strategy that recomputed them from TradingView's fill would be trading
  different geometry from the one the indicator drew.
"""

import textwrap
from decimal import Decimal

from nakagai.strategies.rules.pine.model import (
    PineBundle, PineExits, PineInput, PineProgram, PineRisk,
)

# What the chart guard checks: the chart length as Pine counts it, and the
# words the refusal uses. ONE entry, because there is one driving cadence and
# every export runs on it, whatever the play's own timeframe is. It is a table
# rather than two literals so that the guard is read off the program the
# lowerer built: a chart the lowerer should never have produced raises here
# instead of being guarded with the wrong number. This is a guard rather than a
# decoration, because on any other length the same script is a different
# strategy.
CHART = {"15m": ("15 * 60", "15-minute")}

# The fidelity boundary, in the design's own order. Every bundle carries all
# five, whatever the spec says, because none of them is a property of the spec:
# they are what two data vendors and two broker simulators do differently.
FIDELITY = (
    "TradingView and Alpaca can differ in session coverage, adjustments, "
    "volume, missing bars, and aggregate boundaries.",
    "Nakagai resolves an ambiguous bar pessimistically with stop first. "
    "TradingView infers an intrabar path or uses Bar Magnifier data.",
    "Nakagai uses one basis point of slippage with a one-cent floor. Pine's "
    "built-in slippage is a fixed number of ticks, set to one here, which is "
    "the same cent on a stock at or under $100 and less than Nakagai charges "
    "above it.",
    "Nakagai enforces a T+1 settled-funds ledger. Pine models capital and "
    "margin without an equivalent settlement ledger.",
    "TradingView input changes alter the exported play without changing its "
    "embedded hash.",
)

# One more, conditional on the spec, covering a difference a user would
# otherwise read as a bug in one of the two engines: an exit rule and a time
# stop both leave the engine at the close of the bar that decided, and
# TradingView fills the market order they become at the next bar's open.
NEXT_BAR_CLOSE = ("An exit rule or a time stop closes at the next bar's open "
                  "on TradingView, where the engine closes it at the close of "
                  "the bar that decided.")

_SYNTHETIC = ("a standard {label} candle chart. Heikin Ashi, Renko, Line "
              "Break, Kagi, Point and Figure, Range and every other "
              "synthetic-price chart show prices no market printed, so a "
              "decision read off one would be fiction. This script refuses to "
              "run on one, and on a chart of any other length.")

_ARTIFACT = {
    "indicator": ("indicator. It marks the bars Nakagai would signal on, draws "
                  "the levels that signal froze, and raises one alert for each "
                  "one. It places no orders."),
    "strategy": ("strategy. It runs the same decisions through TradingView's "
                 "broker emulator, entering at the next bar's open and "
                 "carrying the stop and target the signal froze."),
}

_LOCAL_EDIT = ("Editing any input below creates a TradingView-local variation: "
               "its results no longer represent the spec hash above, and "
               "Nakagai cannot read that variation back.")

_EVIDENCE = ("Fidelity boundary. A TradingView result is a separate "
             "simulation and is never Nakagai evidence:")

_WIDTH = 79


# -- literals --------------------------------------------------------------


def _number(value: int | float) -> str:
    """One number as a Pine literal: plain decimal, and round-trip exact.

    A float always keeps its point, because `input.float(2, ...)` is a type
    error on TradingView while `input.float(2.0, ...)` is the same number
    correctly typed. Exponent form is written out rather than emitted, so the
    artifact never depends on how Pine's literal grammar reads `1e-06`.
    """
    if isinstance(value, int):
        return str(value)
    text = repr(float(value))
    return format(Decimal(text), "f") if "e" in text or "E" in text else text


def _string(text: str) -> str:
    """One Pine string literal, with the two characters that would end it early."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _name(program: PineProgram) -> str:
    """The play's own name, made safe to sit inside a Pine string literal.

    Deterministic and total: whitespace collapses, anything unprintable or
    quote-like goes, and a name that sanitizes away entirely falls back to the
    spec hash rather than to an empty title no chart could name.
    """
    text = " ".join(str(program.title).split())
    text = "".join(c for c in text if c.isprintable() and c not in '"\\')[:64]
    return text or program.spec_hash[:12]


def _title(program: PineProgram) -> str:
    """What the script calls itself on a chart."""
    return f"Nakagai · {_name(program)}"


def _prose(text: str, bullet: bool = False) -> list[str]:
    """One sentence as wrapped Pine comment lines."""
    head, rest = ("//   - ", "//     ") if bullet else ("// ", "// ")
    return textwrap.wrap(text, width=_WIDTH, initial_indent=head,
                         subsequent_indent=rest, break_on_hyphens=False,
                         break_long_words=False)


def _indent(lines, depth: int = 1) -> list[str]:
    pad = "    " * depth
    return [pad + line for line in lines]


def _section(title: str, lines) -> list[str]:
    return ["", f"// --- {title} ---", *lines]


# -- the shared half -------------------------------------------------------


def _header(program: PineProgram, kind: str, warnings) -> list[str]:
    label = CHART[program.chart][1]
    lines = ["//@version=6",
             f"// Nakagai Pine export: {_name(program)}"]
    lines += _prose(f"Artifact: {_ARTIFACT[kind]}")
    lines += [f"// Generator version: {program.generator_version}",
              f"// Spec hash: {program.spec_hash}", "//"]
    lines += _prose("Chart: " + _SYNTHETIC.format(label=label)) + ["//"]
    lines += ["// Assumptions:"]
    for text in program.assumptions:
        lines += _prose(text, bullet=True)
    lines += ["//"] + _prose(_EVIDENCE)
    for text in warnings:
        lines += _prose(text, bullet=True)
    return lines + ["//"] + _prose(_LOCAL_EDIT)


def _guards(program: PineProgram) -> list[str]:
    seconds, label = CHART[program.chart]
    return _section("Chart contract", [
        f"if barstate.isfirst and timeframe.in_seconds() != {seconds}",
        "    runtime.error(" + _string(
            f"Nakagai Pine exports require a standard {label} chart.") + ")",
        "if barstate.isfirst and not chart.is_standard",
        "    runtime.error(" + _string(
            "Nakagai Pine exports require a standard candle chart.") + ")",
    ])


def _input(item: PineInput) -> str:
    call = "input.int" if item.kind == "int" else "input.float"
    parts = [_number(item.default), _string(item.label)]
    if item.bounds is not None:
        parts += [f"minval={_number(item.bounds[0])}",
                  f"maxval={_number(item.bounds[1])}"]
    return f"{item.name} = {call}({', '.join(parts)})"


def _calculations(program: PineProgram) -> list[str]:
    """Every top-level statement, with a function body set off from its neighbours."""
    lines: list[str] = []
    for text in program.calculations:
        body = text.splitlines()
        if len(body) > 1 and lines:
            lines.append("")
        lines += body
        if len(body) > 1:
            lines.append("")
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _geometry(risk: PineRisk, side: str) -> tuple[str, str]:
    """The stop and target one side measures from the close of its signal bar.

    stop_target() reads the same two numbers off the same reference: a distance
    away from the close for the stop, and either a multiple of that risked
    distance or a percent of the close for the target.
    """
    long = side == "long"
    away, toward = ("-", "+") if long else ("+", "-")
    distance = (risk.stop if risk.stop_kind == "atr"
                else f"close * {risk.stop} / 100")
    if risk.target_kind == "rr":
        risked = ("(close - nk_long_stop)" if long
                  else "(nk_short_stop - close)")
        target = f"close {toward} {risk.target} * {risked}"
    else:
        target = f"close {toward} close * {risk.target} / 100"
    return f"close {away} {distance}", target


def _decisions(program: PineProgram) -> list[str]:
    """The two identifiers both artifacts read, and the geometry behind them.

    A decision carries its geometry rather than merely its conditions, because
    rr_signal emits NOTHING for a stop that is NaN or on the wrong side of the
    reference. Without that, the opening bars of a chart (where ta.atr is still
    na) would mark a signal the engine never had, alert with NaN levels, and
    size a position from a NaN risk.

    The geometry is measured from `close`, the CHART's close, which is the 15m
    driving close risk.stop_target measures from. The distance beside it came
    off the play's own timeframe. That hybrid is the engine's, not an oversight
    here, and flattening either half would move every stop.

    A play whose timeframe is not the chart's leads with its freshness gate, so
    a decision that would otherwise stand for all four 15m bars of a 1h bar
    fires on the one the engine fires on.
    """
    lines: list[str] = []
    gate = f"{program.decision_gate} and " if program.decision_gate else ""
    for side, entry in (("long", program.long_decision),
                        ("short", program.short_decision)):
        if not entry:
            # Not an omission: the spec has no such side, and both artifacts
            # still read the same two names.
            lines.append(f"nk_{side}_decision = false")
            continue
        stop, target = _geometry(program.risk, side)
        beyond, ahead = ("<", ">") if side == "long" else (">", "<")
        lines += [
            f"nk_{side}_stop = {stop}",
            f"nk_{side}_target = {target}",
            f"nk_{side}_decision = {gate}{entry} and not na(nk_{side}_stop) "
            f"and nk_{side}_stop {beyond} close and nk_{side}_target {ahead} "
            "close",
        ]
    return lines


def _body(program: PineProgram) -> list[str]:
    """Everything both artifacts are one copy of."""
    helpers: list[str] = []
    for helper in program.helpers:
        if helpers:
            helpers.append("")
        helpers += helper.source.splitlines()
    return [
        *_guards(program),
        *_section("Inputs", [_input(item) for item in program.inputs]),
        *(_section("Helpers", helpers) if helpers else []),
        *_section("Calculations", _calculations(program)),
        *_section("Decisions", _decisions(program)),
    ]


def _declaration(program: PineProgram, call: str, *parts: str) -> str:
    history = ([f"max_bars_back={program.max_bars_back}"]
               if program.max_bars_back else [])
    return f"{call}({', '.join([*parts, *history])})"


# -- the indicator ---------------------------------------------------------


def _alert_body(program: PineProgram) -> list[str]:
    """The JSON one decision publishes, built at runtime from frozen levels.

    Every value the receiver could not recompute is in it: which spec decided,
    on which symbol and chart, which way, when, and the three prices. The
    levels arrive as arguments rather than being read off the globals, so the
    function stands on its own wherever it is defined.
    """
    schema = '\\"schema\\":\\"nakagai.pine.signal.v1\\"'
    return [
        "nk_signal_json(direction, reference, stop, target) =>",
        f'    string head = "{{{schema},\\"spec_hash\\":'
        f'\\"{program.spec_hash}\\",\\"symbol\\":\\"" + syminfo.ticker + "\\""',
        '    string where = ",\\"timeframe\\":\\"" + timeframe.period + '
        '"\\",\\"direction\\":\\"" + direction + "\\""',
        '    string when = ",\\"bar_time\\":" + str.tostring(time)',
        '    string levels = ",\\"reference\\":" + str.tostring(reference) + '
        '",\\"stop\\":" + str.tostring(stop) + ",\\"target\\":" + '
        "str.tostring(target)",
        '    head + where + when + levels + "}"',
    ]


def _levels(program: PineProgram) -> list[str]:
    """Freeze the three levels on a decision, long first, and alert once."""
    lines = ["var float nk_reference = na",
             "var float nk_stop = na",
             "var float nk_target = na"]
    branches = [side for side, entry in (("long", program.long_decision),
                                         ("short", program.short_decision))
                if entry]
    for i, side in enumerate(branches):
        lines.append(f"{'if' if i == 0 else 'else if'} nk_{side}_decision")
        lines += _indent([
            "nk_reference := close",
            f"nk_stop := nk_{side}_stop",
            f"nk_target := nk_{side}_target",
            f'alert(nk_signal_json("{side}", nk_reference, nk_stop, '
            "nk_target), alert.freq_once_per_bar_close)",
        ])
    return lines


def _plots(stop: str) -> list[str]:
    return [f"plot(nk_reference, {_string('Nakagai reference')}, "
            "color=color.gray, style=plot.style_linebr)",
            f"plot(nk_stop, {_string(stop)}, color=color.maroon, "
            "style=plot.style_linebr)",
            f"plot(nk_target, {_string('Nakagai target')}, color=color.teal, "
            "style=plot.style_linebr)"]


def _indicator(program: PineProgram, warnings, body: list[str]) -> str:
    markers = [
        f"plotshape(nk_long_decision, {_string('Nakagai long')}, "
        "shape.triangleup, location.belowbar, color.teal, size=size.tiny)",
        f"plotshape(nk_short_decision, {_string('Nakagai short')}, "
        "shape.triangledown, location.abovebar, color.maroon, size=size.tiny)",
    ]
    lines = [
        *_header(program, "indicator", warnings),
        _declaration(program, "indicator", _string(_title(program)),
                     "overlay=true"),
        *body,
        *_section("Markers", markers),
        *_section("Alert body", _alert_body(program)),
        *_section("Signal levels", _levels(program)),
        *_section("Level plots", _plots("Nakagai stop")),
    ]
    return "\n".join(lines) + "\n"


# -- the strategy ----------------------------------------------------------


_SIZING = ('nk_risk_fraction = input.float(1.0, "Sizing · risk per trade '
           '(% of equity)", minval=0.01, maxval=100.0)')


def _entries(program: PineProgram) -> list[str]:
    """Enter only when flat, size from equity and per-share risk, long first.

    The floor and the qty > 0 test are the engine's: it floors the same
    quotient and takes no trade at all when that rounds to nothing.
    """
    lines = ["if strategy.position_size == 0"]
    branches = [(side, entry) for side, entry
                in (("long", program.long_decision),
                    ("short", program.short_decision)) if entry]
    for i, (side, _entry) in enumerate(branches):
        risked = ("(close - nk_long_stop)" if side == "long"
                  else "(nk_short_stop - close)")
        freeze = ["nk_reference := close"]
        if program.exits.breakeven_rr:
            freeze.append(f"nk_signal_stop := nk_{side}_stop")
        freeze += [f"nk_stop := nk_{side}_stop",
                   f"nk_target := nk_{side}_target",
                   f'strategy.entry("Nakagai {side}", strategy.{side}, '
                   f"qty=nk_{side}_size)"]
        lines += _indent([
            f"{'if' if i == 0 else 'else if'} nk_{side}_decision",
            f"    float nk_{side}_size = math.floor(strategy.equity * "
            f"nk_risk_fraction / 100 / {risked})",
            f"    if nk_{side}_size > 0",
            *_indent(freeze, 2),
        ])
    return lines


def _ratchets(exits: PineExits, side: str) -> list[str]:
    """Break-even and trailing, both moving the stop only toward price."""
    toward = "math.max" if side == "long" else "math.min"
    lines: list[str] = []
    if exits.breakeven_rr:
        # Measured against the SIGNAL stop, which never moves: dividing by the
        # ratcheted one would shrink the yardstick as the trade ran.
        gain = ("(close - strategy.position_avg_price)" if side == "long"
                else "(strategy.position_avg_price - close)")
        lines += [
            f"float nk_{side}_risk = math.abs(strategy.position_avg_price - "
            "nk_signal_stop)",
            f"if nk_{side}_risk > 0 and {gain} / nk_{side}_risk >= "
            f"{exits.breakeven_rr}",
            f"    nk_stop := {toward}(nk_stop, strategy.position_avg_price)",
        ]
    if exits.trailing_kind:
        away = "-" if side == "long" else "+"
        distance = (exits.trailing if exits.trailing_kind == "atr"
                    else f"close * {exits.trailing} / 100")
        lines += [
            f"float nk_{side}_trail = close {away} {distance}",
            f"if not na(nk_{side}_trail)",
            f"    nk_stop := {toward}(nk_stop, nk_{side}_trail)",
        ]
    return lines


def _manage(exits: PineExits, side: str) -> list[str]:
    """One open position, managed the way RuleStrategy.manage manages it.

    Same order, and the order is the behavior: an exit rule is asked first, the
    time stop second, and the ratchets only on a bar that is closing neither.
    manage() returns EXIT before it ever reaches them, so a bar that is
    liquidating the position does not also move its stop.
    """
    lines: list[str] = []
    closes = []
    if exits.signal:
        closes.append((exits.signal, "Nakagai exit rule"))
    if exits.time_stop_bars:
        # The fill bar is held bar one: manage() runs in the same loop pass as
        # the fill, one driving bar after the signal.
        closes.append(("bar_index - strategy.opentrades.entry_bar_index(0) + 1 "
                       f">= {exits.time_stop_bars}", "Nakagai time stop"))
    for i, (condition, comment) in enumerate(closes):
        lines += [f"{'if' if i == 0 else 'else if'} {condition}",
                  f'    strategy.close("Nakagai {side}", '
                  f"comment={_string(comment)})"]
    ratchets = _ratchets(exits, side)
    if ratchets:
        lines += ["else", *_indent(ratchets)] if closes else ratchets
    return lines + [f'strategy.exit("Nakagai {side} exit", '
                    f'from_entry="Nakagai {side}", stop=nk_stop, '
                    "limit=nk_target)"]


def _ledger(program: PineProgram) -> list[str]:
    """The three frozen levels, the entry, and one management block per side.

    No blank line ever separates a branch from the `else if` that follows it:
    Pine reads a block by its indentation, and the two halves of one decision
    have to arrive as one statement.
    """
    lines = ["var float nk_reference = na"]
    if program.exits.breakeven_rr:
        lines.append("var float nk_signal_stop = na")
    lines += ["var float nk_stop = na", "var float nk_target = na", ""]
    lines += _entries(program)
    sides = [side for side in ("long", "short")
             if getattr(program, f"{side}_decision")]
    for i, side in enumerate(sides):
        beyond = ">" if side == "long" else "<"
        lines += ([""] if i == 0 else []) + [
            f"{'if' if i == 0 else 'else if'} strategy.position_size "
            f"{beyond} 0"]
        lines += _indent(_manage(program.exits, side))
    return lines


def _strategy(program: PineProgram, warnings, body: list[str]) -> str:
    declaration = _declaration(
        program, "strategy", _string(f"{_title(program)} · strategy"),
        "overlay=true", "initial_capital=10000", "pyramiding=0",
        "calc_on_order_fills=false", "calc_on_every_tick=false",
        "process_orders_on_close=false",
        # The closest honest defaults TradingView has for two costs Nakagai
        # models differently: one tick is the cent Nakagai charges on a stock
        # under $100, and 100% margin is the cash account its settled ledger
        # describes. Both mismatches are stated in the header.
        "slippage=1", "margin_long=100", "margin_short=100")
    lines = [
        *_header(program, "strategy", warnings),
        declaration,
        *body,
        *_section("Sizing", [_SIZING]),
        *_section("Ledger", _ledger(program)),
        *_section("Level plots", _plots("Nakagai stop (live)")),
    ]
    return "\n".join(lines) + "\n"


# -- the seam --------------------------------------------------------------


def _warnings(program: PineProgram) -> tuple[str, ...]:
    conditional = ()
    if program.exits.signal or program.exits.time_stop_bars:
        conditional += (NEXT_BAR_CLOSE,)
    return FIDELITY + conditional + program.warnings


def render(program: PineProgram) -> PineBundle:
    """Both artifacts, or neither. See the module docstring on why."""
    warnings = _warnings(program)
    body = _body(program)
    indicator = _indicator(program, warnings, body)
    strategy = _strategy(program, warnings, body)
    return PineBundle(indicator=indicator, strategy=strategy,
                      spec_hash=program.spec_hash,
                      generator_version=program.generator_version,
                      warnings=warnings, assumptions=program.assumptions)
