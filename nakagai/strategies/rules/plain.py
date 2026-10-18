"""A rule spec in plain sentences, for a person.

describe_spec renders the rule DSL (`close crosses above sma(20)`), which an
agent edits rules with and which must not change. rule_sentence renders the
same spec for a person reading it before they save: "Long on the daily chart
when the close crosses above the 20-day simple moving average." Deterministic,
every number printed in code.

Each term's words live on its Term (`Term.plain`), so an injected vocabulary
brings its own; a term without them reads through a generic fallback. Like
describe_spec this never raises: it is handed whatever arrived.
"""

from collections.abc import Mapping

from nakagai.data.schema import DEFAULT_TIMEFRAMES
from nakagai.strategies.rules.spec import (
    DEFAULT_RISK, LOW_IEX_DISCLOSURE, STOP_ATR_MULT_DEFAULT, STOP_ATR_N_DEFAULT,
    STOP_PCT_DEFAULT, TARGET_PCT_DEFAULT, TARGET_RR_DEFAULT,
    TRAILING_ATR_MULT_DEFAULT, TRAILING_ATR_N_DEFAULT, TRAILING_PCT_DEFAULT,
    is_group_node, names,
)
from nakagai.strategies.rules.vocabulary import (
    Term, Vocabulary, is_condition_rule, plain_number, resolve_vocabulary,
)
from nakagai.strategies.rules.windows import WindowSpec

# One bar of these frames is one natural unit, so a length reads "20-day".
_UNITS = {"1d": "day", "1w": "week", "1h": "hour"}
_FRAMES = {"15m": "15-minute", "1h": "hourly", "4h": "4-hour", "1d": "daily",
           "1w": "weekly"}
_SOURCES = {"close": "the close", "open": "the open", "high": "the high",
            "low": "the low", "volume": "volume"}
_OPS = {">": "is above", "<": "is below", ">=": "is at or above",
        "<=": "is at or below", "crosses_above": "crosses above",
        "crosses_below": "crosses below"}
_MATH = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}
_REDUCERS = {"max": "highest", "min": "lowest", "first": "first",
             "last": "last"}
_PRIOR = {"prior_session": "during the prior session",
          "prior_iso_week": "during the prior week",
          "prior_calendar_month": "during the prior month"}
_DEFAULT_SOURCE = {"src": "close"}


def _frame(tf) -> str:
    return _FRAMES.get(tf, str(tf)) if isinstance(tf, str) else str(tf)


def _join(items: list[str], word: str) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {word} {items[-1]}"


def _possessive(sym: str, phrase: str) -> str:
    """`SPY's close`; a phrase that is not a `the ...` noun takes `for SPY`."""
    if phrase.startswith("the "):
        return f"{sym}'s {phrase[4:]}"
    if phrase in _SOURCES.values():
        return f"{sym}'s {phrase}"
    return f"{phrase} for {sym}"


def _window_text(window: WindowSpec) -> str:
    if window.recurrence in _PRIOR:
        return _PRIOR[window.recurrence]
    city = window.tz.rsplit("/", 1)[-1].replace("_", " ")
    return (f"between {window.start:%H:%M} and {window.end:%H:%M} "
            f"{city} time")


class _Reader:
    def __init__(self, vocabulary: Vocabulary, tf: str):
        self.vocabulary = vocabulary
        self.tf = tf
        self.low_iex = False

    def unit(self, tf) -> str:
        return _UNITS.get(tf, "bar") if isinstance(tf, str) else "bar"

    def scoped(self, node: dict, phrase: str, tf) -> str:
        """The node's own frame and symbol around its phrase. A frame other
        than the spec's is named unless the length already says it."""
        if "tf" in node and node["tf"] != self.tf:
            unit = self.unit(node["tf"])
            if unit == "bar" or f"-{unit}" not in phrase and f" {unit}" not in phrase:
                phrase += f" on {_frame(node['tf'])} bars"
        if isinstance(node.get("sym"), str):
            phrase = _possessive(node["sym"], phrase)
        return phrase

    def expr(self, node, tf) -> str:
        if isinstance(node, bool) or not isinstance(node, (int, float, dict)):
            return repr(node)
        if isinstance(node, (int, float)):
            return plain_number(node)
        tf = node.get("tf", tf)
        if "src" in node:
            return self.source(node)
        if "fact" in node:
            fact = node["fact"]
            return fact.replace("_", " ") if isinstance(fact, str) else repr(fact)
        if "op" in node:
            return self.math(node, tf)
        if "ind" in node:
            return self.indicator(node, tf)
        if "prim" in node:
            return self.primitive(node, tf)
        return repr(node)

    def source(self, node: dict) -> str:
        src = node["src"]
        text = _SOURCES.get(src, repr(src)) if isinstance(src, str) else repr(src)
        if "tf" in node and node["tf"] != self.tf:
            frame = _frame(node["tf"])
            text = (f"the {frame} {text[4:]}" if text.startswith("the ")
                    else f"{frame} {text}")
        if isinstance(node.get("sym"), str):
            text = _possessive(node["sym"], text)
        return text

    def math(self, node: dict, tf) -> str:
        op, args = node["op"], node.get("args")
        if not isinstance(args, list):
            return repr(node)
        parts = [self.expr(a, tf) for a in args]
        if op == "abs" and parts:
            return f"the absolute value of {parts[0]}"
        if op == "min":
            return f"the lower of {_join(parts, 'and')}"
        if op == "max":
            return f"the higher of {_join(parts, 'and')}"
        if names(op, _MATH):
            return "(" + f" {_MATH[op]} ".join(parts) + ")"
        return repr(op)

    def phrase(self, term: Term, args: dict, unit: str, of: str) -> str:
        if term.plain is not None:
            try:
                return str(term.plain(args, unit, of))
            except Exception:  # noqa: BLE001 (an injected phrase must not 500 a read)
                pass
        args = dict(args)
        field = args.pop("field", None)
        text = term.name.replace("_", " ")
        params = ", ".join(f"{str(k).replace('_', ' ')} {plain_number(v)}"
                           for k, v in args.items())
        if params:
            text += f" ({params})"
        if field is not None:
            text += f" {field}"
        return text + (f" of {of}" if of else "")

    def indicator(self, node: dict, tf) -> str:
        name = node["ind"]
        if not names(name, self.vocabulary.indicators):
            return repr(name)
        term = self.vocabulary.indicators[name]
        of_node = node.get("of", _DEFAULT_SOURCE)
        of = ("" if term.kind == "bar" or of_node == _DEFAULT_SOURCE
              else self.expr(of_node, tf))
        if "window" in node and term.window_reduce is not None:
            text = (f"the {_REDUCERS[term.window_reduce]} value of "
                    f"{of or 'the close'} {self.window(node['window'])}")
        else:
            args = {**term.defaults,
                    **{k: v for k, v in node.items()
                       if k not in ("ind", "of", "tf", "window", "sym")}}
            text = self.phrase(term, args, self.unit(tf), of)
        return self.scoped(node, text, tf)

    def primitive(self, node: dict, tf) -> str:
        name = node["prim"]
        if not names(name, self.vocabulary.primitives):
            return repr(name)
        term = self.vocabulary.primitives[name]
        args = {**term.defaults,
                **{k: v for k, v in node.items()
                   if k not in ("prim", "tf", "sym")}}
        for arg, rule in term.args.items():
            if is_condition_rule(rule) and arg in args:
                args[arg] = self.condition(args[arg], tf)
        return self.scoped(node, self.phrase(term, args, self.unit(tf), ""), tf)

    def window(self, name) -> str:
        if not names(name, self.vocabulary.windows):
            return f"during the {name!r} window"
        window = self.vocabulary.windows[name]
        if window.confidence == "low_iex":
            self.low_iex = True
        return _window_text(window)

    def condition(self, cond, tf=None) -> str:
        if not isinstance(cond, dict):
            return repr(cond)
        tf = self.tf if tf is None else tf
        op = cond.get("op")
        text = _OPS[op] if names(op, _OPS) else repr(op)
        return (f"{self.expr(cond.get('lhs'), tf)} {text} "
                f"{self.expr(cond.get('rhs'), tf)}")

    def group(self, group, nested: bool = False) -> str:
        if not is_group_node(group):
            return self.condition(group)
        key, val = next((k, v) for k, v in group.items()
                        if k in ("all", "any", "not"))
        if key == "not":
            inner = self.group(val)
            return f"not ({inner})" if nested else f"it is not true that {inner}"
        if not isinstance(val, list):
            return repr(val)
        items = [self.group(item, nested=True) for item in val]
        text = _join(items, "and" if key == "all" else "or")
        return f"({text})" if nested and len(items) > 1 else text


def _sides(spec: dict) -> tuple[str, str]:
    """Where the stop and a percent target sit relative to entry."""
    long, short = "long" in spec, "short" in spec
    if long and not short:
        return "below entry", "above entry"
    if short and not long:
        return "above entry", "below entry"
    return "from entry", "from entry"


def _exits(reader: _Reader, exits: Mapping) -> list[str]:
    out = []
    unit = reader.unit(reader.tf)
    if "exit" in exits:
        out.append(f"Exit early when {reader.group(exits['exit'])}.")
    trailing = exits.get("trailing")
    if isinstance(trailing, dict):
        if trailing.get("kind") == "atr":
            out.append(
                f"Trail the stop "
                f"{plain_number(trailing.get('mult', TRAILING_ATR_MULT_DEFAULT))}x "
                f"the "
                f"{plain_number(trailing.get('n', TRAILING_ATR_N_DEFAULT))}-{unit} "
                "average true range behind the best price.")
        else:
            out.append(
                f"Trail the stop "
                f"{plain_number(trailing.get('pct', TRAILING_PCT_DEFAULT))}% "
                "behind the best price.")
    time_stop = exits.get("time_stop")
    if isinstance(time_stop, dict):
        # Held bars are counted on the driving frame, whatever the spec's.
        out.append(f"Close the trade after "
                   f"{plain_number(time_stop.get('bars'))} "
                   f"{_frame(DEFAULT_TIMEFRAMES.driving)} bars.")
    breakeven = exits.get("breakeven_at")
    if isinstance(breakeven, dict):
        out.append(f"Move the stop to entry once the trade is up "
                   f"{plain_number(breakeven.get('rr'))}x the risk.")
    return out


def _risk(reader: _Reader, risk: Mapping, spec: dict) -> list[str]:
    stop_side, target_side = _sides(spec)
    unit = reader.unit(reader.tf)
    stop = risk.get("stop", DEFAULT_RISK["stop"])
    target = risk.get("target", DEFAULT_RISK["target"])
    stop = stop if isinstance(stop, dict) else DEFAULT_RISK["stop"]
    target = target if isinstance(target, dict) else DEFAULT_RISK["target"]
    if stop.get("kind") == "atr":
        stop_text = (f"{plain_number(stop.get('mult', STOP_ATR_MULT_DEFAULT))}x "
                     f"the "
                     f"{plain_number(stop.get('n', STOP_ATR_N_DEFAULT))}-{unit} "
                     f"average true range {stop_side}")
    else:
        stop_text = (f"{plain_number(stop.get('pct', STOP_PCT_DEFAULT))}% "
                     f"{stop_side}")
    if target.get("kind") == "rr":
        target_text = (f"{plain_number(target.get('rr', TARGET_RR_DEFAULT))}x "
                       "the risk")
    else:
        target_text = (f"{plain_number(target.get('pct', TARGET_PCT_DEFAULT))}% "
                       f"{target_side}")
    return [f"Stop {stop_text}.", f"Target {target_text}."]


def rule_sentence(spec: dict, vocabulary: Vocabulary | None = None) -> str:
    """Plain sentences restating a rule spec for a person. The spec's name is
    not repeated: every surface showing this already shows the label."""
    vocabulary = resolve_vocabulary(vocabulary)
    if not isinstance(spec, dict):
        return "Not a strategy spec."
    tf = spec.get("timeframe", "1h")
    reader = _Reader(vocabulary, tf)
    chart = f"the {_frame(tf)} chart"
    sentences = []
    for side in ("long", "short"):
        if side in spec:
            sentences.append(f"{side.capitalize()} on {chart} when "
                             f"{reader.group(spec[side])}.")
    exits = spec.get("exits")
    if isinstance(exits, dict):
        sentences.extend(_exits(reader, exits))
    risk = spec.get("risk", DEFAULT_RISK)
    sentences.extend(_risk(reader, risk if isinstance(risk, dict) else {}, spec))
    if reader.low_iex:
        sentences.append(LOW_IEX_DISCLOSURE)
    return " ".join(sentences)
