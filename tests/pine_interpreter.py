"""A bar-by-bar interpreter for the Pine the primitive helpers actually emit.

Why this exists. A causal lowering's whole content is WHEN a value becomes
readable, and that lives in two places a substring assertion cannot see: the
ORDER of a helper's lines, and whether a variable persists across bars. Dropping
`var` from a level that is stamped and carried forward, or rolling a session's
levels after the running aggregate has already been reset, changes what every
bar reads and changes not one character that `"..." in source` looks at. So the
net that establishes these lowerings has to RUN them.

It runs the emitted text, never a Python restatement of it. A hand-written copy
of what a helper is believed to do would agree with itself forever: it would not
move when the emitted source moved, which is the only thing worth catching here.

Scope, deliberately narrow. This is the subset those helpers use and nothing
more: `var` persistence, `:=`, `+=`, typed declarations, if/else if/else, `for`
with `break`, ternaries, history indexing on the built-in series and on a
function's own locals, and the handful of `math.*` and `array.*` calls in the
registry. Two known simplifications, both unreachable from the helpers as
written: a comparison against `na` reads false rather than `na` (Pine reads that
false in a condition, which is where every one of them sits), and a function's
locals are one flat scope per call rather than one per block (every helper
declares in the outer block and assigns inward, which is the only legal shape
for the carry-forward they need). Do not grow this into a general Pine engine:
when a helper needs something new, add exactly that.

Pine semantics that ARE modelled, because a helper leans on each:
- `var` initialises once per call site and survives the bar; a plain local does
  not, and that difference is the swing carry-forward.
- every call SITE gets its own instance, so nk_gap_pct calling nk_new_session
  and nk_prev_session_close (which calls it again) keeps two histories, exactly
  as a chart would.
- `for a to b` counts DOWN when a exceeds b, which is what nk_order_block's
  `shove < lookback - 1` guard exists to prevent.
"""

import math
import re

import numpy as np
import pandas as pd

NA = float("nan")

TYPES = ("int", "float", "bool", "string", "array_int", "array_float")
KEYWORDS = ("var", "if", "else", "for", "to", "break", "and", "or", "not",
            "true", "false")

_TOKEN = re.compile(r"""
      (?P<space>[ \t]+)
    | (?P<number>\d+\.\d*|\.\d+|\d+)
    | (?P<string>"[^"]*")
    | (?P<name>[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*)
    | (?P<op><=|>=|==|!=|:=|\+=|[-+*/%<>?:()\[\],=])
""", re.VERBOSE)

# Pine spells a typed array with angle brackets, which would tokenize as two
# comparisons. Both spellings are rewritten to plain names before tokenizing;
# nothing else in the subset uses `<` next to a name.
_ANGLES = {"array<int>": "array_int", "array<float>": "array_float",
           "array.new<int>": "array.new_int",
           "array.new<float>": "array.new_float"}


class PineError(RuntimeError):
    """The interpreter met something outside the subset, or a real error."""


def _tokenize(line: str) -> list[str]:
    for spelling, plain in _ANGLES.items():
        line = line.replace(spelling, plain)
    out, pos = [], 0
    while pos < len(line):
        match = _TOKEN.match(line, pos)
        if match is None:
            raise PineError(f"cannot tokenize {line[pos:]!r} in {line!r}")
        pos = match.end()
        if match.lastgroup != "space":
            out.append(match.group())
    return out


# -- expressions -----------------------------------------------------------
# Node shapes: ("num", v) ("str", s) ("bool", b) ("na",) ("name", n)
# ("index", name, expr) ("call", name, args, site) ("un", op, x)
# ("bin", op, l, r) ("?:", c, a, b) ("tuple", [nodes])

_BINARY = (("or",), ("and",), ("==", "!="), ("<", ">", "<=", ">="),
           ("+", "-"), ("*", "/", "%"))


class _Parser:
    """One line of Pine, into one expression tree."""

    def __init__(self, tokens: list[str], sites: list):
        self.t, self.i, self.sites = tokens, 0, sites

    def peek(self, ahead: int = 0):
        j = self.i + ahead
        return self.t[j] if j < len(self.t) else None

    def take(self) -> str:
        token = self.t[self.i]
        self.i += 1
        return token

    def accept(self, value: str) -> bool:
        if self.peek() == value:
            self.i += 1
            return True
        return False

    def expect(self, value: str) -> None:
        if not self.accept(value):
            raise PineError(f"expected {value!r}, got {self.peek()!r} "
                            f"in {' '.join(self.t)!r}")

    def done(self) -> bool:
        return self.i >= len(self.t)

    def expr(self):
        node = self._binary(0)
        if self.accept("?"):
            left = self.expr()
            self.expect(":")
            return ("?:", node, left, self.expr())
        return node

    def _binary(self, level: int):
        if level >= len(_BINARY):
            return self._unary()
        node = self._binary(level + 1)
        while self.peek() in _BINARY[level]:
            op = self.take()
            node = ("bin", op, node, self._binary(level + 1))
        return node

    def _unary(self):
        if self.accept("not"):
            return ("un", "not", self._unary())
        if self.accept("-"):
            return ("un", "-", self._unary())
        return self._postfix()

    def _postfix(self):
        node = self._primary()
        while self.peek() == "[":
            self.take()
            if node[0] != "name":
                raise PineError("Pine indexes an identifier, not an expression")
            node = ("index", node[1], self.expr())
            self.expect("]")
        return node

    def _primary(self):
        token = self.peek()
        if token is None:
            raise PineError(f"expression ended early in {' '.join(self.t)!r}")
        if token == "(":
            self.take()
            node = self.expr()
            self.expect(")")
            return node
        if token == "[":
            self.take()
            members = [self.expr()]
            while self.accept(","):
                members.append(self.expr())
            self.expect("]")
            return ("tuple", members)
        self.take()
        if token[0] == '"':
            return ("str", token[1:-1])
        if token[0].isdigit() or token[0] == ".":
            return ("num", float(token) if "." in token else int(token))
        if token in ("true", "false"):
            return ("bool", token == "true")
        if token == "na" and self.peek() != "(":
            return ("na",)
        if self.accept("("):
            args = []
            if not self.accept(")"):
                args.append(self.expr())
                while self.accept(","):
                    args.append(self.expr())
                self.expect(")")
            # One id per call SITE, so each keeps its own instance state.
            self.sites.append(token)
            return ("call", token, args, len(self.sites) - 1)
        return ("name", token)


def _parse_expr(tokens: list[str], sites: list):
    parser = _Parser(tokens, sites)
    node = parser.expr()
    if not parser.done():
        raise PineError(f"trailing tokens in {' '.join(tokens)!r}")
    return node


# -- statements ------------------------------------------------------------
# ("var", name, expr) ("let", name, expr) ("set", name, expr)
# ("add", name, expr) ("if", [(cond|None, body)]) ("for", name, lo, hi, body)
# ("break",) ("value", expr)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _block(lines: list[str], start: int, indent: int, sites: list):
    """Every statement at `indent`, and the index of the first line past it."""
    out, i = [], start
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        here = _indent(line)
        if here < indent:
            break
        if here > indent:
            raise PineError(f"unexpected indent at {line!r}")
        tokens = _tokenize(line.strip())
        head = tokens[0]
        if head in ("if", "else"):
            branches = []
            while i < len(lines) and lines[i].strip():
                tokens = _tokenize(lines[i].strip())
                if _indent(lines[i]) != indent or tokens[0] not in ("if", "else"):
                    break
                if tokens[0] == "if":
                    if branches:
                        break
                    cond = _parse_expr(tokens[1:], sites)
                elif tokens[1:2] == ["if"]:
                    cond = _parse_expr(tokens[2:], sites)
                else:
                    cond = None
                body, i = _block(lines, i + 1, indent + 4, sites)
                branches.append((cond, body))
                if cond is None:
                    break
            out.append(("if", branches))
            continue
        if head == "for":
            name = tokens[1]
            if tokens[2] != "=" or "to" not in tokens:
                raise PineError(f"unsupported for statement {line!r}")
            cut = tokens.index("to")
            lo = _parse_expr(tokens[3:cut], sites)
            hi = _parse_expr(tokens[cut + 1:], sites)
            body, i = _block(lines, i + 1, indent + 4, sites)
            out.append(("for", name, lo, hi, body))
            continue
        i += 1
        if head == "break":
            out.append(("break",))
        elif head == "var":
            name = tokens[2] if tokens[1] in TYPES else tokens[1]
            out.append(("var", name,
                        _parse_expr(tokens[tokens.index("=") + 1:], sites)))
        elif head in TYPES and tokens[2:3] == ["="]:
            out.append(("let", tokens[1], _parse_expr(tokens[3:], sites)))
        elif tokens[1:2] == [":="]:
            out.append(("set", head, _parse_expr(tokens[2:], sites)))
        elif tokens[1:2] == ["+="]:
            out.append(("add", head, _parse_expr(tokens[2:], sites)))
        else:
            out.append(("value", _parse_expr(tokens, sites)))
    return out, i


class PineFunction:
    """One helper source, parsed: its name, its parameters, and its body."""

    def __init__(self, source: str):
        lines = source.split("\n")
        header = _tokenize(lines[0].strip())
        arrow = next((i for i in range(len(header) - 1)
                      if header[i] == "=" and header[i + 1] == ">"), -1)
        if arrow < 0 or header[1] != "(" or header[arrow - 1] != ")":
            raise PineError(f"not a Pine function header: {lines[0]!r}")
        self.name = header[0]
        self.params = [t for t in header[2:arrow - 1] if t != ","]
        self.sites: list[str] = []
        rest = header[arrow + 2:]
        # A one-liner keeps its whole body on the header line; everything else
        # indents under it.
        self.body = ([("value", _parse_expr(rest, self.sites))] if rest
                     else _block(lines, 1, 4, self.sites)[0])


# -- evaluation ------------------------------------------------------------


def _is_na(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _truthy(value) -> bool:
    return False if _is_na(value) else bool(value)


class _Break(Exception):
    pass


class _Frame:
    """One call site's state: what `var` holds, and every local's history."""

    def __init__(self):
        self.persistent: dict[str, object] = {}
        self.history: dict[str, list] = {}

    def commit(self, scope: dict) -> None:
        for name, value in scope.items():
            self.history.setdefault(name, []).append(value)

    def past(self, name: str, back: int):
        past = self.history.get(name, ())
        return past[-back] if len(past) >= back else NA


class Market:
    """The bars a run is executed over, in the shapes Pine reads them."""

    def __init__(self, bars: pd.DataFrame):
        self.series = {name: bars[name].to_numpy(dtype="float64")
                       for name in ("open", "high", "low", "close", "volume")}
        # `time` is the bar's opening timestamp in epoch MILLISECONDS, which is
        # what Pine holds and what every date part below is read off. Measured
        # as a span rather than cast from the index's own integers: pandas 2
        # carries a resolution per index, so a frame built from second-unit
        # stamps would otherwise hand out epoch seconds and every session would
        # collapse into one.
        self.series["time"] = np.asarray(
            (bars.index - pd.Timestamp(0, tz="UTC"))
            // pd.Timedelta(milliseconds=1), dtype="float64")
        self.length = len(bars)
        self._stamps = {}

    def at(self, name: str, bar: int):
        if name == "bar_index":
            return bar
        if name not in self.series:
            raise PineError(f"{name!r} is neither a local nor a bar series")
        if bar < 0 or bar >= self.length:
            return NA
        value = self.series[name][bar]
        return float(value) if name != "time" else int(value)

    def stamp(self, milliseconds, tz: str) -> pd.Timestamp:
        key = (milliseconds, tz)
        if key not in self._stamps:
            self._stamps[key] = pd.Timestamp(
                int(milliseconds), unit="ms", tz="UTC").tz_convert(tz)
        return self._stamps[key]


class Runtime:
    """One program's execution over one market, bar by bar."""

    def __init__(self, market: Market, helpers: dict[str, PineFunction]):
        self.market = market
        self.helpers = helpers
        self.frames: dict[tuple, _Frame] = {}
        self.bar = 0

    def call(self, name: str, args: list, key: tuple):
        function = self.helpers.get(name)
        if function is None:
            raise PineError(f"no helper named {name!r}")
        if len(args) != len(function.params):
            raise PineError(f"{name} takes {function.params}, got {args!r}")
        frame = self.frames.setdefault(key, _Frame())
        scope = dict(zip(function.params, args))
        value = self._body(function.body, scope, frame, key)
        frame.commit(scope)
        return value

    def _body(self, body, scope, frame, key):
        value = None
        for statement in body:
            value = self._statement(statement, scope, frame, key)
        return value

    def _statement(self, statement, scope, frame, key):
        kind = statement[0]
        if kind == "var":
            _, name, node = statement
            if name not in frame.persistent:
                frame.persistent[name] = self._eval(node, scope, frame, key)
            scope[name] = frame.persistent[name]
        elif kind == "let":
            scope[statement[1]] = self._eval(statement[2], scope, frame, key)
        elif kind in ("set", "add"):
            _, name, node = statement
            value = self._eval(node, scope, frame, key)
            if kind == "add":
                value = scope[name] + value
            scope[name] = value
            if name in frame.persistent:
                frame.persistent[name] = value
        elif kind == "if":
            for cond, body in statement[1]:
                if cond is None or _truthy(self._eval(cond, scope, frame, key)):
                    return self._body(body, scope, frame, key)
        elif kind == "for":
            _, name, lo_node, hi_node, body = statement
            lo = int(self._eval(lo_node, scope, frame, key))
            hi = int(self._eval(hi_node, scope, frame, key))
            # Pine counts DOWN when the start is past the end, rather than
            # skipping the loop. nk_order_block's guard exists for this.
            step = 1 if lo <= hi else -1
            for counter in range(lo, hi + step, step):
                scope[name] = counter
                try:
                    self._body(body, scope, frame, key)
                except _Break:
                    break
        elif kind == "break":
            raise _Break()
        elif kind == "value":
            return self._eval(statement[1], scope, frame, key)
        return None

    def _eval(self, node, scope, frame, key):
        kind = node[0]
        if kind in ("num", "str", "bool"):
            return node[1]
        if kind == "na":
            return NA
        if kind == "name":
            name = node[1]
            if name in scope:
                return scope[name]
            return self.market.at(name, self.bar)
        if kind == "index":
            _, name, offset_node = node
            back = int(self._eval(offset_node, scope, frame, key))
            if name in self.market.series or name == "bar_index":
                return (scope[name] if name in scope and back == 0
                        else self.market.at(name, self.bar - back))
            if back == 0:
                return scope[name]
            return frame.past(name, back)
        if kind == "tuple":
            return tuple(self._eval(m, scope, frame, key) for m in node[1])
        if kind == "un":
            value = self._eval(node[2], scope, frame, key)
            return not _truthy(value) if node[1] == "not" else -value
        if kind == "?:":
            _, cond, left, right = node
            branch = left if _truthy(self._eval(cond, scope, frame, key)) else right
            return self._eval(branch, scope, frame, key)
        if kind == "bin":
            return self._binary(node, scope, frame, key)
        if kind == "call":
            _, name, arg_nodes, site = node
            args = [self._eval(a, scope, frame, key) for a in arg_nodes]
            if name in self.helpers:
                return self.call(name, args, (key, site))
            return self._builtin(name, args)
        raise PineError(f"unsupported node {node!r}")

    def _binary(self, node, scope, frame, key):
        _, op, left_node, right_node = node
        left = self._eval(left_node, scope, frame, key)
        if op == "and":
            return _truthy(left) and _truthy(
                self._eval(right_node, scope, frame, key))
        if op == "or":
            return _truthy(left) or _truthy(
                self._eval(right_node, scope, frame, key))
        right = self._eval(right_node, scope, frame, key)
        if op in ("==", "!=", "<", ">", "<=", ">="):
            # A comparison against na is na, which every one of these sits
            # inside a condition to read, and a condition reads na as false.
            if _is_na(left) or _is_na(right):
                return False
            return {"==": left == right, "!=": left != right,
                    "<": left < right, ">": left > right,
                    "<=": left <= right, ">=": left >= right}[op]
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return NA if right == 0 else left / right
        if op == "%":
            return left % right
        raise PineError(f"unsupported operator {op!r}")

    def _builtin(self, name: str, args: list):
        market = self.market
        if name == "na":
            return _is_na(args[0])
        if name == "int":
            return int(args[0])
        if name == "math.max":
            return max(args) if not any(map(_is_na, args)) else NA
        if name == "math.min":
            return min(args) if not any(map(_is_na, args)) else NA
        if name == "math.abs":
            return abs(args[0])
        if name in ("year", "month", "dayofmonth", "hour", "minute",
                    "dayofweek"):
            if _is_na(args[0]):
                return NA
            stamp = market.stamp(args[0], args[1])
            if name == "dayofweek":
                # Pine numbers Sunday 1 through Saturday 7; pandas Monday 0.
                return (stamp.dayofweek + 1) % 7 + 1
            return {"year": stamp.year, "month": stamp.month,
                    "dayofmonth": stamp.day, "hour": stamp.hour,
                    "minute": stamp.minute}[name]
        if name in ("array.new_int", "array.new_float"):
            return []
        if name == "array.push":
            args[0].append(args[1])
            return None
        if name == "array.get":
            return args[0][int(args[1])]
        if name == "array.set":
            args[0][int(args[1])] = args[2]
            return None
        if name == "array.size":
            return len(args[0])
        if name == "array.indexof":
            return args[0].index(args[1]) if args[1] in args[0] else -1
        if name == "array.sort":
            args[0].sort()
            return None
        raise PineError(f"unsupported built-in {name!r}")


def run_helper(sources: dict[str, str], entry: str, bars: pd.DataFrame,
               args=()) -> list:
    """`entry`'s value on every bar of `bars`, as a chart would compute it.

    An argument that is an array of the frame's own length is read per bar,
    which is how a condition reaches nk_bars_since; anything else is a constant
    the whole run, which is how an input reaches every other helper.
    """
    helpers = {helper_id: PineFunction(source)
               for helper_id, source in sources.items()}
    runtime = Runtime(Market(bars), helpers)
    out = []
    for bar in range(len(bars)):
        runtime.bar = bar
        row = [a[bar] if isinstance(a, (np.ndarray, pd.Series)) and len(a) == len(bars)
               else a for a in args]
        out.append(runtime.call(entry, row, ("root", entry)))
    return out


def as_series(values: list, bars: pd.DataFrame) -> pd.Series:
    """One run's output as a float Series on the frame's own index."""
    return pd.Series([NA if v is None or _is_na(v) else float(v)
                      for v in values], index=bars.index, dtype="float64")
