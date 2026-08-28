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

It also runs the TOP-LEVEL program, which is how the cross-timeframe lowering
is established. A play off the driving frame is a request, a gate and a latch
working together, and no one of them means anything alone: the request carries
no offset, which is only sound because the gate reads it where the requested
bar closes, and the latch is what stops any other bar reading it. Whether that
lands on the bars the engine decides on is not a property of any line, so
run_program executes the emitted statements over a chart frame with companion
higher-timeframe frames and models request.security the way TradingView aligns
it: by CONTAINMENT, so a chart bar sees the higher-timeframe bar it falls
inside. That containment is the whole reason the naive confirmed form lands a
bar late, so modelling it rather than assuming it is the point.

Scope, deliberately narrow. This is the subset those helpers use and nothing
more: `var` persistence, `:=`, `+=`, `=`, tuple unpacking, typed declarations,
if/else if/else, `for` with `break`, ternaries, history indexing on the built-in
series and on a function's own locals, and the handful of `math.*`, `array.*`,
`input.*`, `time_close` and `request.security` calls the emitted programs use.
Two known simplifications, both unreachable from the helpers as written: a
comparison against `na` reads false rather than `na` (Pine reads that false in a
condition, which is where every one of them sits), and a function's locals are
one flat scope per call rather than one per block (every helper declares in the
outer block and assigns inward, which is the only legal shape for the
carry-forward they need). Do not grow this into a general Pine engine: when a
helper needs something new, add exactly that. In particular there is no `ta.*`
here, so a program to be run must be built from sources, numbers and the
registry's own helpers.

Pine semantics that ARE modelled, because a helper leans on each:
- `var` initialises once per call site and survives the bar; a plain local does
  not, and that difference is the swing carry-forward.
- every call SITE gets its own instance, so two helpers calling
  nk_new_session keep two histories, exactly as a chart would.
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
            args, named = [], {}
            if not self.accept(")"):
                self._argument(args, named)
                while self.accept(","):
                    self._argument(args, named)
                self.expect(")")
            # One id per call SITE, so each keeps its own instance state.
            self.sites.append(token)
            return ("call", token, args, len(self.sites) - 1, named)
        return ("name", token)

    def _argument(self, args: list, named: dict) -> None:
        """One argument, positional or `name=value`.

        A named argument's VALUE is kept as the tokens that spelled it rather
        than evaluated. Every one the emitted programs pass is a barmerge enum,
        which is a setting rather than a number, and the only reader that cares
        (request.security) wants to check the spelling.
        """
        if (self.peek(1) == "=" and self.peek() is not None
                and self.peek().isidentifier()):
            key = self.take()
            self.expect("=")
            start, depth = self.i, 0
            while not self.done() and not (depth == 0
                                           and self.peek() in (",", ")")):
                depth += {"(": 1, ")": -1}.get(self.take(), 0)
            named[key] = "".join(self.t[start:self.i])
            return
        args.append(self.expr())


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
        closer = tokens.index("]") if head == "[" and "]" in tokens else -1
        if head == "break":
            out.append(("break",))
        elif closer > 0 and tokens[closer + 1:closer + 2] == ["="]:
            out.append(("unpack", [t for t in tokens[1:closer] if t != ","],
                        _parse_expr(tokens[closer + 2:], sites)))
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
        elif tokens[1:2] == ["="]:
            # A bare assignment. No helper writes one (they all declare a type),
            # but every top-level calculation an artifact emits is one.
            out.append(("let", head, _parse_expr(tokens[2:], sites)))
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

    def __init__(self, bars: pd.DataFrame, step: pd.Timedelta | None = None):
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
        # `time_close` is the bar's own close, which is `time` plus one step. It
        # exists only when a step is given, so a program reading it over a frame
        # whose cadence was never declared fails by name instead of silently
        # comparing a bar's open against another bar's close.
        if step is not None:
            self.series["time_close"] = (self.series["time"]
                                         + step / pd.Timedelta(milliseconds=1))
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
        # One Market per requested timeframe, and per timeframe the index of
        # the bar CONTAINING each chart bar, which is how TradingView aligns a
        # request. -1 before the first one exists.
        self.markets: dict[str, Market] = {}
        self.maps: dict[str, np.ndarray] = {}
        # Whether the mapped requested bar actually contains each chart bar.
        # A missing expected bar maps to the previous requested row for
        # gaps_off carry-forward, while gaps_on reads this bit and yields na.
        self.coverage: dict[str, np.ndarray] = {}
        # Calendar-derived closes for time_close(tf). They belong to the chart
        # cadence, so a missing bar in a requested symbol cannot erase the
        # visibility gate for that expected interval.
        self.calendar_closes: dict[str, np.ndarray] = {}
        self._requested: dict[int, list] = {}

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
            self._assign(name, value, scope, frame)
        elif kind == "unpack":
            _, names, node = statement
            values = self._eval(node, scope, frame, key)
            if not isinstance(values, tuple) or len(values) != len(names):
                raise PineError(f"cannot unpack {values!r} into {names}")
            for name, value in zip(names, values):
                self._assign(name, value, scope, frame)
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

    def _assign(self, name: str, value, scope: dict, frame: "_Frame") -> None:
        scope[name] = value
        if name in frame.persistent:
            frame.persistent[name] = value

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
            _, name, arg_nodes, site, named = node
            if name == "request.security":
                # The expression is NOT evaluated here: it belongs to the
                # requested context, which is the whole content of a request.
                return self._request(arg_nodes, named, site)
            args = [self._eval(a, scope, frame, key) for a in arg_nodes]
            if name in self.helpers:
                return self.call(name, args, (key, site))
            return self._builtin(name, args)
        raise PineError(f"unsupported node {node!r}")

    def _request(self, arg_nodes, named, site):
        """request.security, aligned by containment the way TradingView aligns it.

        The requested function runs over the requested frame, once, so a `var`
        inside it and an offset applied to its own locals both behave as they
        would on that timeframe's chart. What each chart bar then reads is the
        value at the higher-timeframe bar it falls INSIDE, which is
        barmerge.lookahead_on: on any bar but the last of that period it is the
        bar's finished value before the bar finished, and that is exactly why
        the emitted program reads it only on the bar where the period ends.
        """
        if named.get("lookahead") != "barmerge.lookahead_on":
            raise PineError("only barmerge.lookahead_on is modelled, got "
                            f"{named.get('lookahead')!r}")
        gaps = named.get("gaps")
        if gaps not in ("barmerge.gaps_on", "barmerge.gaps_off"):
            raise PineError("only barmerge gap modes are modelled, got "
                            f"{gaps!r}")
        if len(arg_nodes) != 3 or arg_nodes[1][0] != "str":
            raise PineError("request.security wants (symbol, literal tf, expr)")
        tf, call = arg_nodes[1][1], arg_nodes[2]
        if call[0] != "call" or call[2]:
            raise PineError("the requested expression must be a bare call")
        if tf not in self.markets:
            raise PineError(f"no frame was supplied for the {tf!r} request")
        if site not in self._requested:
            market = self.markets[tf]
            sub = Runtime(market, self.helpers)
            sub.markets, sub.maps = self.markets, self.maps
            sub.coverage = self.coverage
            sub.calendar_closes = self.calendar_closes
            values = []
            for bar in range(market.length):
                sub.bar = bar
                values.append(sub.call(call[1], [], ("requested", site)))
            self._requested[site] = values
        values = self._requested[site]
        position = int(self.maps[tf][self.bar])
        if (position < 0
                or (gaps == "barmerge.gaps_on"
                    and not self.coverage[tf][self.bar])):
            head = values[0] if values else NA
            return tuple(NA for _ in head) if isinstance(head, tuple) else NA
        return values[position]

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
        if name in ("input.int", "input.float"):
            # A run reads every knob at its shipped default, which is the
            # program a user pastes before touching anything.
            return args[0]
        if name == "time_close":
            # The close of the bar of `args[0]` that this chart bar sits in.
            # No request: Pine derives it from the calendar, which is why the
            # gate it spells costs nothing.
            return self.calendar_closes[args[0]][self.bar]
        if name == "int":
            return int(args[0])
        if name == "math.max":
            return max(args) if not any(map(_is_na, args)) else NA
        if name == "math.min":
            return min(args) if not any(map(_is_na, args)) else NA
        if name == "math.abs":
            return abs(args[0])
        if name in ("year", "month", "weekofyear", "dayofmonth", "hour",
                    "minute", "dayofweek"):
            if _is_na(args[0]):
                return NA
            stamp = market.stamp(args[0], args[1])
            if name == "dayofweek":
                # Pine numbers Sunday 1 through Saturday 7; pandas Monday 0.
                return (stamp.dayofweek + 1) % 7 + 1
            if name == "weekofyear":
                return stamp.isocalendar().week
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


def _split_program(lines: list[str]):
    """One artifact's body, split into its statements and its own functions.

    A generated function sits among the top-level statements rather than above
    them, so the split is by shape: a line at column zero ending in `=>` opens
    one, and everything indented under it is its body.
    """
    statements: list[str] = []
    functions: dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("//"):
            i += 1
            continue
        if _indent(line) == 0 and line.rstrip().endswith("=>"):
            body, i = [line], i + 1
            while i < len(lines) and (not lines[i].strip() or _indent(lines[i])):
                if lines[i].strip():
                    body.append(lines[i])
                i += 1
            functions[body[0].split("(")[0].strip()] = "\n".join(body)
            continue
        statements.append(line)
        i += 1
    return statements, functions


def run_program(sources: dict[str, str], lines: list[str], bars: pd.DataFrame,
                step: pd.Timedelta, frames=None) -> list[dict]:
    """Every top-level statement of an emitted artifact, over `bars`, bar by bar.

    `frames` maps a Pine timeframe string to the (frame, step) a request of it
    reads. A chart bar is mapped to the requested bar it falls inside by plain
    searchsorted, which is TradingView's containment for an intraday request
    and, for a daily one over regular-hours bars, the same grouping the engine
    uses: a daily label is UTC midnight of its New York date, and every session
    bar of that date is later in the same UTC date.

    Returns one dict per bar: every identifier that bar assigned, so a test can
    read the decision the artifact reached rather than the text that reached it.
    """
    statements, inline = _split_program(lines)
    helpers = {name: PineFunction(source)
               for name, source in {**sources, **inline}.items()}
    runtime = Runtime(Market(bars, step), helpers)
    for timeframe, (frame, delta) in (frames or {}).items():
        runtime.markets[timeframe] = Market(frame, delta)
        positions = np.asarray(
            frame.index.searchsorted(bars.index, side="right")) - 1
        runtime.maps[timeframe] = positions
        covered = np.zeros(len(bars), dtype=bool)
        mapped = positions >= 0
        if mapped.any():
            closes = frame.index + delta
            covered[mapped] = np.asarray(
                bars.index[mapped] < closes[positions[mapped]])
        runtime.coverage[timeframe] = covered
        delta_ms = int(delta / pd.Timedelta(milliseconds=1))
        chart_ms = runtime.market.series["time"].astype("int64")
        runtime.calendar_closes[timeframe] = (
            (chart_ms // delta_ms + 1) * delta_ms).astype("float64")
    sites: list[str] = []
    body, _rest = _block(statements, 0, 0, sites)
    state = _Frame()
    out = []
    for bar in range(len(bars)):
        runtime.bar = bar
        scope: dict[str, object] = {}
        runtime._body(body, scope, state, ("program",))
        state.commit(scope)
        out.append(dict(scope))
    return out


def as_series(values: list, bars: pd.DataFrame) -> pd.Series:
    """One run's output as a float Series on the frame's own index."""
    return pd.Series([NA if v is None or _is_na(v) else float(v)
                      for v in values], index=bars.index, dtype="float64")
