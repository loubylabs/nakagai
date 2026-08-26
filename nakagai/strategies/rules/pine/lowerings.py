"""The Pine form of every term, plus the helpers they lean on.

One emit function per shape, wired to its term in vocabulary.py so a name's
Pine form sits in the same statement as its executable one. Each function is
handed the compiler context and the call being lowered, and returns the single
expression the surrounding condition reads.

For the INDICATORS, where Pine ships the same measurement it is used: the
built-ins are what a TradingView user can inspect and trust, and they are what
the platform's own charts already draw. Where the two libraries genuinely
disagree, the difference is closed here rather than papered over, and the
comment says which way: donchian excludes the current bar, ta.supertrend signs
its direction the other way around, and adx reads one member of ta.dmi.

For the PRIMITIVES that rule inverts, and the second half of this file is
written the other way round. A session window, a confirmed swing, a fair value
gap's lifecycle and an order block are Nakagai's own definitions, and each one
encodes a causality rule about which bar may first read the value. Pine ships
functions with those names attached to different rules: ta.pivothigh confirms
elsewhere, the session variables follow the exchange's settings rather than the
New York calendar day the engine groups by. So every primitive helper below is
a translation of the algorithm in strategies/rules/primitives.py (and the ICT
modules it calls), line for line, and the comment on each one names what it
mirrors. A built-in with the same name would pass every shape test and trade
differently.

This module imports the model and nothing else. vocabulary.py imports it, so
anything it reached for from higher up would close an import cycle.
"""

from nakagai.strategies.rules.pine.model import PineExpr, PineHelper

# The zero-safe divide. frame_eval._math maps a zero denominator to NaN rather
# than raising or producing an infinity, because a condition over NaN reads
# False, which is the honest answer for a ratio that does not exist on that
# bar. Pine reads `na` in a comparison the same way.
DIV = "nk_div"

# The primitives' helpers. Each is named after the term it answers, so a
# lowering never reaches for another term's helper and the id a program emits
# says which primitive put it there. The ones without a term of their own are
# the shared machinery below them.
SESSION_KEY = "nk_session_key"
NEW_SESSION = "nk_new_session"
SESSION_OPEN = "nk_session_open"
IN_SESSION = "nk_in_session"
SESSION_OPEN_BAR = "nk_session_open_bar"
WINDOW_ATR = "nk_window_atr"
GAP_PCT = "nk_gap_pct"
DAY_OF_WEEK = "nk_day_of_week"
MINUTES_INTO_SESSION = "nk_minutes_into_session"
RVOL = "nk_rvol"
SWING_HIGH = "nk_swing_high"
SWING_LOW = "nk_swing_low"
BARS_SINCE = "nk_bars_since"
LEG_RETRACE = "nk_leg_retrace"
FVG_NEAREST = "nk_fvg_nearest"
ORDER_BLOCK = "nk_order_block"


def emit_series_call(fn: str, *args: str):
    """A Pine call over the term's source series: ta.sma(close, n)."""

    def emit(ctx, call):
        parts = ", ".join([call.source, *(ctx.arg(call, a) for a in args)])
        return PineExpr(ctx.calc(call, f"{fn}({parts})"))

    return emit


def emit_bar_call(fn: str, *args: str, source: str = ""):
    """A Pine call over the bars themselves: ta.atr(n), or ta.cci(hlc3, n).

    `source` is the fixed series a full-bar term measures, spelled the way the
    engine's own function does: cci and mfi both work off the typical price,
    which is Pine's hlc3.
    """

    def emit(ctx, call):
        parts = ", ".join([*([source] if source else []),
                           *(ctx.arg(call, a) for a in args)])
        return PineExpr(ctx.calc(call, f"{fn}({parts})"))

    return emit


def emit_obv(ctx, call):
    # Both cumulate signed volume, but from different starting points:
    # indicators.obv starts at the first bar the engine loaded, ta.obv at the
    # start of the chart's history. The shape of the line agrees; its level
    # does not, so a condition reading the level (obv > 0) can differ.
    ctx.warn("ta.obv cumulates from the start of the chart's history, while "
             "the engine cumulates from the first bar it loaded, so the two "
             "lines share a shape but not a level.")
    return PineExpr(ctx.calc(call, "ta.obv"))


def emit_zscore(ctx, call):
    n = ctx.arg(call, "n")
    src = call.source
    # ta.stdev is the population deviation by default, which is the ddof=0 that
    # indicators.zscore divides by.
    return PineExpr(ctx.calc(
        call, f"{DIV}({src} - ta.sma({src}, {n}), ta.stdev({src}, {n}))"))


def emit_macd(ctx, call):
    fast, slow, signal = (ctx.arg(call, a) for a in ("fast", "slow", "signal"))
    parts = ctx.destructure(call, ("macd", "signal", "hist"),
                            f"ta.macd({call.source}, {fast}, {slow}, {signal})")
    return PineExpr(ctx.fields(call, parts)[call.field])


def emit_bb(ctx, call):
    n, k = ctx.arg(call, "n"), ctx.arg(call, "k")
    # ta.bb returns [middle, upper, lower]; the destructure follows Pine's
    # order, and the field names follow the term's.
    parts = ctx.destructure(call, ("mid", "upper", "lower"),
                            f"ta.bb({call.source}, {n}, {k})")
    return PineExpr(ctx.fields(call, parts)[call.field])


def emit_donchian(ctx, call):
    n = ctx.arg(call, "n")
    # indicators.donchian shifts the channel by one bar, so a close above the
    # upper edge is a genuine breakout of PAST highs rather than of its own.
    upper = ctx.local(call, "upper", f"ta.highest(high, {n})[1]")
    lower = ctx.local(call, "lower", f"ta.lowest(low, {n})[1]")
    mid = ctx.local(call, "mid", f"({upper} + {lower}) / 2")
    fields = ctx.fields(call, {"upper": upper, "lower": lower, "mid": mid})
    return PineExpr(fields[call.field])


def emit_supertrend(ctx, call):
    n, mult = ctx.arg(call, "n"), ctx.arg(call, "mult")
    # ta.supertrend reports -1 for an up-trend and +1 for a down-trend;
    # indicators.supertrend reports the opposite. Negating here keeps a spec
    # written against the engine reading the same way on the chart.
    parts = ctx.destructure(call, ("line", "raw_direction"),
                            f"ta.supertrend({mult}, {n})")
    direction = ctx.local(call, "direction", f"-{parts['raw_direction']}")
    fields = ctx.fields(call, {"line": parts["line"], "direction": direction})
    return PineExpr(fields[call.field])


def emit_vwap(ctx, call):
    ctx.warn("ta.vwap anchors to the chart's own session, which follows the "
             "exchange's settings rather than the engine's New York session.")
    return PineExpr(ctx.calc(call, "ta.vwap"))


def emit_stoch(ctx, call):
    n, d = ctx.arg(call, "n"), ctx.arg(call, "d")
    k = ctx.local(call, "k", f"ta.stoch(close, high, low, {n})")
    smoothed = ctx.local(call, "d", f"ta.sma({k}, {d})")
    return PineExpr(ctx.fields(call, {"k": k, "d": smoothed})[call.field])


def emit_adx(ctx, call):
    n = ctx.arg(call, "n")
    # indicators.adx smooths the directional index and the ADX itself over the
    # same n, which is ta.dmi's two lengths given the same input.
    parts = ctx.destructure(call, ("di_plus", "di_minus", "adx"),
                            f"ta.dmi({n}, {n})")
    return PineExpr(parts["adx"])


def emit_ichimoku(ctx, call):
    tenkan_n, kijun_n, senkou_n, disp = (
        ctx.arg(call, a) for a in ("tenkan_n", "kijun_n", "senkou_n", "disp"))
    tenkan = ctx.local(call, "tenkan", _midline(tenkan_n))
    kijun = ctx.local(call, "kijun", _midline(kijun_n))
    # disp is an input rather than a constant, so TradingView cannot infer how
    # far back the two senkou lines reach and answers "Pine cannot determine
    # the referencing length of series" unless the script declares
    # max_bars_back. The lines index [disp], so the buffer is disp + 1.
    ctx.needs_history(call, "disp")
    # The senkou lines are displaced forward, so each bar carries the cloud
    # that applies to IT: `[disp]` is exactly indicators.ichimoku's .shift(disp).
    # Both go through a named line first, because Pine indexes an identifier
    # rather than a parenthesized expression.
    a_base = ctx.local(call, "senkou_a_base", f"({tenkan} + {kijun}) / 2")
    senkou_a = ctx.local(call, "senkou_a", f"{a_base}[{disp}]")
    b_base = ctx.local(call, "senkou_b_base", _midline(senkou_n))
    senkou_b = ctx.local(call, "senkou_b", f"{b_base}[{disp}]")
    fields = ctx.fields(call, {"tenkan": tenkan, "kijun": kijun,
                               "senkou_a": senkou_a, "senkou_b": senkou_b})
    return PineExpr(fields[call.field])


def emit_keltner(ctx, call):
    n, mult = ctx.arg(call, "n"), ctx.arg(call, "mult")
    mid = ctx.local(call, "mid", f"ta.ema(close, {n})")
    band = ctx.local(call, "band", f"{mult} * ta.atr({n})")
    upper = ctx.local(call, "upper", f"{mid} + {band}")
    lower = ctx.local(call, "lower", f"{mid} - {band}")
    fields = ctx.fields(call, {"upper": upper, "mid": mid, "lower": lower})
    return PineExpr(fields[call.field])


def _midline(n: str) -> str:
    return f"(ta.highest(high, {n}) + ta.lowest(low, {n})) / 2"


# -- the stateful primitives ----------------------------------------------


def _flag(value: bool) -> str:
    """A choice, as the Pine literal that bakes it into the generated code."""
    return "true" if value else "false"


def emit_primitive(helper: str, *args: str):
    """A primitive as one call of its own helper: nk_gap_pct()."""

    def emit(ctx, call):
        if helper == GAP_PCT:
            previous = _window_value(
                ctx, call, source="close", reducer="last", suffix="prior")
            return PineExpr(ctx.calc(call, f"{helper}({previous})"))
        parts = ", ".join(ctx.arg(call, a) for a in args)
        return PineExpr(ctx.calc(call, f"{helper}({parts})"))

    return emit


def _name(ctx, call, suffix: str) -> str:
    return ctx.claim(f"{call.slot}_{suffix}", call.path)


def _reduce(lines: list[str], current: str, source: str,
            reducer: str) -> None:
    """Append one reducer update over a non-NaN source value."""
    lines.append(f"if not na({source})")
    if reducer == "first":
        lines.append(f"    if na({current})")
        lines.append(f"        {current} := {source}")
    elif reducer == "last":
        lines.append(f"    {current} := {source}")
    elif reducer in ("max", "min"):
        lines.append(
            f"    {current} := na({current}) ? {source} : "
            f"math.{reducer}({current}, {source})")
    else:
        raise ValueError(f"unknown Pine window reducer {reducer!r}")


def _window_value(ctx, call, *, source: str, reducer: str,
                  suffix: str = "window") -> str:
    """Emit the one recurring-window state machine and return its value."""
    window = call.window
    if window is None:
        raise ValueError("a Pine window state machine needs a resolved row")

    tz = f'"{window.tz}"'
    start = window.start.hour * 60 + window.start.minute
    end = window.end.hour * 60 + window.end.minute
    clock = _name(ctx, call, f"{suffix}_clock")
    current = _name(ctx, call, f"{suffix}_current")
    completed = _name(ctx, call, f"{suffix}_completed")
    lines = [
        f"int {clock} = hour(time, {tz}) * 60 + minute(time, {tz})",
        f"var float {current} = na",
        f"var float {completed} = na",
    ]

    if window.recurrence in ("weekday", "xnys_session"):
        owner_time = _name(ctx, call, f"{suffix}_owner_time")
        owner_candidate = _name(ctx, call, f"{suffix}_owner_candidate")
        key = _name(ctx, call, f"{suffix}_key")
        weekday = _name(ctx, call, f"{suffix}_weekday")
        occurrence = _name(ctx, call, f"{suffix}_occurrence")
        closed = _name(ctx, call, f"{suffix}_closed")
        reached = _name(ctx, call, f"{suffix}_reached")
        fresh = _name(ctx, call, f"{suffix}_fresh")
        active = _name(ctx, call, f"{suffix}_active")
        closes = _name(ctx, call, f"{suffix}_closes")
        owner = f"{clock} < {start} ? time - 86400000 : time"
        inside = (f"{clock} >= {start} and {clock} < {end}"
                  if start < end else
                  f"({clock} >= {start} or {clock} < {end})")
        after = (f"{clock} >= {end}"
                 if start < end else
                 f"{clock} >= {end} and {clock} < {start}")
        if window.recurrence == "weekday":
            candidate_weekday = _name(
                ctx, call, f"{suffix}_owner_candidate_weekday")
            weekend_days = _name(ctx, call, f"{suffix}_weekend_days")
            owner_lines = [
                f"int {owner_candidate} = {owner}",
                f"int {candidate_weekday} = dayofweek({owner_candidate}, {tz})",
                f"int {weekend_days} = {candidate_weekday} == 1 ? 2 : "
                f"{candidate_weekday} == 7 ? 1 : 0",
                f"int {owner_time} = {owner_candidate} - "
                f"{weekend_days} * 86400000",
                f"int {key} = year({owner_time}, {tz}) * 10000 + "
                f"month({owner_time}, {tz}) * 100 + "
                f"dayofmonth({owner_time}, {tz})",
                f"int {weekday} = dayofweek({owner_time}, {tz})",
            ]
            observed = f"{weekday} >= 2 and {weekday} <= 6"
            fresh_guard = ""
            active_owner = (
                f" and {candidate_weekday} >= 2 and {candidate_weekday} <= 6")
            closes_when = (
                f"{occurrence} == {key} and "
                f"({after} or {weekend_days} > 0)")
        else:
            calendar_key = _name(ctx, call, f"{suffix}_calendar_key")
            calendar_weekday = _name(
                ctx, call, f"{suffix}_calendar_weekday")
            regular = _name(ctx, call, f"{suffix}_regular")
            observed_session = _name(
                ctx, call, f"{suffix}_observed_session")
            owner_lines = [
                f"int {calendar_key} = year(time, {tz}) * 10000 + "
                f"month(time, {tz}) * 100 + dayofmonth(time, {tz})",
                f"int {calendar_weekday} = dayofweek(time, {tz})",
                f"bool {regular} = {calendar_weekday} >= 2 and "
                f"{calendar_weekday} <= 6 and {clock} >= 570 and "
                f"{clock} < 960",
                f"var int {observed_session} = na",
                f"if {regular}",
                f"    {observed_session} := {calendar_key}",
                f"int {key} = {observed_session}",
            ]
            observed = f"not na({observed_session})"
            fresh_guard = f" and {clock} >= {start}"
            active_owner = f" and {calendar_key} == {observed_session}"
            closes_when = (
                f"not na({occurrence}) and "
                f"({occurrence} != {calendar_key} or {after})")
        lines += [
            *owner_lines,
            f"var int {occurrence} = na",
            f"var bool {closed} = false",
            f"bool {reached} = {observed}",
            f"bool {closes} = {reached} and not {closed} and {closes_when}",
            f"if {closes}",
            f"    {completed} := {current}",
            f"    {closed} := true",
            f"bool {fresh} = {reached}{fresh_guard} and "
            f"(na({occurrence}) or {occurrence} != {key})",
            f"if {fresh}",
            f"    {occurrence} := {key}",
            f"    {current} := na",
            f"    {completed} := na",
            f"    {closed} := false",
            f"bool {active} = {reached}{active_owner} and "
            f"{occurrence} == {key} and {inside}",
            f"if {active}",
        ]
        reduced: list[str] = []
        _reduce(reduced, current, source, reducer)
        lines += [f"    {line}" for line in reduced]
        value = f"{active} ? na : {completed}"
    else:
        period = _name(ctx, call, f"{suffix}_period")
        key = _name(ctx, call, f"{suffix}_key")
        changed = _name(ctx, call, f"{suffix}_changed")
        inside = _name(ctx, call, f"{suffix}_active")
        weekday = _name(ctx, call, f"{suffix}_weekday")
        if window.recurrence == "prior_session":
            key_expr = (f"year(time, {tz}) * 10000 + month(time, {tz}) * 100 + "
                        f"dayofmonth(time, {tz})")
        elif window.recurrence == "prior_iso_week":
            iso_weekday = _name(ctx, call, f"{suffix}_iso_weekday")
            iso_anchor = _name(ctx, call, f"{suffix}_iso_anchor")
            lines += [
                f"int {iso_weekday} = (dayofweek(time, {tz}) + 5) % 7",
                f"int {iso_anchor} = time + (3 - {iso_weekday}) * 86400000",
            ]
            key_expr = (f"year({iso_anchor}, {tz}) * 100 + "
                        f"weekofyear(time, {tz})")
        else:
            key_expr = f"year(time, {tz}) * 100 + month(time, {tz})"
        active_expr = (f"{clock} >= {start} and {clock} < {end}"
                       if start < end else
                       f"({clock} >= {start} or {clock} < {end})")
        lines += [
            f"int {key} = {key_expr}",
            f"int {weekday} = dayofweek(time, {tz})",
            f"var int {period} = na",
            f"bool {changed} = na({period}) or {period} != {key}",
            f"if {changed}",
            f"    if not na({current})",
            f"        {completed} := {current}",
            f"    {current} := na",
            f"    {period} := {key}",
            f"bool {inside} = {weekday} >= 2 and {weekday} <= 6 and "
            f"{active_expr}",
            f"if {inside}",
        ]
        reduced = []
        _reduce(reduced, current, source, reducer)
        lines += [f"    {line}" for line in reduced]
        value = completed

    for line in lines:
        ctx.statement(line)
    answer = _name(ctx, call, f"{suffix}_value")
    ctx.statement(f"{answer} = {value}")
    return answer


def emit_window(ctx, call):
    """Lower any aggregate-capable term through its declared reducer."""
    value = _window_value(
        ctx, call, source=call.source, reducer=call.term.window_reduce)
    return PineExpr(ctx.calc(call, value))


def emit_swing(helper: str):
    """swing_high and swing_low: one confirmed pivot level, carried forward."""

    def emit(ctx, call):
        # The pivot under test sits k bars back and its own left window reaches
        # k further, so the deepest offset the helper indexes is [2k], which is
        # 2k + 1 values. Declaring only k would let TradingView size a buffer
        # the script reads past.
        ctx.needs_history(call, "k", reach=2)
        return PineExpr(ctx.calc(call, f"{helper}({ctx.arg(call, 'k')})"))

    return emit


def emit_leg_retrace(ctx, call):
    # Both swings, so the same 2k reach as the swings themselves.
    ctx.needs_history(call, "k", reach=2)
    want_long = _flag(ctx.choice(call, "direction") == "long")
    return PineExpr(ctx.calc(
        call, f"{LEG_RETRACE}({ctx.arg(call, 'k')}, {want_long})"))


def emit_bars_since(ctx, call):
    # call.source is the condition, already lowered by the walk that owns
    # conditions: an emit function is handed operands, never spec shapes.
    return PineExpr(ctx.calc(call, f"{BARS_SINCE}({call.source})"))


def emit_fvg_nearest(ctx, call):
    # The scan reads `lookback` bars ending at this one, so the count is the
    # buffer: [0] through [lookback - 1].
    ctx.needs_history(call, "lookback", window=True)
    size, lookback = ctx.arg(call, "min_size_atr"), ctx.arg(call, "lookback")
    direction, state = ctx.choice(call, "direction"), ctx.choice(call, "state")
    # "open" reads an unfilled gap of the trade's own direction; "inverted"
    # reads one of the OPPOSITE original direction whose far boundary a later
    # bar closed through, so the zone now supports this direction.
    want_long = _flag((direction == "long") == (state == "open"))
    parts = ctx.destructure(
        call, ("top", "bottom"),
        f"{FVG_NEAREST}({size}, {lookback}, {want_long}, "
        f"{_flag(state == 'inverted')})")
    return PineExpr(ctx.fields(call, _with_mid(ctx, call, parts))[call.field])


def emit_order_block(ctx, call):
    # Same window reading as the gap scan: [0] through [lookback - 1].
    ctx.needs_history(call, "lookback", window=True)
    body, lookback = ctx.arg(call, "body_atr"), ctx.arg(call, "lookback")
    want_long = _flag(ctx.choice(call, "direction") == "long")
    parts = ctx.destructure(call, ("top", "bottom"),
                            f"{ORDER_BLOCK}({body}, {lookback}, {want_long})")
    return PineExpr(ctx.fields(call, _with_mid(ctx, call, parts))[call.field])


def _with_mid(ctx, call, parts: dict[str, str]) -> dict[str, str]:
    """A zone's three readings, from the two boundaries its helper returns."""
    mid = ctx.local(call, "mid", f"({parts['top']} + {parts['bottom']}) / 2")
    return {**parts, "mid": mid}


# The Pine each primitive lowers to. `time` is the bar's opening timestamp in
# both engines, which is what the pandas index labels hold, so every rule below
# reads the same instant its Python counterpart reads.
_PRIMITIVE_HELPERS = (
    # _session_keys: one group per New York calendar day, which the engine
    # spells as that day's 09:30 open. The session is a calendar day in New
    # York, not the exchange's configured trading session, so the key is built
    # from the date parts rather than from `session.*`.
    PineHelper(SESSION_KEY, f"""{SESSION_KEY}() =>
    string tz = "America/New_York"
    year(time, tz) * 10000 + month(time, tz) * 100 + dayofmonth(time, tz)"""),
    # The chart's first bar starts a session: there is no earlier key for it to
    # differ from, and the engine's groupby gives that partial session its own
    # group rather than folding it into nothing.
    PineHelper(NEW_SESSION, f"""{NEW_SESSION}() =>
    int key = {SESSION_KEY}()
    na(key[1]) or key != key[1]""", (SESSION_KEY,)),
    # data/schema.session_opens: the 09:30 New York instant of the bar's OWN
    # session, which is what everything session-anchored measures from. NOT
    # the session's first bar, on either side: the engine's caches are not
    # RTH-only and a chart's extended session opens at 04:00, so the first bar
    # is a pre-market print in both and it is a different one in each.
    #
    # Read off the exchange wall clock, exactly as nk_session_open_bar reads it
    # below and as data/schema.rth_mask reads it on the engine side. hour() and
    # minute() have already resolved the New York offset, so 09:30 stays 09:30
    # through a daylight-saving change rather than drifting an hour, which is
    # the same reason session_opens puts its offset on naive local time. 570 is
    # 09:30 New York in minutes.
    #
    # The subtraction assumes the bar and its own 09:30 share a UTC offset,
    # which every bar of a session does: the US transitions happen at 02:00
    # local, so only a bar between local midnight and 02:00 could straddle one,
    # and both transition days are Sundays.
    PineHelper(SESSION_OPEN, f"""{SESSION_OPEN}() =>
    int into = hour(time, "America/New_York") * 60 + minute(time, "America/New_York") - 570
    time - into * 60000"""),
    # data/schema.rth_mask: whether the bar sits inside the REGULAR session,
    # which is what every session aggregate below is taken over. 570 is 09:30
    # New York in minutes and 960 is 16:00, and the window is half-open on the
    # right because a bar is labeled by its OPEN: the 15:45 bar is the last one
    # the session contains and a bar labeled 16:00 has already crossed into the
    # post-market.
    #
    # This is where the extended-hours difference between the two engines stops
    # mattering. A chart's extended session prints from 04:00 and Alpaca
    # returns the day's first actual trade, so the two disagree about which
    # pre-market bars exist; neither reaches an aggregate now, so the answer is
    # the same on both. rth_mask's session-frame branch has no counterpart
    # here and needs none: an export always charts 15m bars.
    PineHelper(IN_SESSION, f"""{IN_SESSION}() =>
    int into = hour(time, "America/New_York") * 60 + minute(time, "America/New_York")
    into >= 570 and into < 960"""),
    # util.first_bar_of_session: the driving bar a session-aligned play decides
    # on, which is the bar that OPENS a regular session rather than the one
    # that starts a calendar day. The two look equivalent and are not, and the
    # engine says why: the caches are not RTH-only, so on roughly half of a
    # three-year SPY cache the first bar of the date is a pre-market print at
    # 08:00, and gating there fires on a bar no live scanner ever visits. 570
    # is 09:30 New York in minutes. A session whose 09:30 bar is missing fires
    # on the first bar that IS there, one bar late, which is the engine's
    # answer rather than skipping the day.
    PineHelper(SESSION_OPEN_BAR, f"""{SESSION_OPEN_BAR}() =>
    var bool fired = false
    if {NEW_SESSION}()
        fired := false
    bool open_yet = hour(time, "America/New_York") * 60 + minute(time, "America/New_York") >= 570
    bool first = open_yet and not fired
    if first
        fired := true
    first""", (NEW_SESSION,)),
    # gap_pct: 100 * (this session's first REGULAR open - last session's
    # regular close) over that close. No zero guard, because the engine has
    # none either.
    #
    # na until the session's own 09:30 bar has printed, which is the engine's
    # `where(started)`: pandas broadcasts the group's first regular open over
    # the whole group, backwards included, so both sides need saying that a
    # pre-market bar cannot read an open that has not happened. Here it falls
    # out of the carried variable being na until the bar sets it.
    #
    # The two sides DIVERGE on a zero prior close, and that is recorded rather
    # than fixed: pandas answers inf and Pine answers na, so `gap_pct > 2`
    # would read True on the engine and False on the chart. Unreachable, since
    # a zero close is not a price any equity prints. Guarding it here would put
    # a branch in the artifact for a case neither side can meet, and guarding
    # it on both sides would be a change to engine arithmetic to serve an
    # export. If a zero close ever becomes reachable, this is the note.
    PineHelper(GAP_PCT, f"""{GAP_PCT}(previous) =>
    var float session_open = na
    if {NEW_SESSION}()
        session_open := na
    if {IN_SESSION}() and na(session_open)
        session_open := open
    100 * (session_open - previous) / previous""",
               (NEW_SESSION, IN_SESSION)),
    # day_of_week: 0 = Monday, the way pandas numbers a weekday. Pine numbers
    # Sunday 1 through Saturday 7, so the shift is +5 modulo 7.
    #
    # One line answers both of the primitive's branches, and only because the
    # primitive branches on the FRAME rather than on a label's clock. An
    # intraday chart converts the bar's own timestamp to New York, which is the
    # primitive's intraday reading exactly. A daily chart carries one bar per
    # session stamped at that session's start, so converting it to New York
    # lands inside the session and answers the same date the primitive reads
    # off the label's UTC date. The daily half rests on that stamp, which is
    # TradingView's for a US equity and is stated as an assumption on the
    # program rather than assumed silently (see SpecLowerer._assumptions).
    PineHelper(DAY_OF_WEEK, f"""{DAY_OF_WEEK}() =>
    (dayofweek(time, "America/New_York") + 5) % 7"""),
    # minutes_into_session: elapsed minutes from the 09:30 bell, and `na`
    # before it. A pre-market bar reads NEGATIVE against the bell, and a
    # negative passes every `< N` gate a session play writes, so the engine
    # blanks it and so does this: a condition over na is False on both sides,
    # which is what keeps a gated play off the pre-market. Past the close it
    # keeps counting, which the engine's docstring explains and no shipped
    # play reads differently.
    PineHelper(MINUTES_INTO_SESSION, f"""{MINUTES_INTO_SESSION}() =>
    float mins = (time - {SESSION_OPEN}()) / 60000.0
    mins >= 0 ? mins : na""", (SESSION_OPEN,)),
    # rvol: this bar's volume over the MEDIAN volume of the same clock time
    # across the trailing `sessions` sessions, the current one excluded. Three
    # choices the engine documents and this mirrors exactly. The bucket is the
    # EXCHANGE clock, so 09:30 stays one bucket across a daylight-saving
    # change. The baseline is read BEFORE this bar's volume enters its bucket,
    # which is the engine's shift by one OCCURRENCE rather than by one calendar
    # session: a session with no 15:45 bar is simply not in the 15:45 bucket.
    # And it is a median, not a mean, because one halted session inside the
    # window would drag a mean far enough to hide the next genuine surge.
    PineHelper(RVOL, f"""{RVOL}(sessions) =>
    var array<int> clocks = array.new<int>()
    var array<int> counts = array.new<int>()
    var array<float> ring = array.new<float>()
    int clock = hour(time, "America/New_York") * 60 + minute(time, "America/New_York")
    int slot = array.indexof(clocks, clock)
    if slot == -1
        array.push(clocks, clock)
        array.push(counts, 0)
        for i = 0 to sessions - 1
            array.push(ring, na)
        slot := array.size(clocks) - 1
    int base = slot * sessions
    int seen = array.get(counts, slot)
    float baseline = na
    if seen >= sessions
        array<float> window = array.new<float>()
        for i = 0 to sessions - 1
            array.push(window, array.get(ring, base + i))
        array.sort(window)
        int half = int(sessions / 2)
        baseline := sessions % 2 == 1 ? array.get(window, half) : (array.get(window, half - 1) + array.get(window, half)) / 2
    array.set(ring, base + seen % sessions, volume)
    array.set(counts, slot, seen + 1)
    na(baseline) or baseline <= 0 ? na : volume / baseline"""),
    # _swing over _strict_extrema: a pivot at i is strictly beyond every one of
    # the k bars on each side of it, and it is only KNOWN at i + k, so its
    # price is stamped there and carried forward until the next confirmation.
    # The pivot under test is therefore the bar at [k], its neighbours are at
    # [k + i] and [k - i], and the 2k of history the two windows need is what
    # `bar_index >= 2 * k` waits for: _strict_extrema returns an all-false mask
    # while either window is short.
    PineHelper(SWING_HIGH, f"""{SWING_HIGH}(k) =>
    var float level = na
    float pivot = high[k]
    bool ok = bar_index >= 2 * k
    if ok
        for i = 1 to k
            if not (pivot > high[k + i]) or not (pivot > high[k - i])
                ok := false
                break
    if ok
        level := pivot
    level"""),
    PineHelper(SWING_LOW, f"""{SWING_LOW}(k) =>
    var float level = na
    float pivot = low[k]
    bool ok = bar_index >= 2 * k
    if ok
        for i = 1 to k
            if not (pivot < low[k + i]) or not (pivot < low[k - i])
                ok := false
                break
    if ok
        level := pivot
    level"""),
    # leg_retrace: where the close sits inside the last CONFIRMED swing range.
    # 0 is at the swing high for a long, 1 a full retrace to the swing low, and
    # the ICT OTE band is [0.62, 0.79]. na until both swings exist or when the
    # range is not positive.
    PineHelper(LEG_RETRACE, f"""{LEG_RETRACE}(k, want_long) =>
    float hi = {SWING_HIGH}(k)
    float lo = {SWING_LOW}(k)
    float span = hi - lo
    span > 0 ? (want_long ? (hi - close) / span : (close - lo) / span) : na""",
               (SWING_HIGH, SWING_LOW)),
    # bars_since: pos - pos.where(mask).ffill(). Zero on the bar the condition
    # holds, and na before the first one: the no-hit value is not zero, and a
    # condition over na reads false where a condition over zero would fire.
    PineHelper(BARS_SINCE, f"""{BARS_SINCE}(hit) =>
    var float since = na
    since := hit ? 0.0 : (na(since) ? na : since + 1)
    since"""),
    # ict.primitives.atr: the mean of the true ranges inside the LOOKBACK
    # window, at most the last 14 of them. A window shorter than 15 bars
    # averages what it holds rather than reaching outside itself, which is the
    # `dropna().tail(14)` on a frame already cut to `lookback`.
    PineHelper(WINDOW_ATR, f"""{WINDOW_ATR}(bars) =>
    int n = math.min(14, bars - 1)
    float total = 0.0
    for i = 0 to n - 1
        total += math.max(high[i] - low[i], math.max(math.abs(high[i] - close[i + 1]), math.abs(low[i] - close[i + 1])))
    total / n"""),
    # find_fvgs and fvg_nearest: every qualifying 3-candle imbalance in the
    # window, with the lifecycle state a LATER bar gave it, and the one nearest
    # the last close. The scan walks candles newest first so the later-bar
    # extremes accumulate in one pass; `<=` on the distance therefore keeps the
    # OLDEST of two equally near gaps, which is what min() over the engine's
    # oldest-first list returns. A gap exists only once its third candle has
    # closed, so nothing here reads forward of the candle being tested.
    PineHelper(FVG_NEAREST, f"""{FVG_NEAREST}(min_size_atr, lookback, want_long, want_inverted) =>
    float unit = {WINDOW_ATR}(lookback)
    float best = na
    float best_top = na
    float best_bottom = na
    float later_low = na
    float later_high = na
    float later_close_min = na
    float later_close_max = na
    if not na(unit) and unit != 0
        for back = 0 to lookback - 3
            if back > 0
                int m = back - 1
                later_low := na(later_low) ? low[m] : math.min(later_low, low[m])
                later_high := na(later_high) ? high[m] : math.max(later_high, high[m])
                later_close_min := na(later_close_min) ? close[m] : math.min(later_close_min, close[m])
                later_close_max := na(later_close_max) ? close[m] : math.max(later_close_max, close[m])
            float top = na
            float bottom = na
            bool up = false
            if low[back] > high[back + 2] and low[back] - high[back + 2] >= min_size_atr * unit
                top := low[back]
                bottom := high[back + 2]
                up := true
            else if high[back] < low[back + 2] and low[back + 2] - high[back] >= min_size_atr * unit
                top := low[back + 2]
                bottom := high[back]
            if not na(top) and up == want_long
                bool inverted = up ? later_close_min < bottom : later_close_max > top
                bool filled = up ? later_low <= bottom : later_high >= top
                if want_inverted ? inverted : (not inverted and not filled)
                    float gap = math.min(math.abs(close - top), math.abs(close - bottom))
                    if na(best) or gap <= best
                        best := gap
                        best_top := top
                        best_bottom := bottom
    [best_top, best_bottom]""", (WINDOW_ATR,)),
    # order_block: the range of the last opposing candle before the most recent
    # displacement candle (a body of at least body_atr ATRs) inside the
    # lookback window. Both scans stop at the first hit, which is `disp_idx[-1]`
    # and `opp[-1]` read from the near end, and the second never starts before
    # the displacement candle: the opposing candle is strictly earlier.
    PineHelper(ORDER_BLOCK, f"""{ORDER_BLOCK}(body_atr, lookback, want_long) =>
    float unit = {WINDOW_ATR}(lookback)
    float top = na
    float bottom = na
    if not na(unit) and unit != 0
        int shove = -1
        for back = 0 to lookback - 1
            float body = close[back] - open[back]
            if want_long ? body >= body_atr * unit : body <= -body_atr * unit
                shove := back
                break
        if shove >= 0 and shove < lookback - 1
            for back = shove + 1 to lookback - 1
                float body = close[back] - open[back]
                if want_long ? body < 0 : body > 0
                    top := high[back]
                    bottom := low[back]
                    break
    [top, bottom]""", (WINDOW_ATR,)),
)

HELPERS: dict[str, PineHelper] = {
    helper.id: helper for helper in (
        PineHelper(DIV, f"{DIV}(a, b) => b == 0 ? na : a / b"),
        *_PRIMITIVE_HELPERS,
    )
}
