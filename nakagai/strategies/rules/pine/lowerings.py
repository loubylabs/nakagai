"""The Pine form of every ordinary term, plus the helpers they lean on.

One emit function per shape, wired to its term in vocabulary.py so a name's
Pine form sits in the same statement as its executable one. Each function is
handed the compiler context and the call being lowered, and returns the single
expression the surrounding condition reads.

Where Pine ships the same measurement, it is used: the built-ins are what a
TradingView user can inspect and trust, and they are what the platform's own
charts already draw. Where the two libraries genuinely disagree, the difference
is closed here rather than papered over, and the comment says which way:
donchian excludes the current bar, ta.supertrend signs its direction the other
way around, and adx reads one member of ta.dmi.

This module imports the model and nothing else. vocabulary.py imports it, so
anything it reached for from higher up would close an import cycle.
"""

from nakagai.strategies.rules.pine.model import PineExpr, PineHelper

# The zero-safe divide. frame_eval._math maps a zero denominator to NaN rather
# than raising or producing an infinity, because a condition over NaN reads
# False, which is the honest answer for a ratio that does not exist on that
# bar. Pine reads `na` in a comparison the same way.
DIV = "nk_div"

HELPERS: dict[str, PineHelper] = {
    helper.id: helper for helper in (
        PineHelper(DIV, f"{DIV}(a, b) => b == 0 ? na : a / b"),
    )
}


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


def emit_builtin(name: str):
    """A Pine built-in series that takes no arguments: ta.obv."""

    def emit(ctx, call):
        return PineExpr(ctx.calc(call, name))

    return emit


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
