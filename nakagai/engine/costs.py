"""Execution costs: what a fill actually costs beyond the printed price.

Pillar 4 (Replay) in the platform's docs/internal/PILLARS.md. The replay loop's
contract is that its trades are what a real desk would plausibly have got, and
a desk pays two things the tape does not show: it crosses a spread, and it pays
its broker. Both live here so the engine states them as models rather than
burying a constant in an arithmetic expression.

Neither model is trying to be precise. They are trying to be *seams*: a place
where a better estimate can land later without touching the replay loop, and a
guarantee that "we assume zero" is a value somebody chose rather than an
omission nobody noticed.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SlippageModel:
    """Per-share slippage as basis points of the fill price, with a floor.

    A flat cent, which is what the engine used before, is 1bp on a $100 stock
    and 0.02bp on a $500 one. Across a universe with mixed price levels that
    makes the stored numbers incomparable: the same modeled friction is an
    order of magnitude harsher on the cheap names.

    The floor exists because slippage does not shrink to nothing on cheap
    stock; you still cross a spread and the tick is a cent. So below $100 the
    floor binds and above it the proportional term does, which is the behavior
    a desk actually sees.

    Defaults reproduce the old flat cent exactly at $100, so this change moves
    numbers for everything *except* a $100 stock, deliberately.
    """

    bps: float = 1.0
    min_per_share: float = 0.01

    def per_share(self, price: float) -> float:
        return max(self.min_per_share, abs(price) * self.bps / 10_000.0)

    def sell(self, price: float) -> float:
        """Fill price when leaving a long or entering a short: you get less."""
        return price - self.per_share(price)

    def buy(self, price: float) -> float:
        """Fill price when entering a long or covering a short: you pay more."""
        return price + self.per_share(price)


@dataclass(frozen=True)
class FeeModel:
    """Commissions and per-share fees, charged once on entry and once on exit.

    Zero by default, which is correct for the broker this platform trades
    through today. The seam still has to exist: without it the first broker
    that does charge silently invalidates every number in the evidence store,
    instead of changing one constant and re-proving.
    """

    per_trade: float = 0.0
    per_share: float = 0.0

    def charge(self, qty: int) -> float:
        """Round-trip cost for a position of `qty` shares."""
        return 2 * (self.per_trade + self.per_share * abs(qty))
