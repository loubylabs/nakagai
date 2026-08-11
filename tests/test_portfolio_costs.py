"""What one fill costs: the slippage crossed and the fee paid.

The policy and the model are one value. `ExecutionPolicy` carries a `FeeSpec`
and a `SlippageSpec`, and those two price a fill themselves, so there is no
second cost model that could drift from the policy a result reports.

The fee spec prices ONE fill. That is the other thing this file exists for: the
model it replaced returned a round trip from a single call, so a caller that
charged it at entry and again at exit paid four fills and nothing raised. Every
test below states a per-fill number, and the round-trip total only ever appears
as two charges added together.
"""

import pytest

from nakagai.engine.portfolio_types import FeeSpec, SlippageSpec


# ------------------------------------------------------------------- fees


def test_charge_prices_exactly_one_fill():
    spec = FeeSpec(per_fill=1.0, per_share=0.005)

    assert spec.charge(100) == pytest.approx(1.5)


def test_a_round_trip_is_two_separate_charges():
    """Entry and exit each call the spec. Nothing anywhere may double a
    single charge to stand in for the pair."""
    spec = FeeSpec(per_fill=1.0, per_share=0.005)

    entry_fee = spec.charge(704)
    exit_fee = spec.charge(704)

    assert entry_fee == exit_fee == pytest.approx(4.52)
    assert entry_fee + exit_fee == pytest.approx(9.04)


def test_the_per_share_term_scales_with_size():
    spec = FeeSpec(per_fill=0.0, per_share=0.01)

    assert spec.charge(1) == pytest.approx(0.01)
    assert spec.charge(250) == pytest.approx(2.5)


def test_the_flat_term_does_not_scale_with_size():
    spec = FeeSpec(per_fill=2.0, per_share=0.0)

    assert spec.charge(1) == 2.0
    assert spec.charge(10_000) == 2.0


def test_a_fee_is_charged_on_the_magnitude_of_the_quantity():
    """A short sells to open. Its fill costs the same as a buy of the same
    size, so the sign of a quantity cannot refund a commission."""
    spec = FeeSpec(per_fill=1.0, per_share=0.005)

    assert spec.charge(-200) == spec.charge(200)


def test_a_zero_fee_policy_charges_nothing():
    """Correct for the broker this platform trades through today. The values
    are required on every request, so "we assume zero" is a number somebody
    chose rather than a default nobody noticed."""
    assert FeeSpec(per_fill=0.0, per_share=0.0).charge(1_000) == 0.0


def test_the_retired_round_trip_field_is_gone():
    """`per_trade` priced a round trip in one call. Leaving the name behind,
    even as an alias, would let a caller keep paying twice."""
    assert not hasattr(FeeSpec, "per_trade")


def test_a_fee_spec_refuses_a_negative_charge():
    for kwargs in ({"per_fill": -1.0, "per_share": 0.0},
                   {"per_fill": 1.0, "per_share": -0.005}):
        with pytest.raises(ValueError):
            FeeSpec(**kwargs)


# --------------------------------------------------------------- slippage


def test_a_buy_pays_up_and_a_sell_gets_less():
    spec = SlippageSpec(bps=2.0, min_per_share=0.01)

    assert spec.buy(100.0) == pytest.approx(100.02)
    assert spec.sell(100.0) == pytest.approx(99.98)


def test_the_floor_binds_on_a_cheap_share():
    """Two basis points of a five dollar stock is a tenth of a cent, and no
    desk crosses a spread for that. Below the crossover the tick floor is the
    honest estimate."""
    spec = SlippageSpec(bps=2.0, min_per_share=0.01)

    assert spec.per_share(5.0) == 0.01
    assert spec.buy(5.0) == pytest.approx(5.01)


def test_the_proportional_term_binds_on_an_expensive_share():
    """The reason a flat cent had to go: it was one basis point at a hundred
    dollars and a fifth of one at five hundred, so the same modeled friction
    was an order of magnitude harsher on the cheap names."""
    spec = SlippageSpec(bps=2.0, min_per_share=0.01)

    assert spec.per_share(500.0) == pytest.approx(0.1)
    assert spec.sell(500.0) == pytest.approx(499.9)


def test_slippage_is_symmetric_around_the_reference_price():
    spec = SlippageSpec(bps=2.0, min_per_share=0.01)

    assert spec.buy(250.0) - 250.0 == pytest.approx(250.0 - spec.sell(250.0))


def test_a_zero_spec_leaves_the_reference_price_alone():
    """The frictionless policy the ledger tests use has to be exactly that.
    A floor of one cent hiding in a zero specification would move every cash
    assertion by a share count."""
    spec = SlippageSpec(bps=0.0, min_per_share=0.0)

    assert spec.buy(100.0) == 100.0
    assert spec.sell(100.0) == 100.0


def test_slippage_is_charged_on_the_magnitude_of_the_price():
    """The per-share term reads a price's magnitude, so nothing about a
    nonsensical negative reference produces a negative charge."""
    spec = SlippageSpec(bps=5.0, min_per_share=0.02)

    assert spec.per_share(1_000.0) == pytest.approx(0.5)
    assert spec.per_share(-1_000.0) == pytest.approx(0.5)


def test_a_slippage_spec_refuses_a_negative_term():
    for kwargs in ({"bps": -1.0, "min_per_share": 0.0},
                   {"bps": 1.0, "min_per_share": -0.01}):
        with pytest.raises(ValueError):
            SlippageSpec(**kwargs)
