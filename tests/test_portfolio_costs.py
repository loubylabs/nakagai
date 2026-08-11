"""What one fill costs: the slippage crossed and the fee paid.

The fee model prices ONE fill. That is the whole point of this file: the model
it replaced returned a round trip from a single call, so a caller that charged
it at entry and again at exit paid four fills and nothing raised. Every test
below states a per-fill number, and the round-trip total only ever appears as
two charges added together.
"""

import pytest

from nakagai.engine.costs import FeeModel, SlippageModel
from nakagai.engine.portfolio import _fee_model, _slippage_model
from nakagai.engine.portfolio_types import FeeSpec, SlippageSpec


# ------------------------------------------------------------------- fees


def test_charge_prices_exactly_one_fill():
    model = FeeModel(per_fill=1.0, per_share=0.005)

    assert model.charge(100) == pytest.approx(1.5)


def test_a_round_trip_is_two_separate_charges():
    """Entry and exit each call the model. Nothing anywhere may double a
    single charge to stand in for the pair."""
    model = FeeModel(per_fill=1.0, per_share=0.005)

    entry_fee = model.charge(704)
    exit_fee = model.charge(704)

    assert entry_fee == exit_fee == pytest.approx(4.52)
    assert entry_fee + exit_fee == pytest.approx(9.04)


def test_the_per_share_term_scales_with_size():
    model = FeeModel(per_share=0.01)

    assert model.charge(1) == pytest.approx(0.01)
    assert model.charge(250) == pytest.approx(2.5)


def test_the_flat_term_does_not_scale_with_size():
    model = FeeModel(per_fill=2.0)

    assert model.charge(1) == 2.0
    assert model.charge(10_000) == 2.0


def test_a_fee_is_charged_on_the_magnitude_of_the_quantity():
    """A short sells to open. Its fill costs the same as a buy of the same
    size, so the sign of a quantity cannot refund a commission."""
    model = FeeModel(per_fill=1.0, per_share=0.005)

    assert model.charge(-200) == model.charge(200)


def test_fees_are_zero_by_default():
    """Correct for the broker this platform trades through today. The seam
    exists so a broker that does charge changes one value rather than
    invalidating every stored number."""
    assert FeeModel().charge(1_000) == 0.0


def test_the_retired_round_trip_field_is_gone():
    """`per_trade` priced a round trip in one call. Leaving the name behind,
    even as an alias, would let a caller keep paying twice."""
    assert not hasattr(FeeModel, "per_trade")


def test_a_fee_spec_becomes_a_fee_model_field_for_field():
    model = _fee_model(FeeSpec(per_fill=0.35, per_share=0.002))

    assert (model.per_fill, model.per_share) == (0.35, 0.002)
    assert model.charge(50) == pytest.approx(0.45)


# --------------------------------------------------------------- slippage


def test_a_buy_pays_up_and_a_sell_gets_less():
    model = SlippageModel(bps=2.0, min_per_share=0.01)

    assert model.buy(100.0) == pytest.approx(100.02)
    assert model.sell(100.0) == pytest.approx(99.98)


def test_the_floor_binds_on_a_cheap_share():
    """Two basis points of a five dollar stock is a tenth of a cent, and no
    desk crosses a spread for that. Below the crossover the tick floor is the
    honest estimate."""
    model = SlippageModel(bps=2.0, min_per_share=0.01)

    assert model.per_share(5.0) == 0.01
    assert model.buy(5.0) == pytest.approx(5.01)


def test_the_proportional_term_binds_on_an_expensive_share():
    model = SlippageModel(bps=2.0, min_per_share=0.01)

    assert model.per_share(500.0) == pytest.approx(0.1)
    assert model.sell(500.0) == pytest.approx(499.9)


def test_slippage_is_symmetric_around_the_reference_price():
    model = SlippageModel(bps=2.0, min_per_share=0.01)

    assert model.buy(250.0) - 250.0 == pytest.approx(250.0 - model.sell(250.0))


def test_a_zero_spec_leaves_the_reference_price_alone():
    """The frictionless policy the ledger tests use has to be exactly that.
    A floor of one cent hiding in a zero specification would move every cash
    assertion by a share count."""
    model = _slippage_model(SlippageSpec(bps=0.0, min_per_share=0.0))

    assert model.buy(100.0) == 100.0
    assert model.sell(100.0) == 100.0


def test_a_slippage_spec_becomes_a_slippage_model_field_for_field():
    model = _slippage_model(SlippageSpec(bps=5.0, min_per_share=0.02))

    assert (model.bps, model.min_per_share) == (5.0, 0.02)
    assert model.per_share(1_000.0) == pytest.approx(0.5)
