"""The study statistic and the frozen trial set that uses it."""

from nakagai.lab.mutate import literal_trials
from nakagai.lab.study import trial_pf
from tests.lab_helpers import (BASE_SPEC, lab_registry, memory_cache,
                               random_walk_frames, short_windows)


def test_trial_pf_returns_a_float_and_a_trade_count():
    frames = random_walk_frames("TEST", seed=1)
    trial = literal_trials(BASE_SPEC, n=1, seed=1)[0]
    pf, n = trial_pf(memory_cache(frames), trial, ["TEST"],
                     short_windows(frames, "TEST"), lab_registry())
    assert isinstance(pf, float)
    assert isinstance(n, int)
    assert pf >= 0.0
    assert n >= 0


def test_trial_pf_is_repeatable_on_the_same_bars():
    frames = random_walk_frames("TEST", seed=1)
    trial = literal_trials(BASE_SPEC, n=1, seed=1)[0]
    args = (memory_cache(frames), trial, ["TEST"],
            short_windows(frames, "TEST"), lab_registry())
    assert trial_pf(*args) == trial_pf(*args)


def test_trial_pf_with_no_trades_is_zero():
    # A spec whose entry can never fire produces an empty ledger, and an empty
    # ledger must score zero rather than None: a null replay with no trades
    # has to be comparable against a positive observed PF.
    frames = random_walk_frames("TEST", seed=1)
    impossible = dict(BASE_SPEC)
    impossible["long"] = {"all": [
        {"lhs": {"ind": "rsi", "n": 14}, "op": "crosses_above", "rhs": 500}]}
    impossible["name"] = "never_fires"
    from nakagai.lab.mutate import Trial, spec_hash
    trial = Trial(id="x", strategy="rules", spec=impossible,
                  spec_hash=spec_hash(impossible))
    pf, n = trial_pf(memory_cache(frames), trial, ["TEST"],
                     short_windows(frames, "TEST"), lab_registry())
    assert (pf, n) == (0.0, 0)
