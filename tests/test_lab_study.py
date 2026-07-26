"""The study statistic and the frozen trial set that uses it."""

import pandas as pd
import pytest

from nakagai.engine.runner import run_one
from nakagai.lab.mutate import literal_trials
from nakagai.lab.study import StudySpec, run_study, trial_pf
from nakagai.stats import pf_from_trades
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


def test_trial_pf_pools_the_ledger_instead_of_averaging_per_window_pf():
    """trial_pf has to score one ledger built from every window's trades, not
    the mean of each window's own profit factor. The two statistics agree
    only when every window contributes the same trade count, which is not
    this fixture: the three windows place 5, 8, and 6 trades. An averaging
    rewrite would still be deterministic, still type-correct, and would still
    collapse to (0.0, 0) on an empty ledger, so none of the tests above would
    catch it; it would simply score every trial by the wrong number.

    The per-window ledgers are rebuilt here with the same run_one calls
    trial_pf makes internally, but that alone would only mirror the
    implementation. What actually pins pooling is the final assertion: the
    pooled and averaged statistics are computed from the same trades and
    compared against each other, and trial_pf's answer must land on the
    pooled side, not the averaged one.
    """
    from nakagai.lab.mutate import Trial, spec_hash

    frames = random_walk_frames("TEST", seed=1)
    windows = short_windows(frames, "TEST")
    cache = memory_cache(frames)
    registry = lab_registry()
    trial = Trial(id="base", strategy="rules", spec=BASE_SPEC,
                  spec_hash=spec_hash(BASE_SPEC))

    per_window_trades = []
    per_window_pf = []
    for window in windows:
        result = run_one(cache, trial.strategy, {"spec": trial.spec}, "TEST",
                         window, registry=registry, icir=False)
        trades = pd.DataFrame(result["trades"])
        per_window_trades.append(trades)
        per_window_pf.append(pf_from_trades(trades) or 0.0)

    pooled_trades = pd.concat(per_window_trades, ignore_index=True)
    pooled_pf = pf_from_trades(pooled_trades)
    averaged_pf = sum(per_window_pf) / len(per_window_pf)
    # The fixture must actually exercise unequal window sizes, or pooled and
    # averaged would coincide by accident and the test would prove nothing.
    assert pooled_pf != pytest.approx(averaged_pf)

    pf, n = trial_pf(cache, trial, ["TEST"], windows, registry)

    assert n == len(pooled_trades)
    assert pf == pytest.approx(pooled_pf)
    assert pf != pytest.approx(averaged_pf)


def _study(frames, seed=1, n=4):
    return StudySpec(
        trials=tuple(literal_trials(BASE_SPEC, n=n, seed=seed)),
        symbols=("TEST",),
        windows=tuple(short_windows(frames, "TEST")),
        seed=seed)


def test_run_study_scores_every_trial():
    frames = random_walk_frames("TEST", seed=2)
    out = run_study(memory_cache(frames), _study(frames), lab_registry())
    assert out.n_trials == 4
    assert len(out.results) == 4
    assert {r.trial_id for r in out.results} == {t.id for t in _study(frames).trials}


def test_run_study_preserves_trial_order():
    # Order is a documented part of run_study's contract, not an accident of
    # dict/list iteration: Task 9's best_of_n_null compares an observed run
    # against null replays position for position, without sorting first. A
    # set comparison (as in test_run_study_scores_every_trial) or a
    # self-consistency check (as in test_run_study_is_deterministic) would
    # both still pass if run_study sorted its results by trial_id, so
    # neither test pins order. This one does.
    frames = random_walk_frames("TEST", seed=2)
    study = _study(frames)
    out = run_study(memory_cache(frames), study, lab_registry())
    assert [r.trial_id for r in out.results] == [t.id for t in study.trials]


def test_run_study_best_is_the_highest_pf():
    frames = random_walk_frames("TEST", seed=2)
    out = run_study(memory_cache(frames), _study(frames), lab_registry())
    assert out.best is not None
    assert out.best.pf == max(r.pf for r in out.results)


def test_run_study_is_deterministic():
    frames = random_walk_frames("TEST", seed=2)
    a = run_study(memory_cache(frames), _study(frames), lab_registry())
    b = run_study(memory_cache(frames), _study(frames), lab_registry())
    assert [(r.trial_id, r.pf, r.n_trades) for r in a.results] == \
           [(r.trial_id, r.pf, r.n_trades) for r in b.results]


def test_run_study_rejects_an_empty_trial_set():
    frames = random_walk_frames("TEST", seed=2)
    empty = StudySpec(trials=(), symbols=("TEST",),
                      windows=tuple(short_windows(frames, "TEST")), seed=1)
    with pytest.raises(ValueError, match="at least one trial"):
        run_study(memory_cache(frames), empty, lab_registry())
