"""The acceptance gate: is the best-of-N null honest?

Run the entire pipeline on bars with no exploitable structure, where every
trial's true edge is zero, and check that p-values come out roughly uniform
rather than piled up near zero. Then run it on bars with a real, capturable
effect and check that the same machinery finds it.

Calibration runs at REPLICATES = 24, TRIALS = 4, PERMUTATIONS = 16, the
plan's documented fallback, not its headline REPLICATES = 40,
PERMUTATIONS = 24. trial_pf was measured at roughly 0.68 seconds per call
over 2 symbols and 3 windows on this machine, which puts the headline
configuration at roughly 45 minutes and this fallback at roughly 18 minutes
in estimate. The actual measured wall clock for both tests together was
1439 seconds, roughly 24 minutes: budget for that before running this file.

Two symbols are pooled per replicate, not one: a single-symbol trial places
only 19 to 24 trades, thin enough to produce lumpy profit factors, frequent
PF_CLAMP hits, and tied p-values that would corrupt the uniformity this test
measures. TEST and TEST2, built from independent seeds and merged, push a
trial to roughly 37 to 39 trades.

Marked slow: each replicate replays hundreds of backtests. Run with
`uv run pytest -m slow tests/test_lab_calibration.py -v`.
"""

import statistics
from dataclasses import dataclass

import pytest

from nakagai.lab.mutate import literal_trials
from nakagai.lab.null import best_of_n_null, study_verdict
from nakagai.lab.study import StudySpec, run_study
from nakagai.stats import PF_CLAMP
from tests.lab_helpers import (BASE_SPEC, lab_registry, memory_cache,
                               oscillating_frames, random_walk_frames,
                               short_windows)

REPLICATES = 24
TRIALS = 4
PERMUTATIONS = 16

SYMBOLS = ("TEST", "TEST2")


def _pooled_frames(frame_fn, seed_a: int, seed_b: int) -> dict:
    """Merge TEST and TEST2, built from two different seeds so the symbols
    are statistically independent of each other rather than the same path
    twice."""
    return {**frame_fn("TEST", seed=seed_a), **frame_fn("TEST2", seed=seed_b)}


@dataclass(frozen=True)
class _Replicate:
    """One replicate's outcome, kept alongside the p-value so the report can
    show how much of the run was riding the PF_CLAMP resolution floor rather
    than a genuine profit-factor spread."""
    p_value: float
    observed_pf: float
    observed_clamped: bool
    n_null_clamped: int
    n_nulls: int


def _run_replicate(frames, seed: int) -> _Replicate | None:
    """One full pipeline pass: mutate, run, permute, score."""
    registry = lab_registry()
    study = StudySpec(trials=tuple(literal_trials(BASE_SPEC, n=TRIALS, seed=seed)),
                      symbols=SYMBOLS,
                      windows=tuple(short_windows(frames, "TEST")),
                      seed=seed)
    observed = run_study(memory_cache(frames), study, registry)
    if observed.best is None:
        return None
    nulls = best_of_n_null(frames, study, registry, PERMUTATIONS,
                           epoch=f"cal-{seed}")
    total_trades = sum(r.n_trades for r in observed.results)
    # min_trades=1 deliberately: calibration is measuring the p-value
    # DISTRIBUTION, and the production trade floor would refuse most replicates
    # before a p-value ever existed, leaving nothing to measure. The floor is
    # tested separately in test_lab_null.py.
    verdict = study_verdict(observed.best.pf, nulls,
                            min_trades=1, n_trades=total_trades)
    if verdict["p_value"] is None:
        return None
    return _Replicate(
        p_value=verdict["p_value"],
        observed_pf=observed.best.pf,
        observed_clamped=observed.best.pf == PF_CLAMP,
        n_null_clamped=sum(1 for x in nulls if x == PF_CLAMP),
        n_nulls=len(nulls))


@pytest.mark.slow
def test_p_values_are_uniform_on_pure_noise():
    replicates = []
    for i in range(REPLICATES):
        frames = _pooled_frames(random_walk_frames, 1000 + i, 5000 + i)
        r = _run_replicate(frames, seed=i)
        if r is not None:
            replicates.append(r)

    ps = [r.p_value for r in replicates]
    # Printed unconditionally (run with -s to see it) so the acceptance gate's
    # own evidence is on record, not just its pass/fail verdict.
    print(f"\n[calibration] {len(ps)}/{REPLICATES} replicates produced a "
          f"p-value: {ps}")
    if ps:
        print(f"[calibration] mean p = {statistics.fmean(ps):.4f}, "
              f"median p = {statistics.median(ps):.4f}")
    n_observed_clamped = sum(1 for r in replicates if r.observed_clamped)
    n_null_clamped = sum(r.n_null_clamped for r in replicates)
    n_nulls_total = sum(r.n_nulls for r in replicates)
    print(f"[calibration] observed PF_CLAMP hits: {n_observed_clamped}/"
          f"{len(replicates)}; null PF_CLAMP hits: {n_null_clamped}/"
          f"{n_nulls_total}")

    assert len(ps) >= REPLICATES // 2, (
        f"only {len(ps)}/{REPLICATES} replicates produced a p-value; the "
        f"strategy is barely trading on this substrate, so the calibration "
        f"says nothing. Lengthen the bars or loosen BASE_SPEC's entry.")

    mean_p = statistics.fmean(ps)
    # Under a correct null p is uniform, so the mean is 0.5. A null that is too
    # permissive drags this toward 0; one that is too conservative pushes it
    # toward 1.
    assert 0.35 <= mean_p <= 0.65, (
        f"mean p-value {mean_p:.3f} over {len(ps)} noise replicates. The "
        f"best-of-N null is miscalibrated: p should be uniform when there is "
        f"nothing to find. Do NOT build Stage 2 until this holds.")

    # The tail, checked loosely. A badly broken null piles most mass near zero.
    small = sum(1 for p in ps if p <= 0.25) / len(ps)
    assert 0.05 <= small <= 0.50, (
        f"{small:.0%} of noise replicates landed at p <= 0.25; a uniform null "
        f"puts 25% there.")


@pytest.mark.slow
def test_a_real_effect_is_still_detected():
    """The other failure direction. A null so conservative that nothing ever
    survives would pass the uniformity test above by being uniformly useless."""
    replicates = []
    for i in range(8):
        frames = _pooled_frames(oscillating_frames, 2000 + i, 6000 + i)
        r = _run_replicate(frames, seed=i)
        if r is not None:
            replicates.append(r)

    ps = [r.p_value for r in replicates]
    print(f"\n[calibration] positive control p-values: {ps}")
    n_observed_clamped = sum(1 for r in replicates if r.observed_clamped)
    print(f"[calibration] positive control observed PF_CLAMP hits: "
          f"{n_observed_clamped}/{len(replicates)}")

    assert ps, "the positive control produced no p-values at all"
    median_p = statistics.median(ps)
    assert median_p <= 0.25, (
        f"median p {median_p:.3f} on bars with a strong mean-reverting cycle. "
        f"The null is too conservative to detect a real effect, which makes "
        f"the whole lab unable to find anything.")
