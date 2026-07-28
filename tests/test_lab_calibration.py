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
in estimate.

The work is roughly 13,000 replays: 32 replicates, each one study pass plus
PERMUTATIONS null passes, each of those TRIALS trials over 2 symbols and 3
windows. Replicates run across processes (see `_replicates`), so the wall
clock is that work divided by however many workers the box allows, and the
1439 seconds this file once took single-threaded is now the CPU total rather
than the elapsed time. Do not read a wall clock here as a per-replay cost, and
do not compare one machine's number with another's: on the self-hosted CI
runners, which are laptops also running the proving farm, the same work has
ranged over a 2.2x spread from contention alone.

Two symbols are pooled per replicate, not one: a single-symbol trial places
roughly 3 to 26 trades (measured on BASE_SPEC itself, not its mutants, which
trade differently), thin enough to produce lumpy profit factors, frequent
PF_CLAMP hits, and tied p-values that would corrupt the uniformity this test
measures. TEST and TEST2, built from independent seeds and merged, push a
trial's pooled count to roughly 7 to 50 with a median near 24 (again on the
mutants literal_trials actually generates, not BASE_SPEC). Pooling roughly
doubles the count and reduces the thin-ledger lumpiness; it does not
eliminate it, since several trials still land below 20 pooled trades. The
override of the brief's symbols=("TEST",) was still the right call; only the
earlier trade-count figures quoted for it (19 to 24 single, 37 to 39 pooled)
were overstated.

Marked slow: each replicate replays hundreds of backtests. Run with
`uv run pytest -m slow tests/test_lab_calibration.py -v`.
"""

import os
import statistics
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

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

WORKERS_ENV = "NAKAGAI_LAB_WORKERS"

# One worker is a fresh interpreter holding pandas, numpy, this module and one
# replicate's bars: 106 MiB measured at that point. The figure below leaves
# room for the permuted copies a replicate allocates on top of it.
WORKER_FOOTPRINT = 256 * 1024 * 1024


def _memory_limit() -> int | None:
    """This process's memory cap in bytes, or None when it is not capped.

    Sizing a pool by core count alone is wrong wherever cores and memory are
    rationed differently, and the platform's core-integration runner is exactly
    that: a container with no CPU limit, which therefore sees all fifteen of
    the laptop's cores, and a 4 GiB memory cap. Fifteen workers fit the cores
    and not the memory.
    """
    for path in ("/sys/fs/cgroup/memory.max",                    # cgroup v2
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):  # cgroup v1
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            limit = int(raw)
        except ValueError:
            continue
        # v1 reports a huge sentinel rather than "max" when uncapped.
        return None if limit > (1 << 60) else limit
    return None


_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


@contextmanager
def _one_thread_per_worker():
    """Hold numpy to one thread per worker for the life of a pool.

    A replicate is a long run of small pandas and numpy work, so there is
    nothing here for a threaded backend to win, and by default each worker
    opens a thread per core: eight workers times fifteen threads is how a pool
    ends up slower than the loop it replaced, and how a run gets itself killed
    under memory pressure. Measured on a laptop under exactly that
    oversubscription, fifteen workers returned only 4.25x.

    Set in the parent rather than in a pool initializer, because a spawned child
    imports numpy while unpickling the work and has fixed its thread count
    before any initializer of ours could run. Children inherit this environment;
    the parent does no array work while the pool is up.
    """
    saved = {var: os.environ.get(var) for var in _THREAD_VARS}
    os.environ.update({var: "1" for var in _THREAD_VARS})
    try:
        yield
    finally:
        for var, was in saved.items():
            if was is None:
                del os.environ[var]
            else:
                os.environ[var] = was


def _workers(n: int) -> int:
    """How many of n replicates to run at once.

    Bounded by what the operator asked for, the cores this process may use, and
    the memory its container allows. Half the cap rather than all of it: the
    parent holds pytest and its own frames, and a pool that OOM-kills a worker
    turns a slow gate into a failing one.
    """
    override = os.environ.get(WORKERS_ENV)
    if override:
        return max(1, min(int(override), n))
    cores = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
             else os.cpu_count() or 1)
    limit = _memory_limit()
    if limit is not None:
        cores = min(cores, max(1, (limit // 2) // WORKER_FOOTPRINT))
    return max(1, min(cores, n))


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
    # total_trades sums n_trades across every trial in the set (all TRIALS of
    # them), while observed.best.pf comes from a single trial, so the two are
    # a roughly 4x mismatch at TRIALS = 4 (e.g. 84 pooled trades total versus
    # 21 for the winning trial alone on replicate 0). Harmless at
    # min_trades=1, verbatim from the brief, but it would silently defeat the
    # trade floor if min_trades were ever raised in this file: a study could
    # clear the floor on the OTHER trials' trade counts while the best trial
    # itself traded on a near-empty ledger.
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


def _noise_replicate(i: int) -> _Replicate | None:
    """Replicate i of the null calibration. A pure function of i."""
    return _run_replicate(_pooled_frames(random_walk_frames, 1000 + i, 5000 + i),
                          seed=i)


def _effect_replicate(i: int) -> _Replicate | None:
    """Replicate i of the positive control. A pure function of i."""
    return _run_replicate(_pooled_frames(oscillating_frames, 2000 + i, 6000 + i),
                          seed=i)


def _replicates(replicate, n: int) -> list:
    """Every replicate, in index order, across as many cores as we may use.

    Replicates are independent by construction: each builds its own bars from
    its own seeds and each permutation draws from `permutation_seed`, so no
    number here depends on execution order, and `map` returns results in index
    order whatever order the workers finish in. That makes this a pure wall
    clock change, and the reason it is worth making is that the loop it
    replaces was the whole cost of this file: 32 replicates times 17 study
    passes times 4 trials times 6 backtests is roughly 13,000 replays, run one
    at a time on a box with more than a dozen idle cores.

    Sequential on a single core, so a one-core run pays no pool overhead and
    stays trivially debuggable under a profiler.
    """
    workers = _workers(n)
    if workers <= 1:
        return [replicate(i) for i in range(n)]
    with _one_thread_per_worker():
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(replicate, range(n)))


def _small_null(seed: int) -> list[float]:
    """A cheap null computed from a seed alone, for the child-vs-parent check."""
    frames = random_walk_frames("TEST", seed=seed)
    study = StudySpec(trials=tuple(literal_trials(BASE_SPEC, n=2, seed=seed)),
                      symbols=("TEST",),
                      windows=tuple(short_windows(frames, "TEST", count=1)),
                      seed=seed)
    return best_of_n_null(frames, study, lab_registry(), 2, epoch="determinism")


def test_a_child_process_scores_a_null_identically():
    """The property every parallel replicate rests on, held cheaply.

    Each seed in this pipeline is explicit and `permutation_seed` is a sha256 of
    its inputs, so no number here can depend on which process computed it. That
    is what makes `_replicates` a pure wall clock change. If someone later
    reaches for a module-level Generator or the global numpy RNG, the
    replicates quietly stop being reproducible and the slow tests above would
    still pass: this is the test that would not.
    """
    here = _small_null(7)
    with ProcessPoolExecutor(max_workers=1) as pool:
        there = pool.submit(_small_null, 7).result()
    assert here == there


@pytest.mark.slow
def test_p_values_are_uniform_on_pure_noise():
    replicates = [r for r in _replicates(_noise_replicate, REPLICATES)
                  if r is not None]

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
    # Under a correct null p is uniform, so the mean is 0.5 in the continuous
    # limit. A null that is too permissive drags this toward 0; one that is
    # too conservative pushes it toward 1.
    #
    # At PERMUTATIONS = 16 the bias-corrected p-value actually lives on a
    # 17-point grid of k/17 (k = 0..16), so the exact expectation under a
    # correct null is E[p] = 9/17 = 0.5294, not 0.5. A 200k-run simulation of
    # 24 draws from that grid gives SE(mean_p) = 0.0588, which puts the
    # inherited [0.35, 0.65] band (sized in the brief for 40 replicates, not
    # 24) asymmetric around the true expectation: the lower edge (0.35) sits
    # 3.05 SE below it, the upper edge (0.65) only 2.05 SE above it. The
    # resulting spurious-failure rate on correct code is about 3% overall:
    # roughly 1.95% high (mean_p exceeds 0.65, the closer bound), 0.08% low
    # (mean_p falls below 0.35, the farther bound), plus about 1.4% from the
    # tail-fraction band below. Nearly all of that residual risk sits on the
    # "too conservative" (mean_p too high) side, which is exactly the
    # direction a reader would misread as a real defect in the null rather
    # than sampling noise at this N. Bounds are left as-is per the brief;
    # this is a known property, not a defect.
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
    replicates = [r for r in _replicates(_effect_replicate, 8) if r is not None]

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
