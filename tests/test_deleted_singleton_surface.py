"""The retired singleton engine is gone, and no path back to it survives.

A deletion is only finished when nothing can reach the deleted thing. This file
is the check, and it asks four separate questions, because each one fails
differently.

- THE FILES are absent. A module left on disk is a module an import can find.
- THE IMPORTS raise `ModuleNotFoundError`. A file can be gone while a package
  still re-exports its names from somewhere else, which is what a compatibility
  shim looks like from outside.
- THE PUBLIC EXPORTS are exactly the approved contract, as an equality in both
  directions. A retired name that quietly reappears is caught by one direction
  and a contract name that quietly vanishes by the other.
- THE SOURCE carries none of the architecture's deletion vocabulary. The first
  three questions cannot see a helper that survived under a private name, a
  comment that still tells a reader to call `run_grid`, or a doc that describes
  strategy by symbol by window as separate accounts.

The search reads every Python, Markdown, TOML, and YAML file the repository
tracks. Its exclusions are an explicit path list rather than a directory or a
regex, because a broad exclusion is exactly how a live caller hides: the only
two are the historical release notes, which describe what past versions did and
would be falsified by editing, and this file, which has to spell the vocabulary
out in order to search for it.
"""

import re
import subprocess
from pathlib import Path

import pytest

import nakagai.engine as engine

ROOT = Path(__file__).resolve().parents[1]

# Every file the retired engine lived in.
DELETED_FILES = (
    "nakagai/engine/engine.py",
    "nakagai/engine/runner.py",
    "nakagai/engine/provenance.py",
    "nakagai/engine/costs.py",
    "nakagai/icir.py",
    "tests/test_engine_fills.py",
    "tests/test_engine_excursion.py",
    "tests/test_engine_preload.py",
    "tests/test_engine_timeframes.py",
    "tests/test_icir.py",
    "tests/test_icir_runner.py",
    "tests/test_metrics.py",
    "tests/test_runner.py",
    "tests/test_runner_frame_reuse.py",
    "tests/test_rules_cursor.py",
    "tests/test_trades_persistence.py",
)

DELETED_MODULES = (
    "nakagai.engine.engine",
    "nakagai.engine.runner",
    "nakagai.engine.provenance",
    "nakagai.engine.costs",
    "nakagai.icir",
)

# The architecture's deletion vocabulary, as whole words. `Engine` and `Trade`
# are deliberately absent: they are substrings of names that legitimately
# survive (`FrameEval` carries no `Engine`, but `PortfolioTrade` does carry
# `Trade`), so they are covered by the module and export checks above instead.
RETIRED_NAMES = (
    "run_grid",
    "run_one",
    "BacktestResult",
    "buy_and_hold_return",
    "SlippageModel",
    "FeeModel",
    "_SettledLedger",
    "empty_ic_fields",
    "window_icir",
    "icir_fields",
    "runs.parquet",
    "trades.parquet",
    "EvidenceStore",
    "backtest_shard",
    "proving_runs",
    "proving_trades",
)

# One exact path each, and both earn it. The release notes are a record of what
# earlier versions did, so removing the names from them would make the history
# wrong rather than the codebase clean; this file has to write the vocabulary
# down in order to search for it.
EXCLUDED = ("tests/test_deleted_singleton_surface.py",)
RELEASE_NOTES_HEADING = "## Release notes"

SEARCHED_SUFFIXES = (".py", ".md", ".toml", ".yml", ".yaml", ".sh", ".json")


def tracked_files() -> tuple[Path, ...]:
    """Every tracked file of a searched kind, from git rather than a walk.

    `git ls-files` is what makes this honest: a glob would silently skip a
    directory nobody thought of, and it would sweep in build output and virtual
    environments that are not this repository's source. Untracked files count
    too, since a retired name reintroduced in a file nobody has staged yet is
    still a retired name in the tree.
    """
    listed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--cached", "--others",
         "--exclude-standard"],
        capture_output=True, text=True, check=True).stdout.split("\n")
    return tuple(
        ROOT / name for name in listed
        if name and name.endswith(SEARCHED_SUFFIXES) and name not in EXCLUDED)


def searchable_text(path: Path) -> str:
    """One file's text, with the historical release notes cut off the README."""
    text = path.read_text(encoding="utf-8")
    if path.name == "README.md" and RELEASE_NOTES_HEADING in text:
        return text.split(RELEASE_NOTES_HEADING)[0]
    return text


def test_the_search_reads_a_real_and_complete_file_set():
    """The searches below are worthless if this returns nothing.

    A `git ls-files` that failed, a suffix filter that matched nothing, or an
    exclusion that swallowed the tree would all leave every search vacuously
    green. Naming three files that must be in the set is what makes the rest of
    this module a real check.
    """
    files = tracked_files()

    assert len(files) > 50
    for required in ("README.md", "pyproject.toml", "nakagai/engine/replay.py"):
        assert ROOT / required in files, required


# ------------------------------------------------------------------ the files


@pytest.mark.parametrize("name", DELETED_FILES)
def test_a_retired_file_is_absent(name):
    assert not (ROOT / name).exists()


@pytest.mark.parametrize("name", DELETED_MODULES)
def test_a_retired_module_cannot_be_imported(name):
    with pytest.raises(ModuleNotFoundError):
        __import__(name)


# ---------------------------------------------------------------- the exports


APPROVED_EXPORTS = {
    "run_portfolio",
    # inputs
    "AccountPolicy", "BenchmarkSpec", "ExchangeScheduleIdentity",
    "ExecutionPolicy", "FeeSpec", "PlayRequest", "PortfolioBars",
    "PortfolioReplayRequest", "ReplaySchedule", "ReplayWindow",
    "ScheduledBaseInterval", "ScheduledContextBar", "SlippageSpec",
    # the strategy bundle
    "FrozenStrategyRegistry", "StrategyDefinition", "StrategyDependencies",
    "StrategyRegistry", "composite_definition", "rules_definition",
    "spec_definition_digest",
    # the strategy boundary
    "EntryIntent", "ManagementDecision", "PositionView", "Signal",
    # results
    "ARITHMETIC_VERSION", "BenchmarkResult", "EquityPoint", "ExitReason",
    "IcEstimate", "PortfolioMetrics", "PortfolioReplayResult",
    "PortfolioSlice", "PortfolioTrade", "RejectionReason", "ReplayRejection",
    "TradeStats",
    # refusals
    "ReplayInputError", "StrategyOutputError", "StrategyRuntimeError",
    # the codec and the identifier formulas
    "JSONScalar", "JSONValue", "canonical_replay_bytes",
    "decode_replay_request", "decode_replay_result", "decode_replay_schedule",
    "definition_digest", "encode_replay_request", "encode_replay_result",
    "encode_replay_schedule", "expected_candidate_id", "expected_replay_id",
    "rejection_id", "result_digest", "schedule_digest", "trade_id",
}


def test_the_approved_exports_are_exactly_the_portfolio_contract():
    assert set(engine.__all__) == APPROVED_EXPORTS


def test_every_approved_export_resolves():
    """Including the deferred half, which loads on first access."""
    for name in engine.__all__:
        assert getattr(engine, name) is not None, name


def test_no_retired_name_resolves_off_the_engine_package():
    """A deferred lookup must not become a back door.

    The package answers unknown attributes through `__getattr__`, so "the name
    is not in `__all__`" is not by itself proof that it cannot be reached.
    """
    for name in ("Engine", "BacktestResult", "run_one", "run_grid",
                 "summarize", "buy_and_hold_return", "FeeModel",
                 "SlippageModel", "Trade"):
        with pytest.raises(AttributeError):
            getattr(engine, name)


def test_a_composite_has_no_class_bound_membership_door():
    """The one dual path that lived outside the engine package."""
    from nakagai.strategies.composite import CompositeStrategy

    assert not hasattr(CompositeStrategy, "bound")
    assert not hasattr(CompositeStrategy, "MEMBERS")


# ------------------------------------------------------------- the vocabulary


@pytest.mark.parametrize("name", RETIRED_NAMES)
def test_no_tracked_file_mentions_a_retired_name(name):
    """Whole-word, so a longer name that legitimately contains one is safe."""
    pattern = re.compile(rf"(?<![\w.]){re.escape(name)}(?![\w])")
    offenders = [
        str(path.relative_to(ROOT))
        for path in tracked_files()
        if pattern.search(searchable_text(path))
    ]
    assert offenders == []


def test_no_tracked_file_describes_a_run_as_its_own_account():
    """The documentation half of the deletion.

    A doc that still calls a strategy-symbol-window row an independent account
    describes the topology this phase removed, which is a wrong mental model
    rather than a wrong import and nothing above would catch it.
    """
    phrases = ("separate accounts", "independent accounts",
               "one account per run", "per-symbol equity curve")
    offenders = [
        (str(path.relative_to(ROOT)), phrase)
        for path in tracked_files() if path.suffix == ".md"
        for phrase in phrases if phrase in searchable_text(path).lower()
    ]
    assert offenders == []
