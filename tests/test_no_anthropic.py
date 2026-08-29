"""No module in this package may reach for Anthropic.

The compilers used to build `anthropic.Anthropic()` themselves, which made a
vendor SDK a dependency of a package whose whole job is deterministic
evaluation. They take a `model.Complete` callable now (`nakagai/model.py`), so
nothing here knows a provider.

**Two checks, not one.** An import is the obvious shape; an environment gate is
the one that hides. A module reading `ANTHROPIC_API_KEY` to decide whether a
feature is available contains no import at all, so a constructor scan walks
straight past it and the feature goes dark the day the key is removed. The
platform's retirement found exactly that shape in two of its own files after a
constructor-only survey reported them clean.
"""

import ast
import pathlib

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "nakagai"


def _sources():
    return sorted(p for p in PACKAGE.rglob("*.py")
                  if "__pycache__" not in p.parts)


def test_the_scan_reads_enough_to_mean_anything():
    """A floor, so an empty walk cannot pass for a clean one."""
    found = _sources()
    assert len(found) >= 20, (
        f"only {len(found)} modules found under {PACKAGE}; the scan read too "
        "little to prove anything")


@pytest.mark.parametrize("path", _sources(), ids=lambda p: p.name)
def test_no_module_imports_or_constructs_anthropic(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.split(".")[0] == "anthropic", (
                    f"{path.relative_to(ROOT)} imports {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "anthropic", (
                f"{path.relative_to(ROOT)} imports from {node.module}")
        elif isinstance(node, ast.Attribute):
            # `anthropic.Anthropic(...)` with the module bound some other way.
            value = node.value
            if isinstance(value, ast.Name) and value.id == "anthropic":
                raise AssertionError(
                    f"{path.relative_to(ROOT)} reaches anthropic.{node.attr}")


@pytest.mark.parametrize("path", _sources(), ids=lambda p: p.name)
def test_no_module_gates_on_an_anthropic_credential(path):
    """The half a constructor scan cannot see."""
    source = path.read_text()
    assert "ANTHROPIC_API_KEY" not in source, (
        f"{path.relative_to(ROOT)} still gates on an Anthropic credential")


def test_the_manifest_declares_no_anthropic_dependency():
    manifest = (ROOT / "pyproject.toml").read_text()
    assert "anthropic" not in manifest, (
        "pyproject.toml still declares an anthropic dependency")
    lock = (ROOT / "uv.lock").read_text()
    assert "name = \"anthropic\"" not in lock, (
        "uv.lock still resolves anthropic; regenerate it")
