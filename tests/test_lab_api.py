"""The lab's public surface, and the boundaries it must not cross."""

import subprocess
import sys
import textwrap


def test_public_api_is_importable_from_the_package_root():
    import nakagai.lab as lab

    for name in ("Trial", "literal_trials", "composite_trials", "spec_hash",
                 "StudySpec", "StudyResult", "TrialResult", "run_study",
                 "trial_pf", "best_of_n_null", "study_verdict"):
        assert hasattr(lab, name), f"nakagai.lab is missing {name}"


def test_the_lab_never_imports_the_platform_or_the_edge():
    # Run in a subprocess: this process may already have anything imported.
    script = textwrap.dedent("""
        import importlib, json, sys
        for m in ("nakagai.lab", "nakagai.lab.mutate", "nakagai.lab.study",
                  "nakagai.lab.null"):
            importlib.import_module(m)
        print(json.dumps(sorted(sys.modules)))
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, check=True).stdout
    import json
    loaded = set(json.loads(out))
    leaked = sorted(m for m in loaded
                    if m.split(".")[0] in ("nakagai_platform", "nakagai_edge"))
    assert leaked == [], (
        f"nakagai.lab imports {leaked}. The lab is a pure library: the edge "
        f"and the platform both consume it, so it may depend on neither.")


def test_the_lab_reaches_no_network():
    # httpx and the Alpaca provider, NOT socket or urllib: pandas pulls both of
    # those into sys.modules on import, so banning them would fail for reasons
    # that have nothing to do with the lab. httpx is the core's actual HTTP
    # client and nakagai.data.alpaca is its only network-touching module, so
    # those two are the assertion that means something.
    script = textwrap.dedent("""
        import importlib, json, sys
        for m in ("nakagai.lab", "nakagai.lab.mutate", "nakagai.lab.study",
                  "nakagai.lab.null"):
            importlib.import_module(m)
        print(json.dumps(sorted(sys.modules)))
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, check=True).stdout
    import json
    loaded = set(json.loads(out))
    banned = sorted(m for m in loaded
                    if m == "httpx" or m.startswith("httpx.")
                    or m == "nakagai.data.alpaca")
    assert banned == [], (
        f"nakagai.lab imports {banned} at module scope. A study is offline "
        f"compute; whatever fetches its bars lives outside this package.")
