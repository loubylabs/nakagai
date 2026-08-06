"""Core ships four dependencies. This test is what keeps it that way.

The deflated-Sharpe work was the first thing here with a real pull toward
scipy, and it was implemented around it deliberately (math.erf plus one
rational approximation). A future edit that reaches for the easy import
should fail here rather than in a Docker build.
"""
import subprocess
import sys


def test_importing_core_pulls_no_scientific_stack():
    code = (
        "import sys, nakagai.stats, nakagai.engine.metrics;"
        "bad=[m for m in ('scipy','sklearn','statsmodels') if m in sys.modules];"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "", f"core imported: {out.stdout.strip()}"
