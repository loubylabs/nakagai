"""Pine generation: the model, the per-term lowerings, and the compiler.

This package's __init__ deliberately re-exports the MODEL only. vocabulary.py
imports the lowering slot from here, and importing a submodule runs this file
first, so pulling the compiler in at this level would have vocabulary.py import
lower.py import vocabulary.py, half-initialized. The public entry point,
lower_pine, is exported from nakagai.strategies.rules instead.
"""

from nakagai.strategies.rules.pine.model import (
    GENERATOR_VERSION, PineBundle, PineCompileError, PineExpr, PineHelper,
    PineInput, PineLowering, PineProgram, RulePath, TermCall,
)

__all__ = [
    "GENERATOR_VERSION", "PineBundle", "PineCompileError", "PineExpr",
    "PineHelper", "PineInput", "PineLowering", "PineProgram", "RulePath",
    "TermCall",
]
