"""The public seam: one validated spec in, one PineProgram out."""

from nakagai.strategies.rules.pine.lower import SpecLowerer
from nakagai.strategies.rules.pine.model import PineCompileError, PineProgram
from nakagai.strategies.rules.spec import validate_spec
from nakagai.strategies.rules.vocabulary import Vocabulary, resolve_vocabulary


def lower_pine(spec: dict, vocabulary: Vocabulary | None = None) -> PineProgram:
    """Lower a RuleSpec v2 into one target-neutral Pine program.

    The spec is validated first and an invalid one is refused outright: the
    lowering reads a node's shape the way the validator guarantees it, so
    walking an unchecked spec would report a KeyError from somewhere deep in
    the walk instead of the precise per-path errors the caller already has a
    retry loop for.

    `vocabulary` is an ordinary positional here, unlike the seams where it
    sits behind another optional argument, because it follows the one required
    argument directly and cannot bind anywhere else.
    """
    vocabulary = resolve_vocabulary(vocabulary)
    errs = validate_spec(spec, vocabulary)
    if errs:
        raise PineCompileError(
            "pine_invalid_spec",
            "the spec does not validate, so there is nothing to lower: "
            + "; ".join(errs))
    return SpecLowerer(spec, vocabulary).run()
