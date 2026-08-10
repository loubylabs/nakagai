"""Durable identity for replay arithmetic and fill behavior.

These values name the numerical contract executed by the engine. They are
independent of package releases and source revisions. Change either value only
when the corresponding result semantics change.
"""

ARITHMETIC_VERSION = "1"
FILL_MODE = "pessimistic"
