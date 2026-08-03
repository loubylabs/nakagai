from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class PineLowering:
    emit: Callable[[object, object], object]
    helpers: tuple[str, ...] = ()
