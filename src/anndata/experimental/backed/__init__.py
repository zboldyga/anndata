from __future__ import annotations

from typing import TYPE_CHECKING

from ..._core.xarray import Dataset2D
from ._io import read_lazy

if TYPE_CHECKING:
    from ._lazy_arrays import CategoricalArray, MaskedArray

__all__ = ["CategoricalArray", "Dataset2D", "MaskedArray", "read_lazy"]


def __getattr__(name: str) -> object:
    # `_lazy_arrays` subclasses xarray classes, so it is only imported on demand
    if name in {"CategoricalArray", "MaskedArray"}:
        from . import _lazy_arrays

        return getattr(_lazy_arrays, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
