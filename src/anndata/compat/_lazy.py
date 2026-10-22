"""Placeholder types for classes from optional dependencies.

`anndata` dispatches on and checks against classes from optional dependencies
(dask, xarray, awkward, cupy, …) in many places.
Importing those packages just to get at their classes makes `import anndata` slow,
so :func:`lazy_type` creates a placeholder class that stands in for the real one:

- Checks against instances and concrete classes consult `sys.modules` without importing
  the dependency. View factories and resolution callbacks can import supporting modules;
  comparing two placeholders also imports their dependencies to determine inheritance.
- Once the dependency is imported, checks are delegated to the real class.
- Calling the placeholder or accessing a public attribute imports the dependency
  and forwards to the real class (e.g. `XDataset.from_dataframe(...)`).
- :func:`functools.singledispatch` treats the placeholder like a registered
  abstract base class, so `@f.register(DaskArray)` keeps working.
  Caveat: `singledispatch` ranks such virtual bases below *explicit* base classes of the
  real class (e.g. `Mapping` for `xarray.Dataset`) and cannot order two unrelated virtual
  bases (e.g. a placeholder and a runtime protocol). If a function registers both, handle
  the placeholder explicitly in the other implementation (see `_gen_dataframe`).
  Instances should only be dispatched after their dependency finishes importing. If a
  dispatcher caches a fallback while the class is still unpublished, a later check or
  explicit resolution of the placeholder is needed to invalidate that cache.

Placeholders cannot be subclassed; use `factory=` to lazily build a real subclass instead.
"""

from __future__ import annotations

import sys
import threading
from abc import ABCMeta
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType, NotImplementedType

__all__ = ["LazyType", "lazy_type", "lazy_type_for"]

_LAZY_TYPES: list[LazyType] = []
_REAL_TO_LAZY: dict[type, LazyType] = {}
# guards publishing a resolved class (never held while importing or running factories)
_PUBLISH_LOCK = threading.RLock()


class LazyType(ABCMeta):
    """Metaclass of the placeholders created by :func:`lazy_type`."""

    _lazy_module: str
    _lazy_name: str
    _lazy_factory: Callable[[], type] | None
    _lazy_on_resolve: Callable[[type], None] | None
    _lazy_real: type | None
    _lazy_publishing: int | None

    def __new__(
        mcls, name: str, bases: tuple[type, ...], namespace: dict[str, object], /, **kw
    ) -> LazyType:
        if any(isinstance(base, LazyType) for base in bases):
            msg = (
                f"Cannot subclass lazy placeholder type(s) {bases!r}; "
                "import the real class, or use `lazy_type(..., factory=...)`."
            )
            raise TypeError(msg)
        return super().__new__(mcls, name, bases, namespace, **kw)

    def _lazy_get(cls, *, import_: bool = False) -> type | None:
        """Get the real class, or `None` if its module is not loaded.

        If `import_` is `True`, the module is imported if it is importable.
        """
        if (real := cls._lazy_real) is not None:
            return real
        if cls._lazy_publishing == threading.get_ident():  # `ABCMeta.register` re-entry
            return None
        if (mod := cls._lazy_module_obj(import_=import_)) is None:
            return None
        # No lock is held here: factories import modules, and holding a lock across
        # imports can deadlock with the import lock of a thread initializing the module.
        if cls._lazy_factory is not None:
            real = cls._lazy_factory()
        elif (real := getattr(mod, cls._lazy_name, None)) is None:
            return None  # e.g. the module is still being initialized
        with _PUBLISH_LOCK:
            if (published := cls._lazy_real) is not None:
                return published  # another thread won; discard ours
            cls._lazy_publishing = threading.get_ident()
            try:
                # Register the real class as a virtual subclass *before* publishing it:
                # this bumps the ABC cache token, invalidating stale negative `isinstance`
                # results and `singledispatch` caches (also for subclasses of `real`).
                ABCMeta.register(cls, real)
                # The unlocked fast path may expose this class to IO immediately.
                # Publish its normalization entry before making the class visible.
                _REAL_TO_LAZY[real] = cls
                cls._lazy_real = real
            finally:
                cls._lazy_publishing = None
        if cls._lazy_on_resolve is not None:
            cls._lazy_on_resolve(real)
        return real

    def _lazy_module_obj(cls, *, import_: bool) -> ModuleType | None:
        if not import_:
            mod = sys.modules.get(cls._lazy_module)
            # A module is visible before its body finishes. A factory may import it,
            # so a non-importing check must not enter a factory at this point.
            if getattr(getattr(mod, "__spec__", None), "_initializing", False):
                return None
            return mod
        try:
            return import_module(cls._lazy_module)
        except ImportError:
            return None

    def _lazy_require(cls) -> type:
        """Get the real class, importing its module. Raises `ImportError` if not installed."""
        if (real := cls._lazy_get(import_=True)) is None:
            msg = f"{cls.__module__}.{cls.__qualname__} requires `{cls._lazy_module}` to be installed."
            raise ImportError(msg)
        return real

    def __instancecheck__(cls, instance: object) -> bool:
        # resolve first: ABCMeta’s negative cache would otherwise skip `__subclasshook__`
        cls._lazy_get()
        return super().__instancecheck__(instance)

    def __subclasscheck__(cls, subclass: type) -> bool:
        cls._lazy_get()
        return super().__subclasscheck__(subclass)

    def __call__(cls, *args: object, **kwargs: object) -> object:
        return cls._lazy_require()(*args, **kwargs)

    def __getattr__(cls, name: str) -> object:
        # only forward public attributes, so introspection does not trigger imports
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(cls._lazy_require(), name)

    def __getitem__(cls, item: object) -> object:
        """Forward `__class_getitem__` (e.g. `XDataset[int]`) to the real class."""
        return cls._lazy_require().__class_getitem__(item)  # type: ignore[attr-defined]

    def __repr__(cls) -> str:
        return f"<lazy class '{cls.__module__}.{cls.__qualname__}'>"


def _lazy_subclasshook(cls: type, sub: type, /) -> bool | NotImplementedType:
    assert isinstance(cls, LazyType)
    if isinstance(sub, LazyType):  # both symbolic: resolve them for real
        real = cls._lazy_get(import_=True)
        sub_real = sub._lazy_get(import_=True)
        if real is None or sub_real is None:
            return NotImplemented
        return issubclass(sub_real, real)
    if (real := cls._lazy_get()) is None:
        return NotImplemented
    return issubclass(sub, real)


def lazy_type(
    module: str,
    name: str,
    *,
    factory: Callable[[], type] | None = None,
    on_resolve: Callable[[type], None] | None = None,
    module_name: str | None = None,
) -> LazyType:
    """Create a placeholder for class `name` in optional `module`.

    Parameters
    ----------
    module
        Module the real class lives in (for `factory` types: the optional dependency's module).
    name
        Name of the real class in `module`, or of the placeholder when a `factory` is given.
    factory
        If given, builds the real class once `module` is loaded, instead of looking up `name`.
        Used to define subclasses of optional classes without importing them eagerly.
        Concurrent resolution may run the factory more than once; only one result is
        published. Factories must not publish their classes or create instances.
    on_resolve
        Called by the first publisher after the class and its IO mapping are visible.
        Other threads may use the class before this callback returns. Callback errors
        propagate to the publisher but do not undo publication.
    module_name
        `__module__` of the placeholder; defaults to `module`.
    """
    ns = dict(
        __module__=module_name or module,
        __qualname__=name,
        __subclasshook__=classmethod(_lazy_subclasshook),
        _lazy_module=module,
        _lazy_name=name,
        _lazy_factory=factory,
        _lazy_on_resolve=on_resolve,
        _lazy_real=None,
        _lazy_publishing=None,
    )
    typ = LazyType(name, (), ns)
    _LAZY_TYPES.append(typ)
    return typ


def lazy_type_for(typ: type) -> type:
    """Map a real class to its placeholder without importing an unloaded dependency."""
    if (lazy := _REAL_TO_LAZY.get(typ)) is not None:
        return lazy
    if isinstance(typ, LazyType):
        return typ
    top = typ.__module__.partition(".")[0]
    for lazy in _LAZY_TYPES:
        if lazy._lazy_module.partition(".")[0] == top and lazy._lazy_get() is typ:
            return lazy
    return typ
