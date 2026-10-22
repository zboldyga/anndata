"""Tests for lazily resolved optional dependency types (`anndata.compat._lazy`)."""

from __future__ import annotations

import subprocess
import sys
from functools import singledispatch
from importlib.util import find_spec

import numpy as np
import pytest
from scipy import sparse

import anndata as ad
from anndata.compat import CupyArray, DaskArray, XDataset
from anndata.compat._lazy import lazy_type, lazy_type_for

OPTIONAL_DEPS = ["awkward", "cupy", "dask", "xarray", "zappy"]


def test_import_does_not_load_optional_deps():
    """`import anndata` must not import optional dependencies, see #1122."""
    code = (
        "import sys, anndata\n"
        f"loaded = sorted(m for m in {OPTIONAL_DEPS} if m in sys.modules)\n"
        "print(','.join(loaded))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert out.stdout.strip() == "", f"optional deps imported: {out.stdout.strip()}"


def test_placeholder_without_import():
    placeholder = lazy_type("anndata_nonexistent_pkg", "Thing")
    assert isinstance(placeholder, type)
    assert not isinstance(np.ones(1), placeholder)
    assert not issubclass(np.ndarray, placeholder)
    assert repr(placeholder) == "<lazy class 'anndata_nonexistent_pkg.Thing'>"
    with pytest.raises(ImportError, match="requires `anndata_nonexistent_pkg`"):
        placeholder()
    with pytest.raises(ImportError, match="requires `anndata_nonexistent_pkg`"):
        _ = placeholder.some_attr
    # private attributes never trigger an import
    assert not hasattr(placeholder, "_private")
    with pytest.raises(TypeError, match="Cannot subclass"):

        class Sub(placeholder):
            pass


def test_cupy_placeholder_without_cupy():
    if find_spec("cupy") is not None:
        pytest.skip("cupy is installed")
    assert not isinstance(np.ones(1), CupyArray)
    assert isinstance(np.ones(1), np.ndarray | CupyArray) is not False
    assert isinstance(np.ones(1), np.ndarray | CupyArray)


@pytest.mark.skipif(find_spec("dask") is None, reason="dask not installed")
def test_placeholder_resolves_to_real_class():
    import dask.array as da

    x = da.ones((3, 2))
    assert isinstance(x, DaskArray)
    assert issubclass(da.Array, DaskArray)
    assert issubclass(DaskArray, DaskArray)
    assert not isinstance(np.ones(3), DaskArray)
    assert lazy_type_for(da.Array) is DaskArray
    assert lazy_type_for(np.ndarray) is np.ndarray
    # calls and public attributes are forwarded to the real class
    assert DaskArray.__name__ == "Array"
    assert type(ad.AnnData(X=x).X) is da.Array


@pytest.mark.skipif(find_spec("dask") is None, reason="dask not installed")
def test_singledispatch_on_placeholder():
    import dask.array as da

    @singledispatch
    def f(_) -> str:
        return "default"

    @f.register(DaskArray)
    def _(_) -> str:
        return "dask"

    assert f(da.ones(3)) == "dask"
    assert f(np.ones(3)) == "default"
    assert f(ad.AnnData(X=da.ones((3, 2)))[:2].X) == "dask"  # view subclass


@pytest.mark.skipif(find_spec("xarray") is None, reason="xarray not installed")
def test_xdataset_mapping_dispatch():
    """`xarray.Dataset` is a `Mapping`, which must not shadow the `XDataset` impl."""
    import pandas as pd
    import xarray as xr

    from anndata._core.xarray import Dataset2D

    df = pd.DataFrame(dict(a=[1, 2, 3]), index=list("abc"))
    adata = ad.AnnData(X=np.zeros((3, 1)), obs=XDataset.from_dataframe(df))
    assert isinstance(adata.obs, Dataset2D)
    assert isinstance(adata.obs.ds, xr.Dataset)


def test_core_functionality_without_optional_deps():
    """Basic operations must work when the optional packages cannot be imported."""
    code = f"""
import sys
for mod in {OPTIONAL_DEPS}:
    sys.modules[mod] = None  # makes `import <mod>` raise ImportError
import numpy as np, pandas as pd, anndata as ad
a = ad.AnnData(np.ones((2, 2)), obs=pd.DataFrame(index=["a", "b"]))
b = ad.AnnData(np.zeros((3, 2)), obs=pd.DataFrame(index=["c", "d", "e"]))
c = ad.concat([a, b], join="outer")
assert c.shape == (5, 2)
v = c[:2]
assert v.is_view and v.copy().shape == (2, 2)
import tempfile, pathlib
with tempfile.TemporaryDirectory() as d:
    p = pathlib.Path(d) / "a.h5ad"
    c.write_h5ad(p)
    assert ad.read_h5ad(p).shape == (5, 2)
print("ok")
"""
    out = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert out.stdout.strip() == "ok"


def test_resolution_invalidates_negative_cache():
    """A negative `isinstance` cached before the class was available must be revisited."""
    import types

    name = "anndata_test_partial_pkg"
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    try:
        placeholder = lazy_type(name, "Thing")

        class Thing:
            pass

        inst = Thing()
        assert not isinstance(
            inst, placeholder
        )  # module loaded, class not published yet

        @singledispatch
        def f(_) -> str:
            return "default"

        @f.register(placeholder)
        def _(_) -> str:
            return "thing"

        assert f(inst) == "default"
        mod.Thing = Thing
        assert isinstance(inst, placeholder)
        assert f(inst) == "thing"
    finally:
        del sys.modules[name]


def test_factory_runs_once_across_threads():
    import threading
    import types

    name = "anndata_test_factory_pkg"
    sys.modules[name] = types.ModuleType(name)
    try:
        calls = []

        def factory() -> type:
            calls.append(1)
            return type("Built", (), {})

        placeholder = lazy_type(name, "Built", factory=factory)
        barrier = threading.Barrier(8)
        results: list[type | None] = []

        def resolve() -> None:
            barrier.wait()
            results.append(placeholder._lazy_get())

        threads = [threading.Thread(target=resolve) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(calls) == 1
        assert len({id(r) for r in results}) == 1
        assert lazy_type_for(results[0]) is placeholder
    finally:
        del sys.modules[name]


@pytest.mark.skipif(find_spec("dask") is None, reason="dask not installed")
def test_io_registry_accepts_real_types():
    """Registering or looking up write specs with the real class must hit the placeholder key."""
    import dask.array as da
    import h5py

    from anndata._io.specs.registry import _REGISTRY, IORegistry, IOSpec

    spec = _REGISTRY.get_spec(da.ones((2, 2)))
    assert spec == IOSpec("array", "0.2.0")
    assert _REGISTRY.has_write(h5py.Group, DaskArray, frozenset())
    assert _REGISTRY.has_write(h5py.Group, da.Array, frozenset())
    assert _REGISTRY.has_write(h5py.Group, (da.Array, sparse.csr_matrix), frozenset())

    reg = IORegistry()

    @reg.register_write(h5py.Group, (da.Array, np.ndarray), IOSpec("array", "0.2.0"))
    def _write(*_, **__) -> None:
        pass

    assert reg.has_write(h5py.Group, (DaskArray, np.ndarray), frozenset())
    assert reg.get_spec(da.ones((2, 2))) == IOSpec("array", "0.2.0")


@pytest.mark.skipif(find_spec("xarray") is None, reason="xarray not installed")
def test_class_getitem_forwarded():
    import xarray as xr

    assert XDataset[int] == xr.Dataset[int]


@pytest.mark.skipif(find_spec("dask") is None, reason="dask not installed")
def test_gen_adata_accepts_real_types():
    import dask.array as da

    from anndata.tests.helpers import DEFAULT_KEY_TYPES, gen_adata

    adata = gen_adata((5, 4), obsm_types=(*DEFAULT_KEY_TYPES, da.Array))
    assert isinstance(adata.obsm["da"], da.Array)
