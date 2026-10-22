"""Tests for lazily resolved optional dependency types (`anndata.compat._lazy`)."""

from __future__ import annotations

import subprocess
import sys
from functools import singledispatch
from importlib.util import find_spec

import numpy as np
import pytest

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

        class Sub(placeholder):  # type: ignore[misc]
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
