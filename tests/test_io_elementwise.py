"""
Tests that each element in an anndata is written correctly
"""

from __future__ import annotations

import datetime as dt
import re
from contextlib import ExitStack, nullcontext
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, cast

import h5py
import numpy as np
import pandas as pd
import pytest
import zarr
from packaging.version import Version
from scipy import sparse
from zarr.storage import MemoryStore

import anndata as ad
from anndata._core.xarray import Dataset2D
from anndata._io.specs import _REGISTRY, IOSpec, get_spec
from anndata._io.specs.registry import IORegistryError, read_elem_partial
from anndata.compat import CSArray, CSMatrix, DaskArray, _read_attr
from anndata.experimental import read_elem_lazy
from anndata.io import read_elem, write_elem
from anndata.tests.helpers import (
    GEN_ADATA_NO_XARRAY_ARGS,
    as_cupy,
    as_cupy_sparse_dask_array,
    as_dense_cupy_dask_array,
    assert_equal,
    check_all_sharded,
    gen_adata,
    open_store,
    visititems_zarr,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path
    from typing import Literal

    from anndata._types import _GroupStorageType


PANDAS_3 = Version(version("pandas")) >= Version("3rc0")


@pytest.fixture
def exit_stack() -> Generator[ExitStack]:
    with ExitStack() as stack:
        yield stack


@pytest.fixture
def store(
    diskfmt: Literal["h5ad", "zarr"], tmp_path: Path
) -> Generator[h5py.Group | zarr.Group]:
    store: h5py.Group | zarr.Group
    if diskfmt == "h5ad":
        file = h5py.File(tmp_path / "test.h5ad", "w")
        store = cast("h5py.Group", file["/"])
    elif diskfmt == "zarr":
        store = zarr.open_group(MemoryStore(), mode="w")
    else:
        pytest.fail(f"Unknown store type: {diskfmt}")

    try:
        yield store
    finally:
        if diskfmt == "h5ad":
            file.close()


sparse_formats = ["csr", "csc"]
SIZE = 50
DEFAULT_SHAPE = (SIZE, SIZE * 2)


@pytest.fixture(params=sparse_formats)
def sparse_format(request: pytest.FixtureRequest) -> Literal["csr", "csc"]:
    return request.param


def create_dense_store(
    store: h5py.Group | zarr.Group, *, shape: tuple[int, ...] = DEFAULT_SHAPE
) -> h5py.Group | zarr.Group:
    X = np.random.randn(*shape)

    write_elem(store, "X", X)
    return store


def create_sparse_store[G: h5py.Group | zarr.Group](
    sparse_format: Literal["csc", "csr"], store: G, shape=DEFAULT_SHAPE
) -> G:
    """Returns a store

    Parameters
    ----------
    sparse_format
    store

    Returns
    -------
        A store with a key, `X` that is simply a sparse matrix, and `X_dask` where that same array is wrapped by dask
    """
    import dask.array as da

    X = sparse.random(
        shape[0],
        shape[1],
        format=sparse_format,
        density=0.01,
        random_state=np.random.default_rng(),
    )
    X_dask = da.from_array(
        X,
        chunks=(100 if format == "csr" else SIZE, SIZE * 2 if format == "csr" else 100),
    )

    write_elem(store, "X", X)
    write_elem(store, "X_dask", X_dask)
    return store


@pytest.mark.parametrize(
    ("value", "encoding_type"),
    [
        pytest.param(None, "null", id="none"),
        pytest.param("hello world", "string", id="py_str"),
        pytest.param(np.str_("hello world"), "string", id="np_str"),
        pytest.param(np.array([1, 2, 3]), "array", id="np_arr_int"),
        pytest.param(
            np.array(["hello", "world"], dtype=object), "string-array", id="np_arr_str"
        ),
        pytest.param(1, "numeric-scalar", id="py_int"),
        pytest.param(True, "numeric-scalar", id="py_bool"),
        pytest.param(1.0, "numeric-scalar", id="py_float"),
        pytest.param({"a": 1}, "dict", id="py_dict"),
        pytest.param(
            lambda: gen_adata((3, 2), **GEN_ADATA_NO_XARRAY_ARGS),
            "anndata",
            id="anndata",
        ),
        pytest.param(
            sparse.random(5, 3, format="csr", density=0.5),
            "csr_matrix",
            id="sp_mat_csr",
        ),
        pytest.param(
            sparse.random(5, 3, format="csc", density=0.5),
            "csc_matrix",
            id="sp_mat_csc",
        ),
        pytest.param(pd.DataFrame({"a": [1, 2, 3]}), "dataframe", id="pd_df"),
        pytest.param(
            pd.Categorical([*"aabccedd", pd.NA]),
            "categorical",
            id="pd_cat_np_str",
        ),
        pytest.param(
            pd.Categorical(list("aabccedd"), ordered=True),
            "categorical",
            id="pd_cat_np_str_ord",
        ),
        pytest.param(
            pd.array([*"aabccedd", pd.NA], dtype="string").astype("category"),
            "categorical",
            id="pd_cat_pd_str",
        ),
        pytest.param(
            pd.Categorical([1, 2, 1, 3], ordered=True), "categorical", id="pd_cat_num"
        ),
        pytest.param(
            pd.Categorical(np.repeat(pd.date_range("2018", periods=3, freq="YE"), 2)),
            "categorical",
            id="pd_cat_datetime",
        ),
        pytest.param(
            pd.Categorical(pd.date_range("2018", periods=3, tz="Europe/Berlin")),
            "categorical",
            id="pd_cat_datetime_tz",
        ),
        pytest.param(
            pd.date_range("2018-01-01", periods=5, freq="h").to_numpy(),
            "datetime-array",
            id="np_arr_datetime",
        ),
        pytest.param(
            pd.to_timedelta(np.arange(5), unit="s").to_numpy(),
            "timedelta-array",
            id="np_arr_timedelta",
        ),
        pytest.param(
            pd.array(pd.date_range("2018-01-01", periods=5, freq="h", tz="UTC")),
            "datetime-array",
            id="pd_arr_datetime_tz",
        ),
        pytest.param(
            pd.array(["hello", "world"], dtype="string"),
            "nullable-string-array",
            id="pd_arr_str",
        ),
        pytest.param(
            pd.array(["hello", "world", pd.NA], dtype="string"),
            "nullable-string-array",
            id="pd_arr_str_mask",
        ),
        pytest.param(
            pd.arrays.IntegerArray(
                np.ones(5, dtype=int), mask=np.array([True, False, True, False, True])
            ),
            "nullable-integer",
            id="pd_arr_int_mask",
        ),
        pytest.param(pd.array([1, 2, 3]), "nullable-integer", id="pd_arr_int"),
        pytest.param(
            pd.arrays.BooleanArray(
                np.random.randint(0, 2, size=5, dtype=bool),
                mask=np.random.randint(0, 2, size=5, dtype=bool),
            ),
            "nullable-boolean",
            id="pd_arr_bool_mask",
        ),
        pytest.param(
            pd.array([True, False, True, True]), "nullable-boolean", id="pd_arr_bool"
        ),
        pytest.param(
            zarr.ones((100, 100), chunks=(10, 10)),
            "array",
            id="zarr_dense_array",
        ),
        pytest.param(
            create_dense_store(
                h5py.File("test1.h5", mode="w", driver="core", backing_store=False)
            )["X"],
            "array",
            id="h5_dense_array",
        ),
        # pytest.param(bytes, b"some bytes", "bytes", id="py_bytes"), # Does not work for zarr
        # TODO consider how specific encodings should be. Should we be fully describing the written type?
        # Currently the info we add is: "what you wouldn't be able to figure out yourself"
        # but that's not really a solid rule.
        # pytest.param(bool, True, "bool", id="py_bool"),
        # pytest.param(bool, np.bool_(False), "bool", id="np_bool"),
    ],
)
def test_io_spec(store: _GroupStorageType, value, encoding_type) -> None:
    if callable(value):
        value = value()

    key = f"key_for_{encoding_type}"
    with ad.settings.override(allow_write_nullable_strings=True):
        write_elem(store, key, value, dataset_kwargs={})

    assert encoding_type == _read_attr(store[key].attrs, "encoding-type")

    from_disk = read_elem(store[key])
    assert_equal(value, from_disk)
    assert get_spec(store[key]) == _REGISTRY.get_spec(value)


@pytest.mark.parametrize(
    ("value", "encoding_type"),
    [
        pytest.param(np.asarray(1), "numeric-scalar", id="scalar_int"),
        pytest.param(np.asarray(1.0), "numeric-scalar", id="scalar_float"),
        pytest.param(np.asarray(True), "numeric-scalar", id="scalar_bool"),  # noqa: FBT003
        pytest.param(np.asarray("test"), "string", id="scalar_string"),
    ],
)
def test_io_spec_compressed_scalars(
    store: h5py.Group | zarr.Group, value: np.ndarray, encoding_type: str
):
    key = f"key_for_{encoding_type}"
    write_elem(
        store, key, value, dataset_kwargs={"compression": "gzip", "compression_opts": 5}
    )

    assert encoding_type == _read_attr(store[key].attrs, "encoding-type")

    from_disk = read_elem(store[key])
    assert_equal(value, from_disk)


# Can't instantiate cupy types at the top level, so converting them within the test
@pytest.mark.gpu
@pytest.mark.parametrize(
    ("value", "encoding_type"),
    [
        pytest.param(np.array([1, 2, 3]), "array", id="np-1d"),
        pytest.param(np.arange(12).reshape(4, 3), "array", id="np-2d"),
        pytest.param(
            sparse.random(5, 3, format="csr", density=0.5), "csr_matrix", id="csr"
        ),
        pytest.param(
            sparse.random(5, 3, format="csc", density=0.5), "csc_matrix", id="csc"
        ),
    ],
)
@pytest.mark.parametrize("as_dask", [False, True], ids=["local", "dask"])
def test_io_spec_cupy(store, value, encoding_type, as_dask):
    if not as_dask:
        value = as_cupy(value)
    elif isinstance(value, CSMatrix | CSArray):
        value = as_cupy_sparse_dask_array(value, format=encoding_type[:3])
    else:
        value = as_dense_cupy_dask_array(value)

    key = f"key_for_{encoding_type}"
    write_elem(store, key, value, dataset_kwargs={})

    assert encoding_type == _read_attr(store[key].attrs, "encoding-type")

    from_disk = as_cupy(read_elem(store[key]))
    assert_equal(value, from_disk)
    assert get_spec(store[key]) == _REGISTRY.get_spec(value)


def test_dask_write_sparse(sparse_format, store):
    x_sparse_store = create_sparse_store(sparse_format, store)
    X_from_disk = read_elem(x_sparse_store["X"])
    X_dask_from_disk = read_elem(x_sparse_store["X_dask"])

    assert_equal(X_from_disk, X_dask_from_disk)
    assert_equal(dict(x_sparse_store["X"].attrs), dict(x_sparse_store["X_dask"].attrs))

    assert x_sparse_store["X_dask/indptr"].dtype == np.int64
    assert x_sparse_store["X_dask/indices"].dtype == np.int64


def test_read_lazy_2d_dask(sparse_format, store):
    arr_store = create_sparse_store(sparse_format, store)
    X_dask_from_disk = read_elem_lazy(arr_store["X"])
    X_from_disk = read_elem(arr_store["X"])

    assert_equal(X_from_disk, X_dask_from_disk)
    random_int_indices = np.random.randint(0, SIZE, (SIZE // 10,))
    random_int_indices.sort()
    index_slice = slice(0, SIZE // 10)
    for index in [random_int_indices, index_slice]:
        assert_equal(X_from_disk[index, :], X_dask_from_disk[index, :])
        assert_equal(X_from_disk[:, index], X_dask_from_disk[:, index])
    random_bool_mask = np.random.randn(SIZE) > 0
    assert_equal(
        X_from_disk[random_bool_mask, :], X_dask_from_disk[random_bool_mask, :]
    )
    random_bool_mask = np.random.randn(SIZE * 2) > 0
    assert_equal(
        X_from_disk[:, random_bool_mask], X_dask_from_disk[:, random_bool_mask]
    )

    assert arr_store["X_dask/indptr"].dtype == np.int64
    assert arr_store["X_dask/indices"].dtype == np.int64


@pytest.mark.parametrize(
    ("n_dims", "chunks"),
    [
        (1, (10,)),
        (1, (40,)),
        (2, (10, 10)),
        (2, (40, 40)),
        (2, (20, 40)),
        (1, None),
        (2, None),
        (2, (40, -1)),
        (2, (40, None)),
    ],
)
def test_read_lazy_subsets_nd_dask(
    store: h5py.Group | zarr.Group, n_dims, chunks
) -> None:
    arr_store = create_dense_store(store, shape=DEFAULT_SHAPE[:n_dims])
    X_dask_from_disk = read_elem_lazy(arr_store["X"], chunks=chunks)
    X_from_disk = read_elem(arr_store["X"])
    assert isinstance(X_from_disk, np.ndarray)
    assert isinstance(X_dask_from_disk, DaskArray)
    assert_equal(X_from_disk, X_dask_from_disk)

    random_int_indices = np.random.randint(0, SIZE, (SIZE // 10,))
    random_int_indices.sort()
    random_bool_mask = np.random.randn(SIZE) > 0
    index_slice = slice(0, SIZE // 10)
    indices: list[np.ndarray | slice] = [
        random_int_indices,
        index_slice,
        random_bool_mask,
    ]
    for index in indices:
        assert_equal(X_from_disk[index], X_dask_from_disk[index])


@pytest.mark.xdist_group("dask")
@pytest.mark.dask_distributed
def test_read_lazy_h5_cluster(
    sparse_format: Literal["csr", "csc"],
    tmp_path: Path,
    local_cluster_addr: str,
) -> None:
    import dask.distributed as dd

    with h5py.File(tmp_path / "test.h5", "w") as file:
        store = file["/"]
        arr_store = create_sparse_store(sparse_format, store)
        X_dask_from_disk = read_elem_lazy(arr_store["X"])
        X_from_disk = read_elem(arr_store["X"])
    with dd.Client(local_cluster_addr):
        assert_equal(X_from_disk, X_dask_from_disk)


def test_undersized_shape_to_default(store: h5py.Group | zarr.Group) -> None:
    shape = (1000, 50)
    arr_store = create_dense_store(store, shape=shape)
    X_dask_from_disk = read_elem_lazy(arr_store["X"])
    assert isinstance(X_dask_from_disk, DaskArray)
    assert all(c <= s for c, s in zip(X_dask_from_disk.chunksize, shape, strict=True))
    assert X_dask_from_disk.shape == shape


@pytest.mark.parametrize(
    ("arr_type", "chunks", "expected_chunksize"),
    [
        ("dense", (10, 10), (10, 10)),
        ("csc", (SIZE, 1), (SIZE, 1)),
        ("csr", (1, SIZE * 2), (1, SIZE * 2)),
        ("csc", None, (SIZE, 100)),
        ("csr", None, DEFAULT_SHAPE),
        ("csr", (10, -1), (10, SIZE * 2)),
        ("csc", (-1, 10), (SIZE, 10)),
        ("csr", (10, None), (10, SIZE * 2)),
        ("csc", (None, 10), (SIZE, 10)),
        ("csc", (None, None), DEFAULT_SHAPE),
        ("csr", (None, None), DEFAULT_SHAPE),
        ("csr", (-1, -1), DEFAULT_SHAPE),
        ("csc", (-1, -1), DEFAULT_SHAPE),
    ],
)
def test_read_lazy_2d_chunk_kwargs(
    store: h5py.Group | zarr.Group,
    arr_type: Literal["csr", "csc", "dense"],
    chunks: tuple[int | None, int | None] | None,
    expected_chunksize: tuple[int, int],
) -> None:
    if arr_type == "dense":
        arr_store = create_dense_store(store)
        X_dask_from_disk = read_elem_lazy(arr_store["X"], chunks=chunks)
    else:
        arr_store = create_sparse_store(arr_type, store)
        X_dask_from_disk = read_elem_lazy(arr_store["X"], chunks=chunks)
    assert isinstance(X_dask_from_disk, DaskArray)
    assert X_dask_from_disk.chunksize == expected_chunksize
    X_from_disk = read_elem(arr_store["X"])
    assert_equal(X_from_disk, X_dask_from_disk)


def test_read_lazy_bad_chunk_kwargs(tmp_path):
    arr_type = "csr"
    with h5py.File(tmp_path / "test.h5", "w") as file:
        store = file["/"]
        arr_store = create_sparse_store(arr_type, store)
        with pytest.raises(
            ValueError, match=r"`chunks` must be a tuple of two integers"
        ):
            read_elem_lazy(arr_store["X"], chunks=(SIZE,))
        with pytest.raises(ValueError, match=r"Only the major axis can be chunked"):
            read_elem_lazy(arr_store["X"], chunks=(SIZE, 10))


@pytest.mark.parametrize("sparse_format", ["csr", "csc"])
def test_write_indptr_dtype_override(store, sparse_format):
    X = sparse.random(
        100,
        100,
        format=sparse_format,
        density=0.1,
        random_state=np.random.default_rng(),
    )

    write_elem(store, "X", X, dataset_kwargs=dict(indptr_dtype="int64"))

    assert store["X/indptr"].dtype == np.int64
    assert X.indptr.dtype == np.int32
    np.testing.assert_array_equal(store["X/indptr"][...], X.indptr)


@pytest.fixture(params=["indices-data", "shape"])
def bad_sparray(request: pytest.FixtureRequest):
    m = sparse.random_array((100, 100), format="csr", density=0.1)
    match request.param:
        case "indices-data":
            m.indices = np.zeros(len(m.indices) * 2, dtype=m.indices.dtype)
        case "shape":
            m._shape = (10**11, m.shape[1])  # type: ignore[attr-defined]
        case _:
            pytest.fail(f"Unknown matrix type: {request.param}")
    return m


def test_write_bad_sparray(
    diskfmt_store: Path | MemoryStore, bad_sparray: sparse.csr_array
) -> None:
    f = open_store(diskfmt_store)

    with pytest.raises(
        ValueError, match=r"index pointer size|should have the same size"
    ):
        ad.io.write_elem(f, "mtx", bad_sparray)


def test_read_bad_sparray(
    diskfmt_store: Path | MemoryStore, bad_sparray: sparse.csr_array
) -> None:
    f = open_store(diskfmt_store)
    x = f.create_group("mtx")
    x.attrs["encoding-type"] = "csr_matrix"
    x.attrs["encoding-version"] = "0.1.0"
    x.attrs["shape"] = bad_sparray.shape
    c = x.create_dataset if isinstance(x, h5py.Group) else x.create_array
    c("data", data=bad_sparray.data)
    c("indices", data=bad_sparray.indices)
    c("indptr", data=bad_sparray.indptr)

    with pytest.raises(
        ValueError, match=r"index pointer size|should have the same size"
    ):
        ad.io.read_elem(f["mtx"])


@pytest.mark.parametrize(
    ("num_minor_axis", "expected_dtype"),
    [
        pytest.param(1, np.dtype("uint8"), id="one_col-expected_uint8_on_disk"),
        pytest.param(
            np.iinfo(np.uint8).max,
            np.dtype("uint8"),
            id="max_np.uint8-matching_dtype_on_disk",
        ),
        pytest.param(
            np.iinfo(np.int8).max,
            np.dtype("uint8"),
            id="max_np.int8-uint8_on_disk",
        ),
        pytest.param(
            np.iinfo(np.uint16).max,
            np.dtype("uint16"),
            id="max_np.uint16-matching_dtype_on_disk",
        ),
        pytest.param(
            np.iinfo(np.int16).max,
            np.dtype("uint16"),
            id="max_np.int16-uint16_on_disk",
        ),
        pytest.param(
            np.iinfo(np.uint32).max,
            np.dtype("uint32"),
            id="max_np.uint32-matching_dtype_on_disk",
        ),
        pytest.param(
            np.iinfo(np.int32).max,
            np.dtype("uint32"),
            id="max_np.int32-uint32_on_disk",
        ),
        pytest.param(
            np.iinfo(np.uint8).max + 1,
            np.dtype("uint16"),
            id="max_np.uint8_plus_one_cols-expected_uint16_on_disk",
        ),
        pytest.param(
            np.iinfo(np.uint16).max + 1,
            np.dtype("uint32"),
            id="max_np.uint16_plus_one_cols-expected_uint32_on_disk",
        ),
        pytest.param(
            np.iinfo(np.uint32).max + 1,
            np.dtype("uint64"),
            id="max_np.uint32_plus_one_cols-expected_uint64_on_disk",
        ),
        pytest.param(
            np.iinfo(np.int64).max + 1,
            np.dtype("uint64"),
            id="max_np.int64_plus_one_cols-expected_uint64_on_disk",
            marks=pytest.mark.xfail(
                reason="scipy sparse does not support bigger than max(int64) values in indices and there is no uint128."
            ),
        ),
        pytest.param(
            np.iinfo(np.uint64).max + 1,
            np.dtype("uint64"),
            id="max_np.uint64_plus_one_cols-expected_uint64_on_disk",
            marks=pytest.mark.xfail(
                reason="scipy sparse does not support bigger than max(int64) values in indices and there is no uint128."
            ),
        ),
    ],
)
@pytest.mark.parametrize("format", ["csr", "csc"])
def test_write_indices_min(
    store: h5py.Group | zarr.Group,
    num_minor_axis: int,
    expected_dtype: np.dtype,
    format: Literal["csr", "csc"],
):
    minor_axis_index = np.array([num_minor_axis - 1])
    major_axis_index = np.array([10])
    row_cols = (
        (minor_axis_index, major_axis_index)
        if format == "csc"
        else (major_axis_index, minor_axis_index)
    )
    shape = (num_minor_axis, 20) if format == "csc" else (20, num_minor_axis)
    X = getattr(sparse, f"{format}_array")(
        (np.array([10]), row_cols),
        shape=shape,
    )
    assert X.nnz == 1
    with ad.settings.override(write_csr_csc_indices_with_min_possible_dtype=True):
        write_elem(store, "X", X)

    indices = store["X/indices"]
    assert isinstance(indices, h5py.Dataset | zarr.Array)
    assert indices.dtype == expected_dtype
    with ad.settings.override(use_sparse_array_on_read=True):
        result = read_elem(store["X"])
    assert isinstance(result, CSArray)
    assert_equal(result.data, X.data)
    assert_equal(result.indices, X.indices)
    assert_equal(result.indptr, X.indptr)
    assert X.format == result.format
    assert result.shape == X.shape
    # != comparison converts to csr, which allocates a lot of memory or errors out with:
    # ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
    # Because the old, very large, minor axis is now the major axis and so either it fails to create or the indptr is very big.
    # The above tests should be enough to capture the desired equality checks so this is mostly for being extra sure.
    # See https://github.com/scipy/scipy/issues/23826
    if not (format == "csc" and num_minor_axis > np.iinfo(np.uint16).max + 1):
        assert (result != X).nnz == 0


def test_io_spec_raw(store):
    adata = gen_adata((3, 2), **GEN_ADATA_NO_XARRAY_ARGS)
    adata.raw = adata.copy()

    write_elem(store, "adata", adata)

    assert _read_attr(store["adata/raw"].attrs, "encoding-type") == "raw"

    from_disk = read_elem(store["adata"])
    assert_equal(from_disk.raw, adata.raw)


def test_write_anndata_to_root(store):
    adata = gen_adata((3, 2), **GEN_ADATA_NO_XARRAY_ARGS)

    write_elem(store, "/", adata)
    # TODO: see https://github.com/zarr-developers/zarr-python/issues/2716
    if isinstance(store, zarr.Group):
        store = zarr.open_group(store.store)
    from_disk = read_elem(store)

    assert _read_attr(store.attrs, "encoding-type") == "anndata"
    assert_equal(from_disk, adata)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("encoding-type", "floob"),
        ("encoding-version", "10000.0"),
    ],
)
def test_read_iospec_not_found(store, attribute, value):
    adata = gen_adata((3, 2), **GEN_ADATA_NO_XARRAY_ARGS)

    write_elem(store, "/", adata)
    store["obs"].attrs.update({attribute: value})

    with pytest.raises(IORegistryError) as exc_info:
        read_elem(store)
    msg = str(exc_info.value)

    assert "No read method registered for IOSpec" in msg
    assert f"{attribute.replace('-', '_')}='{value}'" in msg


@pytest.mark.parametrize(
    "obj",
    [(b"x",)],
)
def test_write_io_error(store, obj):
    full_pattern = re.compile(
        rf"No method registered for writing {type(obj)} into .*Group"
    )

    with pytest.raises(IORegistryError, match=r"while writing key '/el'") as exc_info:
        write_elem(store, "/el", obj)

    msg = str(exc_info.value)
    assert re.search(full_pattern, msg)


DATETIME_LIKE_CASES = pytest.mark.parametrize(
    ("value", "encoding_type", "tz"),
    [
        pytest.param(
            np.array(["2020-01-01", "NaT", "1969-12-31"], dtype="datetime64[D]"),
            "datetime-array",
            None,
            id="np_datetime_nat",
        ),
        pytest.param(
            np.array(["2020-01-01T00:00:00.123456789", "NaT"], dtype="datetime64[ns]"),
            "datetime-array",
            None,
            id="np_datetime_ns",
        ),
        pytest.param(
            np.array([1, "NaT"], dtype=">M8[ns]"),
            "datetime-array",
            None,
            id="np_datetime_big_endian",
        ),
        pytest.param(
            pd.date_range("2018", periods=6, freq="h").to_numpy().reshape(3, 2),
            "datetime-array",
            None,
            id="np_datetime_2d",
        ),
        pytest.param(
            np.array([1, -2, "NaT"], dtype="timedelta64[ms]"),
            "timedelta-array",
            None,
            id="np_timedelta_nat",
        ),
        pytest.param(
            np.array([1, "NaT"], dtype=">m8[s]"),
            "timedelta-array",
            None,
            id="np_timedelta_big_endian",
        ),
        pytest.param(
            pd.array(pd.to_datetime(["2018-01-01", "NaT", "2018-01-03"])),
            "datetime-array",
            None,
            id="pd_datetime_nat",
        ),
        pytest.param(
            pd.array(
                pd.date_range("2018-03-25", periods=5, freq="h", tz="Europe/Berlin")
            ),
            "datetime-array",
            "Europe/Berlin",
            id="pd_datetime_tz_dst",
        ),
        pytest.param(
            pd.array(pd.date_range("2018", periods=2, freq="h", tz=dt.UTC)),
            "datetime-array",
            "UTC",
            id="pd_datetime_tz_utc",
        ),
        pytest.param(
            # `str()` of this timezone is "CET", which pandas would resolve to the DST observing zone
            pd.array(
                pd.to_datetime(["2020-07-01", "2020-01-01"]).tz_localize(
                    dt.timezone(dt.timedelta(hours=1), "CET")
                )
            ),
            "datetime-array",
            "UTC+01:00",
            id="pd_datetime_tz_fixed_named",
        ),
        pytest.param(
            pd.array(
                pd.date_range(
                    "2018",
                    periods=2,
                    tz=dt.timezone(-dt.timedelta(hours=5, minutes=30)),
                )
            ),
            "datetime-array",
            "UTC-05:30",
            id="pd_datetime_tz_fixed_negative",
        ),
        pytest.param(
            # outside the range of `datetime.datetime`
            pd.array(
                np.array(["12000-01-01", "NaT"], dtype="datetime64[s]")
            ).tz_localize("UTC"),
            "datetime-array",
            "UTC",
            id="pd_datetime_tz_far_future",
        ),
        pytest.param(
            pd.array(pd.date_range("2020", periods=6, tz="UTC")).reshape(2, 3),
            "datetime-array",
            "UTC",
            id="pd_datetime_tz_2d",
        ),
        pytest.param(
            pd.array(pd.to_timedelta([1, None, 3], unit="s")),
            "timedelta-array",
            None,
            id="pd_timedelta_nat",
        ),
    ],
)


def _expected_unit(
    value: np.ndarray | pd.arrays.DatetimeArray | pd.arrays.TimedeltaArray,
) -> str:
    """The unit is whatever the input had (which depends on the pandas version for pandas inputs)."""
    if isinstance(value, pd.arrays.DatetimeArray | pd.arrays.TimedeltaArray):
        return value.unit
    return np.datetime_data(value.dtype)[0]


def _expected_naive(
    value: np.ndarray | pd.arrays.DatetimeArray | pd.arrays.TimedeltaArray,
) -> np.ndarray:
    """The timezone-naive (UTC) numpy equivalent of `value`."""
    if isinstance(value, pd.arrays.DatetimeArray) and value.tz is not None:
        return np.asarray(value.tz_convert("UTC").tz_localize(None))
    return np.asarray(value)


@DATETIME_LIKE_CASES
def test_io_datetime_like(
    store: _GroupStorageType,
    value: np.ndarray | pd.arrays.DatetimeArray | pd.arrays.TimedeltaArray,
    encoding_type: str,
    tz: str | None,
) -> None:
    """Datetime-like arrays round-trip (incl. NaT) with the same layout in both stores."""
    key = f"key_for_{encoding_type}"
    write_elem(store, key, value)

    elem = store[key]
    assert isinstance(elem, h5py.Dataset | zarr.Array)
    assert _read_attr(elem.attrs, "encoding-type") == encoding_type
    assert _read_attr(elem.attrs, "unit") == _expected_unit(value)
    assert _read_attr(elem.attrs, "tz", default=None) == tz
    # On disk, the counts are stored as plain integers
    assert np.dtype(elem.dtype) == np.dtype(np.int64)

    from_disk = read_elem(elem)
    assert isinstance(
        from_disk, np.ndarray | pd.arrays.DatetimeArray | pd.arrays.TimedeltaArray
    )
    assert from_disk.shape == value.shape
    if tz is not None:
        assert isinstance(value, pd.arrays.DatetimeArray)
        assert isinstance(from_disk, pd.arrays.DatetimeArray)
        assert from_disk.unit == value.unit
        assert from_disk.tz is not None
        assert value.tz is not None
        # Same instants, same UTC offsets (the timezone object may differ, e.g. by name)
        pd.testing.assert_extension_array_equal(
            from_disk.tz_convert("UTC"), value.tz_convert("UTC")
        )
        assert from_disk.tz.utcoffset(dt.datetime(2020, 7, 1)) == value.tz.utcoffset(
            dt.datetime(2020, 7, 1)
        )
        assert from_disk.tz.utcoffset(dt.datetime(2020, 1, 1)) == value.tz.utcoffset(
            dt.datetime(2020, 1, 1)
        )
    else:
        assert isinstance(from_disk, np.ndarray)
        assert from_disk.dtype == np.asarray(value).dtype.newbyteorder("=")
        np.testing.assert_array_equal(from_disk, np.asarray(value))

    # Partial reads keep the dtype as well
    partial = read_elem_partial(elem, indices=slice(1, None))
    assert partial.dtype == from_disk.dtype
    assert partial.shape == (len(value) - 1, *value.shape[1:])
    assert read_elem_partial(elem, indices=slice(0, 0)).shape == (0, *value.shape[1:])
    first = read_elem_partial(elem, indices=0)
    expected_first = from_disk[0]
    if tz is not None and value.ndim == 1:
        assert isinstance(first, pd.Timestamp)
        assert isinstance(value, pd.arrays.DatetimeArray)
        assert first == expected_first
        assert first.unit == value.unit
    elif tz is not None:
        assert isinstance(first, pd.arrays.DatetimeArray)
        assert isinstance(expected_first, pd.arrays.DatetimeArray)
        pd.testing.assert_extension_array_equal(first, expected_first)
    else:
        np.testing.assert_array_equal(first, expected_first)


@DATETIME_LIKE_CASES
def test_io_datetime_like_lazy(
    store: _GroupStorageType,
    value: np.ndarray | pd.arrays.DatetimeArray | pd.arrays.TimedeltaArray,
    encoding_type: str,
    tz: str | None,
) -> None:
    """Lazy reads produce dask arrays of the right dtype (timezones can not be represented)."""
    key = f"key_for_{encoding_type}"
    write_elem(store, key, value)
    expected = _expected_naive(value)
    warns_tz = pytest.warns(UserWarning, match=r"returned as timezone-naive UTC")

    with warns_tz if tz is not None else nullcontext():
        lazy = read_elem_lazy(store[key])
    assert isinstance(lazy, DaskArray)
    assert lazy.dtype == expected.dtype.newbyteorder("=")
    np.testing.assert_array_equal(lazy.compute(), expected)

    # Round-tripping a (dask backed) lazily read array goes through the same encoding
    key_copy = f"{key}_copy"
    with warns_tz if tz is not None else nullcontext():
        write_elem(store, key_copy, read_elem_lazy(store[key]))
    assert _read_attr(store[key_copy].attrs, "encoding-type") == encoding_type
    assert _read_attr(store[key_copy].attrs, "unit") == _expected_unit(value)
    np.testing.assert_array_equal(read_elem(store[key_copy]), expected)


def test_read_datetime_partial_orthogonal(store: _GroupStorageType) -> None:
    value = pd.date_range("2018", periods=12, freq="h").to_numpy().reshape(3, 4)
    write_elem(store, "key", value)
    elem = store["key"]
    assert isinstance(elem, h5py.Dataset | zarr.Array)

    # h5py only supports one fancy index per selection, zarr uses orthogonal indexing
    indices = (
        ([0, 2], [1, 3]) if isinstance(elem, zarr.Array) else ([0, 2], slice(1, 4, 2))
    )
    np.testing.assert_array_equal(
        read_elem_partial(elem, indices=indices), value[[0, 2]][:, [1, 3]]
    )


def test_read_big_endian_datetime_dataset(store: _GroupStorageType) -> None:
    """A store written big-endian by another implementation is read correctly."""
    if isinstance(store, zarr.Group):
        arr = store.create_array("key", shape=(2,), dtype=">i8")
        arr[...] = np.array([5, 6], dtype=">i8")
    else:
        arr = store.create_dataset("key", data=np.array([5, 6], dtype=">i8"))
    arr.attrs.update({
        "encoding-type": "datetime-array",
        "encoding-version": "0.1.0",
        "unit": "s",
    })

    expected = np.array([5, 6], dtype="datetime64[s]")
    np.testing.assert_array_equal(read_elem(store["key"]), expected)
    lazy = read_elem_lazy(store["key"])
    assert isinstance(lazy, DaskArray)
    np.testing.assert_array_equal(lazy.compute(), expected)


def test_read_lazy_dataframe_datetime_columns(store: _GroupStorageType) -> None:
    pytest.importorskip("xarray")
    df = pd.DataFrame(
        {
            "datetime": np.array(["2020-01-01", "NaT"], dtype="datetime64[s]"),
            "timedelta": np.array([1, "NaT"], dtype="timedelta64[s]"),
            "datetime_cat": pd.Categorical(
                pd.to_datetime(["2020-01-01", "2021-01-01"])
            ),
            "datetime_tz_cat": pd.Categorical(
                pd.date_range("2020", periods=2, tz="UTC")
            ),
            "timedelta_cat": pd.Categorical(pd.to_timedelta([1, 2], unit="s")),
        },
        index=["a", "b"],
    )
    write_elem(store, "df", df)

    lazy = read_elem_lazy(store["df"])
    assert isinstance(lazy, Dataset2D)
    assert lazy["datetime"].dtype == df["datetime"].dtype
    assert lazy["timedelta"].dtype == df["timedelta"].dtype
    pd.testing.assert_frame_equal(lazy.to_memory(), df)


@pytest.mark.parametrize(
    ("value", "exc", "pattern"),
    [
        pytest.param(
            np.array([1, 2], dtype="datetime64[25s]"),
            ValueError,
            r"explicit unit and a unit step",
            id="non_unit_step",
        ),
        pytest.param(
            np.array(["NaT"], dtype="datetime64"),
            ValueError,
            r"explicit unit and a unit step",
            id="generic_unit",
        ),
        pytest.param(
            np.array(np.datetime64("2020-01-01", "s")),
            NotImplementedError,
            r"zero-dimensional datetime64\[s\] arrays",
            id="zero_dim",
        ),
    ],
)
def test_write_datetime_like_errors(
    store: _GroupStorageType, value: np.ndarray, exc: type[Exception], pattern: str
) -> None:
    with pytest.raises(exc, match=pattern):
        write_elem(store, "key", value)
    assert "key" not in store


@pytest.mark.parametrize("dtype", ["int32", "float64"], ids=["int32", "float64"])
def test_write_datetime_like_dtype_override(store: _GroupStorageType, dtype: str):
    value = pd.date_range("2018", periods=2).to_numpy()
    with pytest.raises(ValueError, match=r"stored as int64 counts"):
        write_elem(store, "key", value, dataset_kwargs={"dtype": dtype})
    assert "key" not in store
    # An explicit int64 is fine
    write_elem(store, "key", value, dataset_kwargs={"dtype": "int64"})
    np.testing.assert_array_equal(read_elem(store["key"]), value)


class _UnnamedTz(dt.tzinfo):
    """A timezone whose string form pandas cannot parse (like `dateutil.tz.tzutc()`)."""

    def utcoffset(self, _: dt.datetime | None) -> dt.timedelta:
        return dt.timedelta(hours=1)

    def dst(self, _: dt.datetime | None) -> dt.timedelta:
        return dt.timedelta(0)

    def tzname(self, _: dt.datetime | None) -> str:
        return "unnamed"


def test_write_datetime_unparseable_tz(store: _GroupStorageType) -> None:
    value = pd.array(pd.date_range("2018", periods=3, freq="h", tz=_UnnamedTz()))
    assert value.tz is not None
    with pytest.raises(ValueError, match=r"timezone.*_UnnamedTz.*tz_convert"):
        write_elem(store, "key", value)
    assert "key" not in store


def test_read_native_zarr_datetime_array() -> None:
    """Zarr arrays written as native datetimes with the `array` encoding (anndata < 0.14) stay readable."""
    value = pd.date_range("2018-01-01", periods=5, freq="h").to_numpy()
    group = zarr.open_group(MemoryStore(), mode="w")
    arr = group.create_array("key", shape=value.shape, dtype=value.dtype)
    arr[...] = value
    arr.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})

    np.testing.assert_array_equal(read_elem(group["key"]), value)
    lazy = read_elem_lazy(group["key"])
    assert isinstance(lazy, DaskArray)
    np.testing.assert_array_equal(lazy.compute(), value)


PAT_IMPLICIT = r"allow_write_nullable_strings.*None.*future\.infer_string.*False"
ALWAYS_INFER_STRING = pytest.mark.skipif(
    PANDAS_3, reason="Can’t disable `future.infer_string` in pandas 3+."
)


@pytest.mark.parametrize(
    ("ad_setting", "pd_setting", "expected_missing", "expected_no_missing"),
    [
        # when explicitly disabled, we expect an error when trying to write an array with missing values
        # and expect an array without missing values to be written in the old, non-nullable format
        *(
            pytest.param(
                False,
                pd_ignored,
                (ValueError, r"missing values.*allow_write_nullable_strings.*False"),
                "string-array",
                id=f"off-explicit-{int(pd_ignored)}",
                marks=() if pd_ignored else ALWAYS_INFER_STRING,
            )
            for pd_ignored in [False, True]
        ),
        # when implicitly disabled, we expect a different message in both cases
        pytest.param(
            None,
            False,
            (RuntimeError, PAT_IMPLICIT),
            (RuntimeError, PAT_IMPLICIT),
            id="off-implicit",
            marks=ALWAYS_INFER_STRING,
        ),
        # when enabled, we expect arrays to be written in the nullable format
        pytest.param(None, True, *(["nullable-string-array"] * 2), id="on-implicit"),
        pytest.param(
            True,
            False,
            *(["nullable-string-array"] * 2),
            id="on-explicit-0",
            marks=ALWAYS_INFER_STRING,
        ),
        pytest.param(True, True, *(["nullable-string-array"] * 2), id="on-explicit-1"),
    ],
)
@pytest.mark.parametrize("missing", [True, False], ids=["missing", "no_missing"])
def test_write_nullable_string(
    *,
    store: _GroupStorageType,
    ad_setting: bool | None,
    pd_setting: bool,
    expected_missing: tuple[type[Exception], str] | str,
    expected_no_missing: tuple[type[Exception], str] | str,
    missing: bool,
) -> None:
    expected = expected_missing if missing else expected_no_missing
    with (
        ad.settings.override(allow_write_nullable_strings=ad_setting),
        (  # `future.infer_string` defaults to True in pandas 3, and setting it is deprecated in pandas 3.1
            nullcontext()
            if PANDAS_3
            else pd.option_context("future.infer_string", pd_setting)
        ),
        (
            nullcontext()
            if isinstance(expected, str)
            else pytest.raises(expected[0], match=expected[1])
        ),
    ):
        write_elem(store, "/el", pd.array([pd.NA if missing else "a"], dtype="string"))

    if isinstance(expected, str):
        assert store["el"].attrs["encoding-type"] == expected


def test_categorical_order_type(store):
    # https://github.com/scverse/anndata/issues/853
    cat = pd.Categorical([0, 1], ordered=True)
    write_elem(store, "ordered", cat)
    write_elem(store, "unordered", cat.set_ordered(False))

    assert isinstance(read_elem(store["ordered"]).ordered, bool)
    assert read_elem(store["ordered"]).ordered is True
    assert isinstance(read_elem(store["unordered"]).ordered, bool)
    assert read_elem(store["unordered"]).ordered is False


def test_override_specification():
    """Test that trying to overwrite an existing encoding raises an error."""
    from copy import deepcopy

    registry = deepcopy(_REGISTRY)

    with pytest.raises(TypeError):

        @registry.register_write(
            zarr.Group, ad.AnnData, IOSpec("some new type", "0.1.0")
        )
        def _(store, key, adata):
            pass


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({"a": 1}, id="dict"),
        pytest.param(
            lambda: gen_adata((3, 2), **GEN_ADATA_NO_XARRAY_ARGS), id="anndata"
        ),
        pytest.param(sparse.random(5, 3, format="csr", density=0.5), id="csr_matrix"),
        pytest.param(sparse.random(5, 3, format="csc", density=0.5), id="csc_matrix"),
        pytest.param(pd.DataFrame({"a": [1, 2, 3]}), id="dataframe"),
        pytest.param(pd.Categorical(list("aabccedd")), id="categorical"),
        pytest.param(
            pd.Categorical(list("aabccedd"), ordered=True), id="categorical-ordered"
        ),
        pytest.param(
            pd.Categorical([1, 2, 1, 3], ordered=True), id="categorical-numeric"
        ),
        pytest.param(
            pd.arrays.IntegerArray(
                np.ones(5, dtype=int), mask=np.array([True, False, True, False, True])
            ),
            id="nullable-integer",
        ),
        pytest.param(pd.array([1, 2, 3]), id="nullable-integer-no-nulls"),
        pytest.param(
            pd.arrays.BooleanArray(
                np.random.randint(0, 2, size=5, dtype=bool),
                mask=np.random.randint(0, 2, size=5, dtype=bool),
            ),
            id="nullable-boolean",
        ),
        pytest.param(
            pd.array([True, False, True, True]), id="nullable-boolean-no-nulls"
        ),
    ],
)
def test_write_to_root(store: _GroupStorageType, value):
    """
    Test that elements which are written as groups can we written to the root group.
    """
    if callable(value):
        value = value()
    write_elem(store, "/", value)
    # See: https://github.com/zarr-developers/zarr-python/issues/2716
    if isinstance(store, zarr.Group):
        store = zarr.open_group(store.store)
    result = read_elem(store)

    assert_equal(result, value)


@pytest.mark.parametrize(
    "consolidated", [True, False], ids=["consolidated", "unconsolidated"]
)
@pytest.mark.zarr_io
def test_read_zarr_from_group(consolidated):
    # https://github.com/scverse/anndata/issues/1056
    store = MemoryStore()
    adata = gen_adata((3, 2), **GEN_ADATA_NO_XARRAY_ARGS)

    z = zarr.open_group(store, mode="w")
    write_elem(z.create_group("table"), "table", adata)

    if consolidated:
        # Catch the warning so we are alerted once it is no longer surfaced i.e., once consolidated metadata stabilizes.
        with pytest.warns(
            zarr.errors.ZarrUserWarning
            if hasattr(zarr, "errors") and hasattr(zarr.errors, "ZarrUserWarning")
            else UserWarning,
            match=r"Consolidated metadata",
        ):
            zarr.consolidate_metadata(z.store)

    read_func = zarr.open_consolidated if consolidated else zarr.open

    z = read_func(store)
    expected = ad.read_zarr(z["table/table"])
    assert_equal(adata, expected)


def test_dataframe_column_uniqueness(store):
    repeated_cols = pd.DataFrame(np.ones((3, 2)), columns=["a", "a"])

    with pytest.raises(
        ValueError,
        match=r"Found repeated column names: \['a'\]\. Column names must be unique\.",
    ):
        write_elem(store, "repeated_cols", repeated_cols)

    index_shares_col_name = pd.DataFrame(
        {"col_name": [1, 2, 3]}, index=pd.Index([1, 3, 2], name="col_name")
    )

    with pytest.raises(
        ValueError,
        match=r"DataFrame\.index\.name \('col_name'\) is also used by a column whose values are different\.",
    ):
        write_elem(store, "index_shares_col_name", index_shares_col_name)

    index_shared_okay = pd.DataFrame(
        {"col_name": [1, 2, 3]}, index=pd.Index([1, 2, 3], name="col_name")
    )

    write_elem(store, "index_shared_okay", index_shared_okay)
    result = read_elem(store["index_shared_okay"])

    assert_equal(result, index_shared_okay)


@pytest.mark.parametrize(
    "copy_on_write",
    [
        pytest.param(True, id="cow"),
        pytest.param(
            False,
            marks=pytest.mark.skipif(
                PANDAS_3, reason="Can’t disable copy-on-write in pandas 3+."
            ),
            id="nocow",
        ),
    ],
)
def test_io_pd_cow(
    *, exit_stack: ExitStack, store: _GroupStorageType, copy_on_write: bool
) -> None:
    """See <https://github.com/zarr-developers/numcodecs/issues/514>."""
    if not PANDAS_3:  # Setting copy_on_write always warns in pandas 3, and does nothing
        exit_stack.enter_context(pd.option_context("mode.copy_on_write", copy_on_write))
    orig = gen_adata((3, 2), **GEN_ADATA_NO_XARRAY_ARGS)
    write_elem(store, "adata", orig)
    from_store = read_elem(store["adata"])
    assert_equal(orig, from_store)


def test_read_sparse_array(
    diskfmt_store: Path | MemoryStore,
    sparse_format: Literal["csr", "csc"],
):
    a = sparse.random(100, 100, format=sparse_format)
    f = open_store(diskfmt_store)
    ad.io.write_elem(f, "mtx", a)
    ad.settings.use_sparse_array_on_read = True
    mtx = ad.io.read_elem(f["mtx"])
    assert issubclass(type(mtx), CSArray)


@pytest.mark.parametrize(
    ("chunks", "expected_chunks"),
    [
        ((1,), lambda s: (1,)),
        ((-1,), lambda s: (120,)),
        (None, lambda s: s.chunks if isinstance(s, h5py.Dataset) else s.shards),
    ],
    ids=[
        "small",
        "minus_one_uses_full",
        "none_uses_ondisk_sharding_if_zarr_else_chunks",
    ],
)
@pytest.mark.parametrize(
    "arr", [np.arange(120), np.array(["a"] * 120)], ids=["numeric", "string"]
)
def test_chunking_1d_array(
    store: h5py.Group | zarr.Group,
    arr: np.ndarray,
    chunks: tuple[int] | None,
    expected_chunks: Callable[[h5py.Dataset | zarr.Array], tuple[int]],
):
    write_elem(store, "foo", arr, dataset_kwargs={"chunks": (25,)})
    lazy_arr = read_elem_lazy(store["foo"], chunks=chunks)
    assert isinstance(lazy_arr, DaskArray)
    assert lazy_arr.chunksize == expected_chunks(
        cast("h5py.Dataset | zarr.Array", store["foo"])
    )


@pytest.mark.parametrize(
    ("chunks", "expected_chunks"),
    [
        ((1, 50), lambda s: (1, 50)),
        ((10, -1), lambda s: (10, 50)),
        ((10, None), lambda s: (10, 50)),
        (None, lambda s: s.chunks if isinstance(s, h5py.Dataset) else s.shards),
    ],
    ids=[
        "small",
        "minus_one_uses_full",
        "none_on_axis_uses_full",
        "none_uses_ondisk_sharding_if_zarr_else_chunks",
    ],
)
def test_chunking_2d_array(
    store: h5py.Group | zarr.Group,
    chunks: tuple[int] | None,
    expected_chunks: Callable[[h5py.Dataset | zarr.Array], tuple[int, int]],
):
    write_elem(
        store,
        "foo",
        np.arange(100 * 50).reshape((100, 50)),
        dataset_kwargs={"chunks": (25, 25)},
    )
    arr = read_elem_lazy(store["foo"], chunks=chunks)
    assert isinstance(arr, DaskArray)
    assert arr.chunksize == expected_chunks(
        cast("h5py.Dataset | zarr.Array", store["foo"])
    )


@pytest.mark.parametrize(
    ("shape", "expected_chunks"),
    [((100, 50), (100, 50)), ((1100, 1100), (1000, 1000))],
    ids=["under_default", "over_default"],
)
def test_h5_unchunked(
    shape: tuple[int, int],
    expected_chunks: tuple[int, int],
    tmp_path: Path,
):
    with h5py.File(tmp_path / "foo.h5ad", mode="w") as f:
        write_elem(
            f,
            "foo",
            np.arange(shape[0] * shape[1]).reshape(shape),
        )
        arr = read_elem_lazy(f["foo"])
    assert isinstance(arr, DaskArray)
    assert arr.chunksize == expected_chunks


@pytest.mark.zarr_io
def test_write_auto_sharded():
    store = MemoryStore()
    adata = gen_adata((100, 10), **GEN_ADATA_NO_XARRAY_ARGS)
    with ad.settings.override(auto_shard_zarr_v3=True):
        adata.write_zarr(store)

    check_all_sharded(zarr.open_group(store))


@pytest.mark.zarr_io
@pytest.mark.skipif(
    Version(version("zarr")) < Version("3.1.4"),
    reason="autosharding with chosen size was not available",
)
def test_write_auto_sharded_size():
    z = zarr.open_group(MemoryStore())
    ad.io.write_elem(z, "two_shards", np.arange(101), dataset_kwargs={"chunks": (7,)})
    two_shards = z["two_shards"]
    assert isinstance(two_shards, zarr.Array)
    assert two_shards.shards is not None
    # i.e., there are at most two shards since one shard will contain two chunks,
    # and the other the last elements, since the target size is 1GB uncompressed.
    assert (two_shards.shape[0] / two_shards.shards[0]) < 2


@pytest.mark.zarr_io
def test_write_shards_by_default():
    store = MemoryStore()
    adata = gen_adata((100, 10), **GEN_ADATA_NO_XARRAY_ARGS)
    ad.settings.reset("auto_shard_zarr_v3")
    adata.write_zarr(store)
    check_all_sharded(zarr.open_group(store))


@pytest.mark.zarr_io
@pytest.mark.skipif(
    Version(version("zarr")) < Version("3.1.4"),
    reason="autosharding with chosen size was not available",
)
def test_write_auto_sharded_size_sparse():
    z = zarr.open_group(MemoryStore())
    mat = sparse.random(
        1000, 1000, density=0.5, format="csr", random_state=np.random.default_rng(42)
    )
    ad.io.write_elem(z, "two_shards_per_sub_element", mat)
    group = z["two_shards_per_sub_element"]
    assert isinstance(group, zarr.Group)
    # i.e., there are at most two shards since one shard will contain two chunks,
    # and the other the last elements, since the target size is 1GB uncompressed.
    for sub_element in ["indices", "data", "indptr"]:
        arr = group[sub_element]
        assert isinstance(arr, zarr.Array)
        assert arr.shards is not None
        assert (arr.shape[0] / arr.shards[0]) < 2, sub_element


@pytest.mark.zarr_io
def test_write_auto_sharded_does_not_override():
    z = zarr.open_group(MemoryStore(), mode="w", zarr_format=3)
    X = sparse.random(
        100, 100, density=0.1, format="csr", random_state=np.random.default_rng(42)
    )
    ad.io.write_elem(z, "X_default", X)
    x_default = z["X_default"]
    assert isinstance(x_default, zarr.Group)
    indices_default = x_default["indices"]
    assert isinstance(indices_default, zarr.Array)
    assert indices_default.shards is not None
    new_shards = indices_default.shards[0] // 2
    new_shards = int(new_shards - new_shards % 2)
    ad.io.write_elem(
        z,
        "X_manually_set",
        X,
        dataset_kwargs={
            "shards": (new_shards,),
            "chunks": (int(new_shards / 2),),
        },
    )

    def visitor(key: str, array: zarr.Group | zarr.Array) -> None:
        assert isinstance(array, zarr.Array)
        assert array.shards == (new_shards,)

    x_manually_set = z["X_manually_set"]
    assert isinstance(x_manually_set, zarr.Group)
    visititems_zarr(x_manually_set, visitor)


@pytest.mark.parametrize("col", ["new", "a"], ids=["append", "overwrite"])
def test_write_dataframe_column_updates_order(
    store: _GroupStorageType, col: str
) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}, index=["c0", "c1"])
    write_elem(store, "df", df)
    group = cast("_GroupStorageType", store["df"])
    write_elem(group, col, np.array([3, 4]))
    expected = df.assign(**{col: [3, 4]})
    assert list(cast("list[str]", group.attrs["column-order"])) == list(
        expected.columns
    )
    pd.testing.assert_frame_equal(cast("pd.DataFrame", read_elem(group)), expected)
