"""Tests for :func:`anndata.merge`, which combines objects sharing obs and var."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import anndata as ad
from anndata import AnnData
from anndata.tests.helpers import GEN_ADATA_DASK_ARGS, assert_equal, gen_adata


def _empty(obs_names, var_names, **kwargs) -> AnnData:
    """An AnnData with only `obs_names` and `var_names` set, plus `kwargs`."""
    return AnnData(
        obs=pd.DataFrame(index=pd.Index(obs_names)),
        var=pd.DataFrame(index=pd.Index(var_names)),
        **kwargs,
    )


@pytest.fixture
def expr() -> AnnData:
    return AnnData(
        X=np.arange(6, dtype=np.float32).reshape(3, 2),
        obs=pd.DataFrame({"group": ["a", "b", "b"]}, index=["s1", "s2", "s3"]),
        var=pd.DataFrame({"n": [1, 2]}, index=["var1", "var2"]),
        layers={"counts": np.ones((3, 2))},
    )


@pytest.fixture
def pca(expr: AnnData) -> AnnData:
    return _empty(
        expr.obs_names,
        expr.var_names,
        obsm={"X_pca": np.arange(6).reshape(3, 2)},
        varm={"PCs": np.arange(4).reshape(2, 2)},
        obsp={"conn": sparse.random(3, 3, density=0.5, format="csr")},
        varp={"cov": np.eye(2)},
        uns={"pca": {"variance": [0.5, 0.5]}},
    )


def test_issue_example(expr: AnnData, pca: AnnData):
    merged = ad.merge([expr, pca])
    assert merged.shape == expr.shape
    assert merged.obs_names.equals(expr.obs_names)
    assert merged.var_names.equals(expr.var_names)
    assert_equal(merged.X, expr.X)
    assert_equal(merged.layers["counts"], expr.layers["counts"])
    assert_equal(merged.obs, expr.obs)
    assert_equal(merged.var, expr.var)
    for attr in ("obsm", "varm", "obsp", "varp"):
        assert_equal(dict(getattr(merged, attr)), dict(getattr(pca, attr)))
    assert_equal(merged.uns, pca.uns)
    # Order of inputs does not matter for disjoint elements
    assert_equal(ad.merge([pca, expr]), merged)


def test_single_and_identical_inputs():
    adata = gen_adata((10, 8), **GEN_ADATA_DASK_ARGS)
    assert_equal(ad.merge([adata]), adata)
    assert_equal(ad.merge([adata, adata]), adata)
    assert_equal(ad.merge([adata, adata], merge="same"), adata)
    assert_equal(ad.merge([adata, adata], merge="first"), adata)


def test_split_and_merge_roundtrip():
    """Splitting an object into partial objects and merging gives the original back."""
    adata = gen_adata((10, 8), **GEN_ADATA_DASK_ARGS)
    axes = (adata.obs_names, adata.var_names)
    parts = [
        AnnData(X=adata.X, obs=adata.obs, var=adata.var),
        _empty(*axes, layers=dict(adata.layers)),
        _empty(*axes, obsm=dict(adata.obsm), varm=dict(adata.varm)),
        _empty(*axes, obsp=dict(adata.obsp), varp=dict(adata.varp)),
        _empty(*axes, uns=dict(adata.uns)),
    ]
    assert_equal(ad.merge(parts), adata)


def test_inputs_not_modified(expr: AnnData, pca: AnnData):
    expr_orig, pca_orig = expr.copy(), pca.copy()
    ad.merge([expr, pca])
    ad.merge([expr, pca[["s3", "s1"]]], join="inner")
    ad.merge([expr, pca[["s3"]]], join="outer")
    assert_equal(expr, expr_orig)
    assert_equal(pca, pca_orig)


def test_reindexed_elements_are_copies(expr: AnnData, pca: AnnData):
    merged = ad.merge([expr, pca[["s3", "s2", "s1"]]])
    x_pca = merged.obsm["X_pca"]
    assert type(x_pca) is np.ndarray
    x_pca[0, 0] = 100
    assert x_pca[0, 0] == 100
    merged.obs["group"] = "z"
    assert pca.obsm["X_pca"][2, 0] != 100
    assert expr.obs["group"].iloc[0] == "a"


def test_unchanged_elements_are_shared(expr: AnnData, pca: AnnData):
    """Like the `AnnData` constructor, `merge` does not copy elements it does not reindex."""
    merged = ad.merge([expr, pca])
    assert merged.X is expr.X
    assert merged.obsm["X_pca"] is pca.obsm["X_pca"]
    assert_equal(ad.merge([expr, pca]).copy(), merged)


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        pytest.param("unique", {"shared_same", "a_only", "b_only"}, id="unique"),
        pytest.param("same", {"shared_same"}, id="same"),
        pytest.param(
            "first", {"shared_same", "shared_diff", "a_only", "b_only"}, id="first"
        ),
        pytest.param("only", {"a_only", "b_only"}, id="only"),
    ],
)
def test_conflict_strategies(strategy, expected):
    var = pd.DataFrame(index=["v1", "v2"])
    ones, zeros = np.ones((2, 2)), np.zeros((2, 2))
    a = AnnData(
        var=var,
        obsm={"shared_same": ones, "shared_diff": ones, "a_only": ones},
        layers={"shared_same": ones, "shared_diff": ones, "a_only": ones},
        obs=pd.DataFrame(
            {"shared_same": [1, 2], "shared_diff": [1, 2], "a_only": [1, 2]},
            index=["s1", "s2"],
        ),
        uns={"shared_same": 1, "shared_diff": 1, "a_only": 1},
    )
    b = AnnData(
        var=var,
        obsm={"shared_same": ones, "shared_diff": zeros, "b_only": zeros},
        layers={"shared_same": ones, "shared_diff": zeros, "b_only": zeros},
        obs=pd.DataFrame(
            {"shared_same": [1, 2], "shared_diff": [3, 4], "b_only": [3, 4]},
            index=["s1", "s2"],
        ),
        uns={"shared_same": 1, "shared_diff": 2, "b_only": 2},
    )
    merged = ad.merge([a, b], merge=strategy, uns_merge=strategy)
    assert set(merged.obsm.keys()) == expected
    assert set(merged.layers.keys()) == expected
    assert set(merged.obs.columns) == expected
    assert set(merged.uns.keys()) == expected
    if "shared_diff" in expected:  # "first" takes the value from `a`
        assert_equal(merged.obsm["shared_diff"], ones)
        assert_equal(merged.layers["shared_diff"], ones)
        assert list(merged.obs["shared_diff"]) == [1, 2]
        assert merged.uns["shared_diff"] == 1


def test_conflicting_x():
    axes = (["s1", "s2"], ["v1", "v2"])
    a = _empty(*axes, X=np.ones((2, 2)))
    b = _empty(*axes, X=np.zeros((2, 2)))
    assert ad.merge([a, b]).X is None
    assert_equal(ad.merge([a, b], merge="first").X, a.X)
    assert ad.merge([a, b], merge="only").X is None
    assert_equal(ad.merge([a, _empty(*axes)]).X, a.X)


def test_uns_merge_recursive_and_none(expr: AnnData):
    axes = (expr.obs_names, expr.var_names)
    a = _empty(*axes, uns={"nested": {"a": 1, "shared": 1}})
    b = _empty(*axes, uns={"nested": {"b": 2, "shared": 2}})
    assert ad.merge([a, b]).uns == {"nested": {"a": 1, "b": 2}}
    assert ad.merge([a, b], uns_merge="first").uns == {
        "nested": {"a": 1, "shared": 1, "b": 2}
    }
    assert ad.merge([a, b], uns_merge="same").uns == {}
    assert ad.merge([a, b], uns_merge=None).uns == {}
    # `uns_merge` is independent of `merge`
    assert ad.merge([a, b], merge="first", uns_merge="same").uns == {}


def test_callable_strategy(expr: AnnData, pca: AnnData):
    def keep_nothing(mappings):
        return {}

    merged = ad.merge([expr, pca], merge=keep_nothing, uns_merge=keep_nothing)
    assert merged.X is None
    assert len(merged.layers) == len(merged.obsm) == len(merged.uns) == 0
    assert list(merged.obs.columns) == []
    assert merged.shape == expr.shape


@pytest.mark.parametrize("x_type", [np.asarray, sparse.csr_matrix, sparse.csr_array])
def test_join_inner_reorders_and_subsets(x_type):
    """Elements of later objects are aligned to the first object's names."""
    a = AnnData(
        X=x_type(np.arange(1, 7, dtype=np.float64).reshape(3, 2)),
        obs=pd.DataFrame({"a_col": [1, 2, 3]}, index=["s1", "s2", "s3"]),
        var=pd.DataFrame(index=["v1", "v2"]),
    )
    b = AnnData(
        obs=pd.DataFrame({"b_col": [30.0, 10.0, 40.0]}, index=["s3", "s1", "s4"]),
        var=pd.DataFrame({"v_col": list("xyz")}, index=["v2", "v1", "v3"]),
        obsm={"m": np.array([[3], [1], [4]])},
        varm={"vm": np.array([[20], [10], [30]])},
        obsp={"p": np.array([[33, 31, 34], [13, 11, 14], [43, 41, 44]])},
        varp={
            "vp": sparse.csr_matrix(
                np.array([[22, 21, 23], [12, 11, 13], [32, 31, 33]])
            )
        },
        layers={"l": x_type(np.array([[32, 31, 33], [12, 11, 13], [42, 41, 43]]))},
    )
    merged = ad.merge([a, b])
    assert list(merged.obs_names) == ["s1", "s3"]
    assert list(merged.var_names) == ["v1", "v2"]
    assert_equal(np.asarray(merged.to_df()), np.array([[1, 2], [5, 6]]))
    assert list(merged.obs["a_col"]) == [1, 3]
    assert list(merged.obs["b_col"]) == [10.0, 30.0]
    assert list(merged.var["v_col"]) == ["y", "x"]
    assert_equal(merged.obsm["m"], np.array([[1], [3]]))
    assert_equal(merged.varm["vm"], np.array([[10], [20]]))
    assert_equal(merged.obsp["p"], np.array([[11, 13], [31, 33]]))
    assert_equal(merged.varp["vp"], np.array([[11, 12], [21, 22]]))
    assert_equal(merged.layers["l"], np.array([[11, 12], [31, 32]]))
    assert isinstance(merged.X, type(a.X))


def test_join_outer():
    a = AnnData(
        X=sparse.csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]])),
        obs=pd.DataFrame({"a_col": [1.0, 2.0]}, index=["s1", "s2"]),
        var=pd.DataFrame(index=["v1", "v2"]),
        obsm={"dense": np.array([[1.0], [2.0]])},
    )
    b = AnnData(
        obs=pd.DataFrame({"b_col": [3.0]}, index=["s3"]),
        var=pd.DataFrame(index=["v3"]),
        layers={"dense": np.array([[9.0]])},
    )
    merged = ad.merge([a, b], join="outer")
    assert list(merged.obs_names) == ["s1", "s2", "s3"]
    assert list(merged.var_names) == ["v1", "v2", "v3"]
    # sparse is padded with zeros, dense with NaN
    assert_equal(merged.X.toarray(), np.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]]))
    assert_equal(merged.obsm["dense"], np.array([[1.0], [2.0], [np.nan]]))
    expected_layer = np.full((3, 3), np.nan)
    expected_layer[2, 2] = 9.0
    assert_equal(merged.layers["dense"], expected_layer)
    assert merged.obs["a_col"].isna().tolist() == [False, False, True]
    assert merged.obs["b_col"].isna().tolist() == [True, True, False]

    filled = ad.merge([a, b], join="outer", fill_value=-1)
    assert_equal(filled.X.toarray(), np.array([[1, 2, -1], [3, 4, -1], [-1, -1, -1]]))
    assert_equal(filled.obsm["dense"], np.array([[1.0], [2.0], [-1.0]]))


def test_x_none_everywhere():
    merged = ad.merge([_empty(["s1"], ["v1"]), _empty(["s1"], ["v1"])])
    assert merged.X is None
    assert merged.shape == (1, 1)


def test_views(expr: AnnData, pca: AnnData):
    merged = ad.merge([expr[1:], pca[1:]])
    assert not merged.is_view
    assert_equal(merged, ad.merge([expr, pca])[1:].copy())


def test_raw(expr: AnnData, pca: AnnData):
    raw_source = AnnData(
        X=np.full((3, 3), 7.0),
        obs=pd.DataFrame(index=expr.obs_names),
        var=pd.DataFrame(index=["r1", "r2", "r3"]),
    )
    expr.raw = raw_source
    merged = ad.merge([expr, pca])
    assert merged.raw is not None
    assert list(merged.raw.var_names) == ["r1", "r2", "r3"]
    assert_equal(merged.raw.X, raw_source.X)

    # raw is reindexed along obs with the rest of the object
    subset = ad.merge([expr, pca[["s3", "s1"]]])
    assert list(subset.obs_names) == ["s1", "s3"]
    assert subset.raw is not None
    assert subset.raw.shape == (2, 3)

    pca.raw = raw_source
    with pytest.warns(UserWarning, match="More than one AnnData object has a `.raw`"):
        merged = ad.merge([expr, pca])
    assert merged.raw is None


def test_errors(expr: AnnData):
    with pytest.raises(ValueError, match="No objects to merge"):
        ad.merge([])
    with pytest.raises(ValueError, match="merge=None"):
        ad.merge([expr], merge=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="`join` must be one of"):
        ad.merge([expr], join="left")  # type: ignore[arg-type]
    with pytest.warns(UserWarning, match="Observation names are not unique"):
        dup = _empty(["s1", "s1"], ["v1"])
    with pytest.raises(
        ValueError, match="`obs_names` of the AnnData object at position 1"
    ):
        ad.merge([expr, dup])


def test_backed_input_is_loaded(expr: AnnData, pca: AnnData, tmp_path):
    path = tmp_path / "expr.h5ad"
    expr.raw = expr
    expr.write_h5ad(path)
    backed = ad.read_h5ad(path, backed="r")
    assert backed.isbacked
    merged = ad.merge([backed, pca])
    assert not merged.isbacked
    assert type(merged.X) is np.ndarray
    assert merged.raw is not None
    assert type(merged.raw.X) is np.ndarray
    assert_equal(merged.X, expr.X)
    reordered = ad.merge([backed, pca[["s2", "s1"]]])
    assert_equal(reordered.X, expr[["s1", "s2"]].X)


def test_dask_elements(expr: AnnData):
    da = pytest.importorskip("dask.array")
    lazy = _empty(
        expr.obs_names, expr.var_names, layers={"lazy": da.ones((3, 2), chunks=(2, 1))}
    )
    merged = ad.merge([expr, lazy])
    assert isinstance(merged.layers["lazy"], da.Array)
    reordered = ad.merge([lazy, expr[["s3", "s1"]]])
    assert isinstance(reordered.layers["lazy"], da.Array)
    assert reordered.layers["lazy"].shape == (2, 2)


@pytest.mark.parametrize("idx", [slice(0, 1), ["b", "a"]], ids=["slice", "reorder"])
def test_view_inputs_do_not_write_through(idx):
    a = AnnData(
        X=np.zeros((2, 2)),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame(index=["x", "y"]),
        obsm={"m": np.zeros((2, 1))},
    )
    view = a[idx]
    merged = ad.merge([view, _empty(view.obs_names, view.var_names)])
    assert type(merged.X) is np.ndarray
    assert type(merged.obsm["m"]) is np.ndarray
    merged.X[0, 0] = 99
    merged.obsm["m"][0, 0] = 99
    assert merged.X[0, 0] == 99
    assert merged.obsm["m"][0, 0] == 99
    assert a.X[0, 0] == 0
    assert a.obsm["m"][0, 0] == 0


def test_outer_join_awkward():
    ak = pytest.importorskip("awkward")
    var = pd.DataFrame(index=["x"])
    a = _empty(["a", "b"], var.index, obsm={"awk": ak.Array([[1], [2, 3]])})
    b = _empty(["c"], var.index)
    merged = ad.merge([a, b], join="outer")
    assert merged.obsm["awk"].to_list() == [[1], [2, 3], None]
    reordered = ad.merge([b, a], join="outer")
    assert list(reordered.obs_names) == ["a", "b", "c"]
    assert reordered.obsm["awk"].to_list() == [[1], [2, 3], None]


@pytest.mark.parametrize("strategy", ["unique", "same"])
def test_equal_dense_and_sparse_x_is_order_independent(strategy):
    axes = (["a", "b"], ["x", "y"])
    dense = _empty(*axes, X=np.ones((2, 2)))
    sp = _empty(*axes, X=sparse.csr_matrix(np.ones((2, 2))))
    assert ad.merge([dense, sp], merge=strategy).X is not None
    assert ad.merge([sp, dense], merge=strategy).X is not None
    other = _empty(*axes, X=np.zeros((2, 2)))
    assert ad.merge([sp, other], merge=strategy).X is None
    assert ad.merge([other, sp], merge=strategy).X is None


def test_raw_without_x():
    a = AnnData(
        obs=pd.DataFrame(index=["a"]), var=pd.DataFrame({"c": [1]}, index=["x"])
    )
    a.raw = a.copy()
    a.X = np.ones((1, 1))
    with pytest.raises(NotImplementedError, match=r"`\.raw` has no `X`"):
        ad.merge([a])


@pytest.mark.parametrize(
    ("dtype", "fill_value", "expected_dtype"),
    [
        pytest.param(np.int64, None, np.float64, id="int-nan"),
        pytest.param(np.int64, -1, np.int64, id="int-neg"),
        pytest.param(np.uint8, -1, np.int16, id="uint8-neg"),
        pytest.param(np.int8, 1000, np.int32, id="int8-large"),
        pytest.param(np.float32, 1e100, np.float64, id="float32-huge"),
    ],
)
@pytest.mark.parametrize("as_sparse", [False, True], ids=["dense", "sparse"])
def test_outer_join_dask_fill_dtype(dtype, fill_value, expected_dtype, as_sparse):
    da = pytest.importorskip("dask.array")
    var = pd.DataFrame(index=["x", "y"])
    x = da.ones((1, 2), dtype=dtype)
    if as_sparse:
        x = x.map_blocks(sparse.csr_matrix)
    a = _empty(["a"], var.index, X=x)
    b = _empty(["b"], var.index)
    merged = ad.merge([a, b], join="outer", fill_value=fill_value)
    assert isinstance(merged.X, da.Array)
    computed = merged.X.compute()
    computed = np.asarray(computed.toarray() if sparse.issparse(computed) else computed)
    if fill_value is None and as_sparse:
        # sparse blocks are padded with 0 and keep their dtype
        assert computed.dtype == dtype
        assert_equal(computed, np.array([[1, 1], [0, 0]]))
        return
    assert computed.dtype == expected_dtype
    if fill_value is None:
        assert_equal(computed, np.array([[1.0, 1.0], [np.nan, np.nan]]))
    else:
        assert_equal(computed, np.array([[1, 1], [fill_value, fill_value]]))


@pytest.mark.parametrize("strategy", ["unique", "same"])
def test_equal_dask_and_sparse_x_is_order_independent(strategy):
    da = pytest.importorskip("dask.array")
    axes = (["a"], ["x"])
    lazy = _empty(*axes, X=da.ones((1, 1)))
    sp = _empty(*axes, X=sparse.csr_matrix([[1.0]]))
    dense = _empty(*axes, X=np.ones((1, 1)))
    lazy_sp = _empty(*axes, X=da.ones((1, 1)).map_blocks(sparse.csr_matrix))
    for x, y in [(lazy, sp), (lazy, dense), (lazy_sp, dense), (lazy_sp, sp)]:
        assert ad.merge([x, y], merge=strategy).X is not None
        assert ad.merge([y, x], merge=strategy).X is not None
    other = _empty(*axes, X=np.zeros((1, 1)))
    assert ad.merge([lazy, other], merge=strategy).X is None
    assert ad.merge([other, lazy], merge=strategy).X is None


def test_awkward_layer_var_alignment():
    ak = pytest.importorskip("awkward")
    a = _empty(["s"], ["x", "y"], layers={"awk": ak.Array(np.array([[1, 2]]))})
    b = _empty(["s"], ["y"])
    assert ad.merge([a, b]).layers["awk"].to_list() == [[2]]
    assert ad.merge([a, _empty(["s"], ["y", "x"])]).layers["awk"].to_list() == [[1, 2]]
    assert ad.merge([_empty(["s"], ["y", "x"]), a]).layers["awk"].to_list() == [[2, 1]]
    outer = ad.merge([a, _empty(["s"], ["z"])], join="outer")
    assert list(outer.var_names) == ["x", "y", "z"]
    assert outer.layers["awk"].to_list() == [[1, 2, None]]


def test_view_of_dask_stays_lazy():
    da = pytest.importorskip("dask.array")
    computed = []
    x = da.ones((4, 2)).map_blocks(lambda b: computed.append(1) or b)
    a = _empty(list("abcd"), ["x", "y"], X=x)
    computed.clear()
    merged = ad.merge([a[:2], a[["b", "a"]]], merge="first")
    assert isinstance(merged.X, da.Array)
    assert computed == []
    assert_equal(merged.X.compute(), np.ones((2, 2)))


def test_outer_join_fill_value_scope():
    var = pd.DataFrame(index=["x", "y"])
    a = AnnData(X=np.ones((1, 2)), obs=pd.DataFrame({"c": [1.0]}, index=["a"]), var=var)
    a.raw = a
    b = _empty(["b"], var.index)
    merged = ad.merge([a, b], join="outer", fill_value=-7)
    assert merged.raw is not None
    assert_equal(merged.raw.X, np.array([[1.0, 1.0], [-7.0, -7.0]]))
    # obs/var columns are always padded with missing values
    assert merged.obs["c"].isna().tolist() == [False, True]


def test_outer_join_index_order_follows_pandas_union():
    var = pd.DataFrame(index=["x"])
    a = _empty(["z", "b"], var.index)
    b = _empty(["a"], var.index)
    assert list(ad.merge([a, b], join="outer").obs_names) == ["a", "b", "z"]
    assert list(ad.merge([a, b]).obs_names) == []


@pytest.mark.parametrize("load_annotation_index", [True, False])
def test_lazy_annotations(load_annotation_index):
    pytest.importorskip("xarray")
    zarr = pytest.importorskip("zarr")
    from anndata.experimental import read_lazy

    eager = AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame({"score": [10, 20]}, index=["a", "b"]),
        var=pd.DataFrame({"v": [1]}, index=["x"]),
    )
    store = zarr.storage.MemoryStore()
    eager.write_zarr(store)
    lazy = read_lazy(store, load_annotation_index=load_annotation_index)

    for merged in (ad.merge([lazy]), ad.merge([eager, lazy]), ad.merge([lazy, eager])):
        assert list(merged.obs_names) == ["a", "b"]
        assert list(merged.var_names) == ["x"]
        assert list(merged.obs["score"]) == [10, 20]
        assert list(merged.var["v"]) == [1]
    # positional views of lazy objects
    merged = ad.merge([eager, lazy[1:]])
    assert list(merged.obs_names) == ["b"]
    assert list(merged.obs["score"]) == [20]
