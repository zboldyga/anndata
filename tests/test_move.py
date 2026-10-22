"""Tests for moving values between X and the axis annotations (anndata#655)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import anndata as ad
from anndata.experimental import move_to_obs, move_to_var, move_to_x
from anndata.tests.helpers import as_dense_dask_array, assert_equal, gen_adata
from anndata.utils import asarray


@pytest.fixture(
    params=[np.array, sparse.csr_matrix, sparse.csc_array, as_dense_dask_array],
    ids=["np_array", "scipy_csr", "scipy_csc_array", "dask_array"],
)
def matrix_type(request):
    return request.param


@pytest.fixture
def adata(matrix_type):
    adata = gen_adata((8, 6), X_type=matrix_type)
    adata.layers["counts"] = np.arange(48).reshape(8, 6)
    adata.uns["info"] = {"nested": [1, 2]}
    return adata


def _orig(adata) -> pd.DataFrame:
    return adata.to_df()


def _uncat(df: pd.DataFrame) -> pd.DataFrame:
    """Subsetting prunes unused categories, so compare categoricals by value."""
    return df.astype({c: object for c in df.columns if df[c].dtype == "category"})


def assert_values_equal(a: pd.DataFrame, b: pd.DataFrame) -> None:
    """Compare annotations by value: growing an axis pads rows, so int columns become float."""
    pd.testing.assert_frame_equal(_uncat(a), _uncat(b), check_dtype=False)


# ---------------------------------------------------------------------------
# X -> obs / var
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "keys",
    [
        pytest.param("gene1", id="str"),
        pytest.param(["gene1", "gene4"], id="list"),
        pytest.param(np.array([False, True, False, False, True, False]), id="mask"),
    ],
)
def test_move_to_obs(adata, keys):
    orig_df = _orig(adata)
    orig_copy = adata.copy()
    moved_names = ["gene1"] if isinstance(keys, str) else ["gene1", "gene4"]

    res = move_to_obs(adata, keys)

    assert_equal(adata, orig_copy)  # input untouched
    assert res.n_vars == adata.n_vars - len(moved_names)
    assert not res.var_names.isin(moved_names).any()
    for name in moved_names:
        assert name in res.obs.columns
        assert not isinstance(res.obs[name].dtype, pd.SparseDtype)
        np.testing.assert_array_equal(
            res.obs[name].to_numpy(), orig_df[name].to_numpy()
        )
    # The remaining elements are what subsetting would give
    expected = adata[:, ~adata.var_names.isin(moved_names)].copy()
    assert_equal(res.to_df(), expected.to_df())
    assert_equal(res.var, expected.var)
    assert_equal(res.layers, expected.layers)
    assert_equal(res.varm, expected.varm)
    assert_equal(res.varp, expected.varp)
    assert_equal(res.obsm, expected.obsm)
    assert_equal(res.obsp, expected.obsp)
    assert_equal(res.uns, expected.uns)
    assert_equal(res.obs[expected.obs.columns], expected.obs)
    assert list(res.obs.columns) == [*expected.obs.columns, *moved_names]


def test_move_to_obs_layer(adata):
    res = move_to_obs(adata, ["gene0", "gene2"], layer="counts")
    counts = adata.to_df("counts")
    np.testing.assert_array_equal(res.obs["gene0"], counts["gene0"])
    np.testing.assert_array_equal(res.obs["gene2"], counts["gene2"])
    assert res.obs["gene0"].dtype == counts["gene0"].dtype
    assert res.layers["counts"].shape == (8, 4)


def test_move_to_obsm(adata):
    res = move_to_obs(adata, ["gene3", "gene0"], obsm_key="moved")
    assert "gene3" not in res.obs.columns
    assert isinstance(res.obsm["moved"], pd.DataFrame)
    assert list(res.obsm["moved"].columns) == ["gene3", "gene0"]  # order of `keys`
    assert_equal(
        res.obsm["moved"].to_numpy(), adata.to_df()[["gene3", "gene0"]].to_numpy()
    )
    assert not any(isinstance(dt, pd.SparseDtype) for dt in res.obsm["moved"].dtypes)
    assert res.n_vars == 4


@pytest.mark.parametrize(
    "keys",
    [
        pytest.param("cell2", id="str"),
        pytest.param(["cell2", "cell7"], id="list"),
    ],
)
def test_move_to_var(adata, keys):
    moved_names = [keys] if isinstance(keys, str) else keys
    orig_df = _orig(adata)

    res = move_to_var(adata, keys)

    assert res.n_obs == adata.n_obs - len(moved_names)
    assert not res.obs_names.isin(moved_names).any()
    for name in moved_names:
        np.testing.assert_array_equal(
            res.var[name].to_numpy(), orig_df.loc[name].to_numpy()
        )
    expected = adata[~adata.obs_names.isin(moved_names)].copy()
    assert_equal(res.to_df(), expected.to_df())
    assert_equal(res.obs, expected.obs)
    assert_equal(res.layers, expected.layers)
    assert_equal(res.obsm, expected.obsm)
    assert_equal(res.obsp, expected.obsp)
    assert_equal(res.var[expected.var.columns], expected.var)


def test_move_to_varm(adata):
    res = move_to_var(adata, "cell0", varm_key="moved")
    assert isinstance(res.varm["moved"], pd.DataFrame)
    np.testing.assert_array_equal(
        res.varm["moved"].to_numpy(), adata.to_df().T[["cell0"]].to_numpy()
    )


def test_move_to_annot_view():
    adata = gen_adata((8, 6))
    view = adata[2:6, 1:]
    assert view.is_view
    res = move_to_obs(view, "gene2")
    assert not res.is_view
    assert res.shape == (4, 4)
    np.testing.assert_array_equal(res.obs["gene2"], view.to_df()["gene2"])
    assert adata.shape == (8, 6)
    assert "gene2" not in adata.obs.columns


def test_move_to_obs_keeps_raw():
    adata = gen_adata((5, 4), obsm_types=(), varm_types=())
    adata.raw = adata.copy()
    res = move_to_obs(adata, "gene0")
    assert res.raw.shape == (5, 4)
    assert_equal(res.raw.to_adata().to_df(), adata.to_df())
    res_var = move_to_var(adata, "cell0")
    assert res_var.raw.shape == (4, 4)


def test_move_to_annot_categorical_names():
    adata = ad.AnnData(np.arange(6.0).reshape(2, 3))
    adata.var_names = pd.CategoricalIndex(["a", "b", "c"])
    res = move_to_obs(adata, "b")
    assert list(res.obs.columns) == ["b"]
    assert res.var_names.tolist() == ["a", "c"]


def test_move_to_annot_duplicate_obs_names():
    adata = ad.AnnData(np.arange(6.0).reshape(3, 2))
    adata.obs_names = ["x", "x", "y"]
    res = move_to_obs(adata, "1")
    np.testing.assert_array_equal(res.obs["1"], [1.0, 3.0, 5.0])


def test_move_to_annot_errors():
    adata = gen_adata((5, 4), obsm_types=(), varm_types=())
    with pytest.raises(KeyError, match="var_names \\['nope'\\] not found"):
        move_to_obs(adata, ["gene0", "nope"])
    with pytest.raises(KeyError, match="obs_names \\['nope'\\] not found"):
        move_to_var(adata, "nope")
    with pytest.raises(ValueError, match="must have shape \\(4,\\)"):
        move_to_obs(adata, np.array([True, False]))
    with pytest.raises(ValueError, match="At least one var must be selected"):
        move_to_obs(adata, np.zeros(4, dtype=bool))
    with pytest.raises(ValueError, match="At least one var name"):
        move_to_obs(adata, [])
    with pytest.raises(ValueError, match="Duplicate var names"):
        move_to_obs(adata, ["gene0", "gene0"])
    clash = adata.copy()
    clash.var_names = ["int64", "g1", "g2", "g3"]
    with pytest.raises(ValueError, match="`obs` already has columns \\['int64'\\]"):
        move_to_obs(clash, "int64")
    adata.obsm["taken"] = np.zeros((5, 2))
    with pytest.raises(ValueError, match="`obsm` already has key 'taken'"):
        move_to_obs(adata, "gene0", obsm_key="taken")
    with pytest.raises(ValueError, match="X is None"):
        move_to_obs(ad.AnnData(obs=adata.obs, var=adata.var), "gene0")
    dup = ad.AnnData(np.zeros((2, 3)))
    dup.var_names = ["a", "a", "b"]
    with pytest.raises(ValueError, match="must be unique to move values by name"):
        move_to_obs(dup, "a")
    with pytest.raises(ValueError, match="not unique"):
        move_to_obs(dup, np.array([True, True, False]))


# ---------------------------------------------------------------------------
# obs / var -> X
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "keys",
    [
        pytest.param("float64", id="str"),
        pytest.param(["float64", "int64", "bool"], id="list"),
    ],
)
def test_move_to_x_from_obs(adata, keys):
    names = [keys] if isinstance(keys, str) else keys
    orig_copy = adata.copy()

    res = move_to_x(adata, keys)

    assert_equal(adata, orig_copy)  # input untouched
    assert res.shape == (adata.n_obs, adata.n_vars + len(names))
    assert res.var_names.tolist() == [*adata.var_names, *names]
    assert not any(n in res.obs.columns for n in names)
    assert_equal(res.obs, adata.obs.drop(columns=names))
    df = res.to_df()
    # X is promoted following NumPy's rules (float32 X + float64/int64 obs columns -> float64)
    np.testing.assert_array_equal(
        df.iloc[:, : adata.n_vars].to_numpy(), adata.to_df().to_numpy()
    )
    for n in names:
        np.testing.assert_array_equal(
            np.asarray(df[n].to_numpy(), dtype=float),
            adata.obs[n].to_numpy().astype(float),
        )
    # sparsity is preserved
    assert sparse.issparse(res.X) == sparse.issparse(adata.X)
    # everything else is kept, padded along var
    assert_equal(res.obsm, adata.obsm)
    assert_equal(res.obsp, adata.obsp)
    assert_equal(res.uns, adata.uns)
    assert set(res.layers) == set(adata.layers)
    assert set(res.varm) == set(adata.varm)
    assert set(res.varp) == set(adata.varp)
    counts = res.layers["counts"]
    np.testing.assert_array_equal(counts[:, : adata.n_vars], adata.layers["counts"])
    assert np.isnan(counts[:, adata.n_vars :]).all()
    assert res.var.loc[names].isna().all().all()
    assert_values_equal(res.var.iloc[: adata.n_vars], adata.var)


def test_move_to_x_from_var(adata):
    res = move_to_x(adata, ["float64", "uint8"], source="var")
    assert res.shape == (adata.n_obs + 2, adata.n_vars)
    assert res.obs_names.tolist() == [*adata.obs_names, "float64", "uint8"]
    assert "float64" not in res.var.columns
    assert_equal(res.var, adata.var.drop(columns=["float64", "uint8"]))
    df = res.to_df()
    np.testing.assert_array_equal(
        asarray(df.iloc[: adata.n_obs]), asarray(adata.to_df())
    )
    np.testing.assert_array_equal(
        df.loc["uint8"].to_numpy(), adata.var["uint8"].to_numpy()
    )
    assert_equal(res.varm, adata.varm)
    assert_equal(res.varp, adata.varp)
    assert set(res.obsm) == set(adata.obsm)


@pytest.mark.parametrize("as_df", [True, False], ids=["df", "array"])
def test_move_to_x_from_obsm(*, as_df: bool):
    adata = gen_adata((5, 3), obsm_types=(), varm_types=())
    values = np.arange(10.0).reshape(5, 2)
    adata.obsm["prot"] = (
        pd.DataFrame(values, index=adata.obs_names, columns=["p1", "p2"])
        if as_df
        else values
    )
    res = move_to_x(adata, "prot", source="obsm")
    new_names = ["p1", "p2"] if as_df else ["prot0", "prot1"]
    assert res.var_names.tolist() == [*adata.var_names, *new_names]
    assert "prot" not in res.obsm
    np.testing.assert_array_equal(asarray(res.X)[:, 3:], values)
    assert_equal(res.obs, adata.obs)


def test_move_to_x_from_varm():
    adata = gen_adata((5, 3), obsm_types=(), varm_types=())
    adata.varm["load"] = np.ones((3, 2))
    res = move_to_x(adata, "load", source="varm")
    assert res.obs_names.tolist() == [*adata.obs_names, "load0", "load1"]
    assert "load" not in res.varm
    np.testing.assert_array_equal(asarray(res.X)[5:], np.ones((2, 3)))


def test_move_to_x_view_and_raw():
    adata = gen_adata((6, 4), obsm_types=(), varm_types=())
    adata.raw = adata.copy()
    view = adata[:3]
    res = move_to_x(view, "float64")
    assert not res.is_view
    assert res.shape == (3, 5)
    assert res.raw is not None
    assert res.raw.shape == (3, 4)
    assert_equal(res.raw.to_adata().to_df(), adata[:3].to_df())
    assert adata.shape == (6, 4)
    with pytest.warns(UserWarning, match="drops `.raw`"):
        res_var = move_to_x(adata, "float64", source="var")
    assert res_var.raw is None


def test_move_to_x_empty_x():
    obs = pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4]}, index=["c1", "c2"])
    adata = ad.AnnData(obs=obs)
    res = move_to_x(adata, ["a", "b"])
    assert res.shape == (2, 2)
    np.testing.assert_array_equal(res.X, [[1.0, 3.0], [2.0, 4.0]])
    assert list(res.obs.columns) == []


def test_move_to_x_nullable_dtypes():
    adata = ad.AnnData(np.zeros((3, 1)))
    adata.obs["n"] = pd.array([1, None, 3], dtype="Int64")
    res = move_to_x(adata, "n")
    col = res.to_df()["n"]
    assert col.iloc[0] == 1
    assert np.isnan(col.iloc[1])


def test_move_to_x_errors():
    adata = gen_adata((5, 4), obsm_types=(), varm_types=())
    with pytest.raises(KeyError, match="Columns \\['nope'\\] not found in `obs`"):
        move_to_x(adata, "nope")
    with pytest.raises(KeyError, match="'nope' not found in `obsm`"):
        move_to_x(adata, "nope", source="obsm")
    with pytest.raises(ValueError, match="Only numeric or boolean values"):
        move_to_x(adata, ["float64", "obs_cat"])
    with pytest.raises(ValueError, match="`source` must be one of"):
        move_to_x(adata, "float64", source="uns")
    with pytest.raises(TypeError, match="must be a single key"):
        move_to_x(adata, ["a", "b"], source="obsm")
    adata.obsm["three_d"] = np.zeros((5, 2, 2))
    with pytest.raises(ValueError, match="must be 2-dimensional"):
        move_to_x(adata, "three_d", source="obsm")
    adata.obs["gene0"] = 1.0
    with pytest.raises(ValueError, match="`var_names` already contain \\['gene0'\\]"):
        move_to_x(adata, "gene0")
    no_x = ad.AnnData(obs=adata.obs, var=adata.var)
    with pytest.raises(ValueError, match="X is None but the object is not empty"):
        move_to_x(no_x, "float64")


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_round_trip_obs(adata):
    names = ["gene1", "gene4"]
    back = move_to_x(move_to_obs(adata, names), names)
    reordered = adata[:, back.var_names].copy()
    np.testing.assert_array_equal(back.to_df().to_numpy(), reordered.to_df().to_numpy())
    assert_equal(back.obs, adata.obs)
    # annotations of the moved variables are discarded by `move_to_obs`
    kept = adata.n_vars - len(names)
    assert_values_equal(back.var.iloc[:kept], reordered.var.iloc[:kept])
    assert back.var.iloc[kept:].isna().all().all()
    assert_equal(back.uns, adata.uns)
    assert_equal(back.obsm, adata.obsm)
    # values of the moved variables in other layers are discarded
    counts = back.layers["counts"]
    np.testing.assert_array_equal(
        counts[:, :kept], reordered.layers["counts"][:, :kept]
    )
    assert np.isnan(counts[:, kept:]).all()


def test_round_trip_var(adata):
    names = ["cell0", "cell5"]
    back = move_to_x(move_to_var(adata, names), names, source="var")
    reordered = adata[back.obs_names].copy()
    np.testing.assert_array_equal(back.to_df().to_numpy(), reordered.to_df().to_numpy())
    assert_equal(back.var, adata.var)
    kept = adata.n_obs - len(names)
    assert_values_equal(back.obs.iloc[:kept], reordered.obs.iloc[:kept])


def test_round_trip_obsm(adata):
    names = ["gene1", "gene4"]
    back = move_to_x(move_to_obs(adata, names, obsm_key="tmp"), "tmp", source="obsm")
    np.testing.assert_array_equal(
        back.to_df().to_numpy(), adata[:, back.var_names].to_df().to_numpy()
    )
    assert set(back.obsm) == set(adata.obsm)
