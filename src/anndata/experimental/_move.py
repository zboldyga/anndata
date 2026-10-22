"""Move values between :attr:`~anndata.AnnData.X` and the axis annotations.

See https://github.com/scverse/anndata/issues/655
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .._core.anndata import AnnData
from .._core.merge import _resolve_axis, concat
from .._warnings import warn

if TYPE_CHECKING:
    from collections.abc import Collection
    from typing import Literal

    from numpy.typing import NDArray


__all__ = ["move_to_obs", "move_to_var", "move_to_x"]


def _normalize_keys(keys: str | Collection[str], what: str) -> list[str]:
    """Turn a single name or a collection of names into a list of strings."""
    if isinstance(keys, str):
        return [keys]
    keys = list(keys)
    if not keys:
        msg = f"At least one {what} must be given."
        raise ValueError(msg)
    if any(not isinstance(k, str) for k in keys):
        msg = f"{what}s must be strings, got {keys!r}."
        raise TypeError(msg)
    if len(set(keys)) != len(keys):
        msg = f"Duplicate {what}s: {keys!r}."
        raise ValueError(msg)
    return keys


def _resolve_axis_mask(
    adata: AnnData,
    keys: str | Collection[str] | NDArray[np.bool_],
    *,
    axis: Literal[0, 1],
) -> tuple[NDArray[np.bool_], list[str]]:
    """Resolve `keys` (names or boolean mask over `axis`) to a mask and the names."""
    axis_name = "obs" if axis == 0 else "var"
    names: pd.Index = getattr(adata, f"{axis_name}_names")
    if not isinstance(keys, str) and getattr(keys, "dtype", None) is not None:
        mask = np.asarray(keys)
        if mask.dtype != bool or mask.shape != (adata.shape[axis],):
            msg = (
                f"Boolean mask for `{axis_name}` must have shape ({adata.shape[axis]},), "
                f"got dtype {mask.dtype} and shape {mask.shape}."
            )
            raise ValueError(msg)
        if not mask.any():
            msg = f"At least one {axis_name} must be selected."
            raise ValueError(msg)
        return mask, list(names[mask])
    names_list = _normalize_keys(keys, f"{axis_name} name")
    if missing := [k for k in names_list if k not in names]:
        msg = f"{axis_name}_names {missing!r} not found."
        raise KeyError(msg)
    if not names.is_unique:
        msg = (
            f"`{axis_name}_names` must be unique to move values by name; "
            f"call `{axis_name}_names_make_unique()` first or pass a boolean mask."
        )
        raise ValueError(msg)
    return names.isin(names_list), names_list


def _move_to_annot(
    adata: AnnData,
    keys: str | Collection[str] | NDArray[np.bool_],
    *,
    axis: Literal[0, 1],
    layer: str | None,
    mapping_key: str | None,
) -> AnnData:
    """Shared implementation of :func:`move_to_obs` (axis=1) and :func:`move_to_var` (axis=0).

    `axis` is the axis that shrinks, i.e. the axis the moved entries are taken from.
    """
    mask, moved = _resolve_axis_mask(adata, keys, axis=axis)
    if layer is None and not adata._has_X():
        msg = "X is None, there are no values to move. Pass `layer` to move values from a layer."
        raise ValueError(msg)
    if len(set(moved)) != len(moved):
        axis_name = "obs" if axis == 0 else "var"
        msg = (
            f"Selected `{axis_name}_names` are not unique: {moved!r}; "
            f"call `{axis_name}_names_make_unique()` first."
        )
        raise ValueError(msg)
    alt_axis_name = "var" if axis == 0 else "obs"
    alt_annot: pd.DataFrame = getattr(adata, alt_axis_name)
    if mapping_key is None:
        if collisions := [k for k in moved if k in alt_annot.columns]:
            msg = f"`{alt_axis_name}` already has columns {collisions!r}; cannot move."
            raise ValueError(msg)
    elif mapping_key in getattr(adata, f"{alt_axis_name}m"):
        msg = f"`{alt_axis_name}m` already has key {mapping_key!r}; cannot move."
        raise ValueError(msg)

    # Extract the values first so the subset below does not need to keep them.
    idx = (slice(None), mask) if axis == 1 else (mask, slice(None))
    df = adata[idx].to_df(layer)
    if axis == 0:
        df = df.T
    df = df.reindex(columns=moved)  # order like `keys`
    if all(isinstance(dt, pd.SparseDtype) for dt in df.dtypes):
        # a sparse `X` gives sparse columns; annotations are dense
        df = pd.DataFrame(df.to_numpy(), index=df.index, columns=df.columns)
    if isinstance(df.columns, pd.CategoricalIndex):  # from a categorical var_names
        df.columns = pd.Index(df.columns.astype(object))

    rest = adata[(slice(None), ~mask) if axis == 1 else (~mask, slice(None))].copy()
    if mapping_key is None:
        # Assign positionally: `df` has the same index as the annotation,
        # and index-aligned joins fail for non-unique names.
        annot = getattr(rest, alt_axis_name).copy()
        for col in df.columns:
            annot[col] = df[col].to_numpy()
        setattr(rest, alt_axis_name, annot)
    else:
        getattr(rest, f"{alt_axis_name}m")[mapping_key] = df
    return rest


def move_to_obs(
    adata: AnnData,
    keys: str | Collection[str] | NDArray[np.bool_],
    *,
    layer: str | None = None,
    obsm_key: str | None = None,
) -> AnnData:
    """Move variables (columns of :attr:`~anndata.AnnData.X`) into :attr:`~anndata.AnnData.obs`.

    The selected variables are removed from the object (``X``, ``layers``, ``var``, ``varm`` and ``varp``
    are subset accordingly) and their values become columns of ``obs``
    named after the variables, or a :class:`~pandas.DataFrame` in ``obsm[obsm_key]``.
    All other elements (``uns``, ``raw``, ``obsm``, ``obsp``) are kept.

    .. warning::
       APIs in the experimental module are currently in development and subject to change at any time.

    Params
    ------
    adata
        The annotated data matrix. It is not modified; a new object is returned.
    keys
        Name(s) of the variables to move, or a boolean mask over ``var_names``.
    layer
        Take the values from ``layers[layer]`` instead of ``X``.
        Values of the moved variables in all *other* layers are discarded.
    obsm_key
        If given, the values are stored as a :class:`~pandas.DataFrame` in ``obsm[obsm_key]``
        instead of as columns of ``obs``.

    Returns
    -------
    A new :class:`~anndata.AnnData` with ``n_vars`` reduced by the number of moved variables.

    See Also
    --------
    move_to_x : the inverse operation.

    Examples
    --------
    >>> import numpy as np, pandas as pd, anndata as ad
    >>> from anndata.experimental import move_to_obs, move_to_x
    >>> adata = ad.AnnData(
    ...     np.arange(6.0).reshape(2, 3),
    ...     var=pd.DataFrame(index=["gene1", "gene2", "age"]),
    ... )
    >>> moved = move_to_obs(adata, "age")
    >>> moved
    AnnData object with n_obs × n_vars = 2 × 2
        obs: 'age'
        layers: None (.X)
    >>> moved.obs
       age
    0  2.0
    1  5.0
    >>> move_to_x(moved, "age").to_df()
       gene1  gene2  age
    0    0.0    1.0  2.0
    1    3.0    4.0  5.0
    """
    return _move_to_annot(adata, keys, axis=1, layer=layer, mapping_key=obsm_key)


def move_to_var(
    adata: AnnData,
    keys: str | Collection[str] | NDArray[np.bool_],
    *,
    layer: str | None = None,
    varm_key: str | None = None,
) -> AnnData:
    """Move observations (rows of :attr:`~anndata.AnnData.X`) into :attr:`~anndata.AnnData.var`.

    The mirror image of :func:`move_to_obs`: the selected observations are removed from the object
    and their values become columns of ``var`` named after the observations,
    or a :class:`~pandas.DataFrame` in ``varm[varm_key]``.

    .. warning::
       APIs in the experimental module are currently in development and subject to change at any time.

    Params
    ------
    adata
        The annotated data matrix. It is not modified; a new object is returned.
    keys
        Name(s) of the observations to move, or a boolean mask over ``obs_names``.
    layer
        Take the values from ``layers[layer]`` instead of ``X``.
        Values of the moved observations in all *other* layers are discarded.
    varm_key
        If given, the values are stored as a :class:`~pandas.DataFrame` in ``varm[varm_key]``
        instead of as columns of ``var``.

    Returns
    -------
    A new :class:`~anndata.AnnData` with ``n_obs`` reduced by the number of moved observations.
    If ``adata.raw`` is set, it is subset along with ``X``.

    See Also
    --------
    move_to_x : the inverse operation (with ``source="var"``).
    """
    return _move_to_annot(adata, keys, axis=0, layer=layer, mapping_key=varm_key)


def _values_from_mapping(
    adata: AnnData, key: str, *, axis_name: Literal["obs", "var"]
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Get the values of `adata.{obs,var}m[key]` as a DataFrame with unique column names."""
    mapping_name = f"{axis_name}m"
    mapping = getattr(adata, mapping_name)
    if key not in mapping:
        msg = f"{key!r} not found in `{mapping_name}`."
        raise KeyError(msg)
    elem = mapping[key]
    if isinstance(elem, pd.DataFrame):
        df = elem
        if isinstance(df.columns, pd.CategoricalIndex):
            df = df.set_axis(pd.Index(df.columns.astype(object)), axis=1)
    else:
        try:
            arr = np.asarray(elem)
        except Exception as e:
            msg = (
                f"`{mapping_name}[{key!r}]` of type {type(elem)} cannot be moved to X."
            )
            raise TypeError(msg) from e
        if arr.ndim != 2:
            msg = f"`{mapping_name}[{key!r}]` must be 2-dimensional, got shape {arr.shape}."
            raise ValueError(msg)
        df = pd.DataFrame(
            arr,
            index=getattr(adata, f"{axis_name}_names"),
            columns=[f"{key}{i}" for i in range(arr.shape[1])],
        )
    if not df.columns.is_unique:
        msg = f"Columns of `{mapping_name}[{key!r}]` must be unique."
        raise ValueError(msg)
    if any(not isinstance(c, str) for c in df.columns):
        df = df.set_axis(df.columns.astype(str), axis=1)
    rest_mapping = {k: v for k, v in mapping.items() if k != key}
    return df, rest_mapping


def _to_numeric_array(df: pd.DataFrame, where: str) -> np.ndarray:
    """Convert numeric/boolean columns to one 2-D array using NumPy's type promotion.

    Nullable extension dtypes are materialized with `NaN` for missing values.
    """
    if bad := [
        c
        for c, dt in df.dtypes.items()
        if not (is_numeric_dtype(dt) or is_bool_dtype(dt))
    ]:
        msg = (
            f"Only numeric or boolean values can be moved to X, "
            f"but {where} column(s) {bad!r} have dtype(s) {[str(df[c].dtype) for c in bad]!r}."
        )
        raise ValueError(msg)
    columns: list[np.ndarray] = []
    for name in df.columns:
        col = df[name]
        if (numpy_dtype := getattr(col.dtype, "numpy_dtype", None)) is not None:
            # nullable extension dtypes (Int64, Float64, boolean)
            if col.isna().any():
                arr = col.to_numpy(dtype=np.float64, na_value=np.nan)
            else:
                arr = col.to_numpy(dtype=numpy_dtype)
        else:  # numpy dtypes, and e.g. SparseDtype (densified)
            arr = col.to_numpy()
        columns.append(np.asarray(arr))
    dtype = np.result_type(*columns) if columns else np.dtype(np.float64)
    if dtype.kind not in "biuf":
        msg = f"Values of {where} cannot be combined into a numeric array (got dtype {dtype})."
        raise ValueError(msg)
    return np.column_stack([c.astype(dtype, copy=False) for c in columns]).reshape(
        len(df), len(columns)
    )


def _resolve_source(
    adata: AnnData,
    keys: str | Collection[str],
    *,
    source: Literal["obs", "var", "obsm", "varm"],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], str]:
    """Get the values to move, plus the annotation and mapping with them removed."""
    aligned_axis_name: Literal["obs", "var"] = (
        "obs" if source.startswith("obs") else "var"
    )
    aligned_annot: pd.DataFrame = getattr(adata, aligned_axis_name)
    aligned_mapping = getattr(adata, f"{aligned_axis_name}m")
    if source in {"obs", "var"}:
        names = _normalize_keys(keys, f"{source} column")
        if missing := [k for k in names if k not in aligned_annot.columns]:
            msg = f"Columns {missing!r} not found in `{source}`."
            raise KeyError(msg)
        return (
            aligned_annot[names],
            aligned_annot.drop(columns=names),
            dict(aligned_mapping),
            f"`{source}`",
        )
    if not isinstance(keys, str):
        msg = f"With {source=!r}, `keys` must be a single key, got {keys!r}."
        raise TypeError(msg)
    values, rest_mapping = _values_from_mapping(
        adata, keys, axis_name=aligned_axis_name
    )
    return values, aligned_annot, rest_mapping, f"`{source}[{keys!r}]`"


def move_to_x(
    adata: AnnData,
    keys: str | Collection[str],
    *,
    source: Literal["obs", "var", "obsm", "varm"] = "obs",
) -> AnnData:
    """Move annotation columns into :attr:`~anndata.AnnData.X`.

    With ``source="obs"`` (default), the ``obs`` columns named in ``keys`` are removed from ``obs``
    and appended to ``X`` as new variables of the same name.
    With ``source="var"`` the ``var`` columns become new observations instead.
    With ``source="obsm"`` / ``"varm"``, ``keys`` is the name of a single element of ``obsm`` / ``varm``
    (a :class:`~pandas.DataFrame` or a 2-D array) whose columns are appended to ``X``;
    columns of an array are named ``f"{key}{i}"``.

    New entries of ``var`` (or ``obs``) are filled with missing values; ``layers``, ``varm``/``obsm``
    and ``varp``/``obsp`` are padded following the rules of :func:`anndata.concat` with ``join="outer"``
    (``NaN`` for dense arrays, which may upcast integers to floats, zeros for sparse arrays).
    ``uns`` is kept. ``raw`` is kept when the observation axis is unchanged,
    otherwise it is dropped with a warning.
    Only numeric and boolean columns can be moved; categorical and string columns raise an error.

    .. warning::
       APIs in the experimental module are currently in development and subject to change at any time.

    Params
    ------
    adata
        The annotated data matrix. It is not modified; a new object is returned.
    keys
        Column name(s) of ``obs``/``var``, or the key of a single ``obsm``/``varm`` element.
    source
        Where to take the values from.

    Returns
    -------
    A new :class:`~anndata.AnnData` with the moved values appended to ``X``.

    See Also
    --------
    move_to_obs, move_to_var : the inverse operations.
    """
    if source not in {"obs", "var", "obsm", "varm"}:
        msg = f"`source` must be one of 'obs', 'var', 'obsm', 'varm', got {source!r}."
        raise ValueError(msg)
    # The axis along which X grows (the moved columns become entries of it)
    # and the axis the values are aligned to (unchanged).
    aligned_axis_name: Literal["obs", "var"] = (
        "obs" if source.startswith("obs") else "var"
    )
    grow_axis, grow_axis_name = _resolve_axis(
        "var" if aligned_axis_name == "obs" else "obs"
    )
    aligned_axis = 1 - grow_axis

    values, rest_annot, rest_mapping, where = _resolve_source(
        adata, keys, source=source
    )
    new_values = _to_numeric_array(values, where)

    grow_names: pd.Index = getattr(adata, f"{grow_axis_name}_names")
    if collisions := [c for c in values.columns if c in grow_names]:
        msg = f"`{grow_axis_name}_names` already contain {collisions!r}; cannot move."
        raise ValueError(msg)
    if not adata._has_X() and adata.shape[grow_axis] > 0:
        msg = (
            "X is None but the object is not empty along the "
            f"`{grow_axis_name}` axis; cannot append to X."
        )
        raise ValueError(msg)

    new_x = new_values if grow_axis == 1 else new_values.T
    new_annot = pd.DataFrame(index=pd.Index(values.columns, name=grow_names.name))
    # Both objects get the same aligned-axis elements, so `merge="first"` keeps them.
    aligned_pairwise = dict(getattr(adata, f"{aligned_axis_name}p"))
    src = adata.copy() if adata.is_view else adata
    if aligned_axis == 0:
        new = AnnData(
            X=new_x,
            obs=rest_annot,
            var=new_annot,
            obsm=rest_mapping,
            obsp=aligned_pairwise,
        )
        base = AnnData(
            X=src.X,
            obs=rest_annot,
            var=src.var,
            layers=dict(src.layers),
            obsm=rest_mapping,
            varm=dict(src.varm),
            obsp=aligned_pairwise,
            varp=dict(src.varp),
        )
    else:
        new = AnnData(
            X=new_x,
            obs=new_annot,
            var=rest_annot,
            varm=rest_mapping,
            varp=aligned_pairwise,
        )
        base = AnnData(
            X=src.X,
            obs=src.obs,
            var=rest_annot,
            layers=dict(src.layers),
            obsm=dict(src.obsm),
            varm=rest_mapping,
            obsp=dict(src.obsp),
            varp=aligned_pairwise,
        )
    result = concat(
        [base, new],
        axis=grow_axis,
        join="outer",
        merge="first",
        uns_merge=None,
        pairwise=True,
    )
    result.uns = deepcopy(dict(adata.uns))
    if adata.raw is not None:
        if aligned_axis == 0:
            result.raw = adata.raw.to_adata()
        else:
            msg = "Moving values to new observations drops `.raw`, as it has no values for them."
            warn(msg, UserWarning)
    return result
