#!/usr/bin/env python
# coding: utf-8



from __future__ import annotations




from py4stats import building_block as build # py4stats のプログラミングを補助する関数群
# from py4stats import eda_tools as eda        # 基本統計量やデータの要約など
import matplotlib.pyplot as plt
import functools
from functools import singledispatch, reduce, partial
import pandas_flavor as pf

import pandas as pd
import numpy as np
import scipy as sp
import itertools
import narwhals
import narwhals as nw
import narwhals.selectors as ncs
from narwhals.typing import FrameT, IntoFrameT, SeriesT, IntoSeriesT, IntoExpr

import pandas_flavor as pf

import warnings

import math
from wcwidth import wcswidth
import scipy.stats as st

from collections import namedtuple




from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    Literal,
    overload,
    NamedTuple
)
from numpy.typing import ArrayLike

# matplotlib の Axes は遅延インポート/前方参照でもOK
try:
    from matplotlib.axes import Axes
except Exception:  # notebook等で未importでも落ちないように
    Axes = Any  # type: ignore

DataLike = Union[pd.Series, pd.DataFrame]




def is_intoframe(obj: object) -> bool:
    try:
        _ = nw.from_native(obj)
        return True
    except Exception:
        return False




def is_intoseries(obj: object) -> bool:
    try:
        _ = nw.from_native(obj, series_only = True)
        return True
    except Exception:
        return False




@singledispatch
def as_nw_datarame(arg: Any, arg_name: str = 'data'):
    try:
        return nw.from_native(arg)
    except TypeError:
        raise TypeError(
            f"Argument `{arg_name}` must be a DataFrame supported by narwhals "
            "(e.g. pandas.DataFrame, polars.DataFrame, pyarrow.Table), "
            f"but got '{type(arg).__name__}'."
        ) from None




@as_nw_datarame.register(list)
def as_nw_datarame_list(arg: List[Any], arg_name: str = 'df_list', max_items: int = 5):
    try:
        return [nw.from_native(df) for df in arg]
    except TypeError:
        not_sutisfy = [
            f"{i} ({type(v).__name__})" 
            for i, v in enumerate(arg) 
            if not is_intoframe(v)
            ]

        not_sutisfy_text = build.oxford_comma_shorten(
            not_sutisfy, quotation = False,
            max_items = max_items, suffix = 'elements'
            )

        raise TypeError(
            f"Argument `{arg_name}` must be a DataFrame supported by narwhals "
            "(e.g. pandas.DataFrame, polars.DataFrame, pyarrow.Table).\n"
                f"{11 * ' '}Elements at indices {not_sutisfy_text} are not supported."
        ) from None




@as_nw_datarame.register(dict)
def as_nw_datarame_dict(arg: Mapping[str, Any], arg_name: str = 'df_dict', max_items: int = 5):
    try:
        return {
            key: nw.from_native(df) for df in arg
            for key, df in arg.items()
            }
    except TypeError:
        not_sutisfy = [
            f"'{i}' ({type(v).__name__})" 
            for i, v in arg.items()
            if not is_intoframe(v)
            ]

        not_sutisfy_text = build.oxford_comma_shorten(
            not_sutisfy, quotation = False,
            max_items = max_items, suffix = 'elements'
            )

        raise TypeError(
            f"Argument `{arg_name}` must be a DataFrame supported by narwhals "
            "(e.g. pandas.DataFrame, polars.DataFrame, pyarrow.Table).\n"
            f"{11 * ' '}Elements at keys {not_sutisfy_text} are not supported."
        ) from None




def as_nw_series(arg: Any, arg_name: str = 'data', **keywargs):
    try:
        return nw.from_native(arg, series_only = True)
    except TypeError:
        
        raise TypeError(
            f"Argument `{arg_name}` must be a Series supported by narwhals "
            "(e.g. pandas.Series, polars.Series, pyarrow.ChunkedArray), "
            f"but got '{type(arg).__name__}'."
        ) from None




def _cast_assignment(key, value, backend):
    if isinstance(value, nw.expr.Expr) : return value
    if is_intoseries(value): return nw.from_native(value, series_only = True)
    if isinstance(value, Iterable) : 
        result = nw.Series.from_iterable(
            key, value, backend = backend
        )
        return result

def assign_nw(
        data: nw.DataFrame, 
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: Mapping[str, Iterable]
        ):
    """Narwhals DataFrame の列に Iterable オブジェクトを代入する
    data_nw = nw.from_native(load_penguins())
    data_nw.pipe(assign_nw,
        nw.lit(1).alias('const'),
        nw.col('body_mass_g') / 1000,
        bill_size = nw.col('bill_length_mm') * nw.col('bill_depth_mm'),
        heavy = nw.col('body_mass_g') > 4000,
        cutted = pd.cut(data_nw['bill_length_mm'], bins = 10, labels = False),
    ).select(ncs.matches('bill|body|heavy|cutted|const')).head(2)
    #> ┌───────────────────────────────────────────────┐
    #> |              Narwhals DataFrame               |
    #> |-----------------------------------------------|
    #> |   body_mass_g  const  bill_size  heavy  cutted|
    #> |0         3.75      1     731.17  False     2.0|
    #> |1         3.80      1     687.30  False     2.0|
    #> └───────────────────────────────────────────────┘
    """
    data_nw = as_nw_datarame(data)

    exprs = [
        as_nw_series(value) if is_intoseries(value)
        else value
        for value in exprs
    ]

    named_exprs = {
        key: _cast_assignment(key, value, data_nw.implementation)
        for key, value in named_exprs.items()
    }

    result = data_nw.with_columns(exprs, **named_exprs)
    
    return result




def _assert_selectors(*args, arg_name = '*args', nullable = False):
    if (not args and nullable) or (all(build.is_missing(args)) and nullable): 
        return None
        
    build.assert_missing(args, arg_name = arg_name)

    is_varid = [
        isinstance(v, str) or
        (build.is_character(v) and isinstance(v, list)) or
        isinstance(v, nw.expr.Expr) or
        isinstance(v, nw.selectors.Selector) 
        for v in args
        ]

    if not all(is_varid):
        invalids = [v for i, v in enumerate(args) if not is_varid[i]]
        message = f"Argument `{arg_name}` must be of type 'str', list of 'str', 'narwhals.Expr' or 'narwhals.Selector'\n"\
        + f"            The value(s) of {build.oxford_comma_and(invalids)} cannot be accepted.\n"\
        + "            Examples of valid inputs: 'x', ['x', 'y'], ncs.numeric(), nw.col('x')"
        
        raise ValueError(message)




def _assert_unique_backend(args, arg_name: str = 'args'):
    if build.length(args) <= 1: return None
    
    unique_type = build.list_unique(
        df.implementation for df in args
        )
    
    if build.length(unique_type) > 1:
        type_text = build.oxford_comma_and(unique_type)
        message = f"Elements of `{arg_name}` must share the same backend, got {type_text}." 
        raise TypeError(message)


# ## enframe



@singledispatch
def enframe(
    data: Any,
    row_id:int = 0,
    name: str = 'name',
    value: str = 'value',
    backend: Optional[Union[str, nw.Implementation]] = None,
    names: Optional[Union[list[str], list[int]]] = None,
    to_native: bool = True,
    **keywarg: Any
) -> IntoFrameT:
    """Convert a row of DataFrame or other iterable object into two-column DataFrame.

    This function transforms an object containing values (such as a Series,
    list, dict, or a single-row DataFrame) into a DataFrame with two columns:
    one for names (keys) and one for values. It is inspired by
    ``tibble::enframe()`` in R and is useful for reshaping aggregation results
    into a tidy, long format.

    The function supports multiple backends via narwhals and can return either
    a native DataFrame or a ``narwhals.DataFrame``.

    Args:
        data (Any):
            Input object to be converted. Supported types include:
            - ``narwhals.DataFrame`` (typically with a single row)
            - ``narwhals.Series``
            - ``list`` or ``tuple``
            - ``dict``
        row_id (int, optional):
            Row index to extract values from when ``data`` is a DataFrame.
            Defaults to 0.
        name (str, optional):
            Column name for variable names (keys). Defaults to ``'name'``.
        value (str, optional):
            Column name for values. Defaults to ``'value'``.
        backend (str or narwhals.Implementation, optional):
            Backend used to construct the output DataFrame.
            If None, the backend is inferred from the input data.
        names (list of str, optional):
            Names corresponding to the values.
            If None, names are inferred from column names, index,
            or keys of the input object.
        to_native (bool, optional):
            If True, convert the result to the native DataFrame type of the
            selected backend. If False, return a narwhals DataFrame.
            Defaults to True.
        **keywarg:
            Additional keyword arguments passed to the internal dispatch methods.

    Returns:
        IntoFrameT or narwhals.DataFrame:
            A two-column DataFrame with one column for names and one for values.
            The return type depends on the value of ``to_native``.

    Raises:
        NotImplementedError:
            If the input object type is not supported.

    Examples:
        Convert a single-row DataFrame produced by an aggregation:

        >>> df.select(ncs.numeric().mean()).pipe(enframe)
              name     value
        0     mpg     20.09
        1     hp     146.69

        Convert a Series:

        >>> enframe(pd.Series([10, 20], index=['a', 'b']))
          name  value
        0    a     10
        1    b     20

        Convert a dictionary:

        >>> enframe({'x': 1, 'y': 2})
          name  value
        0    x      1
        1    y      2
    """
    raise NotImplementedError(f'enframe mtethod for object {type(data)} is not implemented.')




@enframe.register(nw.DataFrame)
@enframe.register(nw.typing.IntoDataFrame)
def enframe_table(
    data: IntoFrameT,
    row_id:int = 0,
    name: str = 'name',
    value: str = 'value',
    names: Optional[Union[list[str], list[int]]] = None,
    backend: Optional[Union[str, nw.Implementation]] = None,
    to_native: bool = True,
    **keywarg: Any
) -> IntoFrameT:
    # 引数のアサーション =======================================================
    build.assert_count(row_id, arg_name = 'row_id', len_arg = 1)
    build.assert_character(name, arg_name = 'name', len_arg = 1)
    build.assert_character(value, arg_name = 'value', len_arg = 1)
    # build.assert_character(names, arg_name = 'names', nullable = True)
    build.assert_logical(to_native, arg_name = 'to_native')
    # =======================================================================
    data = as_nw_datarame(data)

    if backend is None:
        backend = data.implementation
    if names is None:
        nemes = data.columns
    
    result = nw.from_dict({
        name: nemes,
        value: data.row(row_id)
    }, backend = backend)
    
    if to_native: return result.to_native()
    return result




@enframe.register(Union[list, tuple])
def enframe_iterable(
    data: Union[nw.Series, list, tuple],
    name: str = 'name',
    value: str = 'value',
    names: Optional[Union[list[str], list[int]]] = None,
    backend: Optional[Union[str, nw.Implementation]] = None,
    to_native: bool = True,
    **keywarg: Any
) -> IntoFrameT:
    # 引数のアサーション =======================================================
    build.assert_character(name, arg_name = 'name', len_arg = 1)
    build.assert_character(value, arg_name = 'value', len_arg = 1)
    build.assert_logical(to_native, arg_name = 'to_native')
    # =======================================================================
    if backend is None: backend = 'pandas'
    if names is None: names = range(build.length(data))

    result = nw.from_dict({
        name: names,
        value: list(data)
    }, backend = backend)
    
    if to_native: return result.to_native()
    return result




@enframe.register(Union[int, float, bool, None, str, pd._libs.missing.NAType])
def enframe_atomic(
    data: Union[int, float, bool, None, str, pd._libs.missing.NAType],
    name: str = 'name',
    value: str = 'value',
    names: Optional[Union[list[str], list[int]]] = None,
    backend: Optional[Union[str, nw.Implementation]] = None,
    to_native: bool = True,
    **keywarg: Any
) -> IntoFrameT:
    # 引数のアサーション =======================================================
    build.assert_character(name, arg_name = 'name', len_arg = 1)
    build.assert_character(value, arg_name = 'value', len_arg = 1)
    build.assert_logical(to_native, arg_name = 'to_native')
    # =======================================================================
    if backend is None: backend = 'pandas'

    result = nw.from_dict({
        name: 0,
        value: [data]
    }, backend = backend)
    
    if to_native: return result.to_native()
    return result




@enframe.register(nw.typing.IntoSeries)
def enframe_series(
    data: IntoSeriesT,
    name: str = 'name',
    value: str = 'value',
    names: Optional[Union[list[str], list[int]]] = None,
    backend: Optional[Union[str, nw.Implementation]] = None,
    to_native: bool = True,
    **keywarg: Any
) -> IntoFrameT:
    # 引数のアサーション =======================================================
    build.assert_character(name, arg_name = 'name', len_arg = 1)
    build.assert_character(value, arg_name = 'value', len_arg = 1)
    # build.assert_character(names, arg_name = 'names', nullable = True)
    build.assert_logical(to_native, arg_name = 'to_native')
    # =======================================================================
    data = as_nw_series(data)

    if backend is None:
        if hasattr(data, 'implementation'):
            backend = data.implementation
        else:
            backend = 'pandas'
    if names is None:
        if hasattr(data, 'implementation') and (data.implementation.is_pandas()):
            names = data.to_pandas().index.to_list()

    args_dict = locals()
    args_dict.pop('data')
    return enframe_iterable(data, **args_dict)




@enframe.register(dict)
def enframe_dict(
    data: dict,
    name: str = 'name',
    value: str = 'value',
    names: Optional[Union[list[str], list[int]]] = None,
    backend: Optional[Union[str, nw.Implementation]] = None,
    to_native: bool = True,
    **keywarg: Any
) -> IntoFrameT:
    # 引数のアサーション =======================================================
    build.assert_character(name, arg_name = 'name', len_arg = 1)
    build.assert_character(value, arg_name = 'value', len_arg = 1)
    build.assert_logical(to_native, arg_name = 'to_native')
    # =======================================================================
    if backend is None: backend = 'pandas'
    if names is None:   names = data.keys()

    result = nw.from_dict({
        name: names,
        value: data.values()
    }, backend = backend)
    
    if to_native: return result.to_native()
    return result

