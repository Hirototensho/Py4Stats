#!/usr/bin/env python
# coding: utf-8



from __future__ import annotations


# # `eda_tools_nw`： `narwhals` ライブラリを使った実験的実装



from py4stats import bilding_block as bild # py4stats のプログラミングを補助する関数群
from py4stats import eda_tools as eda        # 基本統計量やデータの要約など
import functools
from functools import singledispatch
import pandas_flavor as pf

import pandas as pd
import numpy as np
import scipy as sp
import itertools
import narwhals as nw
import narwhals.selectors as ncs
from narwhals.typing import FrameT, IntoFrameT

import pandas_flavor as pf




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
)
from numpy.typing import ArrayLike

# matplotlib の Axes は遅延インポート/前方参照でもOK
try:
    from matplotlib.axes import Axes
except Exception:  # notebook等で未importでも落ちないように
    Axes = Any  # type: ignore

DataLike = Union[pd.Series, pd.DataFrame]


# # `diagnose()`



def get_dtypes(data: FrameT) -> pd.Series:
    data_nw = nw.from_native(data)
    if hasattr(data, 'dtypes'):
        list_dtypes = data.dtypes
        # if hasattr(list_dtypes, 'to_list'): list_dtypes = list_dtypes.to_list()
    else:
        list_dtypes = [str(data_nw.schema[col]) for col in data_nw.columns]

    s_dtypes = pd.Series(list_dtypes, index = data_nw.columns).astype(str)
    return s_dtypes




@pf.register_dataframe_method
def diagnose_nw(self: FrameT) -> FrameT:
    """Summarize each column of a DataFrame for quick EDA.

    This method computes basic diagnostics for each column:
    - dtype
    - missing_count / missing_percent
    - unique_count / unique_rate

    Returns:
        pandas.DataFrame:
            Summary table indexed by original column names with columns:
            - dtype: pandas dtype of the column.
            - missing_count: number of missing values.
            - missing_percent: percentage of missing values (100 * missing_count / nrow).
            - unique_count: number of unique values (excluding duplicates).
            - unique_rate: percentage of unique values (100 * unique_count / nrow).

    Raises:
        AssertionError:
            If `self` is not a pandas.DataFrame.
    """
    self_nw = nw.from_native(self)

    n = self_nw.shape[0]
    list_dtypes = get_dtypes(self).to_list()

    result = nw.from_dict({
        'columns':self_nw.columns,
        'dtype':list_dtypes,
        'missing_count':self_nw.null_count().row(0),
        'unique_count':[self_nw[col].n_unique() for col in self_nw.columns]
    }, backend = self_nw.implementation)\
    .with_columns(
        (100 * nw.col('missing_count') / n).alias('missing_percent'),
        (100 * nw.col('unique_count') / n).alias('unique_rate')
    )\
    .select('columns', 'dtype', 'missing_count', 'missing_percent', 'unique_count', 'unique_rate')\
    .to_native()
    return result


# ### 異なるデータフレームの列を比較する関数



def is_FrameT(obj: object) -> bool:
    try:
        _ = nw.from_native(obj)
        return True
    except Exception:
        return False




ReturnMatch = Literal["all", "match", "mismatch"]

def compare_df_cols_nw(
    df_list: List[FrameT],
    return_match: Literal["all", "match", "mismatch"] = 'all',
    df_name = None,
    dropna = False,
) -> pd.DataFrame:
  """Compare dtypes of columns with the same names across multiple DataFrames.

  Args:
      df_list (list[pandas.DataFrame]):
          List of DataFrames to compare.
      return_match (str):
          Which rows to return.
          - 'all': return all columns.
          - 'match': return only columns whose dtypes match across all DataFrames.
          - 'mismatch': return only columns whose dtypes do not match.
      df_name (list[str] or None):
          Names for each DataFrame (used as column names in the output).
          If None, auto-generated as ['df1', 'df2', ...].
      dropna (bool):
          Passed to `nunique(dropna=...)` when checking whether dtypes match.

  Returns:
      pandas.DataFrame:
          A table with index = column names (`term`) and one column per DataFrame
          containing the dtype. Additional column:
          - match_dtype (bool): True if all dtypes are identical across DataFrames.

  Raises:
      AssertionError:
          If `df_list` is not a list of pandas.DataFrame.
  """
  # 引数のアサーション ----------------------
  assert isinstance(df_list, list) & \
        all([is_FrameT(v) for v in df_list]), \
        "argument 'df_list' is must be a list of DataFrame."

  return_match = bild.arg_match(
      return_match, ['all', 'match', 'mismatch'],
      arg_name = 'return_match'
      )
  # --------------------------------------
  # df_name が指定されていなければ、自動で作成します。
  if df_name is None:
      df_name = [f'df{i + 1}' for i in range(len(df_list))]

  # df_list = [v.copy() for v in df_list] # コピーを作成
  df_list = [nw.from_native(v) for v in df_list]
  dtype_list = [get_dtypes(v) for v in df_list]
  res = pd.concat(dtype_list, axis = 1)
  res.columns = df_name
  res.index.name = 'term'
  res['match_dtype'] = res.nunique(axis = 1, dropna = dropna) == 1

  if(return_match == 'match'):
    res = res[res['match_dtype']]
  elif(return_match == 'mismatch'):
    res = res[~res['match_dtype']]

  return res




# import narwhals.selectors as ncs
# import itertools
# StatsLike = Union[str, Callable[..., Any]]

def compare_df_stats_nw(
    df_list: List[FrameT],
    return_match: ReturnMatch = "all",
    df_name: Optional[List[str]] = None,
    stats: Callable[..., Any] = nw.mean,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    # **kwargs: Any,
) -> pd.DataFrame:
    """Compare numeric column statistics across multiple DataFrames.

    This function computes a summary statistic (e.g., mean) for numeric columns
    in each DataFrame, then checks whether those statistics are close across
    DataFrames using `numpy.isclose`.

    Args:
        df_list (list[FrameT]):
            List of DataFrames to compare.
        return_match (str):
            Which rows to return.
            - 'all': return all columns.
            - 'match': return only columns whose stats are close across all pairs.
            - 'mismatch': return only columns whose stats are not close.
            Note: this code uses `match_stats` internally; see Returns.
        df_name (list[str] or None):
            Names for each DataFrame (used as column names in the output).
            If None, auto-generated as ['df1', 'df2', ...].
        stats (str or callable):
            Statistic passed to `.agg(stats, **kwargs)` (e.g., 'mean', 'median').
        rtol (float):
            Relative tolerance for `numpy.isclose`.
        atol (float):
            Absolute tolerance for `numpy.isclose`.
        **kwargs:
            Extra keyword arguments forwarded to `.agg(stats, **kwargs)`.

    Returns:
        pandas.DataFrame:
            A table with index = numeric column names (`term`) and one column per DataFrame
            containing the computed statistic. Additional column:
            - match_stats (bool): True if stats are close for all DataFrame pairs.

    Raises:
        AssertionError:
            If `df_list` is not a list of pandas.DataFrame.
    """
    # 引数のアサーション ----------------------
    assert isinstance(df_list, list) & \
            all([is_FrameT(v) for v in df_list]), \
            "argument 'df_list' is must be a list of DataFrame."

    return_match = bild.arg_match(
        return_match,
        ['all', 'match', 'mismatch'],
        arg_name = 'return_match'
        )
    # --------------------------------------
    # df_name が指定されていなければ、自動で作成します。
    if df_name is None:
        df_name = [f'df{i + 1}' for i in range(len(df_list))]

    df_list_nw = [nw.from_native(v) for v in df_list]

    stats_list = [_compute_stats(df, stats) for df in df_list_nw]
    res = pd.concat(stats_list, axis = 1)
    res.columns = df_name
    res.index.name = 'term'

    # データフレームのペア毎に、統計値が近いかどうかを比較します。
    pairwise_comparesion = \
    [pd.Series(
        np.isclose(
            res.iloc[:, i], res.iloc[:, j],
            rtol = rtol, atol = atol
        ), index = res.index)
    for i, j in itertools.combinations(range(len(res.columns)), 2)
        ]

    res['match_stats'] = pd.concat(pairwise_comparesion, axis = 1).all(axis = 1)

    if(return_match == 'match'):
        res = res[res['match_stats']]
    elif(return_match == 'mismatch'):
        res = res[~res['match_stats']]

    return res

def _compute_stats(df, func):
    numeric_vars = df.select(ncs.numeric()).columns
    return df.select(func(numeric_vars)).to_pandas().loc[0, :]


# ## クロス集計表ほか



@pf.register_dataframe_method
def crosstab_nw(
        df_native: IntoFrameT, 
        index: str, columns: str, 
        margins: bool = False,
        margins_name: str = 'All',
        sort_index: bool = True,
        normalize: Union[bool, Literal["all", "index", "columns"]] = False,
        ) -> IntoFrameT:
    df = nw.from_native(df_native)

    out = (
        df.with_columns(__n=nw.lit(1))              # 1を立てる
          .pivot(
              on = columns,
              index = index,
              values = '__n',
              aggregate_function = 'sum',             # セル内の1を合計＝件数
              sort_columns = True,
          )
          # 欠損セルを0にしたい場合（バックエンド依存はあるが一般にOK）
          .with_columns(nw.all().fill_null(0))
    )
    if sort_index:
        out = out.sort(index)
    # return out
    if margins:
        out = out.with_columns(nw.sum_horizontal(ncs.numeric()).alias(margins_name))

        # row_sums の作成と結合
        row_sums = out.select(ncs.numeric().sum())\
            .with_columns(nw.lit(margins_name).alias(index))

        if normalize == 'columns':
            numeric_vars = out.select(ncs.numeric()).columns
            for v in numeric_vars:
                out = out.with_columns(
                    nw.col(v) / row_sums.item(0, v)
                )
        else:
            out = nw.concat([
                    out, row_sums.select(index, ncs.numeric())
                    ], 
                    how = 'vertical'
                    )

        if normalize == 'index':
            out = out.with_columns(
                ncs.numeric() / nw.col(margins_name)
                ).drop(margins_name, strict = False)

        if normalize == 'all':
            total_val = out[margins_name].tail(1).item(0)
            out = out.with_columns(ncs.numeric()/total_val)

        if not normalize:
            out = out.with_columns(ncs.numeric().cast(nw.Int64))

    # return out    
    if out.implementation.is_pandas_like():
        result = nw.to_native(out).set_index(index)
        # .astype(int)
    else:
        result = out.to_native()
    return result




# @pf.register_dataframe_method
def freq_table_nw(
    self: FrameT,
    subset: Union[str, Sequence[str]],
    sort: bool = True,
    descending: bool = False,
    dropna: bool = False,
) -> FrameT:
    """Compute frequency table for one or multiple columns.

    Args:
        self (pandas.DataFrame):
            Input DataFrame.
        subset (str or list[str]):
            Column(s) to count by. Passed to `DataFrame.value_counts(subset=...)`.
        sort (bool):
            Whether to sort counts.
        ascending (bool):
            Sort order.
        dropna (bool):
            Whether to drop NaN from counts.

    Returns:
        pandas.DataFrame:
            Frequency table with columns:
            - freq: counts
            - perc: proportions
            - cumfreq: cumulative counts
            - cumperc: cumulative proportions
    """
    df = nw.from_native(self)
    if dropna:
        df = df.drop_nulls(subset)
    result = df.with_columns(__n=nw.lit(1))\
        .group_by(nw.col(subset))\
        .agg(
            nw.col('__n').sum().alias('freq'),
            )
    if sort:
        result = result.sort(subset, descending = descending)

    result = result.with_columns(
            (nw.col('freq') / nw.col('freq').sum()).alias('perc'),
            nw.col('freq').cum_sum().alias('cumfreq')
        )\
        .with_columns(
            (nw.col('cumfreq') / nw.col('freq').sum()).alias('cumperc'),
        )

    return result.to_native()




@singledispatch
def tabyl_nw(
    self: IntoFrameT,
    index: str,
    columns: str,
    margins: bool = True,
    margins_name: str = "All",
    normalize: Union[bool, Literal["index", "columns", "all"]] = "index",
    dropna: bool = False,
    rownames: Optional[Sequence[str]] = None,
    colnames: Optional[Sequence[str]] = None,
    digits: int = 1,
) -> pd.DataFrame:
    df = nw.from_native(self)
    res = tabyl_nw(
        df.to_pandas(), index = index, columns = columns,
        margins = margins, margins_name = margins_name, normalize = normalize,
        dropna = dropna, rownames = rownames, colnames = colnames, digits = digits
        )
    return res




@pf.register_dataframe_method
@tabyl_nw.register(pd.DataFrame)
def tabyl_pandas(
    self: pd.DataFrame,
    index: str,
    columns: str,
    margins: bool = True,
    margins_name: str = "All",
    normalize: Union[bool, Literal["index", "columns", "all"]] = "index",
    dropna: bool = False,
    rownames: Optional[Sequence[str]] = None,
    colnames: Optional[Sequence[str]] = None,
    digits: int = 1,
) -> pd.DataFrame:
    """Create a crosstab with counts and (optionally) percentages in parentheses.

    This function produces a table similar to `janitor::tabyl()` (R), where the
    main cell is a count and percentages can be appended like: `count (xx.x%)`.

    Args:
        self (pandas.DataFrame):
            Input DataFrame.
        index (str):
            Column name used for row categories.
        columns (str):
            Column name used for column categories.
        margins (bool):
            Add margins (totals) if True.
        margins_name (str):
            Name of the margin row/column.
        normalize (bool or {'index','columns','all'}):
            If False, return counts only.
            Otherwise, compute percentages normalized by the specified axis.
        dropna (bool):
            Whether to drop NaN in the crosstab.
        rownames (list[str] or None):
            Row names passed to `pandas.crosstab`.
        colnames (list[str] or None):
            Column names passed to `pandas.crosstab`.
        digits (int):
            Number of decimal places for percentages.

    Returns:
        pandas.DataFrame:
            Crosstab table. If `normalize` is not False, cells contain strings like
            `"count (xx.x%)"`. Otherwise counts (as strings after formatting).
    """
    if(not isinstance(normalize, bool)):
      normalize = bild.arg_match(
          normalize, ['index', 'columns', 'all'],
          arg_name = 'normalize'
          )

    if self[index].dtype == "bool":
        self[index] = self[index].astype(str)
    if self[columns].dtype == "bool":
        self[columns] = self[columns].astype(str)

    # 度数クロス集計表（最終的な表では左側の数字）
    c_tab1 = pd.crosstab(
        index = self[index], columns = self[columns], values = None,
        rownames = rownames, colnames = colnames,
        aggfunc = None, margins = margins, margins_name = margins_name,
        dropna = dropna, normalize = False
        )

    c_tab1 = c_tab1.apply(bild.style_number, digits = 0)

    if(normalize != False):

      # 回答率クロス集計表（最終的な表では括弧内の数字）
      c_tab2 = pd.crosstab(
          index = self[index], columns = self[columns], values = None,
          rownames = rownames, colnames = colnames,
          aggfunc = None, margins = margins, margins_name = margins_name,
          dropna = dropna, normalize = normalize
          )

      # 2つめのクロス集計表の回答率をdigitsで指定した桁数のパーセントに換算し、文字列化します。
      c_tab2 = c_tab2.apply(bild.style_percent, digits = digits)

      col = c_tab2.columns
      idx = c_tab2.index
      # 1つめのクロス集計表も文字列化して、↑で計算したパーセントに丸括弧と%記号を追加したものを文字列として結合します。
      c_tab1.loc[idx, col] = c_tab1.astype('str').loc[idx, col] + ' (' + c_tab2 + ')'

    return c_tab1

