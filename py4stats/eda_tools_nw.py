#!/usr/bin/env python
# coding: utf-8



from __future__ import annotations


# # `eda_tools_nw`： `narwhals` ライブラリを使った実験的実装



from py4stats import bilding_block as bild # py4stats のプログラミングを補助する関数群
from py4stats import eda_tools as eda        # 基本統計量やデータの要約など
import matplotlib.pyplot as plt
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
    **kwargs: Any,
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
        to_native: bool = True,
        **kwargs: Any
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

    if not to_native:
        return out

    if out.implementation.is_pandas_like():
        result = nw.to_native(out).set_index(index)
    else:
        result = out.to_native()
    return result




@pf.register_dataframe_method
def freq_table_nw(
    self: FrameT,
    subset: Union[str, Sequence[str]],
    sort: bool = True,
    descending: bool = False,
    dropna: bool = False,
    to_native: bool = True,
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

    if to_native:
        return result.to_native()
    else:
        return result




def tabyl_nw(
    self: IntoFrameT,
    index: str,
    columns: str,
    margins: bool = True,
    margins_name: str = 'All',
    normalize: Union[bool, Literal["index", "columns", "all"]] = "index",
    dropna: bool = False,
    # rownames: Optional[Sequence[str]] = None,
    # colnames: Optional[Sequence[str]] = None,
    digits: int = 1,
    **kwargs: Any
) -> pd.DataFrame:
    """Create a crosstab with counts and (optionally) percentages in parentheses.

    This function produces a table similar to `janitor::tabyl()` (R), where the
    main cell is a count and percentages can be appended like: `count (xx.x%)`.

    Args:
        self (IntoFrameT):
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
    args_dict = locals()
    args_dict.pop('normalize')
    c_tab1 = crosstab_nw(
       df_native = self,
       normalize = False,
       **args_dict
       )

    c_tab1 = c_tab1.apply(bild.style_number, digits = 0)

    if(normalize != False):
        # 回答率クロス集計表（最終的な表では括弧内の数字）
        c_tab2 = crosstab_nw(
           df_native = self, 
           normalize = normalize, 
           **args_dict
           )

        # 2つめのクロス集計表の回答率をdigitsで指定した桁数のパーセントに換算し、文字列化します。
        c_tab2 = c_tab2.apply(bild.style_percent, digits = digits)

        col = c_tab2.columns
        idx = c_tab2.index
        c_tab1 = c_tab1.astype('str')
        # 1つめのクロス集計表も文字列化して、↑で計算したパーセントに丸括弧と%記号を追加したものを文字列として結合します。
        c_tab1.loc[idx, col] = c_tab1.astype('str').loc[idx, col] + ' (' + c_tab2 + ')'

    return c_tab1


# ## 完全な空白列 and / or 行の除去



def missing_percent_nw(
        data: FrameT,
        axis: str = 'index',
        pct: bool = True
        ):
    data_nw = nw.from_native(data)
    n = data_nw.shape[0]

    if axis == 'index':
        miss_count = pd.Series(data_nw.null_count().row(0), index = data_nw.columns)
    else:
        miss_count = data_nw.with_columns(
            nw.all().is_null().cast(nw.Int32)
            )\
            .select(
                nw.sum_horizontal(nw.all()).alias('miss_count')
            )['miss_count']

        if data_nw.implementation.is_pandas_like():
            miss_count = pd.Series(miss_count, index = data.index)
        else:
            miss_count = pd.Series(miss_count)

    miss_pct = (100 ** pct) * miss_count / n
    return miss_pct




@pf.register_dataframe_method
def remove_empty_nw(
    self: IntoFrameT,
    cols: bool = True,
    rows: bool = True,
    cutoff: float = 1.0,
    quiet: bool = True,
    to_native: bool = True,
    **kwargs: Any
) -> IntoFrameT:
  """Remove fully (or mostly) empty columns and/or rows.

  Args:
      self (pandas.DataFrame):
          Input DataFrame.
      cols (bool):
          If True, remove empty columns.
      rows (bool):
          If True, remove empty rows.
      cutoff (float):
          Threshold on missing proportion (0-1). A column/row is removed if
          missing proportion is >= cutoff.
          - cutoff=1 removes only completely empty columns/rows.
      quiet (bool):
          If False, print removal summary.

  Returns:
      pandas.DataFrame:
          DataFrame after removing empty columns/rows.
  """
  df_shape = self.shape
  data_nw = nw.from_native(self)
  # 空白列の除去 ------------------------------
  if cols :
    empty_col = missing_percent_nw(self, axis = 'index', pct = False) >= cutoff
    data_nw = data_nw[:, (~empty_col).to_list()]

    if not(quiet) :
      ncol_removed = empty_col.sum()
      col_removed = empty_col[empty_col].index.to_series().astype('str').to_list()
      print(
            f"Removing {ncol_removed} empty column(s) out of {df_shape[1]} columns" +
            f"(Removed: {','.join(col_removed)}). "
            )
  # 空白行の除去 ------------------------------
  if rows :
    empty_rows = missing_percent_nw(self, axis = 'columns', pct = False) >= cutoff

    data_nw = data_nw.filter((~empty_rows).to_list())

    if not(quiet) :
        nrow_removed = empty_rows.sum()
        row_removed = empty_rows[empty_rows].index.to_series().astype('str').to_list()
        print(
              f"Removing {nrow_removed} empty row(s) out of {df_shape[0]} rows" +
              f"(Removed: {','.join(row_removed)}). "
          )

    if to_native: return data_nw.to_native()
    else: return data_nw




@pf.register_dataframe_method
def remove_constant_nw(
    self: IntoFrameT,
    quiet: bool = True,
    to_native: bool = True,
    **kwargs: Any
) -> IntoFrameT:
    """Remove constant columns (columns with only one unique value).

    Args:
        self (pandas.DataFrame):
            Input DataFrame.
        quiet (bool):
            If False, print removal summary.
        dropna (bool):
            Passed to `nunique(dropna=...)`. If False, NaN is counted as a value.

    Returns:
        pandas.DataFrame:
            DataFrame after removing constant columns.
    """
    data_nw = nw.from_native(self)
    df_shape = data_nw.shape

    # データフレーム(data_nw) の行が定数かどうかを判定
    unique_count = pd.Series(data_nw.select(nw.all().n_unique()).row(0), index = data_nw.columns)
    constant_col = unique_count == 1
    data_nw = data_nw[:, (~constant_col).to_list()]

    if not(quiet) :
        ncol_removed = constant_col.sum()
        col_removed = constant_col[constant_col].index.to_series().astype('str').to_list()

        print(
            f"Removing {ncol_removed} constant column(s) out of {df_shape[1]} columns" +
            f"(Removed: {','.join(col_removed)}). "
        )
    if to_native: return data_nw.to_native()
    else: return data_nw


# # パレート図を作図する関数



# パレート図を作成する関数
def Pareto_plot_nw(
    data: IntoFrameT,
    group: str,
    values: Optional[str] = None,
    top_n: Optional[int] = None,
    aggfunc:  Callable[..., Any] = nw.mean,
    ax: Optional[Axes] = None,
    fontsize: int = 12,
    xlab_rotation: Union[int, float] = 0,
    palette: Sequence[str] = ("#478FCE", "#252525"),
) -> None:
    """Plot a Pareto chart.

    If `values` is None, the chart is built from frequency counts of `group`.
    Otherwise, it aggregates `values` by `group` using `aggfunc`.

    Args:
        data (pandas.DataFrame):
            Input data.
        group (str):
            Grouping column (x-axis categories).
        values (str or None):
            Value column to aggregate. If None, uses counts.
        top_n (int or None):
            If specified, plot only top-N categories.
        aggfunc (str or callable):
            Aggregation function used when `values` is provided.
        ax (matplotlib.axes.Axes or None):
            Axes to draw the bar chart on. If None, a new figure/axes is created.
        fontsize (int):
            Base font size.
        xlab_rotation (float or int):
            Rotation angle for x tick labels.
        palette (list[str]):
            Colors for bar and line. (Note: this function explicitly uses colors.)

    Returns:
        None
    """
    # 引数のアサーション
    if(top_n is not None): bild.assert_count(top_n, lower = 1)
    bild.assert_numeric(xlab_rotation)
    bild.assert_character(palette)

    # 指定された変数でのランクを表すデータフレームを作成
    data_nw = nw.from_native(data)
    if values is None:
        shere_rank = freq_table_nw(data_nw, group, dropna = True,  descending = True, to_native = False)
        cumlative = 'cumfreq'
    else:
        shere_rank = make_rank_table_nw(
            data_nw.to_pandas(), 
            group, values, aggfunc = aggfunc,
            to_native = False
            )
        cumlative = 'cumshare'

    shere_rank = shere_rank.to_pandas().set_index(group)
    # 作図
    args_dict = locals()
    make_Pareto_plot(**args_dict)




def make_rank_table_nw(
    data: pd.DataFrame,
    group: str,
    values: str,
    aggfunc:  Callable[..., Any] = nw.mean,
    to_native: bool = True,
) -> pd.DataFrame:
    data_nw = nw.from_native(data)

    rank_table = data_nw\
        .group_by(group)\
        .agg(aggfunc(values))\
        .sort(values, descending = True)\
        .with_columns(share = nw.col(values) / nw.col(values).sum())\
        .with_columns(cumshare = nw.col('share').cum_sum())

    if to_native:
        return rank_table.to_native()
    else:
        return rank_table




def make_Pareto_plot(
    shere_rank: pd.DataFrame,
    group: str,
    cumlative: str,
    values: Optional[str] = None,
    top_n: Optional[int] = None,
    ax: Optional[Axes] = None,
    fontsize: int = 12,
    xlab_rotation: Union[int, float] = 0,
    palette: Sequence[str] = ("#478FCE", "#252525"),
    **kwargs: Any
):
    # グラフの描画
    if ax is None:
        fig, ax = plt.subplots()

    # yで指定された変数の棒グラフ
    # top_n が指定されていた場合、上位 top_n 件を抽出します。
    if top_n is not None:
        shere_rank = shere_rank.head(top_n)

    if values is None:
        ax.bar(shere_rank.index, shere_rank['freq'], color = palette[0])
        ax.set_ylabel('freq', fontsize = fontsize * 1.1)
    else:
        # yで指定された変数の棒グラフ
        ax.bar(shere_rank.index, shere_rank[values], color = palette[0])
        ax.set_ylabel(values, fontsize = fontsize * 1.1)

    ax.set_xlabel(group, fontsize = fontsize * 1.1)

    # 累積相対度数（シェア率）の線グラフ
    ax2 = ax.twinx()
    ax2.plot(
        shere_rank.index, shere_rank[cumlative],
        linestyle = 'dashed', color = palette[1], marker = 'o'
        )

    ax2.set_xlabel(group, fontsize = fontsize * 1.1)
    ax2.set_ylabel(cumlative, fontsize = fontsize * 1.1)

    # x軸メモリの回転
    ax.xaxis.set_tick_params(rotation = xlab_rotation, labelsize = fontsize)
    ax2.xaxis.set_tick_params(rotation = xlab_rotation, labelsize = fontsize);
    ax.yaxis.set_tick_params(labelsize = fontsize * 0.9)
    ax2.yaxis.set_tick_params(labelsize = fontsize * 0.9);

