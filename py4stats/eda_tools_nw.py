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
from narwhals.typing import FrameT, IntoFrameT, SeriesT, IntoSeriesT

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
    else:
        list_dtypes = [str(data_nw.schema[col]) for col in data_nw.columns]
    
    s_dtypes = pd.Series(list_dtypes, index = data_nw.columns).astype(str)
    return s_dtypes




@pf.register_dataframe_method
def diagnose_nw(self: FrameT, to_native: bool = True) -> FrameT:
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
    bild.assert_logical(to_native, arg_name = 'to_native')
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
    .select('columns', 'dtype', 'missing_count', 'missing_percent', 'unique_count', 'unique_rate')
    
    if to_native: return result.to_native()
    return result




def plot_miss_var_nw(
        data: IntoFrameT,
        values: Literal['missing_percent', 'missing_count'] = 'missing_percent', 
        sort: bool = True, 
        miss_only: bool = False, 
        fontsize: int = 12,
        ax: Optional[Axes] = None,
        color: str = '#478FCE',
        **kwargs: Any
        ) -> None:
    """Plot missing-value diagnostics for each variable in a DataFrame.

    This function visualizes the amount of missing data for each column
    as a horizontal bar chart. It supports multiple DataFrame backends
    via narwhals and relies on ``diagnose_nw`` to compute missing-value
    statistics.

    Args:
        data (IntoFrameT):
            Input DataFrame. Any DataFrame type supported by narwhals
            (e.g. pandas, polars, pyarrow) can be used.
        values (Literal['missing_percent', 'missing_count'], optional):
            Metric to plot on the horizontal axis.
            - ``'missing_percent'``: percentage of missing values per column.
            - ``'missing_count'``: absolute number of missing values per column.
            Defaults to ``'missing_percent'``.
        sort (bool, optional):
            Whether to sort columns by the selected metric before plotting.
            Defaults to ``True``.
        miss_only (bool, optional):
            Whether to include only columns that contain at least one
            missing value. If ``True``, columns with no missing values
            are excluded from the plot. Defaults to ``False``.
        fontsize (int, optional):
            Base font size used for axis labels. Defaults to ``12``.
        ax (matplotlib.axes.Axes, optional):
            Matplotlib Axes object to draw the plot on. If ``None``,
            a new figure and axes are created. Defaults to ``None``.
        color (str, optional):
            Color of the bars in the plot. Defaults to ``'#478FCE'``.
        **kwargs:
            Additional keyword arguments passed to
            ``matplotlib.axes.Axes.barh``.

    Returns:
        None:
            This function draws a plot and does not return a value.

    Raises:
        ValueError:
            If ``values`` is not one of the supported options
            (``'missing_percent'`` or ``'missing_count'``).

    Notes:
        This function is intended for exploratory data analysis.
        The underlying missing-value statistics are computed by
        ``diagnose_nw``, and the resulting plot reflects its output.
    """
    values = bild.arg_match(
        values, ['missing_percent', 'missing_count'],
        arg_name = 'values'
    )
    bild.assert_logical(sort, arg_name = 'sort')
    bild.assert_logical(miss_only, arg_name = 'miss_only')
    
    diagnose_tab = diagnose_nw(data, to_native = False)
    if sort: diagnose_tab = diagnose_tab.sort(values)
    if miss_only: diagnose_tab = diagnose_tab.filter(nw.col('missing_percent') > 0)
    
    # グラフの描画
    if ax is None:
        fig, ax = plt.subplots()

    ax.barh(
        y = diagnose_tab['columns'],
        width = diagnose_tab[values],
        color = color,
        **kwargs
    )
    if values == 'missing_percent':
        ax.set_xlabel('percentage of missing recode(%)', fontsize = fontsize * 1.1);
    if values == 'missing_count':
        ax.set_xlabel('number of missing recode', fontsize = fontsize * 1.1);


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
  bild.assert_logical(dropna, arg_name = 'dropna')
  # --------------------------------------
  # df_name が指定されていなければ、自動で作成します。
  if df_name is None:
      df_name = [f'df{i + 1}' for i in range(len(df_list))]

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


# ## グループ別平均（中央値）の比較



def compare_group_means_nw(
    group1: IntoFrameT,
    group2: IntoFrameT,
    group_names: Sequence[str] = ('group1', 'group2'),
) -> pd.DataFrame:
    """Compare group-wise means and derived difference metrics.

    Args:
        group1 (pandas.DataFrame):
            Data for group 1.
        group2 (pandas.DataFrame):
            Data for group 2.
        group_names (list[str]):
            Names used for output columns. Must be length 2.

    Returns:
        pandas.DataFrame:
            DataFrame indexed by variable names with columns:
            - {group_names[0]}: mean of group 1
            - {group_names[1]}: mean of group 2
            - norm_diff: normalized difference using pooled variance
            - abs_diff: absolute difference
            - rel_diff: relative difference defined as
            2*(A-B)/(A+B)

    Notes:
        Constant columns are removed using `remove_constant` before comparison.
        Means/variances use `numeric_only=True`.
    """
    # 引数のアサーション ==============================================
    bild.assert_character(group_names, arg_name = 'group_names')
    # ==============================================================
    group1 = nw.from_native(group1)
    group2 = nw.from_native(group2)
    group1 = remove_constant_nw(group1, to_native = False)
    group2 = remove_constant_nw(group2, to_native = False)

    res = pd.DataFrame({
        group_names[0]:group1.select(ncs.numeric().mean()).to_pandas().loc[0, :],
        group_names[1]:group2.select(ncs.numeric().mean()).to_pandas().loc[0, :]
    })

    s2A = group1.select(ncs.numeric().var()).to_pandas().loc[0, :]
    s2B = group2.select(ncs.numeric().var()).to_pandas().loc[0, :]
    nA = group1.shape[0]
    nB = group2.shape[0]

    s2_pooled = ((nA - 1) * s2A + (nB - 1) * s2B) / (nA + nB - 2)
    res['norm_diff'] = (res[group_names[0]] - res[group_names[1]]) / np.sqrt(s2_pooled)

    res['abs_diff'] = (res[group_names[0]] - res[group_names[1]]).abs()
    res['rel_diff'] = 2 * (res[group_names[0]] - res[group_names[1]]) \
                    /(res[group_names[0]] + res[group_names[1]])
    return res





def compare_group_median_nw(
    group1: IntoFrameT,
    group2: IntoFrameT,
    group_names: Sequence[str] = ("group1", "group2"),
) -> pd.DataFrame:
    """Compare group-wise medians and derived difference metrics.

    Args:
        group1 (IntoFrameT):
            Data for group 1.
        group2 (IntoFrameT):
            Data for group 2.
        group_names (list[str]):
            Names used for output columns. Must be length 2.

    Returns:
        pandas.DataFrame:
            DataFrame indexed by variable names with columns:
            - {group_names[0]}: median of group 1
            - {group_names[1]}: median of group 2
            - abs_diff: absolute difference
            - rel_diff: relative difference defined as
            2*(A-B)/(A+B)

    Notes:
        Constant columns are removed using `remove_constant` before comparison.
        Medians use `numeric_only=True`.
    """
    # 引数のアサーション ==============================================
    bild.assert_character(group_names, arg_name = 'group_names')
    # ==============================================================
    group1 = nw.from_native(group1)
    group2 = nw.from_native(group2)
    group1 = remove_constant_nw(group1, to_native = False)
    group2 = remove_constant_nw(group2, to_native = False)

    res = pd.DataFrame({
        group_names[0]:group1.select(ncs.numeric().median()).to_pandas().loc[0, :],
        group_names[1]:group2.select(ncs.numeric().median()).to_pandas().loc[0, :]
    })

    res['abs_diff'] = (res[group_names[0]] - res[group_names[1]]).abs()
    res['rel_diff'] = 2 * (res[group_names[0]] - res[group_names[1]]) \
                    /(res[group_names[0]] + res[group_names[1]])
    return res




def plot_mean_diff_nw(
    group1: IntoFrameT,
    group2: IntoFrameT,
    stats_diff: Literal["norm_diff", "abs_diff", "rel_diff"] = "norm_diff",
    ax: Optional[Axes] = None,
) -> None:
  """Plot group mean differences for each variable as a stem plot.

  Args:
      group1 (IntoFrameT):
          Data for group 1.
      group2 (IntoFrameT):
          Data for group 2.
      stats_diff (str):
          Which difference metric to plot.
          - 'norm_diff': normalized difference using pooled variance
          - 'abs_diff': absolute difference
          - 'rel_diff': relative difference
      ax (matplotlib.axes.Axes or None):
          Axes to draw on. If None, a new figure/axes is created.

  Returns:
      None
  """
  stats_diff = bild.arg_match(
      stats_diff, ['norm_diff', 'abs_diff', 'rel_diff'],
      arg_name = 'stats_diff'
      )
  group_means = compare_group_means_nw(group1, group2)

  if ax is None:
    fig, ax = plt.subplots()

  ax.stem(group_means[stats_diff], orientation = 'horizontal', basefmt = 'C7--');

  ax.set_yticks(range(len(group_means.index)), group_means.index)

  ax.invert_yaxis();




def plot_median_diff_nw(
    group1: IntoFrameT,
    group2: IntoFrameT,
    stats_diff: Literal["abs_diff", "rel_diff"] = "rel_diff",
    ax: Optional[Axes] = None,
) -> None:
  """Plot group median differences for each variable as a stem plot.

  Args:
      group1 (IntoFrameT):
          Data for group 1.
      group2 (IntoFrameT):
          Data for group 2.
      stats_diff (str):
          Which difference metric to plot.
          - 'abs_diff': absolute difference
          - 'rel_diff': relative difference
      ax (matplotlib.axes.Axes or None):
          Axes to draw on. If None, a new figure/axes is created.

  Returns:
      None
  """
  stats_diff = bild.arg_match(
      stats_diff, ['abs_diff', 'rel_diff']
      )

  group_median = compare_group_median_nw(group1, group2)

  if ax is None:
    fig, ax = plt.subplots()

  ax.stem(group_median[stats_diff], orientation = 'horizontal', basefmt = 'C7--')
  ax.set_yticks(range(len(group_median.index)), group_median.index)
  ax.invert_yaxis();


# ## クロス集計表ほか



@pf.register_dataframe_method
def crosstab_nw(
        df_native: IntoFrameT, 
        index: str, columns: str, 
        margins: bool = False,
        margins_name: str = 'All',
        sort_index: bool = True,
        normalize: Union[bool, Literal['all', 'index', 'columns']] = False,
        to_native: bool = True,
        **kwargs: Any
        ) -> IntoFrameT:
    
    bild.assert_logical(to_native, arg_name = 'to_native')
    bild.assert_logical(margins, arg_name = 'margins')
    
    if not isinstance(normalize, bool):
        normalize = bild.arg_match(
            normalize,
            ['all', 'index', 'columns'],
            arg_name = 'normalize'
        )
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
    
    if not to_native: return out
    
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
    bild.assert_logical(sort, arg_name = 'sort')
    bild.assert_logical(descending, arg_name = 'descending')
    bild.assert_logical(dropna, arg_name = 'dropna')
    bild.assert_logical(to_native, arg_name = 'to_native')
    
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
    if to_native: return result.to_native()
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
    bild.assert_logical(margins, arg_name = 'margins')
    bild.assert_character(margins_name, arg_name = 'margins_name')
    bild.assert_logical(dropna, arg_name = 'dropna')
    bild.assert_count(digits, arg_name = 'digits')
    
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
    # 引数のアサーション ==============================================
    bild.assert_logical(cols, arg_name = 'cols')
    bild.assert_logical(rows, arg_name = 'rows')
    bild.assert_numeric(cutoff, lower = 0, upper = 1)
    bild.assert_logical(quiet, arg_name = 'quiet')
    bild.assert_logical(to_native, arg_name = 'to_native')
    # ==============================================================
    
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
    return data_nw




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
    # 引数のアサーション ==============================================
    bild.assert_logical(quiet, arg_name = 'quiet')
    bild.assert_logical(to_native, arg_name = 'to_native')
    # ==============================================================
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
    return data_nw




# 列名や行名に特定の文字列を含む列や行を除外する関数
@pf.register_dataframe_method
def filtering_out_nw(
    self: IntoFrameT,
    contains: Optional[str] = None,
    starts_with: Optional[str] = None,
    ends_with: Optional[str] = None,
    axis: Union[int, str] = 'columns',
    to_native: bool = True,
) -> FrameT:
    """Filter out rows/columns whose labels match given string patterns.

    Args:
        self (IntoFrameT):
            Input DataFrame.
        contains (str or None):
            Exclude labels that contain this substring.
        starts_with (str or None):
            Exclude labels that start with this substring.
        ends_with (str or None):
            Exclude labels that end with this substring.
        axis (int or str):
            Axis to filter.
            - 1 or 'columns': filter columns by column labels.
            Supported for all DataFrame backends handled by narwhals.
            - 0 or 'index': filter rows by index (row labels).
            This option is only supported when the input DataFrame has
            an explicit index attribute (e.g. pandas.DataFrame).


    Returns:
        FrameT:
            Filtered DataFrame.

    Raises:
        AssertionError:
            If `contains`/`starts_with`/`ends_with` is provided but not a string.
        TypeError:
            If axis is set to 'index' (or 0) but the input DataFrame
            does not support row labels (i.e. has no 'index' attribute).

    Notes:
        Row-wise filtering via axis='index' relies on the presence of an explicit index. Therefore, this option is not available for DataFrame backends that do not expose row labels (e.g. some Arrow-based tables).
    
    """
    # 引数のアサーション ==============================================
    axis = str(axis)
    axis = bild.arg_match(axis, ['1', 'columns', '0', 'index'], arg_name = 'axis')
    bild.assert_logical(to_native, arg_name = 'to_native')
    # ==============================================================
    data_nw = nw.from_native(self)
    drop_table = pd.DataFrame()

    if axis in ("0", "index"):
        if not hasattr(self, "index"):
            raise TypeError(
                f"filtering_out_nw(..., axis='{axis}') requires an input that has"
                "an 'index' (row labels), e.g. pandas.DataFrame.\n"
                f"Got: {type(self)}."
            )

    if((axis == '1') | (axis == 'columns')):
        s_columns = pd.Series(data_nw.columns)
        if contains is not None:
            # assert isinstance(contains, str), "'contains' must be a string."
            bild.assert_character(contains, arg_name = 'contains')
            drop_table['contains'] = s_columns.str.contains(contains)

        if starts_with is not None:
            # assert isinstance(starts_with, str), "'starts_with' must be a string."
            bild.assert_character(starts_with, arg_name = 'starts_with')
            drop_table['starts_with'] = s_columns.str.startswith(starts_with)

        if ends_with is not None:
            # assert isinstance(ends_with, str), "'ends_with' must be a string."
            bild.assert_character(ends_with, arg_name = 'ends_with')
            drop_table['ends_with'] = s_columns.str.endswith(ends_with)
        drop_list = s_columns[drop_table.any(axis = 'columns')].to_list()
        data_nw = data_nw.drop(drop_list)
    
    elif hasattr(self, 'index'):
        if contains is not None: 
            bild.assert_character(contains, arg_name = 'contains')
            drop_table['contains'] = self.index.to_series().str.contains(contains)

        if starts_with is not None: 
            bild.assert_character(starts_with, arg_name = 'starts_with')
            drop_table['starts_with'] = self.index.to_series().str.startswith(starts_with)

        if ends_with is not None:
            bild.assert_character(ends_with, arg_name = 'ends_with')
            drop_table['ends_with'] = self.index.to_series().str.endswith(ends_with)

        keep_list = (~drop_table.any(axis = 'columns')).to_list()
        data_nw = data_nw.filter(keep_list)

    if to_native: return data_nw.to_native()
    return data_nw


# # パレート図を作図する関数



# パレート図を作成する関数
def Pareto_plot_nw(
    data: IntoFrameT,
    group: str,
    values: Optional[str] = None,
    top_n: Optional[int] = None,
    aggfunc: Callable[..., Any] = nw.mean,
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
    # 引数のアサーション ===================================================================
    bild.assert_numeric(fontsize, arg_name = 'fontsize', lower = 0, inclusive = 'right')
    bild.assert_numeric(xlab_rotation, arg_name = 'xlab_rotation')
    bild.assert_character(palette, arg_name = 'palette')
    # ===================================================================================

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
    aggfunc: Callable[..., Any] = nw.mean,
    to_native: bool = True,
) -> pd.DataFrame:
    data_nw = nw.from_native(data)

    # 引数のアサーション ===================================================================
    col_names = data_nw.columns
    group = bild.arg_match(group, values = col_names, arg_name = 'group', multiple = True)
    values = bild.arg_match(values, values = col_names, arg_name = 'values')
    # ===================================================================================
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
        bild.assert_count(top_n, lower = 1, arg_name = 'top_n')
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


# ### 代表値 + 区間推定関数
# 
# ```python
# import pandas as pd
# from palmerpenguins import load_penguins
# penguins = load_penguins() # サンプルデータの読み込み
# 
# from py4stats import eda_tools as eda
# 
# print(penguins['bill_depth_mm'].mean_qi().round(2))
# #>                 mean  lower  upper
# #> variable                          
# #> bill_depth_mm  17.15   13.9   20.0
# 
# print(penguins['bill_depth_mm'].median_qi().round(2))
# #>                median  lower  upper
# #> variable                           
# #> bill_depth_mm    17.3   13.9   20.0
# 
# print(penguins['bill_depth_mm'].mean_ci().round(2))
# #>                 mean  lower  upper
# #> variable                          
# #> bill_depth_mm  17.15  16.94  17.36
# ```



Interpolation = Literal[
    'inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', 
    'interpolated_inverted_cdf', 'hazen', 'weibull', 'linear', 
    'median_unbiased', 'normal_unbiased', 'lower', 'higher', 
    'midpoint', 'nearest'
    ]

interpolation_values = [
    'inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', 
    'interpolated_inverted_cdf', 'hazen', 'weibull', 'linear', 
    'median_unbiased', 'normal_unbiased', 'lower', 'higher', 
    'midpoint', 'nearest'
    ]




@pf.register_dataframe_method
@pf.register_series_method
def mean_qi_nw(
    self: FrameT,
    width: float = 0.975,
    interpolation: Interpolation = 'midpoint',
    to_native: bool = True
) -> pd.DataFrame:
    """Compute mean and quantile interval (QI).

    Args:
        self (pandas.Series or pandas.DataFrame):
            Input data.
        width (float):
            Upper quantile to use (must be in (0, 1)).
            Lower quantile is computed as `1 - width`.
        point_fun (str):
            Currently kept for API compatibility. (Not used in computation.)

    Returns:
        pandas.DataFrame:
            Table indexed by variable names with columns:
            - mean: mean value
            - lower: quantile at `1 - width`
            - upper: quantile at `width`

    Raises:
        AssertionError:
            If `width` is not in (0, 1).
    """
    # 引数のアサーション =======================================================
    bild.assert_numeric(width, lower = 0, upper = 1, inclusive = 'neither')
    bild.assert_logical(to_native, arg_name = 'to_native')
    interpolation = bild.arg_match(
        interpolation, interpolation_values,
        arg_name = 'interpolation'
        )
    # =======================================================================
    
    self_nw = nw.from_native(self, allow_series = True)

    if type(self_nw).__name__ == 'DataFrame':
        return mean_qi_nw_data_frame(
            self_nw, interpolation = interpolation, 
            width = width, to_native = to_native
            )
    if type(self_nw).__name__ == 'Series':
        return mean_qi_nw_series(
            self_nw, interpolation = interpolation, 
            width = width, to_native = to_native
        )
    
    raise NotImplementedError(f'mean_qi_nw mtethod for object {type(self)} is not implemented.')

def mean_qi_nw_data_frame(
    self: Union[pd.Series, pd.DataFrame],
    width: float = 0.975,
    interpolation: Interpolation = 'midpoint',
    to_native: bool = True
) -> pd.DataFrame:
    
    df_numeric = nw.from_native(self).select(ncs.numeric())

    result = nw.from_dict({
        'variable': df_numeric.columns,
        'mean': df_numeric.select(ncs.numeric().mean()).row(0),
        'lower': df_numeric.select(
            ncs.numeric().quantile(1 - width, interpolation = interpolation)
            ).row(0),
        'upper': df_numeric.select(
            ncs.numeric().quantile(width, interpolation = interpolation)
            ).row(0)
        }, backend = df_numeric.implementation
        )
    if to_native: return result.to_native()
    return result

def mean_qi_nw_series(
    self: Union[pd.Series, pd.DataFrame],
    width: float = 0.975,
    interpolation: Interpolation = 'midpoint',
    to_native: bool = True
):
    
    self_nw = nw.from_native(self, allow_series=True)
    
    result = nw.from_dict({
    'variable': [self_nw.name],
    'mean': [self_nw.mean()],
    'lower': [self_nw.quantile(1 - width, interpolation = interpolation)],
    'upper': [self_nw.quantile(width, interpolation = interpolation)]
    }, backend = self_nw.implementation
    )
    if to_native: return result.to_native()
    return result




@pf.register_dataframe_method
@pf.register_series_method
def median_qi_nw(
    self: FrameT,
    width: float = 0.975,
    interpolation: Interpolation = 'midpoint',
    to_native: bool = True
) -> pd.DataFrame:
    """Compute median and quantile interval (QI).

    Args:
        self (pandas.Series or pandas.DataFrame):
            Input data.
        width (float):
            Upper quantile to use (must be in (0, 1)).
            Lower quantile is computed as `1 - width`.
        point_fun (str):
            Currently kept for API compatibility. (Not used in computation.)

    Returns:
        pandas.DataFrame:
            Table indexed by variable names with columns:
            - median: median value
            - lower: quantile at `1 - width`
            - upper: quantile at `width`

    Raises:
        AssertionError:
            If `width` is not in (0, 1).
    """
    # 引数のアサーション =======================================================
    bild.assert_numeric(width, lower = 0, upper = 1, inclusive = 'neither')
    bild.assert_logical(to_native, arg_name = 'to_native')
    interpolation = bild.arg_match(
        interpolation, interpolation_values,
        arg_name = 'interpolation'
        )
    # =======================================================================
    
    self_nw = nw.from_native(self, allow_series = True)

    if type(self_nw).__name__ == 'DataFrame':
        return median_qi_nw_data_frame(
            self_nw, interpolation = interpolation, 
            width = width, to_native = to_native
            )
    if type(self_nw).__name__ == 'Series':
        return median_qi_nw_series(
            self_nw, interpolation = interpolation, 
            width = width, to_native = to_native
        )
    
    raise NotImplementedError(f'median_qi_nw mtethod for object {type(self)} is not implemented.')

def median_qi_nw_data_frame(
    self: Union[pd.Series, pd.DataFrame],
    width: float = 0.975,
    interpolation: Interpolation = 'midpoint',
    to_native: bool = True
) -> pd.DataFrame:
    
    df_numeric = nw.from_native(self).select(ncs.numeric())

    result = nw.from_dict({
        'variable': df_numeric.columns,
        'median': df_numeric.select(ncs.numeric().median()).row(0),
        'lower': df_numeric.select(
            ncs.numeric().quantile(1 - width, interpolation = interpolation)
            ).row(0),
        'upper': df_numeric.select(
            ncs.numeric().quantile(width, interpolation = interpolation)
            ).row(0)
        }, backend = df_numeric.implementation
        )
    if to_native: return result.to_native()
    return result

def median_qi_nw_series(
    self: Union[pd.Series, pd.DataFrame],
    width: float = 0.975,
    interpolation: Interpolation = 'midpoint',
    to_native: bool = True
):
    
    self_nw = nw.from_native(self, allow_series=True)
    
    result = nw.from_dict({
    'variable': [self_nw.name],
    'median': [self_nw.median()],
    'lower': [self_nw.quantile(1 - width, interpolation = interpolation)],
    'upper': [self_nw.quantile(width, interpolation = interpolation)]
    }, backend = self_nw.implementation
    )
    if to_native: return result.to_native()
    return result

