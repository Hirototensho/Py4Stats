#!/usr/bin/env python
# coding: utf-8



from __future__ import annotations


# # `eda_tools`：データセットを要約する関数群 `narwhals` ライブラリによる旧実装

# `eda_tools` モジュールに実装された主要な関数の依存関係
# 
# ``` python
# diagnose()                             # データフレームの基本統計・欠損情報の要約
# └─ missing_percent()                  # 欠損割合を計算
# 
# remove_empty()                         # 完全な空白列・空白行の除去
# └─ missing_percent()                  # 欠損割合を判定に利用
# 
# compare_df_cols()                      # 複数データフレームの列データ型を比較
# └─ build.arg_match()                   # 引数の妥当性チェック
# 
# compare_df_stats()                     # 列の平均・中央値など統計値に基づいて比較
# ├─ build.arg_match()
# └─ np.isclose()                       # 数値の近接性を比較
# 
# compare_df_record()                    # レコード単位で値を比較（数値・非数値を分離して比較）
# ├─ np.isclose()
# └─ pd.concat()                        # 数値・非数値結果の統合
# 
# compare_group_means()                  # グループ別の平均の差異を比較
# └─ remove_constant()                 # 定数列を除去してから計算
# 
# compare_group_median()                # グループ別の中央値の差異を比較
# └─ remove_constant()
# 
# plot_mean_diff()                       # 平均の差を可視化（stem plot）
# └─ compare_group_means()
# 
# plot_median_diff()                     # 中央値の差を可視化（stem plot）
# └─ compare_group_median()
# 
# diagnose_category()                    # カテゴリカル変数の情報（エントロピーなど）を要約
# ├─ is_dummy()                         # 01変数を boolean に変換
# ├─ missing_percent()
# └─ std_entropy()                      # 標準化エントロピー
# 
# std_entropy()                          # 標準化エントロピーを算出
# └─ entropy()
# 
# entropy()                              # 情報エントロピーを算出
# 
# freq_mode()                            # 最頻値の相対頻度を返す
# 
# make_rank_table()                      # パレート図用のランク表を作成
# └─ pd.pivot_table()
# 
# Pareto_plot()                          # パレート図の描画
# ├─ make_rank_table() または freq_table()
# └─ plt.subplots(), twinx() など
# 
# mean_qi()                               # 平均と分位区間（quantile interval）を計算
# 
# median_qi()                             # 中央値と分位区間を計算
# 
# mean_ci()                               # 平均と信頼区間（confidence interval）を計算
# └─ scipy.stats.t.isf()
# 
# set_n_miss()                            # 欠損数を指定して Series に欠損を挿入
# └─ build.arg_match()
# 
# set_prop_miss()                         # 欠損率を指定して Series に欠損を挿入
# └─ build.arg_match()
# 
# check_that()                            # データにルールを適用し、判定集計を返す
# └─ data.eval()                         # pandas の式評価
# 
# check_viorate()                         # 条件違反レコードを検出
# 
# is_dummy()                              # 01ダミー変数判定
# └─ set(self) == set(cording)
# 
# is_number()                             # 文字列が数字かどうかを正規表現で判定
# └─ detect_Kanzi(), is_ymd_like()
# 
# is_ymd()                                 # yyyy-mm-dd 形式の判定
# 
# is_ymd_like()                            # 年月日らしい表現の判定（例：2020年5月1日）
# 
# filtering_out()                         # 列・行名に特定文字列を含む行/列を除外
# └─ build.arg_match()
# 
# remove_constant()                       # 定数列の除去
# 
# crosstab2()                             # クロス集計表を作成（pd.crosstab のラッパー）
# 
# freq_table()                            # 頻度表＋累積度数を作成（カテゴリ変数の概要）
# 
# tabyl()                                 # クロス集計表＋割合＋表示整形
# ├─ pd.crosstab()
# ├─ build.style_number()
# └─ build.style_percent()
# 
# is_complet()                            # 欠損のない行を判定
# 
# Sum(), Mean(), Max(), Min(), Median()  # 行方向の合計・平均・中央値などを計算
# └─ pd.concat(...).sum() など
# ```



from py4stats import building_block as build # py4stats のプログラミングを補助する関数群
import functools
from functools import singledispatch
import matplotlib.pyplot as plt
import pandas_flavor as pf

import pandas as pd
import numpy as np
import scipy as sp

import warnings




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


# ## FutureWarnin用関数の実装



# Pandas ベースの実装を将来的に廃止することを想定して `FutureWarning` を導入します。
def future_warn_pandas_implementation(stacklevel: int = 3):
    warnings.warn(
        "The pandas-based implementation of `eda_tools` is deprecated and will be removed"
        "in a future release. Please use the narwhals-based implementation via"
        "`import py4stats as py4st`, which provides the same public API.",
        FutureWarning,
        stacklevel = stacklevel,
    )


# # `diagnose()`



def missing_percent(
    x: DataLike,
    axis: Union[str, int] = "index",
    pct: bool = True,
) -> Union[pd.Series, pd.DataFrame]:
    return (100 ** pct) * x.isna().mean(axis=axis)




# @pf.register_dataframe_method
@singledispatch
def diagnose(self: pd.DataFrame) -> pd.DataFrame:
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
    future_warn_pandas_implementation()
    assert isinstance(self, pd.DataFrame), '`self` must be a pandas.DataFrame.'
    self = self.copy()
    # 各種集計値の計算 ------------
    result = pd.DataFrame({
      'dtype':self.dtypes,
      'missing_count':self.isna().sum(),
      'missing_percent':missing_percent(self),
      'unique_count':self.nunique(),
      'unique_rate': 100 * self.nunique() / len(self),
    })

    return result




def plot_miss_var_pd(
        data: pd.DataFrame,
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
    as a horizontal bar chart. It relies on ``diagnose`` to compute missing-value
    statistics.

    Args:
        data (pd.DataFrame):
            Input DataFrame.
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
        ``diagnose``, and the resulting plot reflects its output.
    """
    future_warn_pandas_implementation()
    values = build.arg_match(
        values, ['missing_percent', 'missing_count'],
        arg_name = 'values'
    )
    build.assert_logical(sort, arg_name = 'sort')
    build.assert_logical(miss_only, arg_name = 'miss_only')
    
    diagnose_tab = diagnose(data)
    if sort: diagnose_tab = diagnose_tab.sort_values(values)
    if miss_only: diagnose_tab = diagnose_tab.query('missing_percent > 0')
    
    # グラフの描画
    if ax is None:
        fig, ax = plt.subplots()

    ax.barh(
        y = diagnose_tab.index,
        width = diagnose_tab[values],
        color = color,
        **kwargs
    )
    if values == 'missing_percent':
        ax.set_xlabel('percentage of missing recode(%)', fontsize = fontsize * 1.1);
    if values == 'missing_count':
        ax.set_xlabel('number of missing recode', fontsize = fontsize * 1.1);


# ### 異なるデータフレームの列を比較する関数



ReturnMatch = Literal["all", "match", "mismatch"]

def compare_df_cols(
    df_list: List[pd.DataFrame],
    return_match: ReturnMatch = "all",
    df_name: Optional[List[str]] = None,
    dropna: bool = False,
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
  future_warn_pandas_implementation()
  # 引数のアサーション ----------------------
  assert isinstance(df_list, list) & \
        all([isinstance(v, pd.DataFrame) for v in df_list]),\
        "argument 'df_list' is must be a list of pandas.DataFrame."

  return_match = build.arg_match(
      return_match,
       ['all', 'match', 'mismatch'],
      arg_name = 'return_match'
      )
  # --------------------------------------
  # df_name が指定されていなければ、自動で作成します。
  if df_name is None:
      df_name = [f'df{i + 1}' for i in range(len(df_list))]

  df_list = [v.copy() for v in df_list] # コピーを作成
  dtype_list = [v.dtypes for v in df_list]
  res = pd.concat(dtype_list, axis = 1)
  res.columns = df_name
  res.index.name = 'term'
  res['match_dtype'] = res.nunique(axis = 1, dropna = dropna) == 1

  if(return_match == 'match'):
    res = res[res['match_dtype']]
  elif(return_match == 'mismatch'):
    res = res[~res['match_dtype']]

  return res


# ### 平均値などの統計値の近接性で比較するバージョン



import itertools
StatsLike = Union[str, Callable[..., Any]]

def compare_df_stats(
    df_list: List[pd.DataFrame],
    return_match: ReturnMatch = "all",
    df_name: Optional[List[str]] = None,
    stats: StatsLike = "mean",
    rtol: float = 1e-05,
    atol: float = 1e-08,
    **kwargs: Any,
) -> pd.DataFrame:
  """Compare numeric column statistics across multiple DataFrames.

  This function computes a summary statistic (e.g., mean) for numeric columns
  in each DataFrame, then checks whether those statistics are close across
  DataFrames using `numpy.isclose`.

  Args:
      df_list (list[pandas.DataFrame]):
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
  future_warn_pandas_implementation()
  # 引数のアサーション ----------------------
  assert isinstance(df_list, list) & \
        all([isinstance(v, pd.DataFrame) for v in df_list]),\
        "argument 'df_list' is must be a list of pandas.DataFrame."

  return_match = build.arg_match(
      return_match,
       ['all', 'match', 'mismatch'],
      arg_name = 'return_match'
      )
  # --------------------------------------
  # df_name が指定されていなければ、自動で作成します。
  if df_name is None:
      df_name = [f'df{i + 1}' for i in range(len(df_list))]

  df_list = [v.copy() for v in df_list] # コピーを作成
  stats_list = [
      v.select_dtypes(include = ['int', 'float', 'bool'])\
      .dropna(axis = 1, how = 'all').agg(stats, **kwargs)
      for v in df_list
      ]
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




# レコード毎の近接性（数値の場合）または一致性（数値以外）で評価する関数
def compare_df_record(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> pd.DataFrame:
  """Compare two DataFrames record-wise (element-wise).

  - Numeric columns are compared with `numpy.isclose`.
  - Non-numeric columns are compared with equality (`==`).

  Args:
      df1 (pandas.DataFrame):
          First DataFrame.
      df2 (pandas.DataFrame):
          Second DataFrame. Should have the same columns as `df1`.
      rtol (float):
          Relative tolerance for numeric comparison via `numpy.isclose`.
      atol (float):
          Absolute tolerance for numeric comparison via `numpy.isclose`.

  Returns:
      pandas.DataFrame:
          Boolean DataFrame indicating whether each element matches (or is close).
          Columns are ordered to match `df1.columns`.

  Notes:
      This function assumes `df1` and `df2` have compatible columns.
      If you want strict checks, you may add assertions for shape/columns.
  """
  future_warn_pandas_implementation()
  all_columns = df1.columns
  number_col = df1.select_dtypes(include = 'number').columns
  nonnum_col = df1.select_dtypes(exclude = 'number').columns

  res_number_col = [np.isclose(df1[v], df2[v]) for v in number_col]

  res_nonnum_col = [(df1[v] == df2[v]) for v in nonnum_col]

  result = pd.concat([
      pd.DataFrame(res_number_col, index = number_col).T,
      pd.DataFrame(res_nonnum_col, index = nonnum_col).T
  ], axis = 'columns').loc[:, all_columns]

  return result


# ## グループ別平均（中央値）の比較



@singledispatch
def compare_group_means(
    group1: pd.DataFrame,
    group2: pd.DataFrame,
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
  future_warn_pandas_implementation()
  group1 = remove_constant(group1)
  group2 = remove_constant(group2)

  res = pd.DataFrame({
    group_names[0]:group1.mean(numeric_only = True),
    group_names[1]:group2.mean(numeric_only = True)
    })

  s2A = group1.var(numeric_only = True)
  s2B = group2.var(numeric_only = True)
  nA = group1.shape[0]
  nB = group2.shape[0]

  s2_pooled = ((nA - 1) * s2A + (nB - 1) * s2B) / (nA + nB - 2)
  res['norm_diff'] = (res[group_names[0]] - res[group_names[1]]) / np.sqrt(s2_pooled)

  res['abs_diff'] = (res[group_names[0]] - res[group_names[1]]).abs()
  res['rel_diff'] = 2 * (res[group_names[0]] - res[group_names[1]]) \
                    /(res[group_names[0]] + res[group_names[1]])
  return res




@singledispatch
def compare_group_median(
    group1: pd.DataFrame,
    group2: pd.DataFrame,
    group_names: Sequence[str] = ("group1", "group2"),
) -> pd.DataFrame:
  """Compare group-wise medians and derived difference metrics.

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
            - {group_names[0]}: median of group 1
            - {group_names[1]}: median of group 2
            - abs_diff: absolute difference
            - rel_diff: relative difference defined as
              2*(A-B)/(A+B)

    Notes:
        Constant columns are removed using `remove_constant` before comparison.
        Medians use `numeric_only=True`.
    """
  future_warn_pandas_implementation()
  group1 = remove_constant(group1)
  group2 = remove_constant(group2)

  res = pd.DataFrame({
    group_names[0]:group1.median(numeric_only = True),
    group_names[1]:group2.median(numeric_only = True)
    })

  res['abs_diff'] = (res[group_names[0]] - res[group_names[1]]).abs()
  res['rel_diff'] = 2 * (res[group_names[0]] - res[group_names[1]]) \
                    /(res[group_names[0]] + res[group_names[1]])
  return res




def plot_mean_diff(
    group1: pd.DataFrame,
    group2: pd.DataFrame,
    stats_diff: Literal["norm_diff", "abs_diff", "rel_diff"] = "norm_diff",
    ax: Optional[Axes] = None,
) -> None:
  """Plot group mean differences for each variable as a stem plot.

  Args:
      group1 (pandas.DataFrame):
          Data for group 1.
      group2 (pandas.DataFrame):
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
  future_warn_pandas_implementation()
  stats_diff = build.arg_match(
      stats_diff, ['norm_diff', 'abs_diff', 'rel_diff']
      )
  group_means = compare_group_means(group1, group2)

  if ax is None:
    fig, ax = plt.subplots()

  ax.stem(group_means[stats_diff], orientation = 'horizontal', basefmt = 'C7--');

  ax.set_yticks(range(len(group_means.index)), group_means.index)

  ax.invert_yaxis();




def plot_median_diff(
    group1: pd.DataFrame,
    group2: pd.DataFrame,
    stats_diff: Literal["abs_diff", "rel_diff"] = "rel_diff",
    ax: Optional[Axes] = None,
) -> None:
  """Plot group median differences for each variable as a stem plot.

  Args:
      group1 (pandas.DataFrame):
          Data for group 1.
      group2 (pandas.DataFrame):
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
  future_warn_pandas_implementation()
  stats_diff = build.arg_match(
      stats_diff, ['abs_diff', 'rel_diff']
      )

  group_median = compare_group_median(group1, group2)

  if ax is None:
    fig, ax = plt.subplots()

  ax.stem(group_median[stats_diff], orientation = 'horizontal', basefmt = 'C7--')
  ax.set_yticks(range(len(group_median.index)), group_median.index)
  ax.invert_yaxis();


# ## 完全な空白列 and / or 行の除去



# @pf.register_dataframe_method
def remove_empty(
    self: pd.DataFrame,
    cols: bool = True,
    rows: bool = True,
    cutoff: float = 1.0,
    quiet: bool = True,
) -> pd.DataFrame:
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
  future_warn_pandas_implementation()
  df_shape = self.shape

  # 空白列の除去 ------------------------------
  if cols :
    empty_col = missing_percent(self, axis = 'index', pct = False) >= cutoff
    self = self.loc[:, ~empty_col]

    if not(quiet) :
      ncol_removed = empty_col.sum()
      col_removed = empty_col[empty_col].index.to_series().astype('str').to_list()
      print(
            f"Removing {ncol_removed} empty column(s) out of {df_shape[1]} columns" +
            f"(Removed: {','.join(col_removed)}). "
            )
  # 空白行の除去 ------------------------------
  if rows :
    empty_rows = missing_percent(self, axis = 'columns', pct = False) >= cutoff
    self = self.loc[~empty_rows, :]

    if not(quiet) :
        nrow_removed = empty_rows.sum()
        row_removed = empty_rows[empty_rows].index.to_series().astype('str').to_list()
        print(
              f"Removing {nrow_removed} empty row(s) out of {df_shape[0]} rows" +
              f"(Removed: {','.join(row_removed)}). "
          )

  return self


# ## 定数列の除去



# @pf.register_dataframe_method
@singledispatch
def remove_constant(
    self: pd.DataFrame,
    quiet: bool = True,
    dropna: bool = False,
) -> pd.DataFrame:
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
  future_warn_pandas_implementation()
  df_shape = self.shape
  # データフレーム(self) の行が定数かどうかを判定
  constant_col = self.nunique(dropna = dropna) == 1
  self = self.loc[:, ~constant_col]

  if not(quiet) :
    ncol_removed = constant_col.sum()
    col_removed = constant_col[constant_col].index.to_series().astype('str').to_list()

    print(
        f"Removing {ncol_removed} constant column(s) out of {df_shape[1]} columns" +
        f"(Removed: {','.join(col_removed)}). "
     )

  return self




# 列名に特定の文字列を含む列を除外する関数
# @pf.register_dataframe_method
def filtering_out(
    self: pd.DataFrame,
    contains: Optional[str] = None,
    starts_with: Optional[str] = None,
    ends_with: Optional[str] = None,
    axis: Union[int, str] = 1,
) -> pd.DataFrame:
  """Filter out rows/columns whose labels match given string patterns.

    Args:
        self (pandas.DataFrame):
            Input DataFrame.
        contains (str or None):
            Exclude labels that contain this substring.
        starts_with (str or None):
            Exclude labels that start with this substring.
        ends_with (str or None):
            Exclude labels that end with this substring.
        axis (int or str):
            Axis to filter.
            - 1 or 'columns': filter columns by column names
            - 0 or 'index': filter rows by index values

    Returns:
        pandas.DataFrame:
            Filtered DataFrame.

    Raises:
        AssertionError:
            If `contains`/`starts_with`/`ends_with` is provided but not a string.
  """
  future_warn_pandas_implementation()
  axis = str(axis)
  axis = build.arg_match(axis, ['1', 'columns', '0', 'index'], arg_name = 'axis')
  self = self.copy()

  if((axis == '1') | (axis == 'columns')):
      if contains is not None:
        assert isinstance(contains, str), "'contains' must be a string."
        self = self.loc[:, ~self.columns.str.contains(contains)]

      if starts_with is not None:
        assert isinstance(starts_with, str), "'starts_with' must be a string."
        self = self.loc[:, ~self.columns.str.startswith(starts_with)]

      if ends_with is not None:
        assert isinstance(ends_with, str), "'ends_with' must be a string."
        self = self.loc[:, ~self.columns.str.endswith(ends_with)]
  else:
      if contains is not None:
        assert isinstance(contains, str), "'contains' must be a string."
        self = self.loc[~self.index.to_series().str.contains(contains), :]

      if starts_with is not None:
        assert isinstance(starts_with, str), "'starts_with' must be a string."
        self = self.loc[~self.index.to_series().str.startswith(starts_with), :]

      if ends_with is not None:
        assert isinstance(ends_with, str), "'ends_with' must be a string."
        self = self.loc[~self.index.to_series().str.endswith(ends_with), :]

  return self


# ## クロス集計表ほか



# @pf.register_dataframe_method
@singledispatch
def crosstab2(
    data: pd.DataFrame,
    index: str,
    columns: str,
    values: Optional[str] = None,
    rownames: Optional[Sequence[str]] = None,
    colnames: Optional[Sequence[str]] = None,
    aggfunc: Optional[Callable[..., Any]] = None,
    margins: bool = False,
    margins_name: str = "All",
    dropna: bool = True,
    normalize: Union[bool, Literal["all", "index", "columns"]] = False,
) -> pd.DataFrame:
    future_warn_pandas_implementation()
    if values is not None:
        values_vec = data[values]
    else:
        values_vec = None
    res = pd.crosstab(
        index = data[index], columns = data[columns], values = values_vec,
        rownames = rownames, colnames = colnames,
        aggfunc = aggfunc, margins = margins, margins_name = margins_name,
        dropna = dropna, normalize = normalize
        )
    return res




# @pf.register_dataframe_method
@singledispatch
def freq_table(
    self: pd.DataFrame,
    subset: Union[str, Sequence[str]],
    sort: bool = True,
    ascending: bool = False,
    dropna: bool = False,
) -> pd.DataFrame:
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
    future_warn_pandas_implementation()
    count = self.value_counts(
        subset = subset, sort = sort, ascending = ascending,
        normalize=False, dropna = dropna
        )

    rel_count = self.value_counts(
        subset = subset, sort = sort, ascending = ascending,
        normalize=True, dropna = dropna
        )

    res = pd.DataFrame({
            'freq':count,
            'perc':rel_count,
            'cumfreq':count.cumsum(),
            'cumperc':rel_count.cumsum()
        })
    return res




# @pf.register_dataframe_method
@singledispatch
def tabyl(
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
    future_warn_pandas_implementation()
    if(not isinstance(normalize, bool)):
      normalize = build.arg_match(
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

    c_tab1 = c_tab1.apply(build.style_number, digits = 0)

    if(normalize != False):

      # 回答率クロス集計表（最終的な表では括弧内の数字）
      c_tab2 = pd.crosstab(
          index = self[index], columns = self[columns], values = None,
          rownames = rownames, colnames = colnames,
          aggfunc = None, margins = margins, margins_name = margins_name,
          dropna = dropna, normalize = normalize
          )

      # 2つめのクロス集計表の回答率をdigitsで指定した桁数のパーセントに換算し、文字列化します。
      c_tab2 = c_tab2.apply(build.style_percent, digits = digits)

      col = c_tab2.columns
      idx = c_tab2.index
      # 1つめのクロス集計表も文字列化して、↑で計算したパーセントに丸括弧と%記号を追加したものを文字列として結合します。
      c_tab1.loc[idx, col] = c_tab1.astype('str').loc[idx, col] + ' (' + c_tab2 + ')'

    return c_tab1


# ## `diagnose_category()`：カテゴリー変数専用の要約関数



# @pf.register_dataframe_method
# @pf.register_series_method
@singledispatch
def is_dummy(
    self: Union[pd.Series, pd.DataFrame],
    cording: Sequence[Any] = (0, 1),
) -> Union[bool, pd.Series]:
    """Check whether values consist only of dummy codes.

    Args:
        self (pandas.Series or pandas.DataFrame):
            Input data.
        cording (list):
            Allowed set of dummy codes (default: [0, 1]).

    Returns:
        bool or pandas.Series:
            - If Series input: returns bool.
            - If DataFrame input: returns a boolean Series per column.
    """
    future_warn_pandas_implementation()
    return set(self) == set(cording)

@is_dummy.register(pd.DataFrame)
def _(self: pd.DataFrame, cording: Sequence[Any] = (0, 1)) -> pd.Series:
    future_warn_pandas_implementation()
    return self.apply(is_dummy, cording = cording)




# カテゴリカル変数についての集計関数 --------------
# 情報エントロピーと、その値を0から1に標準化したもの --------------
def entropy(X: ArrayLike, base: float = 2.0, axis: int = 0) -> float:
    future_warn_pandas_implementation()
    vc = pd.Series(X).value_counts(normalize = True, sort = False)
    res = sp.stats.entropy(pk = vc,  base = base, axis = axis)
    return res

def std_entropy(X: ArrayLike, axis: int = 0) -> float:
    future_warn_pandas_implementation()
    K = pd.Series(X).nunique()
    res = entropy(X, base = K) if K > 1 else 0.0
    return res

def freq_mode(x: pd.Series, normalize: bool = False) -> Union[int, float]:
    future_warn_pandas_implementation()
    res = x.value_counts(normalize = normalize, dropna = False).iloc[0]
    return res

# カテゴリカル変数についての概要を示す関数
def diagnose_category(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize categorical variables in a DataFrame.

    This function targets object/category/bool columns, converts 0/1 dummy
    variables to boolean, and produces a summary including missing rates,
    unique rates, mode share, and standardized entropy.

    Args:
        data (pandas.DataFrame):
            Input DataFrame.

    Returns:
        pandas.DataFrame:
            Summary table with columns:
            - count: non-missing count (from describe)
            - missing_percent: missing percentage
            - unique: number of unique values (from describe)
            - unique_percent: percentage of unique values
            - top: mode value (from describe)
            - freq: mode count (from describe, cast to int)
            - pct_mode: mode percentage
            - std_entropy: standardized entropy (0-1)
    """
    future_warn_pandas_implementation()
    # 01のダミー変数はロジカル変数に変換
    data = data.copy()
    data.loc[:, is_dummy(data)] = (data.loc[:, is_dummy(data)] == 1)
    # 文字列 or カテゴリー変数のみ抽出
    data = data.select_dtypes(include = [object, 'category', bool])

    n = len(data)
    # describe で集計表の大まかな形を作成
    res = data.describe().T
    res['freq'] = res['freq'].astype('int')
    # 追加の集計値を計算して代入
    res = res.assign(
        unique_percent = 100 * data.nunique(dropna = False) / n,
        missing_percent = missing_percent(data),
        pct_mode = (100 * (res['freq'] / n)),
        std_entropy = data.agg(std_entropy)
    )
    # 見やすいように並べ替え
    res = res.loc[:, [
        'count', 'missing_percent', 'unique', 'unique_percent',
        'top', 'freq', 'pct_mode', 'std_entropy'
        ]]

    return res


# ## その他の補助関数



def weighted_mean(x: pd.Series, w: pd.Series) -> float:
  future_warn_pandas_implementation()
  wmean = (x * w).sum() / w.sum()
  return wmean

def scale(x: pd.Series, ddof: int = 1) -> pd.Series:
    future_warn_pandas_implementation()
    z = (x - x.mean()) / x.std(ddof = ddof)
    return z

def min_max(x: pd.Series) -> pd.Series:
  future_warn_pandas_implementation()
  mn = (x - x.min()) / (x.max() - x.min())
  return mn


# # パレート図を作図する関数



import matplotlib.pyplot as plt

# パレート図に使用するランキングを作成する関数
def make_rank_table(
    data: pd.DataFrame,
    group: str,
    values: str,
    aggfunc: Union[str, Callable[..., Any]] = "sum",
) -> pd.DataFrame:
    future_warn_pandas_implementation()
    # ピボットテーブルを使って、カテゴリー group（例：メーカー）ごとの values （例：販売額）の合計を計算
    p_table = pd.pivot_table(
        data = data,
        index = group,
        values = values,
        aggfunc = aggfunc,
        fill_value = 0
        )
    # values の値に基づいてソート
    rank_table = p_table.sort_values(values, ascending=False)

    # シェア率と累積相対度数を計算
    rank_table['share'] = (rank_table[values] / rank_table[values].sum())
    rank_table['cumshare'] = rank_table['share'].cumsum()
    return rank_table




# パレート図を作成する関数
def Pareto_plot(
    data: pd.DataFrame,
    group: str,
    values: Optional[str] = None,
    top_n: Optional[int] = None,
    aggfunc: Union[str, Callable[..., Any]] = "sum",
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
    future_warn_pandas_implementation()
    # 引数のアサーション
    if(top_n is not None): build.assert_count(top_n, arg_name = 'top_n', lower = 1)
    build.assert_numeric(xlab_rotation, arg_name = 'xlab_rotation')
    build.assert_character(palette, arg_name = 'palette')

    # 指定された変数でのランクを表すデータフレームを作成
    if values is None:
        shere_rank = freq_table(data, group, dropna = True)
        cumlative = 'cumfreq'
    else:
        shere_rank = make_rank_table(data, group, values, aggfunc = aggfunc)
        cumlative = 'cumshare'

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

    # 累積相対度数の線グラフ
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



# @pf.register_dataframe_method
# @pf.register_series_method
def mean_qi(
    self: Union[pd.Series, pd.DataFrame],
    width: float = 0.975,
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
  future_warn_pandas_implementation()
  build.assert_numeric(width, arg_name = 'width', lower = 0, upper = 1, inclusive = 'neither')
  if(isinstance(self, pd.DataFrame)):
    self = self.select_dtypes([int, float])
    var_name = self.columns
  else:
    var_name = [self.name]

  res = pd.DataFrame({
      'mean':self.mean(),
      'lower':self.quantile(1 - width),
      'upper':self.quantile(width),
  }, index = var_name
  )

  res.index.name = 'variable'
  return res




# @pf.register_dataframe_method
# @pf.register_series_method
def median_qi(
    self: Union[pd.Series, pd.DataFrame],
    width: float = 0.975,
    point_fun: str = "median",
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
  future_warn_pandas_implementation()
  build.assert_numeric(width, arg_name = 'width', lower = 0, upper = 1, inclusive = 'neither')
  if(isinstance(self, pd.DataFrame)):
    self = self.select_dtypes([int, float])
    var_name = self.columns
  else:
    var_name = [self.name]

  res = pd.DataFrame({
      'median':self.median(),
      'lower':self.quantile(1 - width),
      'upper':self.quantile(width),
  }, index = var_name
  )

  res.index.name = 'variable'
  return res




from scipy.stats import t
# @pf.register_dataframe_method
# @pf.register_series_method
def mean_ci(
    self: Union[pd.Series, pd.DataFrame],
    width: float = 0.95,
) -> pd.DataFrame:
  """Compute mean and t-based confidence interval (CI).

  Args:
      self (pandas.Series or pandas.DataFrame):
          Input data.
      width (float):
          Confidence level in (0, 1) (e.g., 0.95).

  Returns:
      pandas.DataFrame:
          Table indexed by variable names with columns:
          - mean: sample mean
          - lower: lower bound of CI
          - upper: upper bound of CI

  Raises:
      AssertionError:
          If `width` is not in (0, 1).

  Notes:
      Uses t critical value with df = n - 1:
      `t.isf((1 - width) / 2, df=n-1)`.
  """
  future_warn_pandas_implementation()
  build.assert_numeric(width, arg_name = 'width', lower = 0, upper = 1, inclusive = 'neither')
  if(isinstance(self, pd.DataFrame)):
    self = self.select_dtypes([int, float])
    var_name = self.columns
  else:
    var_name = [self.name]

  n = len(self)
  t_alpha = t.isf((1 - width) / 2, df = n - 1)
  x_mean = self.mean()
  x_std = self.std()

  res = pd.DataFrame({
      'mean':x_mean,
      'lower':x_mean - t_alpha * x_std / np.sqrt(n),
      'upper':x_mean + t_alpha * x_std / np.sqrt(n),
      }, index = var_name
    )
  res.index.name = 'variable'
  return res


# ## 正規表現を文字列関連の論理関数



import regex
def detect_Kanzi(s):
  p = regex.compile(r'.*\p{Script=Han}+.*')
  res = p.fullmatch(s)
  return res is not None




# @pf.register_series_method
def is_ymd(self, na_default = True):
  """与えられた文字列が ymd 形式の日付かどうかを判定する関数"""
  rex_ymd = '[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}'

  self_str = self.copy().astype(str)

  res = self_str.str.contains(rex_ymd, regex = True)

  res[self.isna()] = na_default

  return res

# @pf.register_series_method
def is_ymd_like(self, na_default = True):
  """与えられた文字列が ymd 形式っぽい日付かどうかを判定する関数"""
  rex_ymd_like = '[Script=Han]{0,2}[0-9]{1,4}(?:年|-)[0-9]{1,2}(?:月|-)[0-9]{1,2}(?:日)?'

  self_str = self.copy().astype(str)

  res = self_str.str.contains(rex_ymd_like, regex = True)

  res[self.isna()] = na_default

  return res




# @pf.register_series_method
def is_number(self, na_default = True):
  """文字列が数字であるかどうかを判定する関数"""
  rex_phone = '[0-9]{0,4}(?: |-)[0-9]{0,4}(?: |-)[0-9]{0,4}'
  rex_exponent = r'[0-9]+[Ee][+-][0-9]+'

  self_str = self.copy().astype(str)

  res = (
    self_str.str.contains('[0-9]+', regex = True) & 
    ~ self_str.str.contains(rex_phone, regex = True) &
    ~ self_str.str.contains('[\u3041-\u309F]+', regex = True) & # ひらがな
    ~ self_str.str.contains('[\u30A1-\u30FF]+', regex = True) & # カタカナ
    ~ self_str.str.contains('[\uFF61-\uFF9F]+', regex = True) & # 半角カタカナ
    ~ self_str.str.contains('[\u4E00-\u9FFF]+', regex = True) & # 漢字
    ~ self_str.str.contains('[A-z]+', regex = True) &
    ~ is_ymd_like(self_str)
  )

  exponent = self_str.str.contains(rex_exponent, regex = True)
  res[exponent] = True

  res[self.isna()] = na_default

  return res


# ## set missing values in pd.Series



def set_n_miss(x, n = 10, method = 'random', random_state = None, na_value = pd.NA):
  future_warn_pandas_implementation()
  method = build.arg_match(method, ['random', 'first', 'last'], arg_name = 'method')
  build.assert_count(n, arg_name = 'n', upper = len(x))

  x = x.copy()
  n_miss = x.isna().sum()
  non_miss = x.dropna().index.to_series()

  if(method == 'random'):
    index_to_na = non_miss.sample(n = n - n_miss, random_state = random_state)
  elif(method == 'first'):
    index_to_na = non_miss.head(n - n_miss)
  elif(method == 'last'):
    index_to_na = non_miss.tail(n - n_miss)

  x[index_to_na] = na_value

  return x




def set_prop_miss(x, prop = 0.1, method = 'random', random_state = None, na_value = pd.NA):
  future_warn_pandas_implementation()
  method = build.arg_match(method, ['random', 'first', 'last'], arg_name = 'method')
  build.assert_numeric(prop, arg_name = 'prop', lower = 0, upper = 1)

  x = x.copy()
  prop_miss = x.isna().mean()
  non_miss = x.dropna().index.to_series()

  if(method == 'random'):
    index_to_na = non_miss.sample(frac = prop - prop_miss, random_state = random_state)
  elif(method == 'first'):
    n = round(len(non_miss) * (prop - prop_miss))
    index_to_na = non_miss.head(n)
  elif(method == 'last'):
    n = round(len(non_miss) * (prop - prop_miss))
    index_to_na = non_miss.tail(n)

  x[index_to_na] = na_value

  return x


# - `eda.set_n_miss()`： `pd.Series` の欠損数が `n` 個になるように欠測値を追加します。
# - `eda.set_prop_miss()`： `pd.Series` の欠損率が約 `prop` になるように欠測値を追加します。
# 
# 引数
# 
# - method：**str**</br>
#     - `'random'`（初期設定）：`x` のランダムな位置を、欠損値に変換します。
#     - `'first'`：`x` 冒頭を欠損値に変換します。
#     - `'last'`：`x` 末尾を欠損値に変換します。
# 
# ```python
# from py4stats import eda_tools as eda
# from palmerpenguins import load_penguins
# penguins = load_penguins() # サンプルデータの読み込み
# s = penguins['bill_depth_mm'].copy()
# 
# print(s.isna().sum()) # 当初の欠測値の数
# #> 2
# 
# print(eda.set_n_miss(s, n = 20).isna().sum())
# #> 20
# 
# print(eda.set_n_miss(s, method = 'first'))
# #> 0       NaN
# #> 1       NaN
# #> 2       NaN
# #> 3       NaN
# #> 4       NaN
# #>        ...
# #> 339    19.8
# #> 340    18.1
# #> 341    18.2
# #> 342    19.0
# #> 343    18.7
# #> Name: bill_depth_mm, Length: 344, dtype: float64
# 
# print(eda.set_n_miss(s, method = 'last'))
# #> 0      18.7
# #> 1      17.4
# #> 2      18.0
# #> 3       NaN
# #> 4      19.3
# #>        ...
# #> 339     NaN
# #> 340     NaN
# #> 341     NaN
# #> 342     NaN
# #> 343     NaN
# #> Name: bill_depth_mm, Length: 344, dtype: float64
# 
# print(eda.set_prop_miss(s, prop = 0.2).isna().mean())
# #> 0.19767441860465115
# ```

# # 簡易なデータバリデーションツール



# @pf.register_dataframe_method
def check_that(
    data: pd.DataFrame,
    rule_dict: Union[Mapping[str, str], pd.Series],
    **kwargs: Any,
) -> pd.DataFrame:
  """Evaluate validation rules and summarize pass/fail counts.

  Each rule is an expression evaluated by `DataFrame.eval(...)` and must return
  a boolean array-like of length equal to the number of rows, or a scalar bool.

  Args:
      data (pandas.DataFrame):
          Data to validate.
      rule_dict (dict or pandas.Series):
          Mapping from rule name to expression string (for `DataFrame.eval`).
          If a Series is given, it is converted to dict.
      **kwargs:
          Keyword arguments forwarded to `DataFrame.eval(...)` (e.g., engine, parser).

  Returns:
      pandas.DataFrame:
          Summary table indexed by rule name with columns:
          - item: number of evaluated items (rows)
          - passes: number of True
          - fails: number of False
          - coutna: number of NA (after handling NA rows)
          - expression: the rule expression string

  Raises:
      AssertionError:
          If rule expressions are not strings, or the evaluation result is not boolean.
  """
  future_warn_pandas_implementation()
  if(isinstance(rule_dict, pd.Series)): rule_dict = rule_dict.to_dict()

  [build.assert_character(x, arg_name = 'rule_dict') for x in rule_dict.values()]

  result_list = []
  for i, name in enumerate(rule_dict):
    condition = data.eval(rule_dict[name], **kwargs)
    condition = pd.Series(condition)
    assert build.is_logical(condition),\
    f"Result of rule(s) must be of type 'bool'. But result of '{name}' is '{condition.dtype}'."

    if len(condition) == len(data):
      in_exper = [s in rule_dict[name] for s in data.columns]
      any_na = data.loc[:, in_exper].isna().any(axis = 'columns')
      condition = condition.astype('boolean')
      condition = condition.where(~any_na)

    res_df = pd.DataFrame({
        'item':len(condition),
        'passes':condition.sum(skipna = True),
        'fails':(~condition).sum(skipna = True),
        'coutna':condition.isna().sum(),
        'expression':rule_dict[name]
        }, index = [name])

    result_list.append(res_df)

  result_df = pd.concat(result_list)
  result_df.index.name = 'name'

  return result_df




# @pf.register_dataframe_method
def check_viorate(
    data: pd.DataFrame,
    rule_dict: Union[Mapping[str, str], pd.Series],
    **kwargs: Any,
) -> pd.DataFrame:
  """Return row-wise rule violation indicators for each rule.

  Args:
      data (pandas.DataFrame):
          Data to validate.
      rule_dict (dict or pandas.Series):
          Mapping from rule name to expression string (for `DataFrame.eval`).
          If a Series is given, it is converted to dict.
      **kwargs:
          Keyword arguments forwarded to `DataFrame.eval(...)`.

  Returns:
      pandas.DataFrame:
          Boolean DataFrame with one column per rule indicating violations
          (True means violation). Additional columns:
          - any: True if any rule is violated in the row
          - all: True if all rules are violated in the row

  Raises:
      AssertionError:
          If rule expressions are not strings, or the evaluation result is not boolean.
  """
  future_warn_pandas_implementation()
  if(isinstance(rule_dict, pd.Series)): rule_dict = rule_dict.to_dict()
  [build.assert_character(x, arg_name = 'rule_dict') for x in rule_dict.values()]

  df_viorate = pd.DataFrame()
  for i, name in enumerate(rule_dict):
    condition = data.eval(rule_dict[name], **kwargs)
    assert build.is_logical(condition),\
    f"Result of rule(s) must be of type 'bool'. But result of '{name}' is '{condition.dtype}'."

    df_viorate[name] = ~condition

  df_viorate['any'] = df_viorate.any(axis = 'columns')
  df_viorate['all'] = df_viorate.all(axis = 'columns')

  return df_viorate


# ### helper function for pandas `DataFrame.eval()`



def implies_exper(P, Q):
  return f"{Q} | ~({P})"

# @pf.register_dataframe_method
@singledispatch
def is_complet(self: pd.DataFrame) -> pd.Series:
  return self.notna().all(axis = 'columns')

@is_complet.register(pd.Series)
def _(*arg: pd.Series) -> pd.Series:
  return pd.concat(arg, axis = 'columns').notna().all(axis = 'columns')




def Sum(*arg): return pd.concat(arg, axis = 'columns').sum(axis = 'columns')
def Mean(*arg): return pd.concat(arg, axis = 'columns').mean(axis = 'columns')
def Max(*arg): return pd.concat(arg, axis = 'columns').max(axis = 'columns')
def Min(*arg): return pd.concat(arg, axis = 'columns').min(axis = 'columns')
def Median(*arg): return pd.concat(arg, axis = 'columns').median(axis = 'columns')

