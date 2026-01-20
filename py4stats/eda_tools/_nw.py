#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import annotations


# # `eda_tools`：データセットを要約する関数群 `narwhals` ライブラリを使った実装

# `eda_tools._nw` モジュールに実装された主要な関数の依存関係
# 
# ``` python
# ## 1. 基本診断（欠測・ユニーク数など）
# 
# diagnose()                                   # 各列の dtype / 欠測 / ユニーク数を要約
# ├─ build.assert_logical()                    # 引数チェック
# ├─ get_dtypes()                              # 列ごとの dtype 抽出（backend非依存）
# └─ narwhals 集計処理
# 
# get_dtypes()                                 # DataFrame の dtype を Series として取得
# 
# ## 2. 欠測値の可視化
# 
# plot_miss_var()                              # 変数別欠測率・欠測数の横棒グラフ
# ├─ build.arg_match()                         # values 引数の選択検証
# ├─ build.assert_logical()
# ├─ diagnose()                                # 欠測統計の計算
# └─ matplotlib barh 描画
# 
# ## 3. DataFrame 間の比較（構造・統計）
# 
# ### 列構造（dtype）比較
# 
# compare_df_cols()                            # 複数DF間で列 dtype を比較
# ├─ is_FrameT()                               # DataFrame 互換判定
# ├─ build.arg_match()                         # return_match 指定
# ├─ build.assert_logical()
# ├─ get_dtypes()
# └─ pandas.concat / nunique
# 
# is_FrameT()                                  # narwhals.from_native 可否判定
# 
# ### 統計量の近接性比較
# 
# compare_df_stats()                           # 平均などの統計量の近さで比較
# ├─ is_FrameT()
# ├─ build.arg_match()
# ├─ _compute_stats()                          # 数値列の統計量計算
# ├─ itertools.combinations()                  # DF ペア生成
# └─ numpy.isclose()
# 
# _compute_stats()                             # 数値列のみを選び stats を計算
# 
# ### レコード単位比較
# 
# compare_df_record()                          # 行×列レベルで df1 と df2 を比較
# ├─ build.assert_logical()
# ├─ build.arg_match()                         # columns = 'all' / 'common'
# ├─ build.oxford_comma_and()                  # エラーメッセージ整形
# ├─ numpy.isclose()                           # 数値列比較
# └─ 等値比較（非数値）
# 
# ## 4. グループ間比較（平均・中央値）
# 
# compare_group_means()                        # グループ平均と差分指標
# ├─ build.assert_character()
# ├─ remove_constant()                         # 定数列除去
# ├─ narwhals.mean / var
# └─ 差分指標（norm / abs / rel）
# 
# compare_group_median()                       # グループ中央値と差分
# ├─ build.assert_character()
# ├─ remove_constant()
# └─ abs / rel 差分計算
# 
# plot_mean_diff()                             # 平均差のステムプロット
# ├─ build.arg_match()
# ├─ compare_group_means()
# └─ matplotlib stem
# 
# plot_median_diff()                           # 中央値差のステムプロット
# ├─ build.arg_match()
# ├─ compare_group_median()
# └─ matplotlib stem
# 
# 
# ## 5. クロス集計・度数表
# 
# crosstab()                                   # backend非依存クロス集計
# ├─ build.assert_logical()
# ├─ build.arg_match()                         # normalize
# ├─ narwhals.pivot()
# └─ 周辺合計・正規化処理
# 
# freq_table()                                 # 度数・割合・累積度数表
# ├─ build.arg_match()                         # sort_by
# ├─ build.assert_logical()
# ├─ FutureWarning 処理（sort）
# └─ group_by + 集計
# 
# tabyl()                                      # janitor::tabyl 風クロス集計
# ├─ build.assert_*()
# ├─ crosstab()                                # 分割表
# ├─ build.style_number()
# ├─ build.style_percent()
# └─ 文字列結合（"count (xx%)"）
# 
# ## 6. カテゴリー変数の診断
# 
# diagnose_category()                          # カテゴリー変数専用サマリ
# ├─ build.assert_logical()
# ├─ is_dummy()                                # ダミー変数検出
# ├─ freq_table()                              # モード算出
# ├─ std_entropy()                             # 標準化エントロピー
# └─ narwhals 集計
# 
# is_dummy()                                   # ダミー変数判定（汎用）
# ├─ is_dummy_series()                         # Series 用
# └─ is_dummy_data_frame()                     # DataFrame 用
# 
# entropy()                                    # Shannon エントロピー
# └─ scipy.stats.entropy
# 
# std_entropy()                                # 正規化エントロピー
# └─ entropy()
# 
# ## 7. 欠測・定数・不要列の除去
# 
# missing_percent()                            # 行・列ごとの欠測率
# 
# remove_empty()                               # 空白行・列の除去
# ├─ missing_percent()
# ├─ build.assert_*()
# └─ 条件付きフィルタ
# 
# remove_constant()                            # 定数列の除去
# ├─ build.assert_logical()
# └─ n_unique 判定
# 
# filtering_out()                              # 列名・行名のパターン除外
# ├─ build.arg_match()                         # axis
# ├─ build.assert_character()
# └─ pandas.str.contains 系
# 
# ## 8. 数値変換・スケーリング
# 
# weighted_mean()                              # 重み付き平均
# ├─ build.assert_numeric()
# └─ sum(x*w)/sum(w)
# 
# scale()                                     # Z-score 標準化
# ├─ build.assert_count()
# ├─ build.assert_numeric()
# └─ mean / std
# 
# min_max()                                   # Min-Max 正規化
# ├─ build.assert_numeric()
# └─ (x-min)/(max-min)
# 
# ## 9. パレート図
# Pareto_plot()                               # パレート図（頻度 or 集計）
# ├─ build.assert_*()
# ├─ freq_table() / make_rank_table()
# ├─ make_Pareto_plot()
# └─ matplotlib 描画
# 
# make_rank_table()                           # 集計→順位→累積比率
# 
# make_Pareto_plot()                          # パレート図描画専用
# 
# ## 10. カテゴリー積み上げ棒グラフ
# 
# plot_category()                             # カテゴリー積み上げ棒
# ├─ build.arg_match()
# ├─ make_table_to_plot()                     # 描画用テーブル作成
# ├─ make_categ_barh()                        # 描画処理
# └─ seaborn / matplotlib
# 
# make_table_to_plot()                        # freq_table を縦結合
# ├─ freq_table()
# ├─ relocate()
# └─ 累積比率計算
# 
# make_categ_barh()                           # 積み上げ棒描画
# └─ matplotlib + seaborn
# 
# ## 11. 区間推定（QI / CI）
# 
# mean_qi()                                  # 平均 + 分位区間
# ├─ build.assert_numeric()
# ├─ build.arg_match()
# ├─ mean_qi_data_frame()
# └─ mean_qi_series()
# 
# median_qi()                                # 中央値 + 分位区間
# ├─ build.assert_numeric()
# ├─ build.arg_match()
# ├─ median_qi_data_frame()
# └─ median_qi_series()
# 
# mean_ci()                                  # 平均 + t信頼区間
# ├─ build.assert_numeric()
# ├─ mean_ci_data_frame()
# └─ mean_ci_series()
# 
# ## 12. ルールベース検証
# 
# check_that()                               # ルール評価の要約
# └─ check_that_pandas()
# 
# check_viorate()                            # 行ごとの違反判定
# └─ check_viorate_pandas()
# 
# is_complete()                            # 欠損のない行を判定
# 
# Sum(), Mean(), Max(), Min(), Median()  # 行方向の合計・平均・中央値などを計算
# └─ pd.concat(...).sum() など
# 
# ## 13. 列操作ユーティリティ
# 
# relocate()                                 # 列順の再配置
# ├─ arrange_colnames()
# ├─ build.assert_character()
# └─ narwhals.select()
# 
# arrange_colnames()                         # 列名リストの並び替えロジック
# 
# ```

# In[ ]:


from py4stats import building_block as build # py4stats のプログラミングを補助する関数群
# from py4stats import eda_tools as eda        # 基本統計量やデータの要約など
import matplotlib.pyplot as plt
import functools
from functools import singledispatch
import pandas_flavor as pf

import pandas as pd
import numpy as np
import scipy as sp
import itertools
import narwhals
import narwhals as nw
import narwhals.selectors as ncs
from narwhals.typing import FrameT, IntoFrameT, SeriesT, IntoSeriesT

import pandas_flavor as pf

import warnings


# In[ ]:


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

# In[ ]:


def get_dtypes(data: IntoFrameT) -> pd.Series:
    data_nw = nw.from_native(data)
    implement = data_nw.implementation
    
    if isinstance(data, nw.DataFrame):
        list_dtypes = list(data.schema.values())
    else:
        match str(implement):
            case 'pandas':
                list_dtypes = data.dtypes.to_list()
            case 'polars':
                list_dtypes = list(data.schema.values())
            case 'pyarrow':
                list_dtypes = data.schema.types
            case _: # どのケースにも一致しない場合
                list_dtypes = list(data_nw.schema.values())
    
    list_dtypes = pd.Series(
        [str(v) for v in list_dtypes],
        index = data_nw.columns
        )

    return list_dtypes


# In[ ]:


@pf.register_dataframe_method
def diagnose(data: IntoFrameT, to_native: bool = True) -> IntoFrameT:
    """Summarize each column of a DataFrame for quick EDA.

    This method computes basic diagnostics for each column:
    - dtype
    - missing_count / missing_percent
    - unique_count / unique_rate

    Args:
        data (IntoFrameT):
            Input DataFrame. Any DataFrame type supported by narwhals
            (e.g. pandas, polars, pyarrow) can be used.
        to_native (bool, optional):
            If True, convert the result to the native DataFrame type of the
            selected backend. If False, return a narwhals DataFrame.
            Defaults to True.
    
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
            If `data` is not a pandas.DataFrame.
    """
    build.assert_logical(to_native, arg_name = 'to_native')
    data_nw = nw.from_native(data)

    n = data_nw.shape[0]
    list_dtypes = get_dtypes(data)

    result = nw.from_dict({
        'columns':data_nw.columns,
        'dtype':list_dtypes,
        'missing_count':data_nw.null_count().row(0),
        'unique_count':[s.n_unique() for s in data_nw.iter_columns()]
    }, backend = data_nw.implementation)\
    .with_columns(
        (100 * nw.col('missing_count') / n).alias('missing_percent'),
        (100 * nw.col('unique_count') / n).alias('unique_rate')
    )\
    .select('columns', 'dtype', 'missing_count', 'missing_percent', 'unique_count', 'unique_rate')
    
    if to_native: return result.to_native()
    return result


# In[ ]:


@pf.register_dataframe_method
def plot_miss_var(
        data: IntoFrameT,
        values: Literal['missing_percent', 'missing_count'] = 'missing_percent', 
        sort: bool = True, 
        miss_only: bool = False, 
        top_n: Optional[int] = None,
        fontsize: int = 12,
        ax: Optional[Axes] = None,
        color: str = '#478FCE',
        **kwargs: Any
        ) -> None:
    """Plot missing-value diagnostics for each variable in a DataFrame.

    This function visualizes the amount of missing data for each column
    as a horizontal bar chart. It supports multiple DataFrame backends
    via narwhals and relies on ``diagnose()`` to compute missing-value
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
        top_n (int or None):
            If specified, plot only top-n variables.
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
    values = build.arg_match(
        values, arg_name = 'values',
        values = ['missing_percent', 'missing_count']
    )
    build.assert_logical(sort, arg_name = 'sort')
    build.assert_logical(miss_only, arg_name = 'miss_only')
    
    diagnose_tab = diagnose(data, to_native = False)
    
    if miss_only: diagnose_tab = diagnose_tab.filter(nw.col('missing_percent') > 0)
    if top_n is not None:
        build.assert_count(top_n, lower = 1, arg_name = 'top_n')
        diagnose_tab = diagnose_tab.top_k(top_n, by = values)
    if sort: diagnose_tab = diagnose_tab.sort(values)
    
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

# In[ ]:


def is_FrameT(obj: object) -> bool:
    try:
        _ = nw.from_native(obj)
        return True
    except Exception:
        return False


# In[ ]:


ReturnMatch = Literal["all", "match", "mismatch"]

def compare_df_cols(
    df_list: List[IntoFrameT],
    return_match: Literal["all", "match", "mismatch"] = 'all',
    df_name = None,
    dropna:bool = False,
) -> pd.DataFrame:
    """Compare dtypes of columns with the same names across multiple DataFrames.

    Args:
        df_list (list[IntoFrameT]):
            List of input DataFrame(s). Any DataFrame-like object supported by narwhals
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table) can be used.
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

    return_match = build.arg_match(
        return_match, values = ['all', 'match', 'mismatch'],
        arg_name = 'return_match'
        )
    build.assert_logical(dropna, arg_name = 'dropna')
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


# ### 平均値などの統計値の近接性で比較するバージョン

# In[ ]:


# import narwhals.selectors as ncs
# import itertools
# StatsLike = Union[str, Callable[..., Any]]

def compare_df_stats(
    df_list: List[IntoFrameT],
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
        df_list (list[IntoFrameT]):
            Input DataFrame(s). Any DataFrame-like object supported by narwhals
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table) can be used.
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

    return_match = build.arg_match(
        return_match, arg_name = 'return_match',
        values = ['all', 'match', 'mismatch']
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


# In[ ]:


# レコード毎の近接性（数値の場合）または一致性（数値以外）で評価する関数
def compare_df_record(
    df1: IntoFrameT,
    df2: IntoFrameT,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    sikipna: bool = True,
    columns: Literal['common', 'all'] = 'all',
    to_native: bool = True
) -> pd.DataFrame:
    """Compare two DataFrames record-wise (element-wise).

    Each element is compared row by row:
    - Numeric columns are compared using `numpy.isclose`.
    - Non-numeric columns are compared using equality (`==`).

    The set of columns used for comparison is controlled by `columns`.

    Args:
        df1 (IntoFrameT):
            First DataFrame-like object. Any DataFrame-like object supported by narwhals
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table) can be used.
        df2 (IntoFrameT):
            Second DataFrame-like object.
        rtol (float, optional):
            Relative tolerance for numeric comparison via `numpy.isclose`.
        atol (float, optional):
            Absolute tolerance for numeric comparison via `numpy.isclose`.
        skipna (bool, optional):
            If True(default), Exclude NA/null values when computing the result.
        columns (Literal["common", "all"], optional):
            Policy that determines which columns are compared.

            - `"all"` (default):
                Require `df1` and `df2` to have exactly the same set of columns.
                If columns differ, an error is raised.
            - `"common"`:
                Compare only the intersection of columns present in both
                `df1` and `df2`. Columns that exist in only one DataFrame
                are ignored.
        to_native (bool, optional):
            If True, return the result as a native DataFrame class of 'df1'.
            If False, return a `narwhals.DataFrame`.

    Returns:
        pandas.DataFrame or narwhals.DataFrame:
            A DataFrame of boolean values indicating, for each element,
            whether the values in `df1` and `df2` match (or are close for
            numeric columns). Column order follows `df1.columns`
            (restricted by `columns` if `"common"` is used).

    Raises:
        AssertionError:
            If `df1` and `df2` do not have the same number of rows.
        ValueError:
            If `columns="all"` and `df1` and `df2` do not have identical
            column sets.

    Examples:
        >>> compare_df_record(df1, df2, columns="all")
        >>> compare_df_record(df1, df2, rtol=1e-4, columns="common")
    """
    df1 = nw.from_native(df1)
    df2 = nw.from_native(df2)
    all1 = df1.columns
    all2 = df2.columns
    
    build.assert_logical(sikipna, arg_name = 'sikipna')
    if sikipna:
        df1 = df1.drop_nulls(all1)
        df2 = df2.drop_nulls(all2)

    # 引数のアサーション ----------------------------------------------------------------------------------
    build.assert_logical(to_native, arg_name = 'to_native')

    columns = build.arg_match(
        columns,  arg_name = 'columns',
        values = ['common', 'all']
        )
    
    assert df1.shape[0] == df2.shape[0], (
        "df1 and df2 must have the same number of rows "
        f"(got len(df1)={df1.shape[0]} and len(df2)={df2.shape[0]})."
    )
    if columns == 'all':
        only_in_df1 = sorted(set(all1) - set(all2))
        only_in_df2 = sorted(set(all2) - set(all1))

        if only_in_df1 or only_in_df2:
            messages = [
                "Column sets of df1 and df2 do not match while columns='all' is specified."
            ]
            if only_in_df1:
                messages.append(
                    f"Columns only in df1: {build.oxford_comma_and(only_in_df1)}."
                )
            if only_in_df2:
                messages.append(
                    f"Columns only in df2: {build.oxford_comma_and(only_in_df2)}."
                )
            messages.append(
                "Use columns='common' to compare only shared columns."
            )
            raise ValueError("\n".join(messages))
    # --------------------------------------------------------------------------------------------------
    
    numeric1 = df1.select(ncs.numeric()).columns
    nonnum1 = df1.select(~ncs.numeric()).columns
    numeric2 = df2.select(ncs.numeric()).columns
    nonnum2 = df2.select(~ncs.numeric()).columns

    # df1と df2 の列名の共通部分を抽出します。
    all_columns = [item for item in all1 if item in all2]
    numeric_col = set(numeric1) & set(numeric2)
    nonnum_col = set(nonnum1) & set(nonnum2)

    res_number_col = [
        np.isclose(
            df1[v], df2[v], rtol = rtol, atol = atol
        ) for v in numeric_col
    ]
    if res_number_col:
        res_number_col_df = nw.from_numpy(
            np.vstack(res_number_col).T, 
            backend = df1.implementation
        )
        res_number_col_df = res_number_col_df.rename(
            dict(zip(res_number_col_df.columns, numeric_col))
        )
    else:
        res_number_col_df = None

    res_nonnum_col = [(df1[v] == df2[v]).to_frame() for v in nonnum_col]

    if res_nonnum_col:
        res_nonnum_col_df = nw.concat(res_nonnum_col, how = 'horizontal')
    else:
        res_nonnum_col_df = None
    
    res_list = [res_number_col_df, res_nonnum_col_df]
    res_list = list(filter(None, res_list))

    result = nw.concat(
        res_list, 
        how = 'horizontal'
        )\
        .select(all_columns)


    if to_native: return result.to_native()
    return result


# ## グループ別平均（中央値）の比較

# In[ ]:


def enframe(
    data: Any,
    row_id:int = 0,
    name: str = 'name',
    value: str = 'value',
    backend: Optional[str] = None,
    names: list[str] = None,
    to_native: bool = True,
    **keywarg: Any
) -> IntoFrameT:
    # 引数のアサーション =======================================================
    build.assert_count(row_id, arg_name = 'row_id')
    build.assert_character(name, arg_name = 'name')
    build.assert_character(value, arg_name = 'value')
    build.assert_logical(to_native, arg_name = 'to_native')
    # =======================================================================

    try:
        data = nw.from_native(data, allow_series = True)
    finally:
        args_dict = locals()
        args_dict.pop('data')
        return enframe_default(data, **args_dict)

@singledispatch
def enframe_default(data, **keywarg: Any) -> None:
    raise NotImplementedError(f'enframe mtethod for object {type(data)} is not implemented.')


# In[ ]:


@enframe_default.register(nw.DataFrame)
def enframe_table(
    data: IntoFrameT,
    row_id:int = 0,
    name: str = 'name',
    value: str = 'value',
    names: Union[list[str]] = None,
    backend: Optional[str] = None,
    to_native: bool = True,
    **keywarg: Any
) -> IntoFrameT:
    
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


# In[ ]:


@enframe_default.register(nw.series.Series)
@enframe_default.register(list)
@enframe_default.register(tuple)
def enframe_series(
    data: Union[nw.Series, list, tuple],
    name: str = 'name',
    value: str = 'value',
    names: Union[list[str]] = None,
    backend: Optional[str] = None,
    to_native: bool = True,
    **keywarg: Any
) -> IntoFrameT:
    
    if backend is None:
        if hasattr(data, 'implementation'):
            backend = data.implementation
        else:
            backend = 'pandas'
    if names is None:
        if hasattr(data, 'implementation') and (data.implementation.is_pandas()):
            names = data.to_pandas().index.to_list()
        else:
            names = range(build.length(data))

    result = nw.from_dict({
        name: names,
        value: list(data)
    }, backend = backend)
    
    if to_native: return result.to_native()
    return result


# In[ ]:


@enframe_default.register(dict)
def enframe_dict(
    data: dict,
    name: str = 'name',
    value: str = 'value',
    names: Union[list[str]] = None,
    backend: Optional[str] = None,
    to_native: bool = True,
    **keywarg: Any
) -> IntoFrameT:
    
    if backend is None: backend = 'pandas'
    if names is None:   names = data.keys()

    result = nw.from_dict({
        name: names,
        value: data.values()
    }, backend = backend)
    
    if to_native: return result.to_native()
    return result


# In[ ]:


def _row_to_2col_df(
        data_nw,
        row_id = 0,
        values_to = 'values',
        names_to = 'variable'
):
    result = nw.from_dict({
        names_to:  data_nw.columns,
        values_to: data_nw.row(row_id)
    }, backend = data_nw.implementation)
    
    return result


# In[ ]:


def compare_group_means(
    group1: IntoFrameT,
    group2: IntoFrameT,
    group_names: Sequence[str] = ('group1', 'group2'),
    columns: Literal['common', 'all'] = 'all',
    to_native: bool = True
) -> pd.DataFrame:
    """
    Compare variable-wise means between two groups.

    This function computes the mean of each numeric column for two input
    data frames and combines the results into a single table. It also derives
    simple difference metrics based on the group-wise means.

    The function supports multiple DataFrame backends via narwhals
    (e.g., pandas, polars, pyarrow).

    Args:
        group1 (IntoFrameT):
            Data for the first group. Any DataFrame-like object supported by
            narwhals (e.g., ``pandas.DataFrame``, ``polars.DataFrame``,
            ``pyarrow.Table``) can be used.
        group2 (IntoFrameT):
            Data for the second group.
        group_names (Sequence[str]):
            Names used for the output columns corresponding to the two groups.
            Must be a sequence of length 2.
        columns (Literal['common', 'all']):
            Specifies which variables to include when combining results from
            the two groups.

            - ``"common"``:
              Only variables that appear in *both* groups are included.
            - ``"all"``:
              All variables appearing in either group are included. In this
              case, difference metrics may contain missing values (e.g.,
              ``NaN`` or ``None``) for variables that are present in only one
              group.
        to_native (bool, optional):
            If True, return the result as a native DataFrame class of 'group1'.
            If False, return a `narwhals.DataFrame`.

    Returns:
        IntoFrameT:
            A DataFrame with one row per variable and the following columns:

            - ``{group_names[0]}``: mean value in the first group
            - ``{group_names[1]}``: mean value in the second group
            - norm_diff: normalized difference using pooled variance
            - ``abs_diff``: absolute difference between group means
            - ``rel_diff``: relative difference defined as
              ``2 * (A - B) / (A + B)``

    Notes:
        - Only numeric columns are used when computing means.
        - Constant columns are removed from each group before comparison.
        - When ``columns="all"``, variables that exist in only one group are
          retained, and derived difference metrics may be missing.
        - The function performs a join operation internally to align variables
          across the two groups.
    """
    # 引数のアサーション ==============================================
    build.assert_character(group_names, arg_name = 'group_names', len_arg = 2)
    columns = build.arg_match(
        columns,  arg_name = 'columns',
        values = ['common', 'all']
        )
    if columns == "all": how_join = 'full'
    else: how_join = 'inner'

    # ==============================================================
    group1 = nw.from_native(group1)
    group2 = nw.from_native(group2)
    group1 = remove_constant(group1, to_native = False)
    group2 = remove_constant(group2, to_native = False)

    # 平均値の計算 =========================================================
    stats_df1 = enframe(
        group1.select(ncs.numeric().mean()), 
        name = 'variable', value = group_names[0],
        row_id = 0, to_native = False
        )
    
    stats_df2 = enframe(
        group2.select(ncs.numeric().mean()), 
        name = 'variable', value = group_names[1],
        row_id = 0, to_native = False
        )
    # return stats_df1, stats_df2
    # 分散の計算 =========================================================
    var_df1 = enframe(
        group1.select(ncs.numeric().var()), 
        row_id = 0, name = 'variable', value = 's2A',
        to_native = False
        )
    
    var_df2 = enframe(
        group2.select(ncs.numeric().var()),
        row_id = 0, name = 'variable', value = 's2B',
        to_native = False
        )
    nA = group1.shape[0]
    nB = group2.shape[0]

    var_df = var_df1.join(
        var_df2,
        on = 'variable',
        how = 'inner'
    )\
        .with_columns(
        _s2_pooled = (
            (nA - 1) * nw.col('s2A') + (nB - 1) * nw.col('s2B')) / 
            (nA + nB - 2)
    ).select('variable', '_s2_pooled')
    # データフレームの結合 ===============================================================
    mean_sd2 = stats_df1\
        .join(stats_df2, on = 'variable', how = how_join)
    
    if columns == "all":
        mean_sd2 = mean_sd2.with_columns(
            nw.when(nw.col("variable").is_null()).then("variable_right")\
            .otherwise("variable").alias('variable')
        )
    
    mean_sd2 = mean_sd2.join(var_df, on = 'variable', how = 'left')
    # 差分統計量の計算 ==================================================================
    result = mean_sd2\
        .with_columns(
            norm_diff = (nw.col(group_names[0]) - nw.col(group_names[1]))
                        / nw.col('_s2_pooled').sqrt(),
            abs_diff = (nw.col(group_names[0]) - nw.col(group_names[1])).abs(),
            rel_diff = 2 * (nw.col(group_names[0]) - nw.col(group_names[1])) /
                        (nw.col(group_names[0]) + nw.col(group_names[1]))
    )\
    .select(
        'variable', nw.col(group_names), 'norm_diff', 'abs_diff', 'rel_diff'
    )
    # ================================================================================
    if to_native: return result.to_native()
    return result



# In[ ]:


def compare_group_median(
    group1: IntoFrameT,
    group2: IntoFrameT,
    group_names: Sequence[str] = ('group1', 'group2'),
    columns: Literal['common', 'all'] = 'all',
    to_native: bool = True
) -> IntoFrameT:
    """
    Compare variable-wise medians between two groups.

    This function computes the median of each numeric column for two input
    data frames and combines the results into a single table. It also derives
    simple difference metrics based on the group-wise medians.

    The function supports multiple DataFrame backends via narwhals
    (e.g., pandas, polars, pyarrow).

    Args:
        group1 (IntoFrameT):
            Data for the first group. Any DataFrame-like object supported by
            narwhals (e.g., ``pandas.DataFrame``, ``polars.DataFrame``,
            ``pyarrow.Table``) can be used.
        group2 (IntoFrameT):
            Data for the second group.
        group_names (Sequence[str]):
            Names used for the output columns corresponding to the two groups.
            Must be a sequence of length 2.
        columns (Literal['common', 'all']):
            Specifies which variables to include when combining results from
            the two groups.

            - ``"common"``:
              Only variables that appear in *both* groups are included.
            - ``"all"``:
              All variables appearing in either group are included. In this
              case, difference metrics may contain missing values (e.g.,
              ``NaN`` or ``None``) for variables that are present in only one
              group.
        to_native (bool, optional):
            If True, return the result as a native DataFrame class of 'group1'.
            If False, return a `narwhals.DataFrame`.
            
    Returns:
        IntoFrameT:
            A DataFrame with one row per variable and the following columns:

            - ``{group_names[0]}``: median value in the first group
            - ``{group_names[1]}``: median value in the second group
            - ``abs_diff``: absolute difference between group medians
            - ``rel_diff``: relative difference defined as
              ``2 * (A - B) / (A + B)``

    Notes:
        - Only numeric columns are used when computing medians.
        - Constant columns are removed from each group before comparison.
        - When ``columns="all"``, variables that exist in only one group are
          retained, and derived difference metrics may be missing.
        - The function performs a join operation internally to align variables
          across the two groups.
    """
    # 引数のアサーション ==============================================
    build.assert_character(group_names, arg_name = 'group_names', len_arg = 2)
    columns = build.arg_match(
        columns,  arg_name = 'columns',
        values = ['common', 'all']
        )
    if columns == "all": how_join = 'full'
    else: how_join = 'inner'

    # ==============================================================
    group1 = nw.from_native(group1)
    group2 = nw.from_native(group2)
    group1 = remove_constant(group1, to_native = False)
    group2 = remove_constant(group2, to_native = False)

    # 中央値の計算 =========================================================
    stats_df1 = enframe(
        group1.select(ncs.numeric().mean()), 
        name = 'variable', value = group_names[0],
        row_id = 0, to_native = False
        )
    
    stats_df2 = enframe(
        group2.select(ncs.numeric().mean()), 
        name = 'variable', value = group_names[1],
        row_id = 0, to_native = False
        )
    # データフレームの結合 ===============================================================
    stats_df = stats_df1\
        .join(stats_df2, on = 'variable', how = how_join)
    
    if columns == "all":
        stats_df = stats_df.with_columns(
            nw.when(nw.col("variable").is_null()).then("variable_right")\
            .otherwise("variable").alias('variable')
        )
    
    # 差分統計量の計算 ==================================================================
    result = stats_df\
        .with_columns(
            abs_diff = (nw.col(group_names[0]) - nw.col(group_names[1])).abs(),
            rel_diff = 2 * (nw.col(group_names[0]) - nw.col(group_names[1])) /
                        (nw.col(group_names[0]) + nw.col(group_names[1]))
    )\
    .select(
        'variable', nw.col(group_names), 'abs_diff', 'rel_diff'
    )
    # ================================================================================
    if to_native: return result.to_native()
    return result



# In[ ]:


def plot_mean_diff(
    group1: IntoFrameT,
    group2: IntoFrameT,
    stats_diff: Literal["norm_diff", "abs_diff", "rel_diff"] = "norm_diff",
    ax: Optional[Axes] = None,
) -> None:
    """Plot group mean differences for each variable as a stem plot.

    Args:
        group1 (IntoFrameT):
            Data for group 1.Any DataFrame-like object supported by narwhals
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table) can be used.
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
    stats_diff = build.arg_match(
        stats_diff, arg_name = 'stats_diff',
        values = ['norm_diff', 'abs_diff', 'rel_diff']
        )
    group_means = compare_group_means(
        group1, group2, 
        columns = 'common',
        to_native = False
        ).to_pandas().set_index('variable')

    if ax is None:
        fig, ax = plt.subplots()

    ax.stem(group_means[stats_diff], orientation = 'horizontal', basefmt = 'C7--');

    ax.set_yticks(range(len(group_means.index)), group_means.index)

    ax.invert_yaxis();


# In[ ]:


def plot_median_diff(
    group1: IntoFrameT,
    group2: IntoFrameT,
    stats_diff: Literal["abs_diff", "rel_diff"] = "rel_diff",
    ax: Optional[Axes] = None,
) -> None:
    """Plot group median differences for each variable as a stem plot.

    Args:
        group1 (IntoFrameT):
            Data for group 1.Any DataFrame-like object supported by narwhals
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table) can be used.
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
    stats_diff = build.arg_match(
        stats_diff, values = ['abs_diff', 'rel_diff'],
        arg_name = 'stats_diff'
        )

    group_median = compare_group_median(
        group1, group2, 
        columns = 'common',
        to_native = False
        ).to_pandas().set_index('variable')

    if ax is None:
        fig, ax = plt.subplots()

    ax.stem(group_median[stats_diff], orientation = 'horizontal', basefmt = 'C7--')
    ax.set_yticks(range(len(group_median.index)), group_median.index)
    ax.invert_yaxis();


# ## クロス集計表ほか

# In[ ]:


def crosstab(
        data: IntoFrameT, 
        index: str, columns: str, 
        margins: bool = False,
        margins_name: str = 'All',
        sort_index: bool = True,
        normalize: Union[bool, Literal['all', 'index', 'columns']] = False,
        to_native: bool = True,
        dropna: bool = False,
        **kwargs: Any
        ) -> IntoFrameT:
    # 引数のアサーション -------------------------------------
    build.assert_logical(to_native, arg_name = 'to_native')
    build.assert_logical(margins, arg_name = 'margins')
    build.assert_logical(dropna, arg_name = 'dropna')
    
    if not isinstance(normalize, bool):
        normalize = build.arg_match(
            normalize,
            values = ['all', 'index', 'columns'],
            arg_name = 'normalize'
        )
    # -----------------------------------------------------
    data_nw = nw.from_native(data)
    impl = data_nw.implementation

    if impl.is_pyarrow():
        # 2026年1月11日時点で バックエンドが pyarrow の場合 
        # data_nw.pivot() メソッドが使用できないことへの対処です。
        data_nw = nw.from_native(data_nw.to_polars())

    if dropna: data_nw = data_nw.drop_nulls([index, columns])

    # return data_nw
    result = (
        data_nw.with_columns(
            nw.col(index, columns).cast(nw.String), # nan によるエラー回避のため
            __n=nw.lit(1), # 1を立てる
        )         
          .pivot(
              on = columns,
              index = index,
              values = '__n',
              aggregate_function = 'sum',             # セル内の1を合計＝件数
              sort_columns = True,
          )
          # 欠損セルを0にしたい場合（バックエンド依存はあるが一般にOK）
          .with_columns(ncs.numeric().fill_null(0))
    )
    
    if sort_index:
        result = result.sort(index)
    # return result
    if margins:
        result = result.with_columns(nw.sum_horizontal(ncs.numeric()).alias(margins_name))

        # row_sums の作成と結合
        row_sums = result.select(ncs.numeric().sum())\
            .with_columns(nw.lit(margins_name).alias(index))

        if normalize == 'columns':
            numeric_vars = result.select(ncs.numeric()).columns
            for v in numeric_vars:
                result = result.with_columns(
                    nw.col(v) / row_sums.item(0, v)
                )
        else:
            result = nw.concat([
                    result, row_sums.select(index, ncs.numeric())
                    ], 
                    how = 'vertical'
                    )
        
        if normalize == 'index':
            result = result.with_columns(
                ncs.numeric() / nw.col(margins_name)
                ).drop(margins_name, strict = False)
        
        if normalize == 'all':
            total_val = result[margins_name].tail(1).item(0)
            result = result.with_columns(ncs.numeric()/total_val)
        
        if not normalize:
            result = result.with_columns(ncs.numeric().cast(nw.Int64))
    
    
    if impl.is_pyarrow():
        result = nw.from_native(result.to_arrow())

    if not to_native: return result
    
    if result.implementation.is_pandas_like():
        result = nw.to_native(result).set_index(index)
    else:
        result = result.to_native()
    return result


# In[ ]:


@pf.register_dataframe_method
def freq_table(
    data: IntoFrameT,
    subset: Union[str, Sequence[str]],
    sort_by: Literal['frequency', 'values'] = 'frequency',
    descending: bool = False,
    dropna: bool = False,
    to_native: bool = True,
    *,
    sort: Optional[bool] = None
) -> IntoFrameT:
    """Compute frequency table for one or multiple columns.

    Args:
        data (IntoFrameT):
            Input DataFrame. Any DataFrame-like object supported by narwhals
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table) can be used.
        subset (str or list[str]):
            Column(s) to count by. Passed to `DataFrame.value_counts(subset=...)`.
        sort_by (Literal['frequency', 'values']):
            Sorting rule for the output table.
            - 'frequency': sort by frequency. (default)
            - 'values': sort by the category values.
        descending (bool):
            Sort order. Defaults to False.
        dropna (bool):
            Whether to drop NaN from counts.
        to_native (bool, optional):
            If True, convert the result to the native DataFrame type of the
            selected backend. If False, return a narwhals DataFrame.
            Defaults to True.
        sort (bool):
            **Deprecated.** Use `sort_by` instead.
            This argument is kept for backward compatibility. If provided, a
            `FutureWarning` is emitted. When both `sort` and `sort_by`
            are provided, `sort_by` takes precedence and `sort` is ignored.
            Defaults to None.

    Returns:
        IntoFrameT:
            Frequency table with columns:
            - freq: counts
            - perc: proportions
            - cumfreq: cumulative counts
            - cumperc: cumulative proportions
    """
    # 引数のアサーション ========================================
    sort_by = build.arg_match(
        sort_by, arg_name = 'sort_by',
        values = ['frequency', 'values']
        )
    
    build.assert_logical(descending, arg_name = 'descending')
    build.assert_logical(dropna, arg_name = 'dropna')
    build.assert_logical(to_native, arg_name = 'to_native')
    # =========================================================

    # sort の非推奨処理 ========================================
    if sort is not None:
        build.assert_logical(sort, arg_name = 'sort')
        warnings.warn(
            "`sort` argument of `freq_table()` is deprecated and will be removed in a future release of Py4Stats. "
            "Please use the `sort_by` argument instead (e.g., sort_by='frequency' or sort_by='values').",
            category = FutureWarning,
            stacklevel = 2,
        )
    # =========================================================
    
    data_nw = nw.from_native(data)
    
    if dropna:
        data_nw = data_nw.drop_nulls(subset)

    result = data_nw.with_columns(__n=nw.lit(1))\
        .group_by(nw.col(subset))\
        .agg(nw.col('__n').sum().alias('freq'))
  
    # sort 引数を使った処理将来廃止予定 ============================
    if sort is not None:
        if sort:
            result = result.sort('freq', descending = descending)
        else:
            result = result.sort(subset, descending = descending)
    # =========================================================
    
    match sort_by:
        case 'frequency':
            result = result.sort('freq', descending = descending)
        case 'values':
            result = result.sort(subset, descending = descending)

    result = result.with_columns(
            (nw.col('freq') / nw.col('freq').sum()).alias('perc'),
            nw.col('freq').cum_sum().alias('cumfreq')
        )\
        .with_columns(
            (nw.col('cumfreq') / nw.col('freq').sum()).alias('cumperc'),
        )
    
    if to_native: 
        if result.implementation.is_pandas_like():
            return result.to_native().reset_index(drop=True)
        return result.to_native()
    return result


# In[ ]:


# @pf.register_dataframe_method

# def tabyl(
#     data: IntoFrameT,
#     index: str,
#     columns: str,
#     margins: bool = True,
#     margins_name: str = 'All',
#     normalize: Union[bool, Literal["index", "columns", "all"]] = "index",
#     dropna: bool = False,
#     digits: int = 1,
#     # to_native: bool = True,
#     **kwargs: Any
# ) -> pd.DataFrame:
#     """Create a crosstab with counts and (optionally) percentages in parentheses.

#     This function produces a table similar to `janitor::tabyl()` (R), where the
#     main cell is a count and percentages can be appended like: `count (xx.x%)`.

#     Args:
#         data (IntoFrameT):
#             Input DataFrame. Any DataFrame-like object supported by narwhals
#             (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table) can be used.
#         index (str):
#             Column name used for row categories.
#         columns (str):
#             Column name used for column categories.
#         margins (bool):
#             Add margins (totals) if True.
#         margins_name (str):
#             Name of the margin row/column.
#         normalize (bool or {'index','columns','all'}):
#             If False, return counts only.
#             Otherwise, compute percentages normalized by the specified axis.
#         dropna (bool):
#             Whether to drop NaN from counts.
#         digits (int):
#             Number of decimal places for percentages.

#     Returns:
#         pandas.DataFrame:
#             Crosstab table. If `normalize` is not False, cells contain strings like
#             `"count (xx.x%)"`. Otherwise counts (as strings after formatting).
#     """
#     # 引数のアサーション ==============================================
#     build.assert_logical(margins, arg_name = 'margins')
#     build.assert_character(margins_name, arg_name = 'margins_name')
#     build.assert_logical(dropna, arg_name = 'dropna')
#     build.assert_count(digits, arg_name = 'digits')
#     # build.assert_logical(to_native, arg_name = 'to_native')
#     # ==============================================================
    
#     data_nw = nw.from_native(data)

#     if(not isinstance(normalize, bool)):
#       normalize = build.arg_match(
#           normalize, arg_name = 'normalize',
#           values = ['index', 'columns', 'all']
#           )
    
#     # index または columns に bool 値が指定されていると後続処理でエラーが生じるので、
#     # 文字列型に cast します。
#     data_nw = data_nw[[index, columns]].with_columns(
#        ncs.boolean().cast(nw.String)
#     )

#     # 度数クロス集計表（最終的な表では左側の数字）
#     args_dict = locals()
#     args_dict.pop('normalize')
#     args_dict.pop('data')
#     # args_dict.pop('to_native')
    
#     c_tab1 = crosstab(
#         data = data_nw,
#         normalize = False,
#         to_native = False,
#         **args_dict
#        ).to_pandas().set_index(index)
    
#     c_tab1 = c_tab1.apply(build.style_number, digits = 0)
#     # return c_tab1

#     if(normalize != False):
#         # 回答率クロス集計表（最終的な表では括弧内の数字）
#         c_tab2 = crosstab(
#             data = data_nw, 
#             normalize = normalize, 
#             to_native = False,
#             **args_dict
#            ).to_pandas().set_index(index)

#         # 2つめのクロス集計表の回答率をdigitsで指定した桁数のパーセントに換算し、文字列化します。
#         c_tab2 = c_tab2.apply(build.style_percent, digits = digits)
        
#         # return c_tab2
#         col = c_tab2.columns
#         idx = c_tab2.index
#         c_tab1 = c_tab1.astype('str')
#         # 1つめのクロス集計表も文字列化して、↑で計算したパーセントに丸括弧と%記号を追加したものを文字列として結合します。
#         c_tab1.loc[idx, col] = c_tab1.astype('str').loc[idx, col] + ' (' + c_tab2 + ')'
    
#     return c_tab1

#     # if to_native and data_nw.implementation.is_pandas():
#     #    return c_tab1
    
#     # c_tab1 = c_tab1.reset_index()
#     # dict_list = [c_tab1.loc[i, :].to_dict() for i in c_tab1.index]
#     # result = nw.from_dicts(dict_list, backend = data_nw.implementation)
    
#     # if to_native: return result.to_native()
#     # return result


# <!-- ## `diagnose_category()`：カテゴリー変数専用の要約関数 -->

# In[ ]:


@pf.register_dataframe_method
def tabyl(
    data: IntoFrameT,
    index: str,
    columns: str,
    margins: bool = True,
    margins_name: str = 'All',
    normalize: Union[bool, Literal["index", "columns", "all"]] = "index",
    dropna: bool = False,
    digits: int = 1,
    to_native: bool = True,
    **kwargs: Any
) -> IntoFrameT:
    """Create a crosstab with counts and (optionally) percentages in parentheses.

    This function produces a table similar to `janitor::tabyl()` (R), where the
    main cell is a count and percentages can be appended like: `count (xx.x%)`.

    Args:
        data (IntoFrameT):
            Input DataFrame. Any DataFrame-like object supported by narwhals
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table) can be used.
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
            Whether to drop NaN from counts.
        digits (int):
            Number of decimal places for percentages.
        to_native (bool, optional):
            If True, convert the result to the native DataFrame type of the
            selected backend. If False, return a narwhals DataFrame.
            Defaults to True.

    Returns:
        IntoFrameT:
            Crosstab table. If `normalize` is not False, cells contain strings like
            `"count (xx.x%)"`. Otherwise counts (as strings after formatting).
    """
    # 引数のアサーション ==============================================
    build.assert_logical(margins, arg_name = 'margins')
    build.assert_character(margins_name, arg_name = 'margins_name')
    build.assert_logical(dropna, arg_name = 'dropna')
    build.assert_count(digits, arg_name = 'digits')
    # build.assert_logical(to_native, arg_name = 'to_native')
    # ==============================================================
    
    data_nw = nw.from_native(data)

    if(not isinstance(normalize, bool)):
      normalize = build.arg_match(
          normalize, arg_name = 'normalize',
          values = ['index', 'columns', 'all']
          )
    
    # index または columns に bool 値が指定されていると後続処理でエラーが生じるので、
    # 文字列型に cast します。
    data_nw = data_nw[[index, columns]].with_columns(
       ncs.boolean().cast(nw.String)
    )

    # 度数クロス集計表（最終的な表では左側の数字）
    args_dict = locals()
    args_dict.pop('normalize')
    args_dict.pop('data')
    args_dict.pop('to_native')
    
    c_tab1 = crosstab(
        data = data_nw,
        normalize = False,
        to_native = False,
        **args_dict
       ).to_pandas().set_index(index)
    
    c_tab1 = c_tab1.apply(build.style_number, digits = 0) # .astype('str')

    if(normalize != False):
        # 回答率クロス集計表（最終的な表では括弧内の数字）
        c_tab2 = crosstab(
            data = data_nw, 
            normalize = normalize, 
            to_native = False,
            **args_dict
           ).to_pandas().set_index(index)

        # 2つめのクロス集計表の回答率をdigitsで指定した桁数のパーセントに換算し、文字列化します。
        c_tab2 = c_tab2.apply(build.style_percent, digits = digits)
        
        col = c_tab2.columns
        idx = c_tab2.index
        # 1つめのクロス集計表も文字列化して、↑で計算したパーセントに丸括弧と%記号を追加したものを文字列として結合します。
        c_tab1.loc[idx, col] = c_tab1.loc[idx, col] + ' (' + c_tab2 + ')'
    
    if to_native and data_nw.implementation.is_pandas():
       return c_tab1
    
    c_tab1 = c_tab1.reset_index()
    # バックエンドの書き換え ==============================================
    # これは推奨される実装ではない、安易に使い回さないこと。
    dict_list = [c_tab1.loc[i, :].to_dict() for i in c_tab1.index]
    result = nw.from_dicts(dict_list, backend = data_nw.implementation)
    #==================================================================
    if to_native: return result.to_native()
    return result


# In[ ]:


@pf.register_dataframe_method
@pf.register_series_method
@singledispatch
def is_dummy(
    data: Union[IntoFrameT, IntoSeriesT],
    cording: Sequence[Any] = (0, 1),
    dropna: bool = True,
    to_pd_series: bool = False,
    **kwargs
) -> Union[bool, IntoSeriesT, IntoFrameT]:
    """
    Check whether values consist only of dummy codes.

    This function tests whether the input data contains *only* the specified
    dummy codes. The behavior and return type depend on whether the input is
    Series-like or DataFrame-like.

    The function supports multiple backends via narwhals and is implemented
    using ``singledispatch``.

    Args:
        data:
            Input data to check. Can be a Series-like or DataFrame-like object
            supported by narwhals (e.g., ``pandas.Series``,
            ``pandas.DataFrame``, ``polars.Series``, ``polars.DataFrame``,
            ``pyarrow.Table``).
        cording:
            Sequence of allowed dummy codes. The input is considered valid if
            its unique values exactly match this set.
            Defaults to ``(0, 1)``.
        dropna (bool):
            Whether to drop NaN from data before value check.
        to_pd_series:
            Controls the return type when ``data`` is DataFrame-like.
            If True, returns a ``pandas.Series`` indexed by column names.
            If False, returns a Python list of boolean values.
        **kwargs:
            Additional keyword arguments (reserved for future extensions).

    Returns:
        bool or Series-like or list:
            - If ``data`` is Series-like, returns a single boolean indicating
              whether the Series consists only of the specified dummy codes.
            - If ``data`` is DataFrame-like and ``to_pd_series`` is False,
              returns a list of boolean values, one per column.
            - If ``data`` is DataFrame-like and ``to_pd_series`` is True,
              returns a ``pandas.Series`` with column names as the index.

    Notes:
        - A Series is considered dummy-coded if the set of its values is
          exactly equal to the set specified by ``cording``.
        - The check is purely set-based; value frequency and ordering are
          not considered.
        - Missing values are not explicitly handled and will affect the
          result according to the underlying data representation.
    """
    build.assert_logical(to_pd_series, arg_name = 'to_pd_series')
    build.assert_logical(dropna, arg_name = 'dropna')
    
    data_nw = nw.from_native(data, allow_series = True)
    return is_dummy(data_nw, cording, dropna, to_pd_series)


# In[ ]:


@is_dummy.register(nw.Series)
def is_dummy_series(
    data: IntoSeriesT,
    cording: Sequence[Any] = (0, 1),
    dropna: bool = True,
    to_pd_series: bool = False,
    **kwargs
) -> bool:
    if dropna: data = data.drop_nulls()
    return set(data) == set(cording)


# In[ ]:


@is_dummy.register(list)
def is_dummy_list(
    data: list,
    cording: Sequence[Any] = (0, 1),
    dropna: bool = True,
    to_pd_series: bool = False,
    **kwargs
) -> bool:
    return set(data) == set(cording)


# In[ ]:


@is_dummy.register(nw.DataFrame)
def is_dummy_data_frame(
        data: IntoFrameT, 
        cording: Sequence[Any] = (0, 1),
        dropna: bool = True,
        to_pd_series: bool = False,
        **kwargs
        ) -> Union[IntoFrameT, pd.Series]:
    
    data_nw = nw.from_native(data)
    
    result = data_nw.select(
        nw.all().map_batches(
            lambda x: is_dummy_series(x, cording), 
            return_dtype = nw.Boolean,
            returns_scalar = True
            )
    )
    
    if to_pd_series: 
        return result.to_pandas().loc[0, :]
    return list(result.row(0))


# In[ ]:


import scipy as sp

def entropy(x: IntoSeriesT, base: float = 2.0, dropna: bool = False) -> float:
    build.assert_numeric(base, arg_name = 'base', lower = 0, inclusive = 'right')
    build.assert_logical(dropna, arg_name = 'dropna')

    x_nw = nw.from_native(x, series_only = True)

    if dropna: x_nw = x_nw.drop_nulls()

    prop = x_nw.value_counts(normalize = True, sort = False)['proportion']
    result = sp.stats.entropy(pk = prop,  base = base, axis = 0)
    return result

def std_entropy(x: IntoSeriesT, dropna: bool = False) -> float:
    build.assert_logical(dropna, arg_name = 'dropna')
    
    x_nw = nw.from_native(x, series_only = True)
    
    if dropna: x_nw = x_nw.drop_nulls()

    K = x_nw.n_unique()
    result = entropy(x_nw, base = K) if K > 1 else 0.0
    
    return result


# In[ ]:


@pf.register_dataframe_method
def diagnose_category(data: IntoFrameT, to_native: bool = True) -> IntoFrameT:
    """Summarize categorical variables in a DataFrame.

    This function summarizes columns that represent categorical information,
    including categorical/string/boolean columns and 0/1 dummy columns
    (integer-valued columns restricted to {0, 1}). Dummy columns are cast to
    string before summarization.

    The summary includes missing percentage, number/percentage of unique
    values, mode and its frequency/share, and standardized entropy.

    The implementation is backend-agnostic via narwhals.

    Args:
        data (IntoFrameT):
            Input DataFrame. Any DataFrame type supported by narwhals can be
            used (e.g., pandas, polars, pyarrow).
        to_native (bool, optional):
            If True, convert the result to the native DataFrame type of the
            selected backend. If False, return a narwhals DataFrame.
            Defaults to True.

    Returns:
        IntoFrameT:
            A summary table with one row per selected variable and the
            following columns:

            - variables: variable (column) name
            - count: non-missing count
            - miss_pct: missing percentage
            - unique: number of unique values
            - unique_pct: percentage of unique values (unique / N * 100)
            - mode: most frequent value
            - mode_freq: frequency of the mode value
            - mode_pct: percentage of the mode value (mode_freq / N * 100)
            - std_entropy: standardized entropy in [0, 1]

    Raises:
        TypeError:
            If ``data`` is not a DataFrame/Series type supported by narwhals.
        ValueError:
            If no columns are selected for summarization (i.e., ``data`` has
            no categorical/string/boolean columns and no 0/1 dummy columns).

    Notes:
        - ``N`` denotes the number of rows of the selected DataFrame.
        - ``miss_pct`` is computed as ``null_count / N * 100``.
        - ``unique_pct`` and ``mode_pct`` use ``N`` in the denominator (not
          the non-missing count).
        - Standardized entropy is computed per column via ``std_entropy`` and
          is expected to return a scalar in the range [0, 1].

    Examples:
        Basic usage:

        >>> summary = diagnose_category(df)
        >>> summary.head()

        Keep narwhals output (do not convert back to native):

        >>> summary_nw = diagnose_category(df, to_native=False)

        Dummy columns are included automatically if they are 0/1-valued
        integer columns:

        >>> df = df.assign(dummy=(df["x"] > 0).astype(int))
        >>> diagnose_category(df)
    """
    build.assert_logical(to_native, arg_name = 'to_native')

    data_nw = nw.from_native(data)
    res_is_dummy = is_dummy(data_nw, to_pd_series = True)
    dummy_col = res_is_dummy[res_is_dummy].index.to_list()
    
    df = (
        data_nw
        .with_columns(nw.col(dummy_col).cast(nw.String))\
        .select(
        ncs.categorical(),
        ncs.by_dtype(nw.String), 
        ncs.boolean(), 
        ))
    N = df.shape[0]


    var_name = df.columns
    
    if not var_name:
        raise ValueError(
            "`data` has no columns to summarize.\n"
            "Expected at least one categorical, string, or boolean column,\n"
            "or a 0/1 dummy column (integer values restricted to {0, 1})."
        )
    # return df, var_name

    result  = nw.from_dict({
        'variables': var_name,
        'count':df.select(nw.all().count()).row(0),
        'miss_pct':df.select(nw.all().null_count() * nw.lit(100 / N)).row(0),
        'unique':df.select(nw.all().n_unique()).row(0),
        'mode':[
            freq_table(df, v, descending = True, to_native = False).row(0)[0] 
            for v in var_name
            ],
        'mode_freq':[
            freq_table(df, v, descending = True, to_native = False).row(0)[1] 
            for v in var_name
            ],
        'std_entropy':[std_entropy(s) for s in df.iter_columns()]
        # 'std_entropy':df.select(
        #     nw.all().map_batches(std_entropy, returns_scalar = True)
        #     ).row(0)
        },
        backend = df.implementation
        )\
        .with_columns(
        unique_pct = 100 * nw.col('unique') / N,
        mode_pct = 100 * nw.col('mode_freq') / N
        )\
        .select(
            nw.col([
                'variables', 'count',  'miss_pct', 
                'unique', 'unique_pct',
                'mode', 'mode_freq', 'mode_pct', 
                'std_entropy'
            ])
            )

    if to_native: return result.to_native()
    return result


# ## その他の補助関数

# In[ ]:


def weighted_mean(x: IntoSeriesT, w: IntoSeriesT, dropna:bool = False) -> float:
  """Compute the weighted mean of a numeric series.

  This function computes the weighted mean of a numeric vector `x`
  using weights `w`. Both inputs are converted internally to a
  narwhals Series to support multiple backends.

  Args:
      x (IntoSeriesT):
          Numeric data for which the weighted mean is computed.
          Any series-like object supported by narwhals can be used
          (e.g., pandas.Series, polars.Series).
      w (IntoSeriesT):
          Numeric weights corresponding to `x`.
          Must have the same length as `x`.
      dropna (bool, optional):
          If `True`, observations where either `x` or `w` is missing
          (NaN) are removed before computing the weighted mean.
          If `False`, missing values are not removed.
          Defaults to `False`.

  Returns:
      float:
          The weighted mean, computed as
          ``sum(x * w) / sum(w)``.

  Raises:
      ValueError:
          If `x` or `w` is not numeric.
  """
  x = nw.from_native(x, series_only = True)
  w = nw.from_native(w, series_only = True)

  if dropna:
    non_nan = ~x.is_nan() & ~w.is_nan()
    x = x.filter(non_nan)
    w = w.filter(non_nan)

  build.assert_numeric(x, arg_name = 'x')
  build.assert_numeric(w, arg_name = 'w')
  
  wmean = (x * w).sum() / w.sum()
  return wmean


# In[ ]:


@singledispatch
def scale(x: Union[IntoSeriesT, pd.DataFrame], ddof: int = 1, to_native: bool = True) -> IntoSeriesT:
    """Standardize a numeric series by Z-score scaling.

    This function standardizes numeric data by subtracting the mean
    and dividing by the standard deviation:

        ``(x - mean(x)) / std(x)``

    For non-pandas inputs, the computation is performed using a
    narwhals Series to ensure backend-agnostic behavior.

    Args:
        x (IntoSeriesT or pandas.DataFrame):
            Numeric data to be standardized. Typically a series-like
            object supported by narwhals (e.g., pandas.Series,
            polars.Series). A pandas DataFrame is also supported via
            a specialized implementation.
        ddof (int, optional):
            Delta degrees of freedom used in the calculation of the
            standard deviation. Defaults to `1`.
        to_native (bool, optional):
            If `True`, return the result in the native type corresponding
            to the input (e.g., pandas.Series or polars.Series).
            If `False`, return a narwhals object.
            Defaults to `True`.

    Returns:
        IntoSeriesT:
            Standardized values with mean 0 and standard deviation 1.

    Raises:
        ValueError:
            If `x` is not numeric.
        ValueError:
            If `ddof` is not a non-negative integer.
    """
    build.assert_count(ddof, arg_name = 'ddof')
    build.assert_logical(to_native, arg_name = 'to_native')
    
    x = nw.from_native(x, series_only = True)
    
    build.assert_numeric(x.drop_nulls(), arg_name = 'x')

    z = (x - x.mean()) / x.std(ddof = ddof)
    if to_native: return z.to_native()
    return z

@scale.register(pd.DataFrame)
def scale_pandas(x: pd.DataFrame, ddof: int = 1, to_native: bool = True) -> IntoSeriesT:
    build.assert_count(ddof, arg_name = 'ddof')
    build.assert_logical(to_native, arg_name = 'to_native')

    z = (x - x.mean()) / x.std(ddof = ddof)

    if to_native: return z
    return nw.from_native(z, allow_series = True)


# In[ ]:


@singledispatch
def min_max(x: Union[IntoSeriesT, pd.DataFrame], to_native: bool = True) -> IntoSeriesT:
    """Normalize a numeric series using min-max scaling.

    This function rescales numeric data to the range [0, 1] using
    min-max normalization:

        ``(x - min(x)) / (max(x) - min(x))``

    For non-pandas inputs, the computation is performed using a
    narwhals Series to ensure backend-agnostic behavior.

    Args:
        x (IntoSeriesT or pandas.DataFrame):
            Numeric data to be normalized. Typically a series-like
            object supported by narwhals (e.g., pandas.Series,
            polars.Series). A pandas DataFrame is also supported via
            a specialized implementation.
        to_native (bool, optional):
            If `True`, return the result in the native type corresponding
            to the input.
            If `False`, return a narwhals object.
            Defaults to `True`.

    Returns:
        IntoSeriesT:
            Min-max normalized values in the range [0, 1].

    Raises:
        ValueError:
            If `x` is not numeric.
    """
    build.assert_logical(to_native, arg_name = 'to_native')

    x = nw.from_native(x, series_only = True)
    
    build.assert_numeric(x.drop_nulls(), arg_name = 'x')

    z = (x - x.min()) / (x.max() - x.min())
    if to_native: return z.to_native()
    return z

@min_max.register(pd.DataFrame)
def min_max_pandas(x: pd.DataFrame, to_native: bool = True) -> IntoSeriesT:
    build.assert_logical(to_native, arg_name = 'to_native')

    z = (x - x.min()) / (x.max() - x.min())

    if to_native: return z
    return nw.from_native(z, allow_series = True)


# ## 完全な空白列 and / or 行の除去

# In[ ]:


def missing_percent(
        data: IntoFrameT,
        axis: str = 'index',
        pct: bool = True
        ):
    data_nw = nw.from_native(data)
    n = data_nw.shape[0]

    if axis == 'index':
        n = data_nw.shape[0]
        miss_count = pd.Series(data_nw.null_count().row(0), index = data_nw.columns)
        miss_pct = (100 ** pct) * miss_count / n
        return miss_pct
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
        
        k = data_nw.shape[1]
        miss_pct = (100 ** pct) * miss_count / k
        return miss_pct


# In[ ]:


@pf.register_dataframe_method
def remove_empty(
    data: IntoFrameT,
    cols: bool = True,
    rows: bool = True,
    cutoff: float = 1.0,
    quiet: bool = True,
    to_native: bool = True,
    **kwargs: Any
) -> IntoFrameT:
    """Remove fully (or mostly) empty columns and/or rows.

    Args:
        data (IntoFrameT):
            Input DataFrame. Any DataFrame-like object supported by narwhals
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table) can be used.
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
        to_native (bool, optional):
            If True, convert the result to the native DataFrame type of the
            selected backend. If False, return a narwhals DataFrame.
            Defaults to True.
    
    Returns:
        pandas.DataFrame:
            DataFrame after removing empty columns/rows.
    """
    # 引数のアサーション ==============================================
    build.assert_logical(cols, arg_name = 'cols')
    build.assert_logical(rows, arg_name = 'rows')
    build.assert_numeric(cutoff, lower = 0, upper = 1)
    build.assert_logical(quiet, arg_name = 'quiet')
    build.assert_logical(to_native, arg_name = 'to_native')
    # ==============================================================
    
    df_shape = data.shape
    data_nw = nw.from_native(data)
    # 空白列の除去 ------------------------------
    if cols :
        empty_col = missing_percent(data, axis = 'index', pct = False) >= cutoff
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
        empty_rows = missing_percent(data, axis = 'columns', pct = False) >= cutoff
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


# In[ ]:


@pf.register_dataframe_method
def remove_constant(
    data: IntoFrameT,
    quiet: bool = True,
    to_native: bool = True,
    dropna = False,
    **kwargs: Any
) -> IntoFrameT:
    """Remove constant columns (columns with only one unique value).

    Args:
        data (IntoFrameT):
            Input DataFrame. Any DataFrame-like object supported by narwhals
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table) can be used.
        quiet (bool):
            If False, print removal summary.
        dropna (bool):
            Passed to `nunique(dropna=...)`. If False, NaN is counted as a value.
        to_native (bool, optional):
            If True, convert the result to the native DataFrame type of the
            selected backend. If False, return a narwhals DataFrame.
            Defaults to True.
    
    Returns:
        pandas.DataFrame:
            DataFrame after removing constant columns.
    """
    # 引数のアサーション ==============================================
    build.assert_logical(quiet, arg_name = 'quiet')
    build.assert_logical(to_native, arg_name = 'to_native')
    # ==============================================================
    data_nw = nw.from_native(data)
    df_shape = data_nw.shape
    col_name = data_nw.columns
    
    # データフレーム(data_nw) の行が定数かどうかを判定
    def foo (col, dropna):
        if dropna: 
            return data_nw[col].drop_nulls().n_unique() 
        else:
            return data_nw[col].n_unique()
    # unique_count = pd.Series(data_nw.select(nw.all().n_unique()).row(0), index = data_nw.columns)
    unique_count = pd.Series([foo(col, dropna) for col in col_name])
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


# In[ ]:


# 列名や行名に特定の文字列を含む列や行を除外する関数
@pf.register_dataframe_method
def filtering_out(
    data: IntoFrameT,
    contains: Optional[str] = None,
    starts_with: Optional[str] = None,
    ends_with: Optional[str] = None,
    axis: Union[int, str] = 'columns',
    to_native: bool = True,
) -> IntoFrameT:
    """Filter out rows/columns whose labels match given string patterns.

    Args:
        data (IntoFrameT):
            Input DataFrame. Any DataFrame-like object supported by narwhals
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table) can be used.
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
        to_native (bool, optional):
            If True, convert the result to the native DataFrame type of the
            selected backend. If False, return a narwhals DataFrame.
            Defaults to True.

    Returns:
        IntoFrameT:
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
    axis = build.arg_match(axis, arg_name = 'axis', values = ['1', 'columns', '0', 'index'])
    build.assert_logical(to_native, arg_name = 'to_native')
    # ==============================================================
    data_nw = nw.from_native(data)
    drop_table = pd.DataFrame()

    if axis in ("0", "index"):
        if not hasattr(data, "index"):
            raise TypeError(
                f"filtering_out(..., axis='{axis}') requires an input that has"
                "an 'index' (row labels), e.g. pandas.DataFrame.\n"
                f"Got: {type(data)}."
            )

    if((axis == '1') | (axis == 'columns')):
        s_columns = pd.Series(data_nw.columns)
        if contains is not None:
            # assert isinstance(contains, str), "'contains' must be a string."
            build.assert_character(contains, arg_name = 'contains')
            drop_table['contains'] = s_columns.str.contains(contains)

        if starts_with is not None:
            # assert isinstance(starts_with, str), "'starts_with' must be a string."
            build.assert_character(starts_with, arg_name = 'starts_with')
            drop_table['starts_with'] = s_columns.str.startswith(starts_with)

        if ends_with is not None:
            # assert isinstance(ends_with, str), "'ends_with' must be a string."
            build.assert_character(ends_with, arg_name = 'ends_with')
            drop_table['ends_with'] = s_columns.str.endswith(ends_with)
        drop_list = s_columns[drop_table.any(axis = 'columns')].to_list()
        data_nw = data_nw.drop(drop_list)
    
    elif hasattr(data, 'index'):
        if contains is not None: 
            build.assert_character(contains, arg_name = 'contains')
            drop_table['contains'] = data.index.to_series().str.contains(contains)

        if starts_with is not None: 
            build.assert_character(starts_with, arg_name = 'starts_with')
            drop_table['starts_with'] = data.index.to_series().str.startswith(starts_with)

        if ends_with is not None:
            build.assert_character(ends_with, arg_name = 'ends_with')
            drop_table['ends_with'] = data.index.to_series().str.endswith(ends_with)

        keep_list = (~drop_table.any(axis = 'columns')).to_list()
        data_nw = data_nw.filter(keep_list)

    if to_native: return data_nw.to_native()
    return data_nw


# # パレート図を作図する関数

# In[ ]:


def Pareto_plot(
    data: IntoFrameT,
    group: str,
    values: Optional[str] = None,
    top_n: Optional[int] = None,
    aggfunc: Callable[..., Any] = np.mean,
    ax: Optional[Axes] = None,
    fontsize: int = 12,
    xlab_rotation: Union[int, float] = 0,
    palette: Sequence[str] = ("#478FCE", "#252525"),
) -> None:
    """Plot a Pareto chart.

    If `values` is None, the chart is built from frequency counts of `group`.
    Otherwise, it aggregates `values` by `group` using `aggfunc`.

    Args:
        data (IntoFrameT):
            Input DataFrame. Any DataFrame-like object supported by narwhals
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table) can be used.
        group (str):
            Grouping column (x-axis categories).
        values (str or None):
            Value column to aggregate. If None, uses counts.
        top_n (int or None):
            If specified, plot only top-N categories.
        aggfunc (callable):
            Aggregation function used to compute a single summary value for each group.
            This argument accepts either a general callable (e.g., ``numpy.mean`` or a
            user-defined function) that takes a one-dimensional array-like object
            containing the values of a group and returns a single scalar, or a function
            from ``narwhals.functions`` (e.g., ``narwhals.mean``, ``narwhals.sum``),
            which will be applied directly within the narwhals
            ``group_by().agg()`` workflow.
            Defaults to ``numpy.mean``.
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
    
    Notes:
       The aggregation function is expected to return a single scalar value for
        each group. If the function returns an array-like object or multiple
        values, the resulting output may be invalid or lead to unexpected
        behavior.
    """
    
    # 引数のアサーション ===================================================================
    build.assert_numeric(fontsize, arg_name = 'fontsize', lower = 0, inclusive = 'right')
    build.assert_numeric(xlab_rotation, arg_name = 'xlab_rotation')
    build.assert_character(palette, arg_name = 'palette')
    # ===================================================================================
    data_nw = nw.from_native(data)
    # 指定された変数でのランクを表すデータフレームを作成
    if values is None:
        shere_rank = freq_table(
            data_nw, group, dropna = True, 
            sort_by = 'frequency',
            descending = True, 
            to_native = False
            )
        cumlative = 'cumfreq'
        # None のままだと `.top_k()` メソッドの使用に問題が生じるため
        values = 'freq' 
    else:
        shere_rank = make_rank_table(
            data_nw.to_pandas(), 
            group, 
            values, 
            aggfunc = aggfunc,
            to_native = False
            )
        cumlative = 'cumshare'
    
    if top_n is not None:
        build.assert_count(top_n, lower = 1, arg_name = 'top_n')
        shere_rank = shere_rank.top_k(k = top_n, by = values)
    
    shere_rank = shere_rank.to_pandas().set_index(group)

    # 作図
    args_dict = locals()
    make_Pareto_plot(**args_dict)


# In[ ]:


def make_rank_table(
    data: pd.DataFrame,
    group: str,
    values: str,
    aggfunc: Callable[..., Any] = np.mean,
    to_native: bool = True,
) -> pd.DataFrame:
    data_nw = nw.from_native(data)

    # 引数のアサーション ===================================================================
    col_names = data_nw.columns
    build.assert_scalar(group, arg_name = 'group')
    build.assert_scalar(values, arg_name = 'values')
    group = build.arg_match(group, values = col_names, arg_name = 'group')
    values = build.arg_match(values, values = col_names, arg_name = 'values')
    # ===================================================================================
    
    if aggfunc.__module__ == 'narwhals.functions':
        stat_table = data_nw.group_by(group)\
                .agg(aggfunc(values))
    else:
        group_value = data_nw[group].unique()

        # stat_values = [
        #     aggfunc(data_nw.filter(nw.col(group) == g)[values].drop_nulls().to_native()) 
        #     for g in group_value
        #     ]
        stat_values = [
            aggfunc(
                data_nw.filter(nw.col(group) == g)[values]
                .drop_nulls().to_native()
                ) 
            for g in group_value
            ]
            
        stat_table = nw.from_dict({
            group:group_value, values:stat_values
            }, backend = data_nw.implementation
            )

    rank_table = stat_table.sort(values, descending = True)\
            .with_columns(share = nw.col(values) / nw.col(values).sum())\
            .with_columns(cumshare = nw.col('share').cum_sum())
    
    if to_native:
        return rank_table.to_native()
    else:
        return rank_table


# In[ ]:


def make_Pareto_plot(
    shere_rank: pd.DataFrame,
    group: str,
    cumlative: str,
    values: Optional[str] = None,
    ax: Optional[Axes] = None,
    fontsize: int = 12,
    xlab_rotation: Union[int, float] = 0,
    palette: Sequence[str] = ("#478FCE", "#252525"),
    **kwargs: Any
):
    # グラフの描画
    if ax is None:
        fig, ax = plt.subplots()

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

# In[ ]:


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


# In[ ]:


@pf.register_dataframe_method
@pf.register_series_method
@singledispatch
def mean_qi(
    data: Union[IntoFrameT, SeriesT],
    width: float = 0.975,
    interpolation: Interpolation = 'midpoint',
    to_native: bool = True
) -> IntoFrameT:
    """Compute mean and quantile interval (QI).

    Args:
        data (IntoFrameT, IntoSeriesT):
            Input data. Any DataFrame-like or Series-like object supported by narwhals
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table) can be used.
        width (float):
            Upper quantile to use (must be in (0, 1)).
            Lower quantile is computed as `1 - width`.
        to_native (bool, optional):
            If True, convert the result to the native DataFrame type of the
            selected backend. If False, return a narwhals DataFrame.
            Defaults to True.
    
    Returns:
        IntoFrameT:
            Table indexed by variable names with columns:
            - mean: mean value
            - lower: quantile at `1 - width`
            - upper: quantile at `width`

    Raises:
        AssertionError:
            If `width` is not in (0, 1).
    """
    # 引数のアサーション =======================================================
    build.assert_numeric(width, lower = 0, upper = 1, inclusive = 'neither')
    build.assert_logical(to_native, arg_name = 'to_native')
    interpolation = build.arg_match(
        interpolation, arg_name = 'interpolation',
        values = interpolation_values
        )
    # =======================================================================
    
    data_nw = nw.from_native(data, allow_series = True)
    return mean_qi(
        data_nw, interpolation = interpolation, 
        width = width, to_native = to_native
        )

@mean_qi.register(nw.DataFrame)
def mean_qi_data_frame(
    data: IntoFrameT,
    width: float = 0.975,
    interpolation: Interpolation = 'midpoint',
    to_native: bool = True
    ) -> pd.DataFrame:
    
    df_numeric = nw.from_native(data).select(ncs.numeric())

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

@mean_qi.register(nw.Series)
def mean_qi_series(
    data: SeriesT,
    width: float = 0.975,
    interpolation: Interpolation = 'midpoint',
    to_native: bool = True
    ):
    
    data_nw = nw.from_native(data, allow_series=True)
    
    result = nw.from_dict({
        'variable': [data_nw.name],
        'mean': [data_nw.mean()],
        'lower': [data_nw.quantile(1 - width, interpolation = interpolation)],
        'upper': [data_nw.quantile(width, interpolation = interpolation)]
    }, backend = data_nw.implementation
    )
    if to_native: return result.to_native()
    return result


# In[ ]:


@pf.register_dataframe_method
@pf.register_series_method
@singledispatch
def median_qi(
    data: Union[IntoFrameT, IntoSeriesT],
    width: float = 0.975,
    interpolation: Interpolation = 'midpoint',
    to_native: bool = True
) -> IntoFrameT:
    """Compute median and quantile interval (QI).

    Args:
        data (IntoFrameT, IntoSeriesT):
            Input data. Any DataFrame-like or Series-like object supported by narwhals
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table) can be used.
        width (float):
            Upper quantile to use (must be in (0, 1)).
            Lower quantile is computed as `1 - width`.
        to_native (bool, optional):
            If True, convert the result to the native DataFrame type of the
            selected backend. If False, return a narwhals DataFrame.
            Defaults to True.

    Returns:
        IntoFrameT:
            Table indexed by variable names with columns:
            - median: median value
            - lower: quantile at `1 - width`
            - upper: quantile at `width`

    Raises:
        AssertionError:
            If `width` is not in (0, 1).
    """
    # 引数のアサーション =======================================================
    build.assert_numeric(width, lower = 0, upper = 1, inclusive = 'neither')
    build.assert_logical(to_native, arg_name = 'to_native')
    interpolation = build.arg_match(
        interpolation, arg_name = 'interpolation',
        values = interpolation_values
        )
    # =======================================================================
    
    data_nw = nw.from_native(data, allow_series = True)
    return median_qi(
        data_nw, interpolation = interpolation, 
        width = width, to_native = to_native
        )

@median_qi.register(nw.DataFrame)
def median_qi_data_frame(
    data: IntoFrameT,
    width: float = 0.975,
    interpolation: Interpolation = 'midpoint',
    to_native: bool = True
) -> IntoFrameT:
    
    df_numeric = nw.from_native(data).select(ncs.numeric())

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

@median_qi.register(nw.Series)
def median_qi_series(
    data: IntoSeriesT,
    width: float = 0.975,
    interpolation: Interpolation = 'midpoint',
    to_native: bool = True
) -> IntoFrameT:
    data_nw = nw.from_native(data, allow_series=True)
    
    result = nw.from_dict({
        'variable': [data_nw.name],
        'median': [data_nw.median()],
        'lower': [data_nw.quantile(1 - width, interpolation = interpolation)],
        'upper': [data_nw.quantile(width, interpolation = interpolation)]
    }, backend = data_nw.implementation
    )
    if to_native: return result.to_native()
    return result


# In[ ]:


from scipy.stats import t
@pf.register_dataframe_method
@pf.register_series_method
@singledispatch
def mean_ci(
    data: Union[IntoFrameT, IntoSeriesT],
    width: float = 0.975,
    to_native: bool = True
) -> IntoFrameT:
    """Compute mean and t-based confidence interval (CI).

    Args:
        data (IntoFrameT, IntoSeriesT):
            Input data. Any DataFrame-like or Series-like object supported by narwhals
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table) can be used.
        width (float):
            Confidence level in (0, 1) (e.g., 0.95).
        to_native (bool, optional):
            If True, convert the result to the native DataFrame type of the
            selected backend. If False, return a narwhals DataFrame.
            Defaults to True.

    Returns:
        IntoFrameT:
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
    # 引数のアサーション =======================================================
    build.assert_numeric(width, lower = 0, upper = 1, inclusive = 'neither')
    build.assert_logical(to_native, arg_name = 'to_native')
    # =======================================================================
    
    data_nw = nw.from_native(data, allow_series = True)

    return mean_ci(
        data_nw, width = width, to_native = to_native
    )

@mean_ci.register(nw.DataFrame)
def mean_ci_data_frame(
    data: IntoFrameT,
    width: float = 0.975,
    to_native: bool = True
) -> IntoFrameT:
    df_numeric = nw.from_native(data).select(ncs.numeric())
    n = len(df_numeric)
    t_alpha = t.isf((1 - width) / 2, df = n - 1)
    x_mean = df_numeric.select(ncs.numeric().mean())\
        .to_numpy()[0, :]
    x_std = df_numeric.select(ncs.numeric().std())\
        .to_numpy()[0, :]
    
    result = nw.from_dict({
        'variable': df_numeric.columns,
        'mean':x_mean,
        'lower':x_mean - t_alpha * x_std / np.sqrt(n),
        'upper':x_mean + t_alpha * x_std / np.sqrt(n),
        }, backend = df_numeric.implementation
        )
    if to_native: return result.to_native()
    return result

@mean_ci.register(nw.Series)
def mean_ci_series(
    data: SeriesT,
    width: float = 0.975,
    to_native: bool = True
) -> IntoFrameT:
    data_nw = nw.from_native(data, allow_series=True)
    n = len(data_nw)
    t_alpha = t.isf((1 - width) / 2, df = n - 1)
    x_mean = data_nw.mean()
    x_std = data_nw.std()
    
    result = nw.from_dict({
        'variable': [data_nw.name],
        'mean':[x_mean],
        'lower':[x_mean - t_alpha * x_std / np.sqrt(n)],
        'upper':[x_mean + t_alpha * x_std / np.sqrt(n)],
    }, backend = data_nw.implementation
    )
    if to_native: return result.to_native()
    return result


# ## 正規表現を文字列関連の論理関数

# In[ ]:


@pf.register_series_method
def is_kanzi(data:IntoSeriesT, na_default:bool = True, to_native: bool = True) -> IntoSeriesT:
    """
    Check whether each element contains Kanji characters.

    This method returns a boolean Series indicating whether each string
    element contains at least one Kanji character (Unicode range U+4E00–U+9FFF).

    Args:
        data:
            Input Series containing string-like values.
        na_default:
            Boolean value to use for missing values (e.g., ``None``, ``NaN``).
       to_native (bool, optional):
            If True, convert the result to the native Series type of the
            selected backend. If False, return a narwhals Series.
            Defaults to True.

    Returns:
        Series of boolean values indicating whether each element contains
        Kanji characters.

    Notes:
        - The check is performed using a regular expression.
        - Missing values are filled with ``na_default`` before returning
          the result.
    """
    build.assert_logical(to_native, arg_name = 'to_native')
    build.assert_logical(na_default, arg_name = 'na_default')

    data_nw = nw.from_native(data, allow_series = True)
    
    result = data_nw.str.contains('[\u4E00-\u9FFF]+').fill_null(na_default)
    
    if to_native: return result.to_native()
    return result



# In[ ]:


@pf.register_series_method
def is_ymd(data:IntoSeriesT, na_default:bool = True, to_native: bool = True) -> IntoSeriesT:
    """
    Check whether each element matches a YYYY-MM-DD date format.

    This method tests whether each string matches the pattern
    ``YYYY-M-D`` or ``YYYY-MM-DD`` using a regular expression.
    No validation of actual calendar dates is performed.

    Args:
        data:
            Input Series containing string-like values.
        na_default:
            Boolean value to use for missing values (e.g., ``None``, ``NaN``).
        to_native (bool, optional):
            If True, convert the result to the native Series type of the
            selected backend. If False, return a narwhals Series.
            Defaults to True.

    Returns:
        Series of boolean values indicating whether each element matches
        the YYYY-MM-DD pattern.

    Notes:
        - This function checks only the string pattern, not date validity.
        - Missing values are filled with ``na_default`` before returning
          the result.
    """
    build.assert_logical(to_native, arg_name = 'to_native')
    build.assert_logical(na_default, arg_name = 'na_default')

    rex_ymd = '[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}'

    data_nw = nw.from_native(data, allow_series = True)
    
    result = data_nw.str.contains(rex_ymd)

    result = result.fill_null(na_default)
    if to_native: return result.to_native()
    return result


# In[ ]:


@pf.register_series_method
def is_ymd_like(data:IntoSeriesT, na_default:bool = True, to_native: bool = True) -> IntoSeriesT:
    """
    Check whether each element resembles a date in year-month-day order.

    This method detects strings that resemble date-like expressions such as
    ``2023-1-5``, ``2023年1月5日``, or similar variants using a regular
    expression.

    Args:
        data:
            Input Series containing string-like values.
        na_default:
            Boolean value to use for missing values.
        to_native (bool, optional):
            If True, convert the result to the native Series type of the
            selected backend. If False, return a narwhals Series.
            Defaults to True.

    Returns:
        Series of boolean values indicating whether each element resembles
        a year-month-day style date.

    Notes:
        - The check is based on a regular expression and does not validate
          whether the date actually exists.
        - Missing values are filled with ``na_default`` before returning
          the result.
    """
    build.assert_logical(to_native, arg_name = 'to_native')
    build.assert_logical(na_default, arg_name = 'na_default')

    rex_ymd_like = '[Script=Han]{0,2}[0-9]{1,4}(?:年|-)[0-9]{1,2}(?:月|-)[0-9]{1,2}(?:日)?'

    data_nw = nw.from_native(data, allow_series = True)
    
    result = data_nw.str.contains(rex_ymd_like)

    result = result.fill_null(na_default)
    if to_native: return result.to_native()
    return result


# In[ ]:


@pf.register_series_method
def is_number(data:IntoSeriesT, na_default:bool = True, to_native: bool = True) -> IntoSeriesT:
    """
    Check whether each element represents a numeric value.

    This method evaluates whether each string element can be interpreted
    as a number, including integers and scientific notation, while excluding
    alphabetic characters, kana, kanji, phone-number-like patterns, and
    other non-numeric forms.

    Args:
        data:
            Input Series containing string-like values.
        na_default:
            Boolean value to use for missing values.
        to_native (bool, optional):
            If True, convert the result to the native Series type of the
            selected backend. If False, return a narwhals Series.
            Defaults to True.

    Returns:
        Series of boolean values indicating whether each element represents
        a numeric value.

    Notes:
        - This function uses multiple regular-expression-based heuristics
          rather than numeric type casting.
        - Scientific notation (e.g., ``1e-3``) is treated as numeric.
        - Missing values are filled with ``na_default`` before returning
          the result.
    """
    build.assert_logical(to_native, arg_name = 'to_native')
    build.assert_logical(na_default, arg_name = 'na_default')
    
    data_nw = nw.from_native(data, allow_series = True)

    rex_dict = {
        'exponent': r'[0-9]+[Ee][+-][0-9]+',
        'numeric':'[0-9]+',
        'phone':'[0-9]{0,4}(?: |-)[0-9]{0,4}(?: |-)[0-9]{0,4}',
        'alpha':'[A-z]+',
        'ひらがな': '[\u3041-\u309F]+',
        'カタカナ':'[\u30A1-\u30FF]+',
        '半角カタカナ':'[\uFF61-\uFF9F]+',
        '漢字':'[\u4E00-\u9FFF]+',
        'ymd_like':'[Script=Han]{0,2}[0-9]{1,4}(?:年|-)[0-9]{1,2}(?:月|-)[0-9]{1,2}(?:日)?'
    }

    result_dict = {
        key:~(data_nw.str.contains(rex_val).fill_null(na_default).cast(nw.Boolean))
        for rex_val, key in zip(rex_dict.values(), rex_dict.keys())
    }

    result_dict['numeric'] = ~result_dict['numeric']
    result_dict['exponent'] = ~result_dict['exponent']
    result_table = nw.from_dict(result_dict)
    selected_col = list(rex_dict.keys())[1:]

    result = result_table\
        .with_columns(
            res1 = nw.all_horizontal(nw.col(selected_col), ignore_nulls = True),
            )\
        .with_columns(
                result = nw.col('res1') | nw.col('exponent')
            )['result']
    result

    if to_native: return result.to_native()
    return result


# ## 簡易なデータバリデーションツール

# In[ ]:


def check_that(
    data: IntoFrameT,
    rule_dict: Union[Mapping[str, str], pd.Series],
    to_native: bool = True,
    **kwargs: Any,
) -> IntoFrameT:
    """Evaluate validation rules and summarize pass/fail counts.

    Each rule is an expression evaluated by `pd.DataFrame.eval(...)` and must return
    a boolean array-like of length equal to the number of rows, or a scalar bool.

    Args:
        data (IntoFrameT):
            Input DataFrame. Any DataFrame-like object supported by narwhals
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table) can be used.
        rule_dict (dict or pandas.Series):
            Mapping from rule name to expression string (for `DataFrame.eval`).
            If a Series is given, it is converted to dict.
        to_native (bool, optional):
            If True, convert the result to the native DataFrame type of the
            selected backend. If False, return a narwhals DataFrame.
            Defaults to True.
        **kwargs:
            Keyword arguments forwarded to `DataFrame.eval(...)` (e.g., engine, parser).

    Returns:
        IntoFrameT:
            Summary with columns:
            - rule: name of rules which taken from key of `rule_dict`
            - item: number of evaluated items.
                    For rules evaluated per record, this corresponds to the number of rows
                    in the input data. For rules evaluated at the dataset level (e.g., rules
                    based on aggregated values), this value is 1.
            - passes: number of records that were evaluated and determined to satisfy the rule.
            - fails: number of records that were evaluated and determined not to satisfy the rule.
            - countna: number of records for which the rule could not be evaluated due to missing values.
                        For record-level rules, if any variable used in the rule contains a missing
                        value for a given record, the result is treated as NA, and that record is
                        counted here.
            - expression: the rule expression string
    
    Raises:
        ValueError:
            If a rule expression does not evaluate to a boolean result.
    """
    # 引数のアサーション ===============================================================
    if hasattr(rule_dict, 'to_dict'): rule_dict = rule_dict.to_dict()
    build.assert_character(rule_dict.values(), arg_name = 'rule_dict')
    build.assert_logical(to_native, arg_name = 'to_native')
    # ===============================================================================
    data_nw = nw.from_native(data)
    data_pd = data_nw.to_pandas()
    col_names = data_nw.columns
    N = data_nw.shape[0]

    result_list = []
    for name, rule in zip(rule_dict.keys(), rule_dict.values()):

        passes = pd.Series(data_pd.eval(rule, **kwargs))
        
        # `rule` 評価結果がブール値ではない場合、エラーを出す。
        if not build.is_logical(passes):
            raise ValueError(
                "Result of rule evaluation must be boolean. "
                f"But the result of rule '{name}' has dtype '{passes.dtype}'. "
                "Each rule must evaluate to a boolean expression."
            )

        # 欠測値の代入処理 ==============================================================
        # passes の長さがデータの行数と等しく rule の計算に使用した変数に欠測値が含まれる場合、 
        # そのレコードの passes は欠測値として扱います。これはそのレコードでは、
        # rule の検証ができなかったものとして扱うためです。
        if build.length(passes) == N:
            use_in_rule = [col for col in col_names if col in rule]
            
            any_na = data_pd.loc[:, use_in_rule].isna().any(axis = 'columns')

            passes = passes.astype('boolean').mask(any_na, pd.NA)
        # =============================================================================
        res_dict = {
                    'rule':name,
                    'item':len(passes),
                    'passes':passes.sum(skipna = True),
                    'fails':(~passes).sum(skipna = True),
                    'countna':passes.isna().sum(),
                    'expression':rule
                    }

        result_list.append(res_dict)
    
    result = nw.from_dicts(result_list, backend = data_nw.implementation)

    if to_native: return result.to_native()
    return result


# In[ ]:


def check_viorate(
    data: IntoFrameT,
    rule_dict: Union[Mapping[str, str], pd.Series],
    to_native: bool = True,
    **kwargs: Any,
) -> IntoFrameT:
    """Return row-wise rule violation indicators for each rule.

    Args:
        data (IntoFrameT):
            Input DataFrame. Any DataFrame-like object supported by narwhals
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table) can be used.
        rule_dict (dict or pandas.Series):
            Mapping from rule name to expression string (for `DataFrame.eval`).
            If a Series is given, it is converted to dict.
        to_native (bool, optional):
            If True, convert the result to the native DataFrame type of the
            selected backend. If False, return a narwhals DataFrame.
            Defaults to True.
        **kwargs:
            Keyword arguments forwarded to `DataFrame.eval(...)`.

    Returns:
        IntoFrameT:
            Boolean DataFrame with one column per rule indicating violations 
            (True means violation) or rule evaluation failed due to a missing value. 
            Additional columns:
            - any: True if any rule is violated or failed to evaluation in the row.
            - all: True if all rules are violated or failed to evaluation in the row.

    Raises:
        ValueError:
            If a rule expression does not evaluate to a boolean result.
    """
    # 引数のアサーション ===============================================================
    if hasattr(rule_dict, 'to_dict'): rule_dict = rule_dict.to_dict()
    build.assert_character(rule_dict.values(), arg_name = 'rule_dict')
    build.assert_logical(to_native, arg_name = 'to_native')
    # ===============================================================================
    data_nw = nw.from_native(data)
    data_pd = data_nw.to_pandas()
    value_impl = data_nw.implementation
    N = data_nw.shape[0]
    col_names = data_nw.columns

    result_dict = dict()

    for name, rule in zip(rule_dict.keys(), rule_dict.values()):
        violation = ~data_pd.eval(rule, **kwargs)

        # `rule` 評価結果がブール値ではない場合、エラーを出す。
        if not build.is_logical(violation):
            raise ValueError(
                "Result of rule evaluation must be boolean. "
                f"But the result of rule '{name}' has dtype '{violation.dtype}'. "
                "Each rule must evaluate to a boolean expression."
            )

        # 恐らく起きないと思いますが、violation が長さ N（データのレコード数）のSeries か、
        # スカラー値でなければ、nw.from_dict() でエラーが生じるので、
        # 当てはまらない場合は補正します。
        if isinstance(violation, pd.Series) and build.length(violation) != N:
            violation = violation.iloc[0]

        if not isinstance(violation, pd.Series) and build.length(violation) == 1:
            violation = pd.Series(N * [violation])
        
        result_dict.update({name: violation})
    
    # any と all 列の追加 =============================================================
    result_pd = pd.DataFrame(result_dict)
    result_dict.update({
        'any': result_pd.any(axis = 'columns'),
        'all': result_pd.all(axis = 'columns')
    })
    # ===============================================================================
    # return result_dict
    result = nw.from_dict(
            result_dict, 
            backend = value_impl
        ).select(
            nw.col(list(rule_dict.keys())),
            nw.col('any', 'all')
            ) 
            # 列の並びを key の並びと一致させるため
    
    # return result
    if to_native: return result.to_native()
    return result


# ### helper function for pandas `DataFrame.eval()`

# In[ ]:


def implies_exper(P, Q):
  return f"{Q} | ~({P})"

@singledispatch
def is_complete(data: pd.DataFrame) -> pd.Series:
  return data.notna().all(axis = 'columns')

@is_complete.register(pd.Series)
def _(*arg: pd.Series) -> pd.Series:
  return pd.concat(arg, axis = 'columns').notna().all(axis = 'columns')


# In[ ]:


def Sum(*arg: List[pd.Series]): 
    return pd.concat(arg, axis = 'columns').sum(axis = 'columns')
def Mean(*arg: List[pd.Series]): 
    return pd.concat(arg, axis = 'columns').mean(axis = 'columns')
def Max(*arg: List[pd.Series]): 
    return pd.concat(arg, axis = 'columns').max(axis = 'columns')
def Min(*arg: List[pd.Series]): 
    return pd.concat(arg, axis = 'columns').min(axis = 'columns')
def Median(*arg: List[pd.Series]): 
    return pd.concat(arg, axis = 'columns').median(axis = 'columns')


# ## set missing values in Series

# In[ ]:


def set_miss(
    x: IntoSeriesT, 
    n: Optional[int] = None,
    prop: Optional[float] = None, 
    method: Literal['random', 'first', 'last'] = 'random', 
    random_state: Optional[int] = None, 
    to_native: bool = True
    ):
  x_nw = nw.from_native(x, series_only = True)

  # 引数のアサーション =======================================================
  if not((n is not None) ^ (prop is not None)):
    raise ValueError("Exactly one of `n` and `prop` must be specified.")
  
  build.assert_logical(to_native, arg_name = 'to_native')
  
  n_miss = x_nw.null_count()
  p_miss = n_miss / x_nw.shape[0]
  
  method = build.arg_match(
    method, arg_name = 'method',
    values = ['random', 'first', 'last']
    )
  build.assert_count(
     n, arg_name = 'n',
     lower = 0, upper = len(x), 
     nullable = True, scalar_only = True
     )
  build.assert_numeric(
     prop, arg_name = 'prop',
     lower = 0, upper = 1, 
     nullable = True, scalar_only = True
     )
  # =======================================================================

  x_np = x_nw.to_numpy()
  if hasattr(x, 'index'):
    idx = x.index.to_series()
  else:
    idx = pd.Series(np.arange(len(x_nw)))
  
  non_miss = idx[
    ~x_nw.is_nan() | ~x_nw.is_null()
  ]

  if n is not None: 
    n_to_miss = np.max([n - n_miss, 0])
    if n_to_miss <=0:
      warnings.warn(
         f"Already contained {n_miss}(>= n) missing value(s) in 'x', "
        "no additional missing values were added.",
        category = UserWarning,
        stacklevel = 2
      )
      return x

  elif prop is not None: 
    n_non_miss = non_miss.shape[0]

    n_to_miss = int(np.max([
      np.ceil(n_non_miss * (prop - p_miss)), 0
      ]))
    
    if prop <= p_miss:
      warnings.warn(
        f"Already contained {p_miss:.3f}(>= prop) missing value(s) in 'x', "
        "no additional missing values were added.",
        category = UserWarning,
        stacklevel = 2
      )
      return x  
  
  match method:
    case 'random':
        index_to_na = non_miss.sample(n = n_to_miss, random_state = random_state)
    case 'first':
        index_to_na = non_miss.head(n_to_miss)
    case 'last':
        index_to_na = non_miss.tail(n_to_miss)

  x_np[index_to_na] = np.nan
  result = nw.Series.from_numpy(
      name = x_nw.name,
      values = x_np,
      backend = x_nw.implementation
  )
  
  if to_native: return result.to_native()
  return result


# # `relocate()`

# In[ ]:


def arrange_colnames(colnames, selected, before = None, after = None):
    unselected = [i for i in colnames if i not in selected]
    if before is None and after is None:
        arranged = selected + unselected
    
    if before is not None:
        idx = unselected.index(before)
        col_pre = unselected[:idx]
        col_behind = unselected[idx:]
        arranged = col_pre + selected + col_behind

    if after is not None:
        idx = unselected.index(after) + 1
        col_pre = unselected[:idx]
        col_behind = unselected[idx:]
        arranged = col_pre + selected + col_behind
    
    return arranged


# In[ ]:


def relocate(
        data: IntoFrameT, 
        *args: Union[str, List[str], narwhals.Expr, narwhals.selectors.Selector], 
        before: Optional[str] = None,
        after: Optional[str] = None,
        to_native: bool = True
        ) -> IntoFrameT:
    """Reorder columns in a DataFrame without dropping any columns.

    This function reorders columns in a DataFrame by relocating selected
    columns to a new position, while keeping all other columns intact.
    Selected columns can be specified by column names, lists of names,
    or narwhals expressions/selectors.

    By default, the selected columns are moved to the front of the DataFrame.
    Alternatively, they can be placed immediately before or after a specified
    reference column.

    Internally, the input is converted to a narwhals DataFrame to support
    multiple backends (e.g., pandas, polars, pyarrow), and the reordered
    DataFrame is returned in its native type by default.

    Args:
        data (IntoFrameT):
            Input DataFrame whose columns are to be reordered.
            Any DataFrame-like object supported by narwhals
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table) can be used.
        *args (Union[str, List[str], narwhals.Expr, narwhals.Selector]):
            Columns to relocate. Each element may be:
            - a column name (`str`)
            - a list of column names
            - a narwhals expression
            - a narwhals selector
            The order of columns specified here is preserved in the output.
        before (Optional[str], optional):
            Name of a column before which the selected columns should be placed.
            Cannot be specified together with `after`. Defaults to `None`.
        after (Optional[str], optional):
            Name of a column after which the selected columns should be placed.
            Cannot be specified together with `before`. Defaults to `None`.
        to_native (bool, optional):
            If `True`, return the result as the native DataFrame type
            corresponding to the input (e.g., pandas or polars).
            If `False`, return a narwhals DataFrame. Defaults to `True`.

    Returns:
        IntoFrameT:
            A DataFrame with the same columns as `data`, reordered according
            to the specified rules.

    Raises:
        ValueError:
            If `*args` contains unsupported types.
        ValueError:
            If both `before` and `after` are specified.
        ValueError:
            If `before` or `after` is not a valid column name.

    Notes:
        - This function only changes the order of columns; no columns are
          added or removed.
        - If neither `before` nor `after` is specified, the selected columns
          are moved to the beginning of the DataFrame.
        - Column order among the selected columns follows the order specified
          in `*args`.

    Examples:
        >>> import py4stats as py4st
        >>> import narwhals.selectors as ncs
        >>> from palmerpenguins import load_penguins
        >>> penguins = load_penguins()

        Move columns to the front:
        >>> py4st.relocate(penguins, "year", "sex")

        Relocate columns using a selector:
        >>> py4st.relocate(penguins, ncs.numeric())

        Place columns before a specific column:
        >>> py4st.relocate(penguins, "year", before="island")
    """
    # 引数のアサーション ======================================
    build.assert_logical(to_native, arg_name = 'to_native')
    
    is_varid = [
        isinstance(v, str) or
        (build.is_character(v) and isinstance(v, list)) or
        isinstance(v, nw.expr.Expr)
        for v in args
        ]

    if not all(is_varid):
        invalids = [v for i, v in enumerate(args) if not is_varid[i]]
        message = "Argument '*args' must be of type 'str', list of 'str', 'narwhals.Expr' or 'narwhals.Selector'\n"\
        + f"            The value(s) of {build.oxford_comma_and(invalids)} cannot be accepted.\n"\
        + "            Examples of valid inputs: 'x', ['x', 'y'], ncs.numeric(), nw.col('x')"
        
        raise ValueError(message)
    
    build.assert_character(before, arg_name = 'before', nullable = True, scalar_only = True)
    build.assert_character(after, arg_name = 'after', nullable = True, scalar_only = True)
    
    if (before is not None) and (after is not None):
        raise ValueError("Please specify either 'before' or 'after'.")
    # ======================================================
    
    data_nw = nw.from_native(data)
    colnames = data_nw.columns
    selected = data_nw.select(args).columns
    arranged = arrange_colnames(colnames, selected, before, after)

    result = data_nw.select(nw.col(arranged))
    
    if to_native: return result.to_native()
    return result


# # カテゴリー変数の積み上げ棒グラフ

# In[ ]:


def make_table_to_plot(
        data: IntoFrameT, 
        sort_by: Literal['values', 'frequency'] = 'values',
        to_native: bool = True
        ) -> None:
    data_nw = nw.from_native(data)
    
    variables = data_nw.columns
    def foo(v):
        res_ft = freq_table(
                data_nw, v, 
                dropna = True, 
                sort_by = sort_by,
                descending = sort_by == 'frequency',
                to_native = False
            ).rename({v:'value'})\
            .with_columns(
                bottom = nw.col('cumperc').shift(n = 1).fill_null(0)
                )
        
        res_ft = res_ft.with_columns(nw.lit(v).alias('variable'))
        
        return res_ft

    table_to_plot = nw.concat(
        [foo(v) for v in variables],
        how = 'vertical'
    )
    table_to_plot = relocate(table_to_plot, nw.col('variable'), to_native = False)
    if to_native: return table_to_plot.to_native()
    return table_to_plot


# In[ ]:


import seaborn as sns
from matplotlib import patches as mpatches
from matplotlib.ticker import FuncFormatter

def make_categ_barh(
        table_to_plot, 
        list_values,
        palette: Optional[Sequence] = None,
        legend_type: Literal['horizontal', 'vertical', 'none'] = 'horizontal',
        show_vline: bool = True,
        ax: Optional[Axes] = None
        ):
    table_to_plot = table_to_plot.to_pandas() # seaborn が Pandas のみ対応しているため
    k_categ = len(list_values)
    ## カラーパレットの生成 ==========================
    if palette is None:
        palette = sns.diverging_palette(
            h_neg = 183, h_pos = 64, 
            s = 70, l = 75, 
            n = k_categ, 
            as_cmap = False
        )
    
    value = list_values[0]
    patch_list = []
    
    if ax is None:
        fig, ax = plt.subplots()
    ## 積み上げ棒グラフの作成 =========================
    for i, value in enumerate(list_values):

        ax.barh(
            y = table_to_plot.query('value == @value')['variable'], 
            width = table_to_plot.query('value == @value')['perc'],
            left = table_to_plot.query('value == @value')['bottom'],
            color = palette[i]
        )
        
        patch = mpatches.Patch(color = palette[i], label = value)
        patch_list.append(patch)
    
    ## 軸ラベルと垂直線の設定 =========================
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{abs(100 * x):.0f}%")
        )
        
    ax.set_xlim(0, 1)
    ax.set_ylabel('')
    ax.set_xlabel('Percentage')
    ax.invert_yaxis()
    
    if show_vline:
        ax.axvline(0.5, color = "gray", linewidth = 1, linestyle = "--")
    
    ## 凡例の設定 ===============================
    if legend_type != 'none':
        if legend_type == 'horizontal':
            arg_dict = {
                'loc': 'upper center',
                'bbox_to_anchor': (0.5, -0.1),
                'ncol':k_categ,
                'reverse': True
            }
        else:
            arg_dict = {
                'loc': 'upper left',
                'bbox_to_anchor': (1, 1),
                'ncol': 1
            }
        plt.legend(handles = patch_list, frameon = False, **arg_dict);


# In[ ]:


@pf.register_dataframe_method
def plot_category(
    data: IntoFrameT,
    sort_by: Literal['values', 'frequency'] = 'values',
    palette: Optional[sns.palettes._ColorPalette] = None,
    legend_type: Literal["horizontal", "vertical", "none"] = "horizontal",
    show_vline: bool = True,
    ax: Optional[Axes] = None,
) -> None:
    """Plot 100% stacked horizontal bar charts for categorical variables.

    This function summarizes multiple categorical variables and visualizes
    their response distributions as 100% stacked horizontal bar charts
    (one bar per variable). It is suitable for Likert-style survey items and
    other categorical response data where all variables share a common
    coding scheme.

    Internally, the function aggregates the input data into a plotting table
    using `make_table_to_plot()`, then renders the stacked bars via
    `make_categ_barh()` with matplotlib.

    Args:
        data (IntoFrameT):
            Input DataFrame containing categorical variables (one column per item).
            Any DataFrame type supported by narwhals can be used
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table).
            All columns must share the same set of category labels.
        sort_by (Literal["frequency", "values"], optional):
            Rule used to order response categories before plotting.
            - `"values"`: sort by the category values themselves.
            - `"frequency"`: sort by response frequency (descending).
            Defaults to `"values"`.
        palette (Optional[sns.palettes._ColorPalette], optional):
            Color palette used for the response categories.
            If `None`, a default diverging palette is generated internally.
            The length of the palette must match the number of categories.
            Defaults to `None`.
        legend_type (Literal["horizontal", "vertical", "none"], optional):
            Placement and layout of the legend.
            - `"horizontal"`: place the legend below the plot in a single row.
            - `"vertical"`: place the legend to the right of the plot.
            - `"none"`: do not draw a legend.
            Defaults to `"horizontal"`.
        show_vline (bool, optional):
            If `True`, draw a vertical reference line at 0.5 (50%),
            which can serve as a visual midpoint for proportions.
            Defaults to `True`.
        ax (Optional[matplotlib.axes.Axes], optional):
            Matplotlib axes to draw the plot on. If `None`, a new figure
            and axes are created. Defaults to `None`.

    Returns:
        None:
            The function draws the plot on the provided or newly created
            matplotlib axes and returns `None`.

    Raises:
        ValueError:
            If the categorical variables in `data` do not share a common
            coding scheme (i.e., their category labels are not identical).

    Notes:
        - When `sort_by="values"` is specified, the function relies on the presence of
          explicit category order information (e.g., ordered categoricals in pandas
          or Enum categories in polars) to determine the plotting order.
        - **Recommended:** When using `sort_by="values"`, it is recommended to provide
          the input as a `pandas.DataFrame` with columns of type `pd.Categorical`
          (with an explicit order), or as a `polars.DataFrame` with columns defined
          as `Enum` categories.
        - For `polars.Categorical` columns, category order may not be preserved as
          expected, and categories can be displayed in lexicographical order
          (e.g., "Agree", "Disagree", "Strongly agree", "Strongly disagree").
        - When a `pyarrow.Table` is provided as input, `sort_by="values"` may raise
          an error due to limitations of dictionary-encoded types in pyarrow.
          In such cases, use `sort_by="frequency"` instead.
        - Missing values are dropped when computing category frequencies.
        - Category order is determined by `sort_by` and then reversed before
          plotting so that the first category appears at the bottom of the bar.
        - The x-axis represents proportions in the range [0, 1], formatted
          as percentages.
        - This function assumes that `make_table_to_plot()` produces the
          columns `"variable"`, `"value"`, `"perc"`, and `"bottom"`.
          
    Examples:
        >>> import py4stats as py4st
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "Q1": ["Agree", "Disagree", "Agree", "Strongly agree"],
        ...     "Q2": ["Disagree", "Disagree", "Agree", "Agree"],
        ... })
        >>> py4st.plot_category(df, sort_by="values", legend_type="horizontal")
    """
    data_nw = nw.from_native(data)
    variables = data_nw.columns
    # 引数のアサーション ==============================================
    legend_type = build.arg_match(
      legend_type, values = ['horizontal', 'vertical', 'none'],
      arg_name = 'legend_type'
      )
    build.assert_logical(show_vline, arg_name = 'sort')

    if data_nw.implementation.is_pyarrow() and sort_by == "values":
        raise ValueError(
            "`sort_by='values&` is not supported in pyarrow.Table."
            "Please try one of the following:\n"\
            "             - Specify sort_by = 'frequency'\n"\
            "             - Use a `pandas.DataFrame` and set the `pd.Categorical` column as an ordered category\n"\
            "             - Use a `polars.DataFrame` and set the `Enum-type` column as an ordered category"\
        )

    # カテゴリー変数のコーディング確認 ==================================
    cording = data_nw[variables[0]].unique().to_list()
    is_common_cording = all([
        s.unique().is_in(cording).all() 
        for s in data_nw.iter_columns()
    ])

    if not is_common_cording:
        messages = "This function assumes that all columns contained in `data` share a common coding scheme."
        raise ValueError(messages)
    
    # データの集計 ==================================================

    table_to_plot = make_table_to_plot(
        data_nw, sort_by = sort_by,
        to_native = False
        )
    
    list_values = table_to_plot['value'].unique().to_list()
    # if nw.is_ordered_categorical(table_to_plot['value']): 
    #     list_values = table_to_plot['value'].cat.get_categories().to_list() 
    # else: 
    #     list_values = table_to_plot['value'].unique().to_list() 

    list_values = list_values[::-1]
    
    # グラフの作図 ==================================================
    make_categ_barh(
        table_to_plot,
        list_values = list_values,
        palette = palette,
        legend_type = legend_type,
        show_vline = show_vline,
        ax = ax
    )

