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
from functools import singledispatch, reduce, partial
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

import math
from wcwidth import wcswidth
import scipy.stats as st

from collections import namedtuple


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
    NamedTuple
)
from numpy.typing import ArrayLike

# matplotlib の Axes は遅延インポート/前方参照でもOK
try:
    from matplotlib.axes import Axes
except Exception:  # notebook等で未importでも落ちないように
    Axes = Any  # type: ignore

DataLike = Union[pd.Series, pd.DataFrame]


# In[ ]:


def is_intoframe(obj: object) -> bool:
    try:
        _ = nw.from_native(obj)
        return True
    except Exception:
        return False


# In[ ]:


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


# In[ ]:


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


# In[ ]:


@as_nw_datarame.register(dict)
def as_nw_datarame_dict(arg: Mapping[str, Any], arg_name: str = 'df_dict', max_items: int = 5):
    try:
        return {
            key: nw.from_native(df) for df in arg
            for key, df in zip(arg.keys(), arg.values())
            }
    except TypeError:
        not_sutisfy = [
            f"'{i}' ({type(v).__name__})" 
            for i, v in zip(arg.keys(), arg.values())
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


# In[ ]:


def as_nw_series(arg: Any, arg_name: str = 'data', **keywargs):
    try:
        return nw.from_native(arg, series_only = True)
    except TypeError:

        raise TypeError(
            f"Argument `{arg_name}` must be a Series supported by narwhals "
            "(e.g. pandas.Series, polars.Series, pyarrow.ChunkedArray), "
            f"but got '{type(arg).__name__}'."
        ) from None


# In[ ]:


def assign_nw(data_nw: nw.DataFrame, **assignment: Mapping[str, Iterable]):
    """Narwhals DataFrame の列に Iterable オブジェクトを代入する
    >>> data_nw = nw.from_native(load_penguins())
    >>> data_nw.pipe(assign_nw,
    ...    body_mass_kg = data_nw['body_mass_g'] / 1000,
    ...    bill_length_mm = pd.cut(data_nw['bill_length_mm'], bins = 10, labels = False)
    ...    ).select(ncs.matches('bill|body')).head(2)
        ┌───────────────────────────────────────────────────────────┐
        |                    Narwhals DataFrame                     |
        |-----------------------------------------------------------|
        |   bill_length_mm  bill_depth_mm  body_mass_g  body_mass_kg|
        |0             2.0           18.7       3750.0          3.75|
        |1             2.0           17.4       3800.0          3.80|
        └───────────────────────────────────────────────────────────┘
    """
    result = data_nw.clone()
    for key, val in zip(assignment.keys(), assignment.values()):

        val_nw = nw.Series.from_iterable(
            key, values = val, 
            backend = data_nw.implementation
            )

        result = result.with_columns(val_nw.alias(key))

    return result


# # `diagnose()`

# In[ ]:


def get_dtypes(data: IntoFrameT) -> pd.Series:
    data_nw = as_nw_datarame(data)
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
        IntoFrameT:
            A summary table with one row per variable and the following columns:
            - columns: names of columns in original DataFrame
            - dtype: pandas dtype of the column.
            - missing_count: number of missing values.
            - missing_percent: percentage of missing values (100 * missing_count / nrow).
            - unique_count: number of unique values (excluding duplicates).
            - unique_rate: percentage of unique values (100 * unique_count / nrow).
    """
    build.assert_logical(to_native, arg_name = 'to_native')
    data_nw = as_nw_datarame(data)

    n = data_nw.shape[0]
    list_dtypes = get_dtypes(data).to_list()

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
    # 引数のアサーション ====================================================================
    values = build.arg_match(
        values, arg_name = 'values',
        values = ['missing_percent', 'missing_count']
    )
    build.assert_logical(sort, arg_name = 'sort', len_arg = 1)
    build.assert_logical(miss_only, arg_name = 'miss_only', len_arg = 1)
    build.assert_count(top_n, lower = 1, arg_name = 'top_n', len_arg = 1, nullable = True)
    # ====================================================================================

    diagnose_tab = diagnose(data, to_native = False)

    if miss_only: diagnose_tab = diagnose_tab.filter(nw.col('missing_percent') > 0)

    if top_n is not None:
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


def _join_comparsion(result_list, on):
    redundant_col = f"{on}_right"
    result = reduce(
            lambda df1, df2: (
                df1.join(df2, on = on, how = 'full')
                .with_columns(
                    nw.when(nw.col(on).is_null()).then(redundant_col)\
                    .otherwise(on).alias(on)
                )), result_list
                )
    result = filtering_out(
            result, redundant_col, 
            to_native = False
        )
    return result


# In[ ]:


def compare_df_cols(
    df_list: Union[List[IntoFrameT], Mapping[str, IntoFrameT]],
    df_name: Optional[List[str]] = None,
    return_match: Literal["all", "match", "mismatch"] = 'all',
    dropna:bool = False,
    to_native: bool = True
) -> IntoFrameT:
    """
    Compare column dtypes across multiple DataFrames.

    This function compares the dtypes of columns with the same names across
    multiple DataFrame-like objects and summarizes the results in a single
    table. The function supports multiple backends via narwhals
    (e.g., pandas, polars, pyarrow).

    Args:
        df_list:
            A list or mapping of input DataFrames. Any DataFrame-like object
            supported by narwhals (e.g., ``pandas.DataFrame``,
            ``polars.DataFrame``, ``pyarrow.Table``) can be used.
            If a mapping is provided, its keys are used as DataFrame names.
        df_name:
            Names for each DataFrame, used as column names in the output.
            Must have the same length as ``df_list``.
            If None, names are automatically generated as
            ``['df1', 'df2', ...]``.
        return_match:
            Specifies which rows to return based on dtype consistency.

            - ``"all"``: return all columns.
            - ``"match"``: return only columns whose dtypes match across
              all DataFrames.
            - ``"mismatch"``: return only columns whose dtypes do not match.
        dropna:
            Passed to ``nunique(dropna=...)`` when checking dtype consistency.
            Controls whether missing values are ignored when determining
            whether dtypes match.
        to_native:
            If True, returns the result as a native DataFrame of the input
            backend. If False, returns a ``narwhals.DataFrame``.

    Returns:
        IntoFrameT:
            A DataFrame where each row corresponds to a column name shared
            across the input DataFrames. The result contains the following
            columns:

            - ``term``: column (variable) name.
            - one column per input DataFrame, containing the dtype of that
              column in each DataFrame.
            - ``match_dtype``: boolean flag indicating whether the dtypes
              are identical across all DataFrames for that column.

    Notes:
        - The column name ``term`` is included as a regular column in the
          output table (not as the index).
        - Internally, the function aligns results using a join operation
          on ``term``.
        - The function performs dtype comparison only; it does not compare
          values.
    """
    # df_name が指定されていなければ、自動で作成します。
    if df_name is None:
        if isinstance(df_list, dict):
            df_name = list(df_list.keys())
            df_list = list(df_list.values())
        else:
            df_name = [f'df{i + 1}' for i in range(len(df_list))]

    # 引数のアサーション ----------------------
    _ = as_nw_datarame(df_list)

    return_match = build.arg_match(
        return_match, values = ['all', 'match', 'mismatch'],
        arg_name = 'return_match'
        )

    build.assert_logical(dropna, arg_name = 'dropna')

    build.assert_character(
        df_name, arg_name = 'df_name', 
        nullable = True, len_arg = build.length(df_list)
    )
    build.assert_logical(to_native, arg_name = 'to_native')
    # --------------------------------------
    implement = nw.from_native(df_list[0]).implementation

    # dtype の集計 =============================================
    dtype_list = [
        enframe(
            get_dtypes(dt), 
            name = 'term', value = val,
            to_native = False,
            backend = implement
            )
        for val, dt in zip(df_name, df_list)
        ]

    # 結果の結合 ==============================================
    result = _join_comparsion(dtype_list, on = 'term')

    # dtype の一致性を確認 ======================================
    match_dtype = (
        result[:, 1:].to_pandas()
        .nunique(axis = 'columns', dropna = dropna) == 1
        )

    match_dtype = nw.Series.from_iterable(
        name = 'match_dtype',
        values = match_dtype.to_list(),
        backend = implement
    )
    result = result.with_columns(match_dtype)

    if(return_match == 'match'):
        result = result.filter(nw.col('match_dtype') == True)
    elif(return_match == 'mismatch'):
        result = result.filter(nw.col('match_dtype') == False)

    if to_native: return result.to_native()
    return result


# ### 平均値などの統計値の近接性で比較するバージョン

# In[ ]:


def compare_df_stats(
    df_list: Union[List[IntoFrameT], Mapping[str, IntoFrameT]],
    df_name: Optional[List[str]] = None,
    return_match: Literal["all", "match", "mismatch"] = "all",
    stats: Callable[..., Any] = np.mean,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    to_native: bool = True,
    **kwargs: Any,
) -> IntoFrameT:
    """
    Compare summary statistics of numeric columns across multiple DataFrames.

    This function computes a summary statistic (e.g., mean or median) for each
    numeric column in multiple DataFrame-like objects and compares those
    statistics across DataFrames. The results are combined into a single table,
    along with an indicator showing whether the statistics are numerically
    close across all DataFrames.

    The function supports multiple DataFrame backends via narwhals
    (e.g., pandas, polars, pyarrow).

    Args:
        df_list:
            A list or mapping of input DataFrames. Any DataFrame-like object
            supported by narwhals (e.g., ``pandas.DataFrame``,
            ``polars.DataFrame``, ``pyarrow.Table``) can be used.
            If a mapping is provided, its keys are used as DataFrame names.
        df_name:
            Names for each DataFrame, used as column names in the output.
            Must have the same length as ``df_list``.
            If None, names are automatically generated as
            ``['df1', 'df2', ...]``.
        return_match:
            Specifies which rows to return based on the comparison of
            statistics across DataFrames.

            - ``"all"``: return all variables.
            - ``"match"``: return only variables whose statistics are close
              across all DataFrames.
            - ``"mismatch"``: return only variables whose statistics are not
              close.
        stats:
            Aggregation function used to compute a single summary value for each
            numeric column.

            This argument accepts either a general callable (e.g.,
            ``numpy.mean`` or a user-defined function) that takes a
            one-dimensional array-like object and returns a single scalar, or a
            function from ``narwhals.functions`` (e.g., ``narwhals.mean``,
            ``narwhals.sum``), which will be applied directly within the
            narwhals expression workflow. Defaults to ``numpy.mean``.
        rtol:
            Relative tolerance parameter passed to ``numpy.isclose`` when
            comparing statistics.
        atol:
            Absolute tolerance parameter passed to ``numpy.isclose`` when
            comparing statistics.
        to_native:
            If True, returns the result as a native DataFrame of the input
            backend. If False, returns a ``narwhals.DataFrame``.
        **kwargs:
            Additional keyword arguments passed to the aggregation function
            when applicable.

    Returns:
        IntoFrameT:
            A DataFrame where each row corresponds to a numeric column and the
            result contains the following columns:

            - ``term``: column (variable) name.
            - one column per input DataFrame, containing the computed statistic
              for that column.
            - ``match_stats``: boolean flag indicating whether the statistics
              are numerically close across all DataFrames according to
              ``numpy.isclose``.

    Notes:
        - Only numeric columns are included in the comparison.
        - The column name ``term`` is included as a regular column in the
          output table (not as the index).
        - When ``stats`` is a function from ``narwhals.functions``, the
          computation is performed using narwhals expressions. Otherwise, the
          function is applied to non-missing values converted to native Python
          objects.
        - Statistical comparison is based on the minimum and maximum values
          across DataFrames for each variable.
    """
    # df_name が指定されていなければ、自動で作成します。
    if df_name is None:
        if isinstance(df_list, dict):
            df_name = list(df_list.keys())
            df_list = list(df_list.values())
        else:
            df_name = [f'df{i + 1}' for i in range(len(df_list))]
    # 引数のアサーション ==========================================
    return_match = build.arg_match(
        return_match, arg_name = 'return_match',
        values = ['all', 'match', 'mismatch']
        )
    build.assert_character(
        df_name, arg_name = 'df_name', 
        nullable = True, len_arg = build.length(df_list)
    )
    build.assert_logical(to_native, arg_name = 'to_native')
    # ==========================================================

    df_list_nw = as_nw_datarame(df_list)

    # 統計値の計算 =============================================
    stats_list = [
        _compute_stats(df, stats, name) 
        for df, name in zip(df_list_nw, df_name)
        ]

    # 計算結果の結合 ==============================================
    result = _join_comparsion(stats_list, on = 'term')

    # 統計値が近いかどうかを比較 ==============================================
    match_stats = [
        np.isclose(np.min(x), np.max(x), rtol = rtol, atol = atol) 
        for x in result[:, 1:].iter_rows()
    ]

    implement = df_list_nw[0].implementation
    match_stats = nw.Series.from_iterable(
        name = 'match_stats',
        values = match_stats,
        backend = implement
    )
    result = result.with_columns(match_stats)

    # 結果の出力 =================================================================

    if(return_match == 'match'):
        result = result.filter(nw.col('match_dtype') == True)
    elif(return_match == 'mismatch'):
        result = result.filter(nw.col('match_dtype') == False)

    if to_native: return result.to_native()
    return result

def _compute_stats(df, stats, name):
    numeric_vars = df.select(ncs.numeric()).columns

    if stats.__module__ == 'narwhals.functions':
        stats_val = df.select(stats(numeric_vars))
    else:
        stats_val = {
            col: stats(df[:, col].drop_nulls().to_list())
            for col in numeric_vars
        }

    stats_df = enframe(
        stats_val , name = 'term', value = name, 
        to_native = False,
        backend = df.implementation
        )

    return stats_df


# ## 2つのデータフレームをレコード単位で比較する関数

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
) -> IntoFrameT:
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
        IntoFrameT or narwhals.DataFrame:
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
    df1 = as_nw_datarame(df1, arg_name = 'df1')
    df2 = as_nw_datarame(df2, arg_name = 'df2')
    all1 = df1.columns
    all2 = df2.columns

    build.assert_logical(sikipna, arg_name = 'sikipna')
    if sikipna:
        df1 = df1.drop_nulls(all1)
        df2 = df2.drop_nulls(all2)

    # 引数のアサーション ==================================================================================
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
    # ==================================================================================================

    numeric1 = df1.select(ncs.numeric()).columns
    nonnum1 = df1.select(~ncs.numeric()).columns
    numeric2 = df2.select(ncs.numeric()).columns
    nonnum2 = df2.select(~ncs.numeric()).columns

    # df1と df2 の列名の共通部分を抽出します。
    all_columns = [item for item in all1 if item in all2]
    numeric_col = set(numeric1) & set(numeric2)
    nonnum_col = set(nonnum1) & set(nonnum2)

    # 類似性の評価 ==========================================================
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
    # カテゴリ変数の類似性評価 ================================================
    res_nonnum_col = [(df1[v] == df2[v]).to_frame() for v in nonnum_col]

    if res_nonnum_col:
        res_nonnum_col_df = nw.concat(res_nonnum_col, how = 'horizontal')
    else:
        res_nonnum_col_df = None

    # 結果の結合と出力 =======================================================
    res_list = [res_number_col_df, res_nonnum_col_df]
    res_list = list(filter(None, res_list))

    result = nw.concat(
        res_list, 
        how = 'horizontal'
        )\
        .select(all_columns)


    if to_native: return result.to_native()
    return result


# ## enframe

# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# ## グループ別平均（中央値）の比較

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
    group1 = as_nw_datarame(group1, arg_name = 'group1')
    group2 = as_nw_datarame(group2, arg_name = 'group2')
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
    group1 = as_nw_datarame(group1, arg_name = 'group1')
    group2 = as_nw_datarame(group2, arg_name = 'group1')
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
    data_nw = as_nw_datarame(data)
    impl = data_nw.implementation

    if impl.is_pyarrow():
        # 2026年1月11日時点で バックエンドが pyarrow の場合 
        # data_nw.pivot() メソッドが使用できないことへの対処です。
        data_nw = nw.from_native(data_nw.to_polars())

    if dropna: data_nw = data_nw.drop_nulls([index, columns])

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

    if margins:
        result = result.with_columns(
            nw.sum_horizontal(ncs.numeric()).alias(margins_name)
            )

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

    data_nw = as_nw_datarame(data)

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
    build.assert_logical(to_native, arg_name = 'to_native')
    # ==============================================================

    data_nw = as_nw_datarame(data)

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
    # これは非推奨の実装なので、安易に使い回さないこと。
    dict_list = {col: c_tab1[col].to_list() for col in c_tab1.columns}
    result = nw.from_dict(dict_list, backend = data_nw.implementation)
    #==================================================================
    if to_native: return result.to_native()
    return result


# ## カテゴリー変数の要約
# ### is_dummy

# In[ ]:


@pf.register_dataframe_method
@pf.register_series_method
@singledispatch
def is_dummy(
    data: Union[list, IntoFrameT, IntoSeriesT],
    cording: Sequence[Any] = (0, 1),
    dropna: bool = True,
    to_pd_series: bool = False,
    to_native: bool = True,
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
            Input data to check. Can be a list or Series-like or DataFrame-like 
            object supported by narwhals (e.g., ``pandas.Series``,
            ``pandas.DataFrame``, ``polars.Series``, ``polars.DataFrame``,
            ``pyarrow.Table``).
        cording:
            Sequence of allowed dummy codes. The input is considered valid if
            its unique values exactly match this set.
            Defaults to ``(0, 1)``.
        dropna (bool, optional):
            Whether to drop NaN from data before value check.
        to_pd_series (bool, optional)::
            Controls the return type when ``data`` is DataFrame-like.
            If True, returns a ``pandas.Series`` indexed by column names.
            If False, returns a Python list of boolean values.
        to_native (bool, optional):
            Controls the return type when ``data`` is DataFrame-like.
            If True, returns the result as a native Series of the input
            backend. If False, returns a ``narwhals.Series``. Defaults to `True`.
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
    raise NotImplementedError(f'is_dummy mtethod for object {type(data)} is not implemented.')



# In[ ]:


@is_dummy.register(nw.Series)
@is_dummy.register(nw.typing.IntoSeries)
def is_dummy_series(
    data: IntoSeriesT,
    cording: Sequence[Any] = (0, 1),
    dropna: bool = True,
    **kwargs
) -> bool:
    build.assert_logical(dropna, arg_name = 'dropna')

    data_nw = as_nw_series(data)
    if dropna: data_nw = data_nw.drop_nulls()

    return set(data_nw) == set(cording)

@is_dummy.register(nw.DataFrame)
@is_dummy.register(nw.typing.IntoDataFrame)
def is_dummy_data_frame(
    data: IntoFrameT,
    cording: Sequence[Any] = (0, 1),
    dropna: bool = True,
    to_pd_series: bool = False,
    to_native: bool = True,
    **kwargs
) -> IntoSeriesT:
    build.assert_logical(dropna, arg_name = 'dropna')
    build.assert_logical(to_pd_series, arg_name = 'to_pd_series')
    build.assert_logical(to_native, arg_name = 'to_native')

    data_nw = as_nw_datarame(data)

    result_df = data_nw.select(
        nw.all().map_batches(
            lambda x: is_dummy_series(x, cording, dropna = dropna), 
            return_dtype = nw.Boolean,
            returns_scalar = True
            )
    )
    if to_pd_series: return result_df.to_pandas().loc[0, :]

    result = nw.Series.from_iterable(
        name = 'is_dummy',
        values = result_df.row(0),
        backend = data_nw.implementation
    )
    if to_native: return result.to_native()
    return result

@is_dummy.register(list)
@is_dummy.register(tuple)
def is_dummy_list(
    data: list,
    cording: Sequence[Any] = (0, 1),
    dropna: bool = True,
    **kwargs
) -> bool:
    build.assert_logical(dropna, arg_name = 'dropna')

    if dropna:
        data = [v for v in data if not build.is_missing(v).all()]

    return set(data) == set(cording)


# ### entropy

# In[ ]:


import scipy as sp

def entropy(x: IntoSeriesT, base: float = 2.0, dropna: bool = True) -> float:
    build.assert_numeric(base, arg_name = 'base', lower = 0, inclusive = 'right')
    build.assert_logical(dropna, arg_name = 'dropna')

    x_nw = as_nw_series(x)

    if dropna: x_nw = x_nw.drop_nulls()

    prop = x_nw.value_counts(normalize = True, sort = False)['proportion']
    result = sp.stats.entropy(pk = prop,  base = base, axis = 0)
    return result

def normalized_entropy(x: IntoSeriesT, dropna: bool = True) -> float:
    build.assert_logical(dropna, arg_name = 'dropna')

    x_nw = as_nw_series(x)
    if dropna: x_nw = x_nw.drop_nulls()

    K = x_nw.n_unique()
    result = entropy(x_nw, base = K, dropna = dropna) if K > 1 else 0.0

    return result


# In[ ]:


@pf.register_dataframe_method
def diagnose_category(data: IntoFrameT, dropna: bool = True, to_native: bool = True) -> IntoFrameT:
    """Summarize categorical variables in a DataFrame.

    This function summarizes columns that represent categorical information,
    including categorical/string/boolean columns and 0/1 dummy columns
    (integer-valued columns restricted to {0, 1}). Dummy columns are cast to
    string before summarization.

    The summary includes missing percentage, number/percentage of unique
    values, mode and its frequency/share, and evenness.

    The implementation is backend-agnostic via narwhals.

    Args:
        data (IntoFrameT):
            Input DataFrame. Any DataFrame type supported by narwhals can be
            used (e.g., pandas, polars, pyarrow).
        dropna (bool):
            Whether to drop NaN from data before computation.
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
            - evenness: category evenness in [0,1], where 1 indicates a 
              uniform distribution and 0 indicates complete concentration 
              in a single category.

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
        - ``evenness`` is a standardized measure of category dispersion, defined 
          as Shannon entropy normalized to the range [0,1]. It is computed by 
          setting the logarithm base to the number of unique categories (unique), 
          which is equivalent to dividing the entropy (with base 2) by ``log_2(unique)``. 
          This quantity is also known as *normalized entropy*.

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
    build.assert_logical(dropna, arg_name = 'dropna')

    data_nw = as_nw_datarame(data)
    res_is_dummy = is_dummy(data_nw, to_pd_series = True)
    dummy_col = res_is_dummy[res_is_dummy].index.to_list()

    if dummy_col:
        data_nw = data_nw.with_columns(nw.col(dummy_col).cast(nw.String))

    df = (
        data_nw
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

    result  = nw.from_dict({
        'variables': var_name,
        'count':df.select(nw.all().count()).row(0),
        'miss_pct':df.select(nw.all().null_count() * nw.lit(100 / N)).row(0),
        'unique':df.select(nw.all().drop_nulls().n_unique()).row(0),
        'mode':[
            freq_table(df, v, descending = True, to_native = False, dropna = dropna)[v][0] 
            for v in var_name
            ],
        'mode_freq':[
            freq_table(df, v, descending = True, to_native = False, dropna = dropna)['freq'][0] 
            for v in var_name
            ],
        'mode_pct':[
            freq_table(df, v, descending = True, to_native = False, dropna = dropna)['perc'][0] 
            for v in var_name
            ],
        'evenness':[normalized_entropy(s, dropna = dropna) for s in df.iter_columns()]
        },
        backend = df.implementation
        )\
        .with_columns(
            unique_pct = 100 * nw.col('unique') / N,
            mode_pct = 100 * nw.col('mode_pct')
        )\
        .select(
            nw.col([
                'variables', 'count',  'miss_pct', 
                'unique', 'unique_pct',
                'mode', 'mode_freq', 'mode_pct', 
                'evenness'
            ])
            )

    if to_native: return result.to_native()
    return result


# ### info_gain: 情報理論の指標に基づいたカテゴリー変数間の相関分析（実験的実装）

# In[ ]:


def binning_for_ig(data, col, max_unique:int = 20, n_bins: Optional[int] = None):
    data_nw = as_nw_datarame(data)
    n = data_nw.shape[0]

    x = data_nw[col]
    if x.n_unique() >= max_unique:
        # n_bins が未指定なら、スタージェスの公式で計算
        if n_bins is None: n_bins = min(int(np.log2(1 + n)), max_unique)

        bined = pd.qcut(x, q = n_bins, labels = False)

        bined = nw.Series.from_iterable(
            col, values = bined, 
            backend = data_nw.implementation
            )

        data_nw = data_nw.with_columns(bined.alias(col))
    return data_nw


# In[ ]:


# from collections import namedtuple
# IGResult = namedtuple('IGResult', ['h_before', 'h_after', 'info_gain', 'ig_ratio'])

# def ig_compute(
#         data, target: str, 
#         feature: str, 
#         use_bining: bool = True,
#         max_unique: int = 20,
#         n_bins: Optional[int] = None,
#         base: float = 2.0
#         ):
#     data_pd = as_nw_datarame(data).to_pandas().copy()

#     if build.is_numeric(data_pd[feature]) and use_bining:
#         n = data_pd.shape[0]

#         if n_bins is None: n_bins = min(int(np.log2(1 + n)), max_unique)

#         data_pd[feature] = pd.qcut(
#             data_pd[feature], 
#             q = n_bins, 
#             labels = False,
#             duplicates = 'drop'
#             )

#     entropy_b = partial(entropy, base = base)
#     h_before = entropy_b(data_pd[target])

#     res = data_pd\
#         .groupby(feature, observed = True)[target]\
#         .agg([entropy_b, 'count'])

#     h_after = weighted_mean(res['entropy'], res['count'])
#     info_gain = np.max([h_before - h_after, 0])
#     ig_ratio = info_gain / h_before if h_before > 0 else np.nan

#     return IGResult(h_before, h_after, info_gain, ig_ratio)


# In[ ]:


from collections import namedtuple
IGResult = namedtuple(
    'IGResult', 
    ['target', 'feature', 'h_target', 'h_feature', 'h_cond', 'joint_ent', 'info_gain', 'ig_ratio']
    )

def ig_compute(
        data, 
        target: str, 
        feature: str, 
        use_bining: bool = True,
        max_unique: int = 20,
        n_bins: Optional[int] = None,
        base: float = 2.0
        ):
    data_nw = as_nw_datarame(data)


    if build.is_numeric(data_nw[feature]) and use_bining:
        data_nw = binning_for_ig(
            data_nw, col = feature,
            max_unique = max_unique,
            n_bins = n_bins
            )

    h_target = entropy(data_nw[target], base = base)
    h_feature = entropy(data_nw[feature], base = base)

    colpaire = build.list_unique([feature, target])
    freq = freq_table(data_nw, colpaire, to_native = False)['freq']
    joint_ent = st.entropy(freq, base = base)

    h_cond = joint_ent - h_feature
    info_gain = np.max([h_target + h_feature - joint_ent, 0.0]) # 相互情報量の非負性を満たすため

    ig_ratio = info_gain / h_target if h_target > 0 else np.nan
    result = IGResult(
        target, feature, h_target, h_feature, 
        h_cond, joint_ent, info_gain, ig_ratio
        )
    return result


# In[ ]:


def info_gain(
        data: IntoFrameT, 
        target: Union[str, List[str]],
        features: Optional[Union[str, List[str]]] = None,
        use_bining: bool = True,
        n_bins: Optional[int] = None,
        max_unique: int = 20,
        base: float = 2.0,
        to_native: bool = True
        ):
    """
    Compute information gain for multiple target-feature pairs.

    This function evaluates the information gain (mutual information)
    and its normalized form (uncertainty coefficient) for all
    combinations of specified target and feature variables.

    It is designed for exploratory screening in settings such as
    questionnaire analysis, where multiple target variables (e.g.,
    survey questions) are compared against one or more attributes
    (e.g., demographic variables).

    Args:
        data (IntoFrameT):
            Input DataFrame. Any DataFrame type supported by narwhals
            (e.g. pandas, polars, pyarrow) can be used.
        target (Union[List[str], str]):
            One or more categorical target variable names.
        features (Optional[Union[List[str]]], optional):
            Feature variable names. If None, all columns except targets
            are used as features. Defaults to None.
        use_bining (bool, optional):
            Whether to discretize numeric features before computing
            information gain. Defaults to True.
        max_unique (int, optional):
            Threshold for triggering discretization of numeric features
            and upper bound for the number of bins. Defaults to 20.
        n_bins (Optional[int], optional):
            Number of bins for discretization. If None, the number of bins
            is determined automatically. Defaults to None.
        base (float, optional):
            The logarithmic base of entropy, defaults to 2.0.
        to_native (bool, optional):
            If True, convert the result to the native DataFrame type of
            the backend. If False, return a narwhals DataFrame.
            Defaults to True.

    Returns:
        IntoFrameT:
            A DataFrame with one row per target-feature pair containing:

            - target (str):
                Target variable name.
            - features (str):
                Feature variable name.
            - h_target (float):
                Entropy of the target variable.
            - h_feature (float):
                Entropy of the feature.
            - h_cond (float):
                Conditional entropy given the feature.
            - joint_ent (float):
                Joint entropy of the target variable and feature.
            - info_gain (float):
                Information gain (mutual information).
            - ig_ratio (float):
                Normalized information gain (uncertainty coefficient).

    Notes:
        - Information gain is equivalent to mutual information.
        - The normalized information gain (IG ratio) corresponds to
          the uncertainty coefficient (Theil's U).
        - The IG ratio allows comparison across different target
          variables by expressing explained uncertainty as a proportion.
        - Numeric features may be discretized prior to computation.
    """
    # 引数のアサーション ====================================================
    build.assert_character(target, arg_name = 'target')
    build.assert_character(features, arg_name = 'features', nullable = True)
    build.assert_logical(use_bining, arg_name = 'use_bining', len_arg = 1)
    build.assert_count(max_unique, arg_name = 'max_unique', len_arg = 1)
    build.assert_count(n_bins, arg_name = 'n_bins', len_arg = 1, nullable = True)
    build.assert_logical(to_native, arg_name = 'to_native', len_arg = 1)
    build.assert_numeric(base, arg_name = 'base', len_arg = 1)
    # ====================================================================
    data_nw = as_nw_datarame(data)

    if isinstance(target, str): target = [target]

    if features is None: 
        features = build.list_diff(data_nw.columns, target)
    elif isinstance(features, str): features = [features]

    colpair = itertools.product(target, features)

    result_dicts = []

    for cols in colpair:
        res = ig_compute(
            data_nw, cols[0], cols[1], 
            use_bining = use_bining,
            max_unique = max_unique,
            n_bins = n_bins,
            base = base
            )
        result_dicts += [res._asdict()]

    result = nw.from_dicts(
        result_dicts,
        backend = data_nw.implementation
    )
    if to_native: return result.to_native()
    return result


# ## その他の補助関数

# In[ ]:


def weighted_mean(x: IntoSeriesT, w: IntoSeriesT, dropna: bool = False) -> float:
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
  x = as_nw_series(x)
  w = as_nw_series(w)

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
def scale(
    x: Union[IntoSeriesT, pd.DataFrame], 
    ddof: int = 1, 
    to_native: bool = True
    ) -> IntoSeriesT:
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
    raise NotImplementedError(f'scale mtethod for object {type(x)} is not implemented.')

@scale.register(nw.Series)
@scale.register(nw.typing.IntoSeries)
def scale_series(x: IntoSeriesT, ddof: int = 1, to_native: bool = True) -> IntoSeriesT:
    build.assert_count(ddof, arg_name = 'ddof')
    build.assert_logical(to_native, arg_name = 'to_native')

    x = as_nw_series(x)

    build.assert_numeric(x, arg_name = 'x', any_missing = True)

    z = (x - x.mean()) / x.std(ddof = ddof)
    if to_native: return z.to_native()
    return z


@scale.register(pd.DataFrame)
def scale_pandas(x: pd.DataFrame, ddof: int = 1, to_native: bool = True) -> IntoSeriesT:
    build.assert_count(ddof, arg_name = 'ddof')
    build.assert_logical(to_native, arg_name = 'to_native')

    z = (x - x.mean(numeric_only = True)) / x.std(ddof = ddof, numeric_only = True)

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
    raise NotImplementedError(f'min_max mtethod for object {type(x)} is not implemented.')

@min_max.register(nw.Series)
@min_max.register(nw.typing.IntoSeries)
def min_max_series(x: Union[IntoSeriesT, pd.DataFrame], to_native: bool = True) -> IntoSeriesT:
    build.assert_logical(to_native, arg_name = 'to_native')

    x = as_nw_series(x)

    build.assert_numeric(x, arg_name = 'x', any_missing = True)

    z = (x - x.min()) / (x.max() - x.min())
    if to_native: return z.to_native()
    return z

@min_max.register(pd.DataFrame)
def min_max_pandas(x: pd.DataFrame, to_native: bool = True) -> IntoSeriesT:
    build.assert_logical(to_native, arg_name = 'to_native')

    z = (x - x.min(numeric_only = True)) / \
        (x.max(numeric_only = True) - x.min(numeric_only = True))

    if to_native: return z
    return nw.from_native(z, allow_series = True)


# ## 完全な空白列 and / or 行の除去

# In[ ]:


def missing_percent(
        data: IntoFrameT,
        axis: str = 'index',
        pct: bool = True
        ) -> pd.Series:
    data_nw = as_nw_datarame(data)
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

        if data_nw.implementation.is_pandas_like() and hasattr(data, 'index'):
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
    build.assert_logical(cols, arg_name = 'cols', len_arg = 1)
    build.assert_logical(rows, arg_name = 'rows', len_arg = 1)
    build.assert_numeric(cutoff, lower = 0, upper = 1, len_arg = 1)
    build.assert_logical(quiet, arg_name = 'quiet', len_arg = 1)
    build.assert_logical(to_native, arg_name = 'to_native', len_arg = 1)
    # ==============================================================

    data_nw = as_nw_datarame(data)
    df_shape = data.shape
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
            # ↓ missing_percent() -> pd.Series なので成立します。
            row_removed = empty_rows[empty_rows].index.to_series().astype('str').to_list()
            print(
                    f"Removing {nrow_removed} empty row(s) out of {df_shape[0]} rows" +
                    f"(Removed: {','.join(row_removed)}). "
                )

    if to_native: return data_nw.to_native()
    return data_nw


# In[ ]:


def is_constant(data: IntoSeriesT, dropna: bool = True) -> bool:
    data = as_nw_series(data)
    if dropna: 
        return data.drop_nulls().n_unique() == 1
    else:
        return data.n_unique() == 1


# In[ ]:


@pf.register_dataframe_method
def remove_constant(
    data: IntoFrameT,
    quiet: bool = True,
    dropna = False,
    to_native: bool = True,
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
        IntoFrameT:
            DataFrame after removing constant columns.
    """
    # 引数のアサーション ==============================================
    build.assert_logical(quiet, arg_name = 'quiet', len_arg = 1)
    build.assert_logical(dropna, arg_name = 'dropna', len_arg = 1)
    build.assert_logical(to_native, arg_name = 'to_native', len_arg = 1)
    # ==============================================================
    data_nw = as_nw_datarame(data)
    col_name = data_nw.columns

    bool_constant = [is_constant(col, dropna) for col in data_nw.iter_columns()]

    constant_columns = [
        col for i, col in enumerate(col_name) 
        if bool_constant[i]
        ]

    if not(quiet) :
        n_removed = len(constant_columns)
        removed = build.oxford_comma_and(constant_columns, quotation = False)
        print(
            f"Removing {n_removed} constant column(s) out of {len(col_name)} columns" +
            f"(Removed: {removed}). "
        )

    selected = build.list_diff(col_name, constant_columns)
    result = data_nw.select(selected)

    if to_native: return result.to_native()
    return result


# ## 列名や行名に特定の文字列を含む列や行を除外する関数

# In[ ]:


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


# In[ ]:


# 列名や行名に特定の文字列を含む列や行を除外する関数
# @pf.register_dataframe_method
def filtering_out(
    data: IntoFrameT,
    *args: Union[str, List[str], narwhals.Expr, narwhals.selectors.Selector], 
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
        *args (Union[str, List[str], narwhals.Expr, narwhals.Selector]):
            Columns or index to exclude. Each element may be:
            - a column name (`str`)
            - a list of column names or index
            - a narwhals expression (for `axis = 'columns'`)
            - a narwhals selector   (for `axis = 'columns'`)
            The order of columns specified here is preserved in the output.
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
        Row-wise filtering via axis='index' relies on the presence of an explicit index. 
        Therefore, this option is not available for DataFrame backends that do not expose 
        row labels (e.g. some Arrow-based tables).

    """
    # 引数のアサーション ==============================================
    axis = str(axis)
    axis = build.arg_match(axis, arg_name = 'axis', values = ['1', 'columns', '0', 'index'])
    build.assert_logical(to_native, arg_name = 'to_native')
    build.assert_character(contains, arg_name = 'contains', nullable = True)
    build.assert_character(starts_with, arg_name = 'starts_with', nullable = True)
    build.assert_character(ends_with, arg_name = 'ends_with', nullable = True)
    _assert_selectors(*args, arg_name = 'args', nullable = True)
    # ==============================================================
    data_nw = as_nw_datarame(data)

    # columns に基づく除外処理 =================================================================
    if axis in ("0", "index"):
        drop_table = pd.DataFrame()
        if not hasattr(data, "index"):
            raise TypeError(
                f"filtering_out(..., axis='{axis}') requires an input that has"
                "an 'index' (row labels), e.g. pandas.DataFrame.\n"
                f"Got: {type(data)}."
            )
    if((axis == '1') | (axis == 'columns')):
        drop_table = pd.DataFrame()

        if args:
            drop_list = data_nw.select(args).columns
        else: drop_list = []

        list_columns = data_nw.columns

        if contains is not None:
            drop_list += build.list_subset(list_columns, lambda x: contains in x)

        if starts_with is not None:
            drop_list += build.list_subset(list_columns, lambda x: x.startswith(starts_with))

        if ends_with is not None:
            drop_list += build.list_subset(list_columns, lambda x: x.endswith(ends_with))

        if contains is not None or starts_with is not None or ends_with is not None:
            drop_list = build.list_unique(drop_list)

        data_nw = data_nw.drop(drop_list)
        if to_native: return data_nw.to_native()
        return data_nw

    # index に基づく除外処理 =================================================================
    elif isinstance(data, pd.DataFrame):
        list_index = data.index.to_list()
        drop_list = []
        if args: 
            args = list(build.list_flatten(args))
            drop_list += [i for i in args if i in list_index]

        if contains is not None: 
            drop_list += build.list_subset(list_index, lambda x: contains in x)

        if starts_with is not None: 
            drop_list += build.list_subset(list_index, lambda x: x.startswith(starts_with))

        if ends_with is not None:
            drop_list += build.list_subset(list_index, lambda x: x.endswith(ends_with))

        if args or contains is not None or starts_with is not None or ends_with is not None:
            keep_list = [i not in drop_list for i in list_index]
            data_nw = data_nw.filter(keep_list)

        if to_native: return data_nw.to_native()
        return data_nw


# ## パレート図を作図する関数

# In[ ]:


def Pareto_plot(
    data: IntoFrameT,
    group: str,
    values: Optional[str] = None,
    top_n: Optional[int] = None,
    aggfunc: Callable[[IntoSeriesT], Union[int, float]] = np.mean,
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
    data_nw = as_nw_datarame(data)
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
    data: IntoSeriesT,
    group: str,
    values: str,
    aggfunc: Callable[[IntoSeriesT], Union[int, float]] = np.mean,
    to_native: bool = True,
) -> pd.DataFrame:
    data_nw = as_nw_datarame(data)

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
        stat = group_map(
            data_nw, group,
            func = lambda df: aggfunc(df[values]),
        )
        stat_table = nw.from_dict({
                group: stat.groups[group], 
                values: stat.mapped
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


# ## 代表値 + 区間推定関数
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


# ### mean_qi 

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
    raise NotImplementedError(f'mean_qi mtethod for object {type(data)} is not implemented.')


# In[ ]:


@mean_qi.register(nw.DataFrame)
@mean_qi.register(nw.typing.IntoDataFrame)
def mean_qi_data_frame(
    data: IntoFrameT,
    width: float = 0.975,
    interpolation: Interpolation = 'midpoint',
    to_native: bool = True
    ) -> pd.DataFrame:
    # 引数のアサーション =======================================================
    build.assert_numeric(width, lower = 0, upper = 1, inclusive = 'neither', len_arg = 1)
    build.assert_logical(to_native, arg_name = 'to_native', len_arg = 1)
    interpolation = build.arg_match(
        interpolation, arg_name = 'interpolation',
        values = interpolation_values
        )
    # =======================================================================
    df_numeric = as_nw_datarame(data).select(ncs.numeric())

    result = nw.from_dict({
        'variable': df_numeric.columns,
        'mean': df_numeric.select(nw.all().mean()).row(0),
        'lower': df_numeric.select(
            nw.all().quantile(1 - width, interpolation = interpolation)
            ).row(0),
        'upper': df_numeric.select(
            nw.all().quantile(width, interpolation = interpolation)
            ).row(0)
        }, backend = df_numeric.implementation
        )
    if to_native: return result.to_native()
    return result


# In[ ]:


@mean_qi.register(nw.Series)
@mean_qi.register(nw.typing.IntoSeries)
def mean_qi_series(
    data: SeriesT,
    width: float = 0.975,
    interpolation: Interpolation = 'midpoint',
    to_native: bool = True
    ):
    # 引数のアサーション =======================================================
    build.assert_numeric(width, lower = 0, upper = 1, inclusive = 'neither', len_arg = 1)
    build.assert_logical(to_native, arg_name = 'to_native', len_arg = 1)
    interpolation = build.arg_match(
        interpolation, arg_name = 'interpolation',
        values = interpolation_values
        )
    # =======================================================================
    data_nw = as_nw_series(data)
    if data_nw.name: variable = data_nw.name
    else: variable = 'x'

    result = nw.from_dict({
        'variable': [variable],
        'mean': [data_nw.mean()],
        'lower': [data_nw.quantile(1 - width, interpolation = interpolation)],
        'upper': [data_nw.quantile(width, interpolation = interpolation)]
    }, backend = data_nw.implementation
    )
    if to_native: return result.to_native()
    return result


# ### median_qi

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
    raise NotImplementedError(f'median_qi mtethod for object {type(data)} is not implemented.')


# In[ ]:


@median_qi.register(nw.DataFrame)
@median_qi.register(nw.typing.IntoDataFrame)
def median_qi_data_frame(
    data: IntoFrameT,
    width: float = 0.975,
    interpolation: Interpolation = 'midpoint',
    to_native: bool = True
) -> IntoFrameT:
    # 引数のアサーション =======================================================
    build.assert_numeric(width, lower = 0, upper = 1, inclusive = 'neither', len_arg = 1)
    build.assert_logical(to_native, arg_name = 'to_native', len_arg = 1)
    interpolation = build.arg_match(
        interpolation, arg_name = 'interpolation',
        values = interpolation_values
        )
    # =======================================================================

    df_numeric = as_nw_datarame(data).select(ncs.numeric())

    result = nw.from_dict({
        'variable': df_numeric.columns,
        'median': df_numeric.select(nw.all().median()).row(0),
        'lower': df_numeric.select(
            nw.all().quantile(1 - width, interpolation = interpolation)
            ).row(0),
        'upper': df_numeric.select(
            nw.all().quantile(width, interpolation = interpolation)
            ).row(0)
        }, backend = df_numeric.implementation
        )
    if to_native: return result.to_native()
    return result


@median_qi.register(nw.Series)
@median_qi.register(nw.typing.IntoSeries)
def median_qi_series(
    data: IntoSeriesT,
    width: float = 0.975,
    interpolation: Interpolation = 'midpoint',
    to_native: bool = True
) -> IntoFrameT:
    # 引数のアサーション =======================================================
    build.assert_numeric(width, lower = 0, upper = 1, inclusive = 'neither', len_arg = 1)
    build.assert_logical(to_native, arg_name = 'to_native', len_arg = 1)
    interpolation = build.arg_match(
        interpolation, arg_name = 'interpolation',
        values = interpolation_values
        )
    # =======================================================================

    data_nw = as_nw_series(data)
    if data_nw.name: variable = data_nw.name
    else: variable = 'x'

    result = nw.from_dict({
        'variable': [variable],
        'median': [data_nw.median()],
        'lower': [data_nw.quantile(1 - width, interpolation = interpolation)],
        'upper': [data_nw.quantile(width, interpolation = interpolation)]
    }, backend = data_nw.implementation
    )
    if to_native: return result.to_native()
    return result


# ### mean_ci

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
    raise NotImplementedError(f'mean_ci mtethod for object {type(data)} is not implemented.')


# In[ ]:


@mean_ci.register(nw.DataFrame)
@mean_ci.register(nw.typing.IntoDataFrame)
def mean_ci_data_frame(
    data: IntoFrameT,
    width: float = 0.975,
    to_native: bool = True
) -> IntoFrameT:
    # 引数のアサーション =======================================================
    build.assert_numeric(width, lower = 0, upper = 1, inclusive = 'neither', len_arg = 1)
    build.assert_logical(to_native, arg_name = 'to_native', len_arg = 1)
    # =======================================================================
    df_numeric = as_nw_datarame(data).select(ncs.numeric())
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
@mean_ci.register(nw.typing.IntoSeries)
def mean_ci_series(
    data: SeriesT,
    width: float = 0.975,
    to_native: bool = True
) -> IntoFrameT:
    # 引数のアサーション =======================================================
    build.assert_numeric(width, lower = 0, upper = 1, inclusive = 'neither', len_arg = 1)
    build.assert_logical(to_native, arg_name = 'to_native', len_arg = 1)
    # =======================================================================
    data_nw = as_nw_series(data)
    if data_nw.name: variable = data_nw.name
    else: variable = 'x'

    n = len(data_nw)
    t_alpha = t.isf((1 - width) / 2, df = n - 1)
    x_mean = data_nw.mean()
    x_std = data_nw.std()

    result = nw.from_dict({
        'variable': [variable],
        'mean':[x_mean],
        'lower':[x_mean - t_alpha * x_std / np.sqrt(n)],
        'upper':[x_mean + t_alpha * x_std / np.sqrt(n)],
    }, backend = data_nw.implementation
    )
    if to_native: return result.to_native()
    return result


# In[ ]:


def plot_dot_line(
    data: IntoFrameT,
    x: str, y: str,
    lower: str = 'lower',
    upper: str = 'upper',
    ax: Optional[Axes] = None,
    color: Sequence[str] = "#1b69af",
    **keywargs
) -> None:
    data_nw = as_nw_datarame(data)
    # 引数のアサーション ==============================================
    build.assert_character(color, arg_name = 'color')

    columne_name = data_nw.columns
    x = build.arg_match(
        x, arg_name = 'x', values = columne_name
    )
    y = build.arg_match(
        y, arg_name = 'y', values = columne_name
    )
    lower = build.arg_match(
        lower, arg_name = 'lower', values = columne_name
    )
    upper = build.arg_match(
        upper, arg_name = 'upper', values = columne_name
    )
    # ==============================================================    
    if ax is None:
        fig, ax = plt.subplots()

    # 図の描画 -----------------------------
    # エラーバーの作図
    ax.hlines(
        y = data_nw[y], xmin = data_nw[lower], xmax = data_nw[upper],
        linewidth = 1.5,
        color = color
        )
    # 点推定値の作図
    ax.scatter(
      x = data_nw[x],
      y = data_nw[y],
      c = color,
      s = 60
    )
    ax.invert_yaxis()
    ax.set_ylabel(y);


# ## 正規表現と文字列関連の論理関数

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

    data_nw = as_nw_series(data)

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


@pf.register_dataframe_method
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
            - rule: name of rules which taken from keys of `rule_dict`
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
    data_nw = as_nw_datarame(data)
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


@pf.register_dataframe_method
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
    data_nw = as_nw_datarame(data)
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


@pf.register_series_method
def set_miss(
    x: IntoSeriesT, 
    n: Optional[int] = None,
    prop: Optional[float] = None, 
    method: Literal['random', 'first', 'last'] = 'random', 
    random_state: Optional[int] = None, 
    na_value: Any = None,
    to_native: bool = True
    ):
    """Insert missing values into a Series.

    This function replaces a specified number or proportion of non-missing
    elements in a Series with missing values. It supports multiple Series
    backends via narwhals and is primarily intended for generating test data
    or simulating missingness.

    Exactly one of `n` or `prop` must be specified.

    Args:
    x (IntoSeriesT):
        Input Series. Any Series-like object supported by narwhals
        (e.g., pandas.Series, polars.Series, pyarrow.ChunkedArray)
        can be used.
    n (int, optional):
        Target number of missing values in the Series after processing.
        If the Series already contains `n` or more missing values,
        no additional missing values are added and a warning is issued.
    prop (float, optional):
        Target proportion of missing values in the Series after processing.
        Must be between 0 and 1. If the current proportion of missing
        values is greater than or equal to `prop`, no additional missing
        values are added and a warning is issued.
    method ({'random', 'first', 'last'}, optional):
        Strategy for selecting elements to be replaced with missing values.
        - ``'random'``: randomly select non-missing elements.
        - ``'first'``: select from the beginning of the Series.
        - ``'last'``: select from the end of the Series.
        Defaults to ``'random'``.
    random_state (int, optional):
        Random seed used when ``method='random'`` to ensure reproducibility.
    na_value (Any, optional):
        Value used to represent missing data. Defaults to ``None``.
    to_native (bool, optional):
        If True, return the result as a native Series class of 'x'.
        If False, return a `narwhals.Series`.

    Returns:
    IntoSeriesT or narwhals.Series:
        Series with additional missing values inserted. The return type
        depends on the value of ``to_native``.

    Raises:
    ValueError:
        If neither or both of `n` and `prop` are specified.

    Warns:
    UserWarning:
        If the input Series already contains the specified number or
        proportion of missing values and no additional missing values
        are added.

    Examples:
    >>> import pandas as pd
    >>> import py4stats as py4st
    >>> s = pd.Series([1, 2, 3, 4, 5])
    >>> py4st.set_miss(s, n=2, method='first')
    0    NaN
    1    NaN
    2    3.0
    3    4.0
    4    5.0
    dtype: float64

    >>> py4st.set_miss(s, prop=0.4, method='random', random_state=0)
    0    1.0
    1    NaN
    2    3.0
    3    NaN
    4    5.0
    dtype: float64
    """
    x_nw = as_nw_series(x)

    # 引数のアサーション ==================================================================
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

    # 欠測値代入個数の計算 =================================================================
    idx = pd.Series(np.arange(len(x_nw)))
    non_miss = idx[~build.is_missing(x_nw)]

    if n is not None: 
        n_to_miss = np.max([n - n_miss, 0])

        if n_to_miss <=0:
            warnings.warn(
                f"Already contained {n_miss}(>= n) missing value(s) in `x`, "
                "no additional missing values were added.",
            category = UserWarning,
            stacklevel = 2
            )
            if to_native: return x_nw.to_native()
            return x_nw

    elif prop is not None: 
        n_non_miss = non_miss.shape[0]

        n_to_miss = int(np.max([
        np.ceil(n_non_miss * (prop - p_miss)), 0
        ]))

        if prop <= p_miss:
            warnings.warn(
            f"Already contained {p_miss:.3f}(>= prop) missing value(s) in `x`, "
            "no additional missing values were added.",
            category = UserWarning,
            stacklevel = 2
            )
            if to_native: return x_nw.to_native()
            return x_nw

    # 欠測値代入位置の決定 =====================================================================

    match method:
        case 'random':
            index_to_na = non_miss.sample(n = n_to_miss, random_state = random_state)
        case 'first':
            index_to_na = non_miss.head(n_to_miss)
        case 'last':
            index_to_na = non_miss.tail(n_to_miss)
    # 欠測値の代入と結果の出力 ===================================================================
    x_with_na = [na_value if i in index_to_na else v 
            for i, v in enumerate(x_nw)]

    result = nw.Series.from_iterable(
        name = x_nw.name,
        values = x_with_na,
        backend = x_nw.implementation
    )
    if not to_native: return result

    result_native = nw.to_native(result)
    if x_nw.implementation.is_pandas_like():
        result_native.index = x.index
    return result_native


# # `relocate()`

# In[ ]:


def arrange_colnames(
        colnames: list[str], 
        selected: list[str], 
        before: Optional[str] = None, 
        after: Optional[str] = None,
        place: Optional[Literal['first', 'last']] = None,
        ):
    unselected = [i for i in colnames if i not in selected]
    if before is None and after is None:
        if place is None: place = 'first' 

        if place == 'first':
            return selected + unselected
        else:
            return unselected + selected

    if before is not None:
        idx = unselected.index(before)
        col_pre = unselected[:idx]
        col_behind = unselected[idx:]
        return col_pre + selected + col_behind

    elif after is not None:
        idx = unselected.index(after) + 1
        col_pre = unselected[:idx]
        col_behind = unselected[idx:]
        return col_pre + selected + col_behind


# In[ ]:


def _is_before_after_selected(selected: list[str], value:Optional[str] = None):
    result = isinstance(value, str) and value in selected \
        and build.length(selected) == 1 \
            and selected[0] == value
    return result


# In[ ]:


@pf.register_dataframe_method
def relocate(
        data: IntoFrameT, 
        *args: Union[str, List[str], narwhals.Expr, narwhals.selectors.Selector], 
        before: Optional[str] = None,
        after: Optional[str] = None,
        place: Optional[Literal["first", "last"]] = None,
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
        place (Optional[Literal["first", "last"]], optional):
            Destination where the selected columns are placed when neither
            `before` nor `after` is specified.
            - `"first"`: place the selected columns at the beginning of the DataFrame.
            - `"last"`: place the selected columns at the end of the DataFrame.
            Cannot be used together with `before` or `after`.
            Defaults to `None` (equivalent to `"first"`).
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
        ValueError:
            If `place` is specified together with `before` or `after`.
        ValueError:
            If `before`/`after` is the same as the only relocated column.


    Notes:
        - This function only changes the order of columns; no columns are added
          or removed.
        - If neither `before` nor `after` is specified, selected columns are placed
          according to `place` (default: `"first"`).
        - When `before`/`after` is specified and it is also included in the selected
          columns (e.g., via a selector), the reference column is excluded from the
          relocation target to avoid an undefined ordering.
        - Column order among the selected columns follows the order specified in
          `*args`.

    Examples:
        >>> import py4stats as py4st
        >>> import narwhals.selectors as ncs
        >>> from palmerpenguins import load_penguins
        >>> penguins = load_penguins()

        Move columns to the front (default behavior):
        >>> py4st.relocate(penguins, "year", "sex")

        Move selected columns to the end:
        >>> py4st.relocate(penguins, "year", "sex", place="last")

        Relocate columns using a selector:
        >>> py4st.relocate(penguins, ncs.numeric())

        Place columns before a specific column:
        >>> py4st.relocate(penguins, "year", before="island")

        Place selected columns after a reference column (and exclude the reference
        column from relocation if it was selected via a selector):
        >>> py4st.relocate(penguins, ncs.numeric(), after="year")
    """
    # 引数のアサーション ======================================
    build.assert_logical(to_native, arg_name = 'to_native')
    _assert_selectors(*args)

    build.assert_character(before, arg_name = 'before', nullable = True, scalar_only = True)
    build.assert_character(after, arg_name = 'after', nullable = True, scalar_only = True)

    if (before is not None) and (after is not None):
        raise ValueError("Please specify either `before` or `after`, not both.")

    place = build.arg_match(
        place, arg_name= 'place',
        values = ['first', 'last'],
        nullable = True
    )
    if (place is not None) and ((before is not None) or (after is not None)):
        raise ValueError("Please specify either `place` or `before`/`after`, not both.")
    # ======================================================

    data_nw = as_nw_datarame(data)
    colnames = data_nw.columns
    selected = data_nw.select(args).columns

    # before/after に指定された列が arg に含まれている場合への対処 =============================
    # selected の要素が1つで、befor/after と等しいなならエラー（並べ替え方が定義できないので）
    if _is_before_after_selected(selected, before):
        raise ValueError("`before` cannot be the same as the relocated column.")
    if _is_before_after_selected(selected, after):
        raise ValueError("`after` cannot be the same as the relocated column.")

    # selected が複数の要素を持ち、befor/after 含まれるなら除外 ==================================
    if after is not None:
        selected = [c for c in selected if c != after]
    if before is not None:
        selected = [c for c in selected if c != before]

    # selected が空のリストになった場合の安全処置
    # selected = [] なら arrange_colnames() は colnames をそのまま返すと思いますが念のため。
    if not selected:
        return data_nw.to_native() if to_native else data_nw

    # 列を並べ替えて出力 =====================================================================
    arranged = arrange_colnames(colnames, selected, before, after, place)
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
    data_nw = as_nw_datarame(data)

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
    data_nw = as_nw_datarame(data)
    variables = data_nw.columns
    # 引数のアサーション ==============================================
    legend_type = build.arg_match(
      legend_type, values = ['horizontal', 'vertical', 'none'],
      arg_name = 'legend_type'
      )
    build.assert_logical(show_vline, arg_name = 'sort')

    if data_nw.implementation.is_pyarrow() and sort_by == "values":
        raise ValueError(
            "`sort_by = 'values` is not supported in pyarrow.Table."
            "Please try one of the following:\n"\
            "             - Specify `sort_by = 'frequency`'\n"\
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


# ### `review_wrangling()`

# In[ ]:


def _assert_same_backend(data1, data2, funcname = 'review_wrangling', data_name = ['before', 'after']):
      if data1.implementation is not data2.implementation:
        raise TypeError(
            f"{funcname}() requires `{data_name[0]}` and `{data_name[1]}`` to use the same backend.\n"
            f"Got {data_name[0]}={data1.implementation!r}, {data_name[1]}={data2.implementation!r}.\n"
            f"Please make sure that  `{data_name[0]}` and `{data_name[1]}` are the same backend (e.g., both pandas, both polars)."
        )


# In[ ]:


def review_casting(before: IntoFrameT, after: IntoFrameT) -> str:
    """Review changes in column data types between two DataFrames.

    This function compares the data types of columns in two DataFrame-like
    objects representing the state of a dataset before and after wrangling,
    and reports columns whose types have changed.

    Only existing columns with mismatched types are reported. Columns that
    are newly added or removed are not included in this review.

    Args:
        before (IntoFrameT):
            The DataFrame before wrangling.
        after (IntoFrameT):
            The DataFrame after wrangling.

    Returns:
        str:
            A human-readable, multi-line string describing columns whose
            data types have changed. If no type changes are detected,
            a message indicating this is returned.

    Examples:
        >>> import py4stats as py4st
        >>> print(py4st.review_casting(before, after))
        The following columns have changed their type:
          species object -> category
          year    int64 -> float64

    """
    res_compare = compare_df_cols(
        [before, after], return_match = 'mismatch', 
        to_native = False
        ).drop_nulls()

    name_w = res_compare.select(
        len = nw.col("term").str.len_chars()
        )['len'].max()

    if build.length(res_compare) >= 1:
        col_cast = [
            f"  {row[0]:<{name_w}} {row[1]} -> {row[2]}"
            for row in res_compare.iter_rows()
            ]
        cast_message = f'The following columns have changed their type:\n{"\n".join(col_cast)}'
    else: 
        cast_message = 'No existing columns had their type changed.'
    return cast_message


# In[ ]:


def review_col_addition(
        before: IntoFrameT, after: IntoFrameT,
        abbreviate: bool = True,
        max_columns: Optional[int] = None,
        max_width: int = 80
        ) -> str:
    """Review added and removed columns between two DataFrames.

    This function compares column names in two DataFrame-like objects and
    reports which columns were added or removed as a result of wrangling.

    For readability, long lists of column names can be abbreviated based on
    either the number of columns or the formatted text width.

    Args:
        before (IntoFrameT):
            The DataFrame before wrangling.
        after (IntoFrameT):
            The DataFrame after wrangling.
        abbreviate (bool, optional):
            Whether to abbreviate long lists of column names in the output.
            Defaults to True.
        max_columns (int or None, optional):
            Maximum number of column names to display when reporting added
            or removed columns. If None, truncation is based on text width.
        max_width (int, optional):
            Maximum text width (in characters) used for truncation when
            `max_columns` is None. Defaults to 80.

    Returns:
        str:
            A formatted string summarizing column additions and removals.
            If no columns were added or removed, an explanatory message
            is returned.

    Examples:
        >>> import py4stats as py4st
        >>> print(py4st.review_col_addition(before, after))
        Column additions and removals:
        added:   'heavy' and 'const'
        removed: 'sex'
    """
    # 引数のアサーション =======================================================
    build.assert_logical(abbreviate, arg_name = 'abbreviate')
    build.assert_count(
        max_columns, arg_name = 'max_columns', 
        len_arg = 1, nullable = True)
    build.assert_count(
        max_width, arg_name = 'max_width', 
        len_arg = 1)
    # =======================================================================
    columns_before = as_nw_datarame(before, arg_name = 'before').columns
    columns_after  = as_nw_datarame(after, arg_name = 'after').columns

    added = build.list_diff(columns_after, columns_before)
    removed = build.list_diff(columns_before, columns_after)

    if added or removed:
        col_adition = ["Column additions and removals:"]
        if added:
            col_adition += [f"  added:   {build.oxford_comma_shorten(
                added, suffix = 'column(s)', 
                abbreviate = abbreviate, max_items = max_columns, max_width = max_width
                )}"]
        else:
            col_adition += ['  No columns were added. ']
        if removed:
            col_adition += [f"  removed: {build.oxford_comma_shorten(
                removed, suffix = 'column(s)',
                abbreviate = abbreviate, max_items = max_columns, max_width = max_width
                )}"]
        else:
            col_adition += ['  No columns were removed.']

        return '\n'.join(col_adition)
    else: return 'No columns were added or removed.'


# In[ ]:


def format_missing_lines(miss_table):
    rows_list = [
        dict(zip(miss_table.columns, row)) 
        for row in miss_table.iter_rows()
        ]

    names = [f"  {row.get('columns')} " for row in rows_list]
    name_w = max(wcswidth(n) for n in names)  # 列名の表示幅
    # 欠測数の表示幅
    w_count_be = len(f"{miss_table['missing_count'].max():,}")
    w_count_af = len(f"{miss_table['missing_count_after'].max():,}")

    # 欠測率の表示幅
    w_pct_be = len(f"{miss_table['missing_percent'].max():.2f}")
    w_pct_af = len(f"{miss_table['missing_percent_after'].max():.2f}")

    col_miss = []
    for i, row in enumerate(rows_list):
        c_before, p_before = row.get('missing_count'), row.get('missing_percent')
        c_after, p_after = row.get('missing_count_after'), row.get('missing_percent_after')
        col_miss += [
                f"{names[i]:<{name_w}} "+\
                f"before {c_before:>{w_count_be},} ({p_before:>{w_pct_be}.2f}%) -> " +\
                f"after {c_after:>{w_count_af},} ({p_after:>{w_pct_af}.2f}%)"
                ]
    return col_miss


# In[ ]:


def review_missing(before: IntoFrameT, after: IntoFrameT) -> str:
    """Review changes in missing values between two DataFrames.

    This function compares the number and proportion of missing values
    for each column in two DataFrame-like objects and reports columns
    where missingness has increased or decreased.

    Columns with no change in missing values are not listed.

    Args:
        before (IntoFrameT):
            The DataFrame before wrangling.
        after (IntoFrameT):
            The DataFrame after wrangling.

    Returns:
        str:
            A human-readable report describing increases and decreases
            in missing values. If no changes are detected, a message
            indicating this is returned.

    Examples:
        >>> import py4stats as py4st
        >>> print(py4st.review_missing(before, after))
        Increase in missing values:
        flipper_length_mm  before 2 (0.58%) -> after 22 (11.76%)

        Decrease in missing values:
        bill_length_mm  before 2 (0.58%) -> after 0 (0.00%)
        bill_depth_mm   before 2 (0.58%) -> after 0 (0.00%)
        body_mass_g     before 2 (0.58%) -> after 0 (0.00%)    
    """
    compare_miss = diagnose(before, to_native = False)\
        .select('columns', 'missing_count', 'missing_percent')\
        .join(
            diagnose(after, to_native = False)\
            .select('columns', 'missing_count', 'missing_percent'),
            on = 'columns', suffix = "_after"
            )

    increased = compare_miss.filter(
        nw.col('missing_percent') < nw.col('missing_percent_after')
    )
    decreased = compare_miss.filter(
        nw.col('missing_percent') > nw.col('missing_percent_after')
    )
    if build.length(increased) == 0 and build.length(decreased) == 0:
        return 'No existing columns decreased the number of missing values.'

    # return increased, decreased
    miss_review = []
    if build.length(increased) >= 1:
        col_miss = format_missing_lines(increased)
        miss_review += [f'Increase in missing values:\n{"\n".join(col_miss)}']
    else: 
        miss_review += ['No existing columns increased the number of missing values.']

    if build.length(decreased) >= 1:
        col_miss = format_missing_lines(decreased)
        miss_review += [f'Decrease in missing values:\n{"\n".join(col_miss)}']
    else: 
        miss_review += ['None of the existing columns decreases in the number of missing values.']
    result = '\n\n'.join(miss_review)
    return result


# In[ ]:


def shape_change(before: int, after: int) -> str:
    if after > before: return f" (+{after - before:,})"
    if after < before: return f" ({after - before:,})"
    return f" (No change)"


# In[ ]:


def review_shape(before: IntoFrameT, after: IntoFrameT) -> str:
    """Review changes in the shape of a DataFrame.

    This function compares the number of rows and columns in two
    DataFrame-like objects representing the state of a dataset before
    and after wrangling.

    Args:
        before (IntoFrameT):
            The DataFrame before wrangling.
        after (IntoFrameT):
            The DataFrame after wrangling.

    Returns:
        str:
            A formatted string summarizing changes in the number of rows
            and columns.
    """
    before_nw = as_nw_datarame(before, arg_name = 'before')
    after_nw  = as_nw_datarame(after, arg_name = 'after')
    row_o, col_o = before_nw.shape
    row_n, col_n = after_nw.shape
    d_o = len(f"{np.max([row_o, col_o]):,}")
    d_n = len(f"{np.max([row_n, col_n]):,}")

    shpe_message = f"The shape of DataFrame:\n" + \
                f"   Rows: before {row_o:>{d_o},} -> after {row_n:>{d_n},}{shape_change(row_o, row_n)}\n" + \
                f"   Cols: before {col_o:>{d_o},} -> after {col_n:>{d_n},}{shape_change(col_o, col_n)}"
    return shpe_message


# In[ ]:


def review_category(
        before: IntoFrameT, 
        after: IntoFrameT,
        abbreviate: bool = True,
        max_categories: Optional[int] = None,
        max_width: int = 80
        ) -> str:
    """Review changes in category levels between two DataFrames.

    This function compares the observed category levels of string,
    categorical, and boolean columns in two DataFrame-like objects and
    reports additions and removals of category levels.

    Only columns present in both `before` and `after` are considered.
    Category comparisons are based on the unique values observed in
    the data, not on backend-specific category definitions.

    Args:
        before (IntoFrameT):
            The DataFrame before wrangling.
        after (IntoFrameT):
            The DataFrame after wrangling.
        abbreviate (bool, optional):
            Whether to abbreviate long lists of category levels in the
            output. Defaults to True.
        max_categories (int or None, optional):
            Maximum number of category levels to display when reporting
            additions or removals. If None, truncation is based on text width.
        max_width (int, optional):
            Maximum text width (in characters) used for truncation when
            `max_categories` is None. Defaults to 80.

    Returns:
        str:
            A formatted string summarizing category-level additions and
            removals. If no category changes are detected, a message
            indicating this is returned.

    Raises:
        TypeError:
            If `before` and `after` use different DataFrame backends.

    Examples:
        >>> import py4stats as py4st
        >>> print(py4st.review_category(before, after))
        The following columns show changes in categories:
        species:
            addition:  None
            removal:  'Adelie'
        island:
            addition:  None
            removal:  'Torgersen'
    """
    before_nw = as_nw_datarame(before, arg_name = 'before')
    after_nw  = as_nw_datarame(after, arg_name = 'after')
    # 引数のアサーション =======================================================
    _assert_same_backend(before_nw, after_nw, 'review_category')

    build.assert_logical(abbreviate, arg_name = 'abbreviate')
    build.assert_count(
        max_categories, arg_name = 'max_columns', 
        len_arg = 1, nullable = True)
    build.assert_count(
        max_width, arg_name = 'max_width', 
        len_arg = 1)
    # =======================================================================

    cols1 = before_nw.select(ncs.string(), ncs.categorical(), ncs.boolean()).columns
    cols2 = after_nw.select(ncs.string(), ncs.categorical(), ncs.boolean()).columns
    cols_compare = [c for c in cols1 if c in cols2]

    change_category = ['The following columns show changes in categories:']

    args_dict = {
        'suffix':'categories', 'abbreviate':abbreviate, 
        'max_items':max_categories, 'max_width':max_width
        }

    for col in cols_compare:
        unique_before = before_nw[col].cast(nw.String).unique().to_list()
        unique_after = after_nw[col].cast(nw.String).unique().to_list()

        added = build.list_diff(unique_after, unique_before)
        removed = build.list_diff(unique_before, unique_after)

        if added or removed:
            change_category += [f"  {col}:"]
            if added:
                added_text = build.oxford_comma_shorten(added, **args_dict)
                change_category += [f"    addition: {added_text}"]
            else:
                change_category += [f"    addition:  None"]
            if removed:
                removed_text = build.oxford_comma_shorten(removed, **args_dict)
                change_category += [f"    removal:  {removed_text}"]
            else:
                change_category += [f"    removal:   None"]

    if len(change_category) > 1:
        return '\n'.join(change_category)
    else: return 'No columns had categories added or removed.'


# ### review_numeric の実験的実装

# In[ ]:


import numpy as np

def draw_ascii_boxplot(data, range_min = None, range_max = None, width = 30):
    """データセットから文字列の箱ひげ図を作成する"""
    data = as_nw_series(data)

    # 五数要約の計算
    min_val = data.min()
    q1 = data.quantile(0.25, 'midpoint')
    median = data.quantile(0.5, 'midpoint')
    q3 = data.quantile(0.75, 'midpoint')
    max_val = data.max()

    # 描画のための計算

    if range_min is not None and range_max is not None:
        data_range = range_max - range_min
    else:
        data_range = max_val - min_val
        range_min = min_val 

    if data_range == 0:
        return "Data is constant".center(width, ' ')

    def scale(val):
        return int((val - range_min) / data_range * (width - 1))

    # 文字列の箱を組み立て
    plot = [' '] * width

    # ひげ (Whiskers)
    start = scale(min_val)
    end = scale(max_val)
    q1_idx = scale(q1)
    q3_idx = scale(q3)
    med_idx = scale(median)

    for i in range(start, q1_idx): plot[i] = '-'
    for i in range(q3_idx, end + 1): plot[i] = '-'

    # 箱 (Box)
    for i in range(q1_idx, q3_idx + 1): plot[i] = '='

    # 中央値 (Median)
    plot[med_idx] = ':'

    # キャップ (Caps)
    plot[start] = '|'
    plot[end] = '|'

    return "".join(plot)


# In[ ]:


def make_boxplot_with_label(before, after, col, space_left = 7, space_right = 7, width = 30, digits = 2):
    before = as_nw_datarame(before, arg_name = 'before')
    after = as_nw_datarame(after, arg_name = 'after')

    min_be = before[col].min()
    max_be = before[col].max()
    min_af = after[col].min()
    max_af = after[col].max()
    min_all = min(min_be, min_af)
    max_all = max(max_be, max_af)

    boxplot_before = draw_ascii_boxplot(before[col], min_all, max_all, width = width)
    boxplot_after = draw_ascii_boxplot(after[col], min_all, max_all, width = width)

    review = [
        f'  {col}',
        f"{' '*6}before: {min_be:>{space_left},.{digits}f}{boxplot_before}{max_be:>{space_right},.{digits}f}",
        f"{' '*6}after:  {min_af:>{space_left},.{digits}f}{boxplot_after }{max_af:>{space_right},.{digits}f}"
    ]
    result = '\n'.join(review)

    result = '\n'.join(review)
    return result


# In[ ]:


def review_numeric(
        before: IntoFrameT, 
        after: IntoFrameT,
        digits: int = 2,
        width_boxplot: int = 30
        ):
    """
    Generate a textual review of numeric variables before and after preprocessing.

    This function compares numeric columns shared by the ``before`` and ``after``
    datasets and summarizes their distributional changes using ASCII-art
    boxplots. The output is intended for human-readable reviews (e.g., logs,
    console output, or reports), providing a compact visual reference of how
    preprocessing steps affected numeric variables.

    For each numeric column, the function displays:
    - minimum and maximum values (formatted with thousands separators),
    - an ASCII boxplot representing the five-number summary,
    - side-by-side comparison of distributions before and after preprocessing.

    The boxplots for ``before`` and ``after`` are drawn on a common scale per
    column, enabling visual comparison of shifts in location and spread.

    Args:
        before (IntoFrameT):
            The DataFrame before wrangling. Any rows consisting entirely of
            missing values are removed internally. Only numeric columns are
            considered.
        after (IntoFrameT):
            The DataFrame after wrangling.
            Only numeric columns present in both datasets are reviewed.
        digits (int, optional):
            Number of decimal places used when formatting numeric values.
            Defaults to 2.
        width_boxplot (int, optional):
            Width of the ASCII boxplot (number of characters used to draw the
            box and whiskers). Defaults to 30.

    Returns:
        str:
            A multi-line string containing an ASCII-art boxplot review of all
            numeric variables common to ``before`` and ``after``. The first line
            is a header, followed by one block per variable.

    Notes:
        - Empty or all-missing columns are removed prior to comparison.
        - This function is designed for qualitative inspection and debugging.
          The boxplots are provided for reference and are not intended as a
          precise statistical visualization.
        - If no common numeric columns exist between ``before`` and ``after``,
        a descriptive message is returned instead of boxplot output.

    Examples:
        >>> import py4stats as py4st
        >>> print(py4st.review_numeric(before, after))
        Boxplot of Numeric values (for reference):
        bill_length_mm
            before:    32.10|------======:====-----------|   59.60
            after:     40.90         |----==:===---------|   59.60
        ...
    """
    # 引数のアサーション =======================================================
    build.assert_count(digits, arg_name = 'digits', len_arg = 1)
    build.assert_count(width_boxplot, arg_name = 'width_boxplot', len_arg = 1)
    # _ = as_nw_datarame(before, arg_name = 'before')
    # _ = as_nw_datarame(after, arg_name = 'after')
    # =======================================================================

    # before_nw = remove_empty(before, to_native = False)\
    #     .select(ncs.numeric())
    # after_nw = remove_empty(after, to_native = False)\
    #     .select(ncs.numeric())

    before_nw = as_nw_datarame(before, arg_name = 'before')\
        .select(ncs.numeric())\
        .pipe(remove_empty, to_native = False)
    after_nw = as_nw_datarame(after, arg_name = 'after')\
        .select(ncs.numeric())\
        .pipe(remove_empty, to_native = False)

    cols1 = before_nw.columns
    cols2 = after_nw.columns

    cols = [x for x in cols2 if x in cols1]
    if not cols:
        return "No common numeric columns exist between `before` and `after`"

    before_nw = before_nw.select(cols)
    after_nw = after_nw.select(cols)

    max_min = max(max([
        before_nw.select(ncs.all().min()).row(0),
        after_nw.select(ncs.all().min()).row(0),
    ]))

    max_max = max(max([
        before_nw.select(ncs.all().max()).row(0),
        after_nw.select(ncs.all().max()).row(0),
    ]))

    space_left = len(f"{max_min:,.{digits}f}")
    space_right = len(f"{max_max:,.{digits}f}")

    review = [
        make_boxplot_with_label(
            before_nw, after_nw, col, 
            space_left = space_left, 
            space_right = space_right, 
            width = width_boxplot, 
            digits = digits
            )
        for col in cols
    ]

    review = ['Boxplot of Numeric values (for reference):'] + review

    return '\n'.join(review)


# In[ ]:


def make_header(text: str, title: str) -> str:
    max_len = max([wcswidth(s) for s in text.split('\n')])
    len_header = math.ceil(max_len / 2.0) * 2
    return title.center(len_header, '=')


# ### review_wrangling の本体

# In[ ]:


def review_wrangling(
        before: IntoFrameT, 
        after: IntoFrameT, 
        items: Union[List[str], str] = ['shape', 'col_addition', 'casting', 'missing', 'category'],
        title: str = 'Review of wrangling',
        abbreviate: bool = True,
        max_columns: Optional[int] = None,
        max_categories: Optional[int] = None,
        max_width: int = 80,
        ) -> str:
    """Review and summarize differences introduced by data wrangling.

    This function compares two DataFrame-like objects representing the state
    of a dataset before and after wrangling, and returns a human-readable
    review report as a formatted string.

    The review summarizes structural and content-level changes, including:
      - Changes in the shape of the DataFrame (rows and columns).
      - Added and removed columns.
      - Changes in column data types.
      - Increases and decreases in the proportion of missing values.
      - Additions and removals of levels in categorical variables.

    To maintain readability, lists of columns or categories can be abbreviated.
    Abbreviation behavior can be controlled via `abbreviate`, `max_columns`,
    `max_categories`, and `max_width`.

    Internally, the function uses `narwhals` to support multiple DataFrame
    backends (e.g. pandas, polars), while type comparisons rely on the
    original input objects.

    Args:
        before (IntoFrameT):
            The original DataFrame before wrangling.
            Any DataFrame-like object supported by narwhals
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table) can be used.
        items (Union[List[str], str], optional):
            A list specifying which review items to include in the output, and
            in what order they should appear.

            This argument may be either a list of strings or a single string.
            Each review section is generated only if its corresponding keyword
            is included in ``items``.

            Supported values are:

            - "shape": Changes in the number of rows and columns.
            - "col_addition": Added and removed columns.
            - "casting": Changes in column data types.
            - "missing": Increases and decreases in missing values.
            - "category": Additions and removals of category levels.
            - "numeric": Distributional changes in numeric variables, summarized
            using ASCII boxplots.

            Special values:
            - "all":
                Include all available review sections, including "numeric".
                This may be specified either as ``items="all"`` or by including
                "all" in a list.

            The review sections are generated in the order given in ``items``.
            If omitted, a default subset of review sections is used, excluding
            "numeric".
        after (IntoFrameT):
            The DataFrame after wrangling.
        title (str, optional):
            Title shown in the header and footer of the review output.
            If an empty string is provided, no header or footer is added.
        abbreviate (bool, optional):
            Whether to abbreviate long lists of column names or category levels
            in the output. If False, all items are shown without truncation.
            Defaults to True.
        max_columns (int or None, optional):
            Maximum number of column names to display when reporting added or
            removed columns. If specified, truncation is based on the number
            of columns only and is applied only when `abbreviate=True`.
            If None, truncation is based on text width.
        max_categories (int or None, optional):
            Maximum number of category levels to display when reporting changes
            in categorical variables. If specified, truncation is based on the
            number of categories only and is applied only when `abbreviate=True`.
            If None, truncation is based on text width.
        max_width (int, optional):
            Maximum width (in characters) used for text-based truncation when
            `abbreviate=True` and the corresponding `max_*` argument is None.
            Defaults to 80.

    Returns:
        str:
            A formatted, multi-line string summarizing the differences between
            `before` and `after`.

    Raises:
        TypeError:
            If `before` and `after` use different DataFrame backends
            (e.g. pandas vs. polars). Mixing backends is not supported.

    Examples:
        >>> import py4stats as py4st
        >>> from palmerpenguins import load_penguins
        >>> before = load_penguins()
        >>> after = before.copy().dropna()
        >>> print(py4st.review_wrangling(before, after))
            =================== Review of wrangling ====================
            The shape of DataFrame:
               Rows: before 344 -> after 333 (-11)
               Cols: before   8 -> after   8 (No change)
            ...
            ============================================================
    """

    after_nw = as_nw_datarame(after, arg_name = 'after')
    before_nw = as_nw_datarame(before, arg_name = 'before')
    if isinstance(items, str): items = [items]
    # 引数のアサーション =======================================================
    _assert_same_backend(before_nw, after_nw)

    value_items = [
         "shape", "col_addition", "casting", 
         "missing", "category", "numeric", "all"
         ]

    build.arg_match(
         items, values = value_items,
         arg_name = 'items', multiple = True
    )

    build.assert_character(title, arg_name = 'title', len_arg = 1)

    build.assert_logical(abbreviate, arg_name = 'abbreviate')

    build.assert_count(
        max_columns, arg_name = 'max_columns', 
        len_arg = 1, nullable = True)
    build.assert_count(
        max_categories, arg_name = 'max_categories', 
        len_arg = 1, nullable = True)
    build.assert_count(
        max_width, arg_name = 'max_width', 
        len_arg = 1)

    # レビューの作成と整形=========================================================
    if 'all' in items: items = value_items

    review = []

    for item in items:
        match item:
            case 'shape':
                review += [review_shape(before_nw, after_nw)]
            case 'col_addition':
                    review += [review_col_addition(
                        before_nw, after_nw, abbreviate = abbreviate, 
                        max_columns = max_columns, max_width = max_width
                        )]
            case 'casting':
                review += [review_casting(before, after)]
            case 'missing':
                review += [review_missing(before_nw, after_nw)]
            case 'category':
                    review += [review_category(
                        before_nw, after_nw, abbreviate = abbreviate, 
                        max_categories = max_categories, max_width = max_width
                    )]
            case 'numeric':
                  review += [review_numeric(before, after)]

    result = '\n\n'.join(review)
    # ヘッダーとフッターの追加
    if title:
        result = f"{make_header(result, f' {title} ')}\n{result}"
        result = f"{result}\n{make_header(result, '=')}"

    return result


# # grouped operation

# In[ ]:


GroupSplitResult = namedtuple('GroupSplitResult', ['data', 'groups'])

def group_split(
        data: IntoFrameT,
        *group_cols: Union[str, List[str], narwhals.Expr, narwhals.selectors.Selector], 
        keep: bool = True,
        drop_na_groups: bool = True,
        sort_groups: bool = True,
        to_native: bool = True, 
        ) -> Tuple[List[IntoFrameT], IntoFrameT]:
    """
    Split a data frame into a list of group-wise data frames.

    This function partitions the input data frame according to the values
    of one or more grouping columns, and returns:

    - a list of data frames, one for each group, and
    - a data frame describing the group keys corresponding to each element
      of the list.

    The behavior is conceptually similar to `dplyr::group_split()`, but is
    implemented in a backend-agnostic way using narwhals.

    Args:
        data (IntoFrameT):
            Input DataFrame. Any DataFrame type supported by narwhals
            (e.g. pandas, polars, pyarrow) can be used.
        *group_cols (Union[str, List[str], narwhals.Expr, narwhals.Selector]):
            Columns used to define groups. Each element may be:
            - a column name (`str`)
            - a list of column names
            - a narwhals expression
            - a narwhals selector
        keep (bool, optional):
            Whether to keep the grouping columns in each split data frame.
            If False, grouping columns are removed from the returned data frames.
            Defaults to True.
        drop_na_groups (bool, optional):
            Whether to drop groups with missing values in the grouping columns.
            Defaults to True.
        sort_groups (bool, optional):
            Whether to sort groups by the grouping columns before splitting.
            This ensures a stable and reproducible group order.
            Defaults to True.
        to_native (bool, optional):
            Whether to return native backend data frames.
            If False, narwhals DataFrame objects are returned.
            Defaults to True.

    Returns:
        A NamedTuple with two fields:

        - `data`:list of data frames, one per group.
        - `groups`: a data frame describing the grouping keys.

        The i-th element of `data` corresponds to the i-th row of `groups`.

    Examples:
        >>> import py4stats as py4st
        >>> from palmerpenguins import load_penguins
        >>> penguins = load_penguins()

        >>> splited = py4st.group_split(penguins, 'species', 'island')
        >>> print(len(splited.data))
        5
        >>> print([df.shape[0] for df in splited.data])
        [44, 56, 52, 68, 124]
    """
    # 引数のアサーション =======================================================
    _assert_selectors(*group_cols, arg_name = 'args')
    build.assert_logical(sort_groups, arg_name = 'sort_groups')
    build.assert_logical(drop_na_groups, arg_name = 'drop_na_groups')
    build.assert_logical(to_native, arg_name = 'to_native')
    # ========================================================================

    data_nw = as_nw_datarame(data)
    # group_cols で区別されるデータに対するグループ番号を生成・付与
    group_tab = data_nw.select(*group_cols).unique()

    cols_selected = group_tab.columns

    if drop_na_groups:
        group_tab = group_tab.drop_nulls()

    if sort_groups:
        group_tab = group_tab.sort(*group_cols)

    group_tab = group_tab.with_row_index('_group_id')

    with_group_id = data_nw.join(group_tab, on = cols_selected, how = 'left')

    if not keep:
        with_group_id = with_group_id.drop(cols_selected)

    # データセットをグループ番号別に分割
    list_dfs = [
        with_group_id.filter(nw.col('_group_id') == g).drop('_group_id')
        for g in group_tab['_group_id']
        ]
    if to_native:
        list_dfs = [df.to_native() for df in list_dfs]
        group_tab = group_tab.to_native()

    result = GroupSplitResult(data = list_dfs, groups = group_tab)

    return result


# In[ ]:


GroupMapResult = namedtuple('GroupSplitResult', ['mapped', 'groups'])

def group_map(
        data: IntoFrameT,
        *group_cols: Union[str, List[str], narwhals.Expr, narwhals.selectors.Selector], 
        func: Callable[[IntoFrameT], Any], 
        drop_na_groups: bool = True,
        sort_groups: bool = True
        ) -> Tuple[List[IntoFrameT], IntoFrameT]:
    """
    Apply a function to each group and return the results without combining them.

    This function splits the input data frame into groups (using
    `group_split()`), applies a function to each group, and returns the
    mapped results along with the corresponding group information.

    Unlike `group_modify()`, this function does not attempt to combine
    the results into a single data frame. It is intended for cases where
    maximum flexibility is desired.

    This function is conceptually similar to `dplyr::group_map()`.

    Args:
        data (IntoFrameT):
            Input DataFrame. Any DataFrame type supported by narwhals
            (e.g. pandas, polars, pyarrow) can be used.
        *group_cols (Union[str, List[str], narwhals.Expr, narwhals.Selector]):
            Columns used to define groups. Each element may be:
            - a column name (`str`)
            - a list of column names
            - a narwhals expression
            - a narwhals selector
        func (callable):
            A function that takes a data frame as its argument.
            The function is applied separately to each group.
        drop_na_groups (bool, optional):
            Whether to drop groups with missing values in the grouping columns.
            Defaults to True.
        sort_groups (bool, optional):
            Whether to sort groups by the grouping columns before splitting.
            Defaults to True.

    Returns:
        A NamedTuple with two fields:

        - `mapped`: a list containing the result of applying `func`
          to each group.
        - `groups`: a data frame describing the grouping keys.

        The i-th element of `mapped` corresponds to the i-th row of `groups`.

    Note:
        `func` can return any Python object. The results are not combined
        or coerced into a tabular structure. This makes `group_map()`
        suitable for use cases such as returning model objects,
        plots, or other non-tabular results.

    Examples:
        >>> import py4stats as py4st
        >>> from palmerpenguins import load_penguins
        >>> penguins = load_penguins()

        >>> res = py4st.group_map(penguins, "species", func=lambda df: df.shape[0])
        >>> res.mapped
        [152, 68, 124]
        >>> res.groups
    """
    # 引数のアサーション =======================================================
    _assert_selectors(*group_cols, arg_name = 'args')
    build.assert_function(func, arg_name = 'func', len_arg = 1)
    build.assert_logical(drop_na_groups, arg_name = 'drop_na_groups')
    build.assert_logical(sort_groups, arg_name = 'sort_groups')
    # ========================================================================
    list_dfs, group_tab = group_split(
        data, *group_cols, 
        sort_groups = sort_groups,
        drop_na_groups = drop_na_groups,
        to_native = True,
        )

    mapped = [func(df) for df in list_dfs]

    result = GroupMapResult(mapped = mapped, groups = group_tab)
    return result


# In[ ]:


def group_modify(
        data: IntoFrameT,
        *group_cols: Union[str, List[str], narwhals.Expr, narwhals.selectors.Selector], 
        func: Callable[[IntoFrameT], Union[IntoFrameT, IntoSeriesT, int, float, bool, str, None]],
        drop_na_groups: bool = True,
        sort_groups: bool = True,
        to_native: bool = True
        ) -> IntoFrameT:
    """
    Apply a function to each group and combine the results into a data frame.

    This function splits the input data frame into groups, applies a function
    to each group, and combines the results into a single data frame by
    attaching the grouping variables.

    The function supplied to `func` is always called with a *native*
    data frame (e.g. pandas or polars), so existing analysis functions
    can be reused without modification.

    This function is conceptually similar to `dplyr::group_modify()`.

    Args:
        data:
            A data-frame-like object supported by narwhals.
        *group_cols:
            Columns used to define groups.
        *group_cols (Union[str, List[str], narwhals.Expr, narwhals.Selector]):
            Columns used to define groups. Each element may be:
            - a column name (`str`)
            - a list of column names
            - a narwhals expression
            - a narwhals selector
        func (callable):
            A function that takes a data frame as its argument.
            The function is applied separately to each group.
        drop_na_groups (bool, optional):
            Whether to drop groups with missing values in the grouping columns.
            Defaults to True.
        sort_groups (bool, optional):
            Whether to sort groups by the grouping columns before splitting.
            Defaults to True.
        to_native (bool, optional):
            If True, convert the result to the native DataFrame type of the
            selected backend. If False, return a narwhals DataFrame.
            Defaults to True.

    Returns:
        A data frame where the results of `func` are combined and
        augmented with the grouping columns.

    Note:
        The function supplied to `func` must return a value that can be
        coerced into a data frame. Supported return types include:

        - data frame-like objects
        - Series-like objects
        - scalar values (int, float, bool, str)
        - None

        Objects that cannot be converted into a tabular structure
        (e.g., fitted model objects or plot objects) are not supported.
        For such use cases, consider using `group_map()` instead.

    Examples:
        >>> import py4stats as py4st
        >>> from palmerpenguins import load_penguins
        >>> penguins = load_penguins()

        >>> result = group_modify(
        ...     penguins, 'species', 'island',
        ...     func = lambda df: df[['bill_length_mm']].mean()
        ... )
        >>> result.head()
    """
    # 引数のアサーション =======================================================
    _assert_selectors(*group_cols, arg_name = 'args')
    build.assert_function(func, arg_name = 'func', len_arg = 1)
    build.assert_logical(drop_na_groups, arg_name = 'drop_na_groups')
    build.assert_logical(sort_groups, arg_name = 'sort_groups')
    build.assert_logical(to_native, arg_name = 'to_native')

    # 実装メモ ================================================================
    # <3> における結合処理の一貫性と、<2> で pd.DataFrame など native class の 
    # DataFrame と同じメソッドが使えることを両立するために、<1> で `to_native = False`
    # を指定し、<2> で `df.to_native()` を使っています。
    # ========================================================================
    impl = as_nw_datarame(data).implementation

    list_dfs, group_tab = group_split(
        data, *group_cols, 
        sort_groups = sort_groups,
        drop_na_groups = drop_na_groups,
        to_native = False,                                   # <1>
        ) # -> Tuple[List[nw.DataFrame], nw.DataFrame]

    list_result1 = [func(df.to_native()) for df in list_dfs] # <2>

    list_result2 = [
        nw.from_native(v).with_columns(nw.lit(id).alias('_group_id')) 
        if is_intoframe(v) 
        else enframe(v, backend = impl, to_native = False)
            .with_columns(nw.lit(id).alias('_group_id')) 
        for v, id in zip(list_result1, group_tab['_group_id'])
        ] # -> List[nw.DataFrame]

    result = group_tab.join(                                  # <3>
            nw.concat(list_result2),
            on = '_group_id',
            how = 'left'
        ).drop('_group_id')

    if to_native: return result.to_native()
    return result


# ## bind_rows

# In[ ]:


def _assert_unique_backend(args, arg_name: str = 'args'):
    if build.length(args) <= 1: return None

    unique_type = build.list_unique(
        df.implementation for df in args
        )

    if build.length(unique_type) > 1:
        type_text = build.oxford_comma_and(unique_type)
        message = f"Elements of `{arg_name}` must share the same backend, got {type_text}." 
        raise TypeError(message)


# In[ ]:


@singledispatch
def bind_rows(
        *args: Union[IntoFrameT, List[IntoFrameT], Mapping[str, IntoFrameT]], 
        names: Optional[Sequence[Union[str, int, float, bool]]] = None,
        id: str = 'id', to_native: bool = True,
        **keywargs
) -> IntoFrameT:
    """Row-bind (concatenate) multiple data frames while optionally preserving provenance.

    This function concatenates data frames vertically (row-wise) and can optionally
    add an identifier column that indicates which input each row came from, similar
    to `dplyr::bind_rows()` with the `.id` argument.

    Inputs can be provided as:

    - Multiple data frames: `bind_rows(df1, df2, ...)`
    - A list/tuple of data frames: `bind_rows([df1, df2, ...])`
    - A mapping of name -> data frame: `bind_rows({"a": df1, "b": df2, ...})`

    When `id` is not None, an identifier column is added:

    - If `names` is provided (data frames / list input only), its elements are used.
    - If `names` is None (data frames / list input), `range(n)` is used.
    - If a mapping is provided, mapping keys are used (and `names` is ignored if given).

    Concatenation is performed with `how="diagonal"` (i.e., union of columns).
    Columns missing in an input are created and filled with nulls.


    Args:
        *args:
            One of the supported input forms:
            - One or more data frames.
            - A single list/tuple of data frames.
            - A single mapping (e.g., dict) whose values are data frames.

            Each data frame must be a DataFrame supported by narwhals 
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table).
        names (list of str or int or float or bool, optional):
            Optional list of identifier values to attach to each input data frame.
            This is only valid when `*args` are data frames or a list/tuple of data
            frames. If omitted (`None`), identifiers default to `range(n)`.

            `names` must satisfy (when not None):
            - All elements must share the same type.
            - The element type must be one of: `str`, `int`, `float`, or `bool`.
            - Length must match the number of input data frames.

            This argument is only meaningful for the data frame / list input forms.
            If a mapping is provided, mapping keys are used instead.

        id (str, optional):
            Name of the identifier column to add. If `None`, no identifier column
            is created.
        to_native (bool, optional):
            If True, convert the result to the native DataFrame type of the
            selected backend. If False, return a narwhals DataFrame.
            Defaults to True.
        **keywargs:
            Extra keyword arguments reserved for forward compatibility.
            (Currently ignored.)

    Returns:
        A row-bound DataFrame. The return type depends on `to_native`:
        - If `to_native=True`, a native DataFrame is returned.
        - If `to_native=False`, a narwhals DataFrame is returned.

    Raises:
        NotImplementedError:
            If called with unsupported input types for `*args`.
        TypeError:
            If `id` is not a string or `None`, or if `names` / mapping keys contain
            invalid element types (not in `str`, `int`, `float`, `bool`) or contain
            mixed types.
        TypeError:
            If input data frames (including values of a mapping input)
            are backed by different backends (e.g., mixing pandas and
            polars). All inputs must share the same backend.
            Mixing backends is not supported.

        ValueError:
            If `names` is provided but its length does not match the number of
            input data frames.

    Examples:
        Basic usage (positional inputs):

            >>> py4st.bind_rows(df1, df2)

        Provide explicit names:

            >>> py4st.bind_rows(df1, df2, names=["train", "test"])

        Use a custom id column name:

            >>> py4st.bind_rows([df1, df2], names=[2020, 2021], id="year")

        Use a mapping (keys become identifiers):

            >>> py4st.bind_rows({"table1": df1, "table2": df2})

        No identifier column:

            >>> py4st.bind_rows(df1, df2, id=None)
    """
    raise NotImplementedError(f'bind_rows mtethod for object {type(args)} is not implemented.')


# In[ ]:


@bind_rows.register(Union[nw.typing.IntoDataFrame, list])
def bind_rows_df(
        *args: Union[IntoFrameT, List[IntoFrameT]],
        names: Optional[Sequence[Union[str, int, float, bool]]] = None,
        id: str = 'id', to_native: bool = True,
        **keywargs
        )-> IntoFrameT:
    # 引数のアサーション =======================================================
    args = list(build.list_flatten(args))
    args = as_nw_datarame_list(args, arg_name = 'args')
    _assert_unique_backend(args)
    build.assert_character(id, arg_name = 'id', len_arg = 1, nullable = True)
    build.assert_same_type(names, arg_name = 'names')
    build.assert_literal(names, arg_name = 'names', nullable = True)
    # =======================================================================
    if id is None: 
        result = nw.concat(args, how = "diagonal")
        if to_native: return result.to_native()
        return result

    if names is None: names = range(len(args))

    else: build.assert_length(names, arg_name = 'names', len_arg = len(args))

    tabl_list = [
        df.with_columns(nw.lit(key).alias(id))
        for key, df in zip(names, args)
    ]
    result = nw.concat(tabl_list, how = "diagonal")\
        .pipe(relocate, id, to_native = to_native)
    return result


# In[ ]:


@bind_rows.register(dict)
def bind_rows_dict(
    args: Mapping[str, IntoFrameT], 
    id: str = 'id', 
    to_native: bool = True, 
    **keywargs
    )-> IntoFrameT:
    # 引数のアサーション =======================================================
    build.assert_literal_kyes(args, arg_name = 'args')
    args = as_nw_datarame_dict(args, arg_name = 'args')
    _assert_unique_backend(args.values())
    build.assert_character(id, arg_name = 'id', len_arg = 1, nullable = True)
    # =======================================================================
    if id is None: 
        result = nw.concat(args.values(), how = "diagonal")
        if to_native: return result.to_native()
        return result

    tabl_list = [
        df.with_columns(nw.lit(key).alias(id))
        for key, df in zip(args.keys(), args.values())
    ]
    result = nw.concat(tabl_list, how = "diagonal")\
        .pipe(relocate, id, to_native = to_native)
    return result

