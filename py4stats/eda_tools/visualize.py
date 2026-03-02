#!/usr/bin/env python
# coding: utf-8



from __future__ import annotations


# # `eda_tools` の可視化関数



from .. import building_block as build
from . import _utils as eda_utils
from . import operation as eda_ops

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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


# ## 欠測値の可視化



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

    diagnose_tab = eda_ops.diagnose(data, to_native = False)
    
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


# ## パレート図を作図する関数



def make_rank_table(
    data: IntoSeriesT,
    group: str,
    values: str,
    aggfunc: Callable[[IntoSeriesT], Union[int, float]] = np.mean,
    to_native: bool = True,
) -> pd.DataFrame:
    data_nw = eda_utils.as_nw_datarame(data)

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
        stat = eda_ops.group_map(
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
    data_nw = eda_utils.as_nw_datarame(data)
    # 指定された変数でのランクを表すデータフレームを作成
    if values is None:
        shere_rank = eda_ops.freq_table(
            data_nw, group, dropna = True, 
            sort_by = 'frequency',
            descending = True, 
            to_native = False
            )
        cumlative = 'cumperc'
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


# ## グループ間の統計値の差異の可視化



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
    group_means = eda_ops.compare_group_means(
        group1, group2, 
        columns = 'common',
        to_native = False
        ).to_pandas().set_index('variable')

    if ax is None:
        fig, ax = plt.subplots()

    ax.stem(group_means[stats_diff], orientation = 'horizontal', basefmt = 'C7--');

    ax.set_yticks(range(len(group_means.index)), group_means.index)

    ax.invert_yaxis();




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

    group_median = eda_ops.compare_group_median(
        group1, group2, 
        columns = 'common',
        to_native = False
        ).to_pandas().set_index('variable')

    if ax is None:
        fig, ax = plt.subplots()

    ax.stem(group_median[stats_diff], orientation = 'horizontal', basefmt = 'C7--')
    ax.set_yticks(range(len(group_median.index)), group_median.index)
    ax.invert_yaxis();


# # カテゴリー変数の水平棒グラフ



def plot_count_h(
        data: IntoFrameT,
        valiable: str,
        sort_by: Literal['frequency', 'values'] = 'frequency',
        descending: bool = False,
        dropna: bool = False,
        color = "#478FCE",
        fontsize: int = 12,
        padding: int = 3,
        digits: int = 1,
        expand_x_limits: Union[float, int] = 1.5,
        ax: Optional[Axes] = None,
        ):
    """Plot a horizontal bar chart of frequency counts for a categorical variable.

    This function computes a frequency table for the specified categorical
    variable and visualizes it as a horizontal bar chart. Both absolute
    frequency and relative frequency (percentage) are displayed as labels
    on each bar.

    The frequency table is internally computed using `eda_ops.freq_table()`,
    with missing values excluded by default.

    Args:
        data (IntoFrameT):
            Input DataFrame containing categorical variables (one column per item).
            Any DataFrame type supported by narwhals can be used
            (e.g., pandas.DataFrame, polars.DataFrame, pyarrow.Table).
        valiable (str):
            Column name of the categorical variable to be summarized.
        sort_by (Literal['frequency', 'values']):
            Sorting rule for the output table.
            - 'frequency': sort by frequency. (default)
            - 'values': sort by the category values.
        descending (bool, optional):
            Whether to sort in descending order. Defaults to False.
                dropna (bool):
            Whether to drop NaN from counts. Defaults to False.
        color (str, optional):
            Color of the bars. Accepts any Matplotlib-compatible color
            specification. Defaults to "#478FCE".
        fontsize (int, optional):
            Font size of the bar labels. Defaults to 12.
        padding (int, optional):
            Distance (in points) between the bar and its label.
            Passed to `ax.bar_label()`. Defaults to 3.
        digits (int, optional):
            Number of decimal places for percentage labels.
            Defaults to 1.
        expand_x_limits (float, optional):
            Multiplicative factor used to expand the upper limit of the
            x-axis to ensure sufficient space for labels.
            Defaults to 1.5.
        ax (Optional[Axes], optional):
            Existing Matplotlib Axes object to draw the plot on.
            If None, a new figure and axes are created. Defaults to None.

    Returns:
        matplotlib.axes.Axes:
            The Matplotlib Axes object containing the plot.

    Raises:
        ValueError:
            If `sort_by` is not one of the allowed values.
        TypeError:
            If input arguments do not satisfy the required type constraints.

    Examples:
        >>> import py4stats as p4s
        >>> from palmerpenguins import load_penguins
        >>> penguins = load_penguins()
        >>> ax = p4s.plot_count_h(penguins, valiable="species")
    """
    # 引数のアサーション ========================================
    sort_by = build.arg_match(
        sort_by, arg_name = 'sort_by',
        values = ['frequency', 'values']
        )
    build.assert_scalar(valiable, arg_name = 'valiable')
    build.assert_logical(descending, arg_name = 'descending', len_arg = 1)
    build.assert_logical(dropna, arg_name = 'dropna', len_arg = 1)
    build.assert_count(fontsize, arg_name = 'fontsize', len_arg = 1)
    build.assert_count(padding, arg_name = 'padding', len_arg = 1)
    build.assert_count(digits, arg_name = 'digits', len_arg = 1)
    build.assert_numeric(expand_x_limits, arg_name = 'expand_x_limits', len_arg = 1)
    # =========================================================

    count_table = eda_ops.freq_table(
                data, valiable, 
                dropna = dropna, 
                sort_by = sort_by,
                descending = descending,
                to_native = False
                )
    if not dropna:
        count_table = count_table.with_columns(nw.col(valiable).cast(nw.String))
        count_table = eda_ops.replace_na(count_table, {valiable: 'NA'}, to_native = False)
    # 度数(相対頻度%) のラベルを作成
    label_list = [
        f'{freq:,} ({perc:.{digits}%})' for freq, perc in 
        zip(count_table['freq'], count_table['perc'])
    ]
    
    count_table = eda_utils.assign_nw(
        count_table, 
        label = label_list
        )
    
    if ax is None:
        fig, ax = plt.subplots()

    bars = ax.barh(count_table[valiable], count_table['freq'], color = color)
    
    # ここでラベルを作図
    ax.bar_label(
        bars, labels = count_table['label'], 
        padding = padding, 
        fontsize = fontsize
        );

    ax.set_xlim(0, count_table['freq'].max() * expand_x_limits)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: '{:,}'.format(int(x)))
        )
    ax.set_xlabel('frequency');


# # カテゴリー変数の積み上げ棒グラフ



def make_table_to_plot(
        data: IntoFrameT, 
        sort_by: Literal['values', 'frequency'] = 'values',
        to_native: bool = True
        ) -> None:
    data_nw = eda_utils.as_nw_datarame(data)
    
    variables = data_nw.columns
    def foo(v):
        res_ft = eda_ops.freq_table(
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
    table_to_plot = eda_ops.relocate(table_to_plot, nw.col('variable'), to_native = False)
    if to_native: return table_to_plot.to_native()
    return table_to_plot




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
    data_nw = eda_utils.as_nw_datarame(data)
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

