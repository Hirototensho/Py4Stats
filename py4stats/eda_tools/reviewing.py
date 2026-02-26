#!/usr/bin/env python
# coding: utf-8



from __future__ import annotations




from .. import building_block as build
from . import _utils as eda_utils
from . import operation as eda_ops

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


# ### `review_wrangling()`



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
    res_compare = eda_ops.compare_df_cols(
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
    columns_before = eda_utils.as_nw_datarame(before, arg_name = 'before').columns
    columns_after  = eda_utils.as_nw_datarame(after, arg_name = 'after').columns

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
    compare_miss = eda_ops.diagnose(before, to_native = False)\
        .select('columns', 'missing_count', 'missing_percent')\
        .join(
            eda_ops.diagnose(after, to_native = False)\
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




def shape_change(before: int, after: int) -> str:
    if after > before: return f" (+{after - before:,})"
    if after < before: return f" ({after - before:,})"
    return f" (No change)"




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
    before_nw = eda_utils.as_nw_datarame(before, arg_name = 'before')
    after_nw  = eda_utils.as_nw_datarame(after, arg_name = 'after')
    row_o, col_o = before_nw.shape
    row_n, col_n = after_nw.shape
    d_o = len(f"{np.max([row_o, col_o]):,}")
    d_n = len(f"{np.max([row_n, col_n]):,}")

    shpe_message = f"The shape of DataFrame:\n" + \
                f"   Rows: before {row_o:>{d_o},} -> after {row_n:>{d_n},}{shape_change(row_o, row_n)}\n" + \
                f"   Cols: before {col_o:>{d_o},} -> after {col_n:>{d_n},}{shape_change(col_o, col_n)}"
    return shpe_message




def _assert_same_backend(data1, data2, funcname = 'review_wrangling', data_name = ['before', 'after']):
      if data1.implementation is not data2.implementation:
        raise TypeError(
            f"{funcname}() requires `{data_name[0]}` and `{data_name[1]}`` to use the same backend.\n"
            f"Got {data_name[0]}={data1.implementation!r}, {data_name[1]}={data2.implementation!r}.\n"
            f"Please make sure that  `{data_name[0]}` and `{data_name[1]}` are the same backend (e.g., both pandas, both polars)."
        )




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
    before_nw = eda_utils.as_nw_datarame(before, arg_name = 'before')
    after_nw  = eda_utils.as_nw_datarame(after, arg_name = 'after')
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



import numpy as np

def draw_ascii_boxplot(data, range_min = None, range_max = None, width = 30):
    """データセットから文字列の箱ひげ図を作成する"""
    data = eda_utils.as_nw_series(data)

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




def make_boxplot_with_label(before, after, col, space_left = 7, space_right = 7, width = 30, digits = 2):
    before = eda_utils.as_nw_datarame(before, arg_name = 'before')
    after = eda_utils.as_nw_datarame(after, arg_name = 'after')

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
    # =======================================================================

    before_nw = eda_utils.as_nw_datarame(before, arg_name = 'before')\
        .select(ncs.numeric())\
        .pipe(eda_ops.remove_empty, to_native = False)
    after_nw = eda_utils.as_nw_datarame(after, arg_name = 'after')\
        .select(ncs.numeric())\
        .pipe(eda_ops.remove_empty, to_native = False)

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




def make_header(text: str, title: str) -> str:
    max_len = max([wcswidth(s) for s in text.split('\n')])
    len_header = math.ceil(max_len / 2.0) * 2
    return title.center(len_header, '=')


# ### review_wrangling の本体



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

    after_nw = eda_utils.as_nw_datarame(after, arg_name = 'after')
    before_nw = eda_utils.as_nw_datarame(before, arg_name = 'before')
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


# ## 簡易なデータバリデーションツール



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
    data_nw = eda_utils.as_nw_datarame(data)
    data_pd = data_nw.to_pandas()
    col_names = data_nw.columns
    N = data_nw.shape[0]

    result_list = []
    for name, rule in rule_dict.items():

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
    data_nw = eda_utils.as_nw_datarame(data)
    data_pd = data_nw.to_pandas()
    value_impl = data_nw.implementation
    N = data_nw.shape[0]
    col_names = data_nw.columns

    result_dict = dict()

    for name, rule in rule_dict.items():
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



def implies_exper(P, Q):
  return f"{Q} | ~({P})"

@singledispatch
def is_complete(data: pd.DataFrame) -> pd.Series:
  return data.notna().all(axis = 'columns')

@is_complete.register(pd.Series)
def _(*arg: pd.Series) -> pd.Series:
  return pd.concat(arg, axis = 'columns').notna().all(axis = 'columns')




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

