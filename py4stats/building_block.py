#!/usr/bin/env python
# coding: utf-8



from __future__ import annotations


# # `building_block`: `py4stats` のプログラミングを補助する関数群
# 
# `py4stats` ライブラリの実装に使用するアサーション関数やユーティリティ関数を提供します。
# `building_block` モジュール自体は外部から呼び出すことなく内部実装に使用することを想定しています。

# ## 関数の依存関係 
# 
# `building_block` モジュールに実装された主要な関数の依存関係
# 
# ```python
# ## 1. 文字列列挙（Oxford comma）
# 
# oxford_comma()                    # 文字列（列）を英文の並列表現に整形
# ├─ oxford_comma_and()             # "and" を使った並列表現
# │  └─ oxford_comma()
# └─ oxford_comma_or()              # "or" を使った並列表現
#    └─ oxford_comma()
# 
# ## 2. 長さ・スカラー性の基本ユーティリティ
# length()                          # 引数の「要素数」を汎用的に取得
# 
# assert_length()                   # 引数の長さ制約（len, min/max）を検証
# └─ length()
# 
# assert_scalar()                   # 配列的オブジェクトでないことを検証
# 
# ## 3. 欠測値（None / NA / Null）関連
# is_pl_null()                      # polars の Null 判定（存在すれば）
# 
# is_missing()                      # pandas / polars を含む欠測値判定
# ├─ is_pl_null()
# └─ (pd.Series.isna)
# 
# assert_missing()                  # 欠測値の許容ルールを検証
# ├─ is_missing()
# └─ oxford_comma_and()             # エラーメッセージ用
#    └─ oxford_comma()
# 
# ## 4. 選択式引数（match.arg 系）
# 
# match_arg()                       # 単一文字列の部分一致マッチ（R風）
# ├─ oxford_comma_or()
# │  └─ oxford_comma()
# └─ (部分一致ロジック)
# 
# arg_match0()                      # 単一要素専用の厳密マッチ＋候補提示
# ├─ varname.argname()              # 引数名の自動取得
# └─ oxford_comma_or()
#    └─ oxford_comma()
# 
# arg_match()                       # 単数 / 複数対応の高水準マッチ関数
# ├─ varname.argname()
# ├─ assert_missing()               # 欠測値の扱いを統一
# │  └─ is_missing()
# ├─ is_missing()                   # any_missing 時のフィルタ
# └─ arg_match0()
# 
# ## 5. dtype 判定（predicate 関数）
# 
# is_character()                    # 文字列型かどうか
# is_logical()                      # bool 型かどうか
# is_numeric()                      # 数値型かどうか
# is_integer()                      # 整数型かどうか
# is_float()                        # 浮動小数点型かどうか
# ### （いずれも **assert_* 系の材料**）
# 
# ## 6. 汎用 assert ファクトリ（型のみ）
# 
# make_assert_type()                # 型チェック用 assert_* を生成
# └─ func()                         # 実体：複合アサーション
#    ├─ assert_scalar()             # scalar_only 制約
#    ├─ assert_missing()            # 欠測値制約
#    ├─ assert_length()             # 要素数制約
#    ├─ is_missing()                # 欠測除外（any_missing）
#    └─ oxford_comma_or()           # エラーメッセージ整形
# 
# assert_character()                # 文字列型アサーション
# └─ make_assert_type(is_character)
# 
# assert_logical()                  # bool 型アサーション
# └─ make_assert_type(is_logical)
# 
# ## 7. 数値用 assert ファクトリ（範囲チェック込み）
# 
# make_assert_numeric()             # 数値＋範囲チェック用 assert_* を生成
# └─ func()
#    ├─ assert_scalar()             # scalar_only 制約
#    ├─ assert_missing()            # 欠測値制約
#    ├─ assert_length()             # 要素数制約
#    ├─ is_missing()                # 欠測除外（any_missing）
#    ├─ oxford_comma_or()           # 型エラー文
#    └─ oxford_comma_and()          # 位置つき範囲エラー文
# 
# assert_numeric()                  # 数値型（int / float）＋範囲
# └─ make_assert_numeric(is_numeric)
# 
# assert_integer()                  # 整数型アサーション
# └─ make_assert_numeric(is_integer)
# 
# assert_count()                    # 非負整数（カウント用）
# └─ make_assert_numeric(is_integer)
# 
# assert_float()                    # float 専用アサーション
# └─ make_assert_numeric(is_float)
# 
# ## 8. p-value / 数値スタイリング
# 
# ### p-value 表現
# 
# p_stars()                         # p値を有意水準の★表記に変換
# ├─ assert_numeric()               # p値の妥当性検証
# └─ pd.cut()                       # 区間分類
# 
# style_pvalue()                    # p値を文字列表現に整形
# ├─ assert_numeric()               # p値の範囲検証
# ├─ assert_count()                 # 小数点桁数検証
# └─ (mask / round / astype)
# 
# ### 数値フォーマット（vectorize）
# 
# num_comma()                       # 桁区切り付き数値フォーマット
# ├─ assert_count()
# └─ arg_match()                    # 区切り記号の検証
# 
# num_currency()                    # 通貨表記フォーマット
# ├─ assert_count()
# └─ arg_match()
# 
# num_percent()                     # パーセント表記（単純）
# 
# ### pandas.Series ベース整形
# 
# style_number()                    # 数値列の桁区切り整形
# ├─ assert_numeric()
# ├─ assert_count()
# └─ arg_match()
# 
# style_currency()                  # 数値列の通貨表記
# ├─ assert_numeric()
# ├─ assert_count()
# └─ arg_match()
# 
# style_percent()                   # 数値列の百分率表記
# ├─ assert_numeric()
# └─ assert_count()
# 
# ## 9. その他ユーティリティ
# 
# pad_zero()                        # 小数点以下のゼロ埋め
# add_big_mark()                    # 3桁区切りの追加
# ```



import pandas as pd
import numpy as np
import scipy as sp
# from varname import argname
import varname
import collections
import textwrap
# from functools import partial


# ## 型ヒントの準備



from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
    Literal,
    overload,
)

T = TypeVar("T")
StrSeq = list[str]
StrOrStrSeq = Union[str, StrSeq]

# match系の戻り（single / multiple）
MatchArgReturn = Union[str, List[str]]

Inclusive = Literal["both", "neither", "left", "right"]

# 数値っぽい入力を広めに許容（pandas Series / numpy array / Python number）
NumberLike = Union[int, float, np.number]
ArrayLike = Union[NumberLike, Sequence[NumberLike], np.ndarray, pd.Series]

# 文字列配列っぽい入力
# StrArrayLike = Union[str, list[str], np.ndarray, pd.Series]

# p-value などは 0..1 の数値配列として扱うことが多い
# Sequence[int, float, np.number] = ArrayLike


# ## `oxford_comma()`
# 
# 文字列のリストを与えると、英文の並列の形に変換する関数です。表記法については[Wikipedia Serial comma](https://en.wikipedia.org/wiki/Serial_comma)を参照し、コードについては[stack overflow:Grammatical List Join in Python [duplicate]](https://stackoverflow.com/questions/19838976/grammatical-list-join-in-python)を参照しました。
# 
# ```python
# choices = ['apple', 'orange', 'grape']
# oxford_comma_or(choices)
# #> 'apple, orange or grape'
# ```



def oxford_comma(x: Union[str, List[str]], sep_last: str = "and", quotation: bool = True) -> str:
    """Format a string or list of strings using an Oxford-comma style.

    This utility converts a single string or a list of strings into a
    human-readable textual representation, such as:
    `'a', 'b' and 'c'` or `'a', 'b' or 'c'`.

    It is primarily intended for constructing clear error messages or
    user-facing text that enumerates valid options.

    The function also provides two convenience wrappers:
    `oxford_comma_and()` and `oxford_comma_or()`, which fix the final
    separator to ``"and"`` and ``"or"``, respectively.

    Args:
        x (Union[str, List[str]]):
            A single string or a list of strings to be formatted.
        sep_last (str, optional):
            The conjunction placed before the final element.
            Typical values are ``"and"`` or ``"or"``. Defaults to ``"and"``.
        quotation (bool, optional):
            Whether to wrap each element in single quotes. Defaults to ``True``.

    Returns:
        str:
            A formatted string using Oxford-comma conventions.

    Examples:
        >>> from py4stats import building_block as build
        >>> build.oxford_comma(["a", "b", "c"])
        "'a', 'b' and 'c'"

        >>> build.oxford_comma(["a", "b", "c"], sep_last="or")
        "'a', 'b' or 'c'"

        >>> build.oxford_comma("a")
        "'a'"

        >>> build.oxford_comma_and(["x", "y"])
        "'x' and 'y'"

        >>> build.oxford_comma_or(["x", "y"])
        "'x' or 'y'"
    """
    if isinstance(x, str):
      if(quotation): return f"'{x}'"
      else: return x
    else:
      if(quotation): x = [f"'{s}'" for s in x]
    if(len(x) == 1):
      return f"{x[0]}"
    else:
      return ", ".join(x[:-1]) + f" {sep_last} " + x[-1]




def oxford_comma_and(x: Union[str, Sequence[str]], quotation: bool = True) -> str:
  return oxford_comma(x, quotation = quotation, sep_last = 'and')

def oxford_comma_or(x: Union[str, Sequence[str]], quotation: bool = True) -> str:
  return oxford_comma(x, quotation = quotation, sep_last = 'or')

oxford_comma_and.__doc__ = oxford_comma.__doc__
oxford_comma_or.__doc__ = oxford_comma.__doc__




def oxford_comma_shorten(
      x: Union[str, List[str]],
      sep_last: str = 'and', 
      quotation: bool = True, 
      suffix: str = 'items', 
      max_items: Optional[int] = None,
      max_width: int = 80,
      abbreviate: bool = True
      ) -> str:
    """Format a list of strings using Oxford-comma style with optional abbreviation.

    This function is a wrapper around :func:`oxford_comma` that additionally
    abbreviate the resulting text if it exceeds a given character width.
    When shortening occurs, the output indicates how many items were omitted,
    e.g. ``"and other 10 items"``.

    It is designed for use in contexts such as error messages or warnings,
    where overly long lists would reduce readability.

    Args:
        x (List[str]):
            A list of strings to be formatted.
        sep_last (str, optional):
            The conjunction used before the last visible item.
            Defaults to ``'and'``.
        quotation (bool, optional):
            Whether to wrap each item in single quotes.
            Defaults to ``True``.
        suffix (str, optional):
            A label describing the omitted elements.
            Defaults to ``'items'``.
        max_items (int, optional):
            Maximum number of items to display when `abbreviate=True`.
            If specified and smaller than the number of items, the output will
            show the first `max_items` items and then append 
            "and other N {suffix}".
            If None, truncation is based on text width.
        max_width (int, optional):
            Maximum allowed character width of the output string.
            Defaults to ``80``.
        abbreviate (bool, optional):
            Whether shortening is allowed when the text exceeds ``max_width``.
            If ``False``, the full text is always returned.
            Defaults to ``True``.

    Returns:
        str:
            A formatted (and possibly shortened) string representation
            of the input list.

    Examples:
        >>> from py4stats import building_block as build
        >>> import string
        >>> alpha = list(string.ascii_lowercase)

        >>> build.oxford_comma_shorten(alpha)
        "'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l' and other 14 items"

        >>> build.oxford_comma_shorten(alpha, max_width=40)
        "'a', 'b', 'c', 'd' and other 22 items"

        >>> build.oxford_comma_shorten(alpha[:10])
        "'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i' and 'j'"
    """
    if isinstance(x, str): return x
    n_items = len(x)

    item_text = oxford_comma(x, sep_last = sep_last, quotation = quotation)

    if not abbreviate: return item_text

    # 省略処理 =================================================================
    if max_items is None:
        item_text = textwrap.shorten(
            item_text, 
            width = max_width - len(f"and other {len(x)} {suffix}"), 
            placeholder = ''
            )
        in_text = [s for s in x if str(s) in item_text]
        n_remain = n_items - len(in_text)

    elif max_items < n_items:
        if(quotation): x = [f"'{s}'" for s in x]
        item_text = ', '.join(x[:max_items])
        n_remain = n_items - max_items

    elif max_items >= n_items: return item_text

    if n_remain >= 1:
        return f"{item_text} {sep_last} other {n_remain} {suffix}"\
            .replace(f', {sep_last} other', f' {sep_last} other')

    return item_text


# ## 引数の要素数に関するアサーション



def length(x):
    if x is None:
        return 0
    if isinstance(x, str):
        return 1
    if isinstance(x, collections.abc.Sized):
        return len(x)       # list, dict, pandas.Series, etc.
    return 1




# def assert_length(
#         arg: Any, 
#         arg_name: str,
#         len_arg: Optional[int] = None,
#         len_min: int = 1,
#         len_max: Optional[int] = None
#         ):
#         arg_length = length(arg)
#         if len_arg is not None:
#             if arg_length != len_arg:
#                 raise ValueError(
#                      f"Argument `{arg_name}` must have length {len_arg}, "
#                      f"but has length {arg_length}."
#                 )
#         if (len_max is not None) and (len_min is not None):
#             if not(len_min <= arg_length <= len_max):
#                 raise ValueError(
#                      f"Argument `{arg_name}` must have length {len_min} <= n <= {len_max}, "
#                      f"but has length {arg_length}."
#             )




# def assert_length(
#         arg: Any, 
#         arg_name: str,
#         len_arg: Optional[int] = None,
#         len_min: int = 1,
#         len_max: Optional[int] = None
#         ):
#         arg_length = length(arg)
#         if len_arg is not None:
#             if arg_length != len_arg:
#                 raise ValueError(
#                      f"Argument `{arg_name}` must have length {len_arg}, "
#                      f"but has length {arg_length}."
#                 )

#         if arg_length < len_min:
#             if len_max is None:
#                 length_message = f'{len_min} <= n <= {len_max}'
#             else:
#                 length_message = f'length n >= {len_min}'

#             raise ValueError(
#                 f"Argument `{arg_name}` must have length {length_message}, "
#                 f"but has length {arg_length}."
#             )

#         if len_max is not None and arg_length > len_max:
#             raise ValueError(
#                 f"Argument `{arg_name}` must have length {len_min} <= n <= {len_max}, "
#                 f"but has length {arg_length}."
#             )




def assert_length(
        arg: Any, 
        arg_name: str,
        len_arg: Optional[int] = None,
        len_min: int = 1,
        len_max: Optional[int] = None
        ):
        arg_length = length(arg)
        if len_arg is not None:
            if arg_length != len_arg:
                raise ValueError(
                     f"Argument `{arg_name}` must have length {len_arg}, "
                     f"but has length {arg_length}."
                )
        if len_max is None: len_max = float('inf')
        if not(len_min <= arg_length <= len_max):
            raise ValueError(
                f"Argument `{arg_name}` must have length {len_min} <= n <= {len_max}, "
                f"but has length {arg_length}."
            )




def assert_scalar(arg: Any, arg_name:str = 'arg'):
    if not isinstance(arg, str):
        if isinstance(arg, collections.abc.Sized):
            raise ValueError(
                    f"Argument `{arg_name}` must be a scalar value, not an array-like object."
                )


# ## None, Null など引数の missing values を判定



def is_pl_null(x: Any):
    try:
        import polars as pl
        if x is pl.Null or isinstance(x, pl.datatypes.Null):
            return True
        else: return False
    except Exception:
        return False

def is_missing(arg):
    arg = pd.Series(arg)
    result = arg.isna() | arg.apply(is_pl_null)
    return result




def assert_missing(
        arg: Any, 
        arg_name:str = 'arg', 
        any_missing:bool = False,
        all_missing:bool = False
        ):
    arg = pd.Series(arg)
    missing = is_missing(arg)
    not_sutisfy = arg[missing].index.astype(str).to_list()

    if not all_missing and all(missing): 
       raise ValueError(
            f"Argument `{arg_name}` contains only missing values."
        )

    if not any_missing and any(missing): 
       raise ValueError(
            f"Argument `{arg_name}` contains missing values (element {oxford_comma_and(not_sutisfy)})."
        )


# ## 選択式のアサーション



def match_arg(arg: str, values: list[str], arg_name: str = "argument") -> str:
    """
    Simulates the functionality of R's match.arg() function with partial matching in Python.

    Args:
    - arg: The arg to match against the values (partially).
    - values: List of valid values.

    Returns:
    - The matched arg if found in values (partially), otherwise raises an ArgumentError.
    """
    if(arg in values):
      return arg
    else:
      matches = [c for c in values if arg.lower() in c.lower()]
      if len(matches) == 1:
          return matches[0]
      elif len(matches) > 1:
          raise ValueError(
              f"""'{arg}' is ambiguous arg for `{arg_name}`. Matches multiple values: {', '.join(matches)}.
              `{arg_name}` must be one of {oxford_comma_or(values)}."""
              )
      else:
          raise ValueError(f"`{arg_name}` must be one of {oxford_comma_or(values)}, not '{arg}'.")





def arg_match0(arg: str, values: list[str], arg_name: Optional[str] = None) -> str:
    """
    Simulates the functionality of R's rlang::arg_match() function with partial matching in Python.

    Args:
    - arg: The arg to match against the values.
    - values: List of valid values.

    Returns:
    - The matched arg if found in values, otherwise raises an ArgumentError.
    """
    if(arg_name is None):
      arg_name = varname.argname('arg')

    if(arg in values):
      return arg
    else:
      matches = [c for c in values if arg.lower() in c.lower()]
      if len(matches) >= 1:
       raise ValueError(
            f"""`{arg_name}` must be one of {oxford_comma_or(values)}, not '{arg}'.
             Did you mean {oxford_comma_or(matches)}?"""
        )
      else:
        raise ValueError(f"`{arg_name}` must be one of {oxford_comma_or(values)}, not '{arg}'.")





def arg_match(
    arg: Union[str, list[str], pd.Series, np.ndarray],
    values: list[str],
    arg_name: Optional[str] = None,
    multiple: bool = False,
    any_missing: bool = False,
    all_missing: bool = False,
    nullable: bool = False
) -> Union[str, List[str]]:
    """
    Simulates the functionality of R's rlang::arg_match() function with partial matching in Python.

    Args:
    - arg: The list or str of arg to match against the values.
    - values: List of valid values.
    - arg_name : name of argument.
    - multiple: Whether multiple values are allowed for the arg.
    - any_missing: If True, allows the presence of missing values.
    - nullable: If True, allows the argument itself to be None.

    Returns:
    - The matched arg if found in values, otherwise raises an ArgumentError.
    """
    if(arg_name is None):
        arg_name = varname.argname('arg')

    if (arg is None) and nullable: return None

    assert_missing(
      arg, arg_name = arg_name, 
      any_missing = any_missing,
      all_missing = all_missing
      )

    arg = pd.Series(arg)
    if any_missing: 
      arg = arg[~is_missing(arg)]

    if(multiple):
    # 複数選択可の場合
        arg = [arg_match0(val, values = values, arg_name = arg_name) for val in arg]
        return arg
    else:
        arg = arg_match0(arg[0], values = values, arg_name = arg_name)
        return arg


# ## タイプチェックを行う関数



import pandas.api.types

def is_character(x: Any) -> bool:
  return pandas.api.types.is_string_dtype(pd.Series(x))

def is_logical(x: Any) -> bool:
  return pandas.api.types.is_bool_dtype(pd.Series(x))

def is_numeric(x: Any) -> bool:
  return pandas.api.types.is_numeric_dtype(pd.Series(x))

def is_integer(x: Any) -> bool:
  return pandas.api.types.is_integer_dtype(pd.Series(x))

def is_float(x: Any) -> bool:
  return pandas.api.types.is_float_dtype(pd.Series(x))

def is_function(x):
    res = all([isinstance(v, Callable) for v in pd.Series(x)])
    return res




def make_assert_type(
    predicate_fun: Callable[[Any], bool],
    func_name: str,
    valid_type: list[str],
) -> Callable[..., None]:
  """
  Factory to build assertion functions.

  Args:
      predicate_fun: A predicate that returns True if arg has the expected type.
      valid_type: Human-readable type name(s) used in error messages.

  Returns:
      A function that asserts the type condition.
  """

  def func(
      arg: Any, 
      arg_name: Optional[str] = None,
      len_arg: Optional[int] = None,
      len_min: int = 1,
      len_max: Optional[int] = None,
      any_missing: bool = False,
      all_missing: bool = False,
      nullable: bool = False,
      scalar_only: bool = False
      ) -> None:
    """Assert that an argument is specific type and satisfies value and shape constraints.
        `assert_*` is a high-level assertion that combines
        type checking, missing-value handling, length constraints,
        and range validation for numeric arguments.

    Args:
        arg:
            The argument to validate. Can be a scalar or an
            array-like object containing numeric values.
        arg_name:
            Name of the argument, used in error messages. If None, the variable
            name of ``arg`` is inferred when possible.
        lower:
            Lower bound for allowed values.
        upper:
            Upper bound for allowed values.
        inclusive:
            Specifies which bounds are inclusive when checking the value range.

            - ``"both"``: ``lower <= x <= upper``
            - ``"neither"``: ``lower < x < upper``
            - ``"left"``: ``lower <= x < upper``
            - ``"right"``: ``lower < x <= upper``
        len_arg:
            Exact number of elements required. 
            If specified, the argument must contain exactly this number of elements.
            The element count is evaluated on the original input and includes missing
            values (e.g., ``None``, ``NaN``).
        len_min:
            Minimum allowed number of elements.
        len_max:
            Maximum allowed number of elements.
        any_missing:
            If True, allows the presence of missing values.
        all_missing:
            If True, allows all values to be missing.
        nullable:
            If True, allows the argument itself to be None.
        scalar_only:
            If True, only scalar values are allowed. Array-like inputs (even
            those with a single element) are rejected.

    Raises:
        ValueError:
            If the argument is not numeric, violates the specified value range,
            contains disallowed missing values, does not satisfy length
            constraints, or does not conform to the scalar/array-like
            requirements.

    Notes:
        - When ``scalar_only`` is False, the input is internally converted to a
            ``pandas.Series`` for validation.
        - All numeric values must satisfy the specified range constraints.
            If multiple elements violate the condition, their positions are
            reported in the error message.
        - This function performs validation only and returns None if all checks
            pass.

    Example:
        from py4stats import building_block as build
        x = [1, 2, 3]
        y = ['A', 'B', 'C']

        build.assert_character(x, arg_name = 'x')
        #> ValueError: Argument 'x' must be of type 'str'.

        build.assert_character(y, arg_name = 'y')

        build.assert_numeric(x, arg_name = 'x')

        build.assert_numeric(y, arg_name = 'y')
        #> ValueError: Argument 'y' must be of type 'int' or 'float' with value(s) -inf <= x <= inf.

        z = [0.1, 0.3, 0.6]
        build.assert_numeric(z, arg_name = 'z', lower = 0, upper = 1)

        z.extend([2, 3])
        build.assert_numeric(z, arg_name = 'z', lower = 0, upper = 1)
        #> ValueError: Argument 'z' must have value 0 <= x <= 1
        #> element '3' and '4' of 'z' not sutisfy the condtion.

        z = 1
        build.assert_numeric(z, arg_name = 'z', lower = 0, upper = 1, inclusive = 'left')
        #> ValueError: Argument 'z' must have value 0 <= x < 1.
    """
    if(arg_name is None):
      arg_name = varname.argname('arg')


    # 欠測値に関するアサーション ============================================
    if (arg is None) and nullable: return None
    if scalar_only: assert_scalar(arg, arg_name = arg_name)

    arg = pd.Series(arg)

    assert_missing(
      arg, arg_name = arg_name, 
      any_missing = any_missing,
      all_missing = all_missing
      )

    # 引数の要素数に関するアサーション ============================================
    assert_length(
      arg, arg_name, 
      len_arg = len_arg,
      len_min = len_min,
      len_max = len_max
      )

    if any_missing: 
      arg = arg[~is_missing(arg)]

    if not predicate_fun(arg):
      messages = f"Argument `{arg_name}` must be of type {oxford_comma_or(valid_type)}." 
      raise TypeError(messages)
  func.__name__ = func_name
  func.__qualname__ = func_name
  return func




assert_character = make_assert_type(is_character, 'assert_character', valid_type = ['str'])
assert_logical = make_assert_type(is_logical, 'assert_logical', valid_type = ['bool'])
assert_function = make_assert_type(is_function, 'assert_function', valid_type = ['Callable'])




assert_literal = make_assert_type(
    lambda x: is_numeric(x) or is_character(x) or is_logical(x), 
    func_name = 'assert_literal', 
    valid_type = ['str', 'bool', 'int', 'float']
    )


# ### 数値用の `assert_*()` 関数



def make_range_message(
    lower: float = -float("inf"),
    upper: float = float("inf"),
    inclusive: Literal["both", "neither", "left", "right"] = "both"
) -> str:
    arg_match(
        inclusive, arg_name = 'inclusive',
        values = ["both", "neither", "left", "right"]
    )
    inclusive_dict = {
      'both':'<= x <=',
      'neither':'< x <',
      'left':'<= x <',
      'right':'< x <='
    }
    range_message = f"{lower} {inclusive_dict.get(inclusive)} {upper}"
    return range_message




def assert_value_range(
    arg, arg_name:str,
    lower: float = -float("inf"),
    upper: float = float("inf"),
    inclusive: Literal["both", "neither", "left", "right"] = "both",
    # range_message: str = '-inf <= x <= inf'
    ):
    arg = pd.Series(arg)

    range_message = make_range_message(lower, upper, inclusive = inclusive)
    cond = arg.between(lower, upper, inclusive = inclusive)

    not_sutisfy = arg[~cond].index.astype(str).to_list()
    if(len(arg) > 1):
      if not cond.all():
        not_sutisfy_text = oxford_comma_and(not_sutisfy)
        message = (
            f"Argument `{arg_name}` must have value {range_message}.\n"  +
            f"            element {not_sutisfy_text} of `{arg_name}` not sutisfy the condtion."
            )# ↑ "ValueError: " の分のインデント
        raise ValueError(message)
    else:
      if not cond.all():
       message =  f"Argument `{arg_name}` must have value {range_message}."
       raise ValueError(message)




def assert_numeric_dtype(
        arg:Any, 
        arg_name: str,
        predicate_fun: Callable[[Any], bool],
        valid_type: list[str],
        lower: float = -float("inf"),
        upper: float = float("inf"),
        inclusive: Literal["both", "neither", "left", "right"] = "both"
        ):
        range_message = make_range_message(lower, upper, inclusive = inclusive)

        if not predicate_fun(arg): 
            message = f"Argument `{arg_name}` must be of" +\
                f" type {oxford_comma_or(valid_type)}" + \
                f" with value(s) {range_message}."
            raise TypeError(message)




def make_assert_numeric(
    predicate_fun: Callable[[Any], bool],
    func_name: str,
    valid_type: list[str],
    lower: float = -float("inf"),
    upper: float = float("inf"),
) -> Callable[..., None]:
  """
  Factory to build numeric assertion functions with range checks.

  Args:
      predicate_fun: Predicate for numeric dtype check.
      valid_type: Human-readable type label(s) used in messages.
      lower: Default lower bound.
      upper: Default upper bound.

  Returns:
      A function that asserts: numeric dtype and range condition.
  """
  def func(
        arg: Any,
        arg_name: Optional[str] = None,
        lower: float = lower,
        upper: float = upper,
        inclusive: Literal["both", "neither", "left", "right"] = "both",
        len_arg: Optional[int] = None,
        len_min: int = 1,
        len_max: Optional[int] = None,
        any_missing: bool = False,
        all_missing: bool = False,
        nullable: bool = False,
        scalar_only: bool = False
  ) -> None:
    """Assert that an argument is specific type and satisfies value and shape constraints.
    Args:
        arg:
            The argument to validate. Can be a scalar numeric value or an
            array-like object containing numeric values.
        arg_name:
            Name of the argument, used in error messages. If None, the variable
            name of ``arg`` is inferred when possible.
        lower:
            Lower bound for allowed values.
        upper:
            Upper bound for allowed values.
        inclusive:
            Specifies which bounds are inclusive when checking the value range.

            - ``"both"``: ``lower <= x <= upper``
            - ``"neither"``: ``lower < x < upper``
            - ``"left"``: ``lower <= x < upper``
            - ``"right"``: ``lower < x <= upper``
        len_arg:
            Exact number of elements required. 
            If specified, the argument must contain exactly this number of elements.
            The element count is evaluated on the original input and includes missing
            values (e.g., ``None``, ``NaN``).
        len_min:
            Minimum allowed number of elements.
        len_max:
            Maximum allowed number of elements.
        any_missing:
            If True, allows the presence of missing values.
        all_missing:
            If True, allows all values to be missing.
        nullable:
            If True, allows the argument itself to be None.
        scalar_only:
            If True, only scalar values are allowed. Array-like inputs (even
            those with a single element) are rejected.

    Raises:
        ValueError:
            If the argument is not numeric, violates the specified value range,
            contains disallowed missing values, does not satisfy length
            constraints, or does not conform to the scalar/array-like
            requirements.

    Notes:
        - When ``scalar_only`` is False, the input is internally converted to a
            ``pandas.Series`` for validation.
        - All numeric values must satisfy the specified range constraints.
            If multiple elements violate the condition, their positions are
            reported in the error message.
        - This function performs validation only and returns None if all checks
            pass.

    Example:
        from py4stats import building_block as build
        x = [1, 2, 3]
        y = ['A', 'B', 'C']

        build.assert_character(x, arg_name = 'x')
        #> ValueError: Argument 'x' must be of type 'str'.

        build.assert_character(y, arg_name = 'y')

        build.assert_numeric(x, arg_name = 'x')

        build.assert_numeric(y, arg_name = 'y')
        #> ValueError: Argument 'y' must be of type 'int' or 'float' with value(s) -inf <= x <= inf.

        z = [0.1, 0.3, 0.6]
        build.assert_numeric(z, arg_name = 'z', lower = 0, upper = 1)

        z.extend([2, 3])
        build.assert_numeric(z, arg_name = 'z', lower = 0, upper = 1)
        #> ValueError: Argument 'z' must have value 0 <= x <= 1
        #> element '3' and '4' of 'z' not sutisfy the condtion.

        z = 1
        build.assert_numeric(z, arg_name = 'z', lower = 0, upper = 1, inclusive = 'left')
        #> ValueError: Argument 'z' must have value 0 <= x < 1.
    """
    if(arg_name is None):
      arg_name = varname.argname('arg')

    # 欠測値に関するアサーション ============================================
    if (arg is None) and nullable: return None
    if scalar_only: assert_scalar(arg, arg_name = arg_name)

    arg = pd.Series(arg)

    assert_missing(
      arg, arg_name = arg_name, 
      any_missing = any_missing,
      all_missing = all_missing
      )

    # 引数の要素数に関するアサーション ============================================
    assert_length(
      arg, arg_name, 
      len_arg = len_arg,
      len_min = len_min,
      len_max = len_max
      )

    # 引数の型に関するアサーション ===============================================
    # 欠損値の除外は型ベースの検証のためだけに行います。
    # （長さおよび形状のチェックは元の入力に対して実行されます）
    if any_missing: 
      arg = arg[~is_missing(arg)]

    assert_numeric_dtype(
        arg, arg_name = arg_name,
        predicate_fun = predicate_fun,
        valid_type = valid_type,
        lower = lower, upper = upper,
        inclusive = inclusive
        )
    # 引数値の範囲に関するアサーション ============================================
    assert_value_range(
      arg, arg_name = arg_name, 
      lower = lower, upper = upper,
      inclusive = inclusive,
      )

  func.__name__ = func_name
  func.__qualname__ = func_name
  return func




assert_numeric = make_assert_numeric(is_numeric, 'assert_numeric', valid_type = ['int', 'float'])
assert_integer = make_assert_numeric(is_integer, 'assert_integer', valid_type = ['int'])
assert_count = make_assert_numeric(is_integer, 'assert_count', valid_type = ['positive integer'], lower = 0)
assert_float = make_assert_numeric(is_float, 'assert_float', valid_type = ['float'])




def assert_same_type(arg, arg_name: str = 'arg'):
    if length(arg) <= 1: return None
    first_type = type(arg[0])
    mismatched = [
        f"{i} ({type(v).__name__})" 
        for i, v in enumerate(arg) 
        if type(v) is not first_type
        ]
    if mismatched:
        not_sutisfy_text = oxford_comma_and(
            mismatched, quotation = False
        )

        message = f"Elements of `{arg_name}` must share the same type.\n" +\
                  f"{11 * ' '}Found at indices {not_sutisfy_text}."
        raise TypeError(message)




def assert_literal_kyes(arg, arg_name: str = 'arg'):
    keys = list(arg.keys())
    if length(keys) > 1:
        unique_type = list_unique([type(v).__name__ for v in arg])

        if length(unique_type) > 1:
            type_text = oxford_comma_and(unique_type)
            message = f"Keys of `{arg_name}` must share the same type, got {type_text}." 
            raise TypeError(message)

    if not (is_numeric(keys) or is_character(keys) or is_logical(keys)):
        valid_type = ['str', 'int', 'float']
        messages = f"Keys of `{arg_name}` must be of type {oxford_comma_or(valid_type)}." 
        raise TypeError(messages)


# ## 数値などのフォーマット



def p_stars(
    p_value: Sequence[int, float, np.number],
    stars: Optional[Mapping[str, float]] = None,
) -> pd.Series:
    """
    Map p-values to significance stars.

    Args:
        p_value: Scalar or array-like of p-values.
        stars: Mapping from star label to cutoff (upper bound). If None (defaults) to
            {'***': 0.01, '**': 0.05, '*': 0.1}.

    Returns:
        pandas.Series: Star labels for each p-value.
    """
    # stars のラベルに上限値を追加
    #   stars = stars.copy()
    if stars is None:
        stars = {"***": 0.01, "**": 0.05, "*": 0.1}
    stars2 = dict(stars)
    stars2[""] = float("inf")

    # pd.Series に変換した上で、
    # pd.cut() の bin として使用するときにエラーを生じないよう、昇順にソート
    stars2 = pd.Series(stars2, dtype = 'float64').sort_values()

    assert_numeric(p_value, lower = 0)
    assert_numeric(stars2, lower = 0)

    # 0.0 に stars2 の値を追加したものを bins とします。
    bins = [0.0, *stars2.to_list()]

    styled = pd.cut(
        p_value,
        bins = bins,
        labels = stars2.index,
        ordered = True,
        duplicates = 'drop'
        ).astype(str)

    return pd.Series(styled)




def style_pvalue(
    p_value: Sequence[int, float, np.number],
    digits: int = 3,
    prepend_p: bool = False,
    p_min: float = 0.001,
    p_max: float = 0.9
) -> pd.Series:
  """
  Format p-values into strings with optional clipping and prefix.

  Args:
        p_value: Scalar or array-like of p-values.
        digits: Number of decimals.
        prepend_p: If True, prepend 'p' or 'p='.
        p_min: Lower clipping threshold.
        p_max: Upper clipping threshold.

  Returns:
        pandas.Series: Formatted p-values as strings.
  """
  assert_numeric(p_value, lower = 0)
  assert_count(digits, lower = 1)
  assert_numeric(p_min, lower = 0, upper = 1)
  assert_numeric(p_max, lower = 0, upper = 1)

  if(prepend_p): prefix = ['p', 'p=']
  else: prefix = ['', '']


  p_value = pd.Series(p_value)
  styled = prefix[1] + p_value.copy().round(digits).astype(str)

  styled = styled.mask(p_value < p_min, f'{prefix[0]}<{p_min}')\
        .mask(p_value > p_max, f'{prefix[0]}>{p_max}')

  return styled




@np.vectorize
def num_comma(x: NumberLike, digits: int = 2, big_mark: str = ",") -> str:
  assert_count(digits)
  arg_match(big_mark, [',', '_', ''])
  return f'{x:{big_mark}.{digits}f}'

@np.vectorize
def num_currency(x: NumberLike, symbol: str = "$", digits: int = 0, big_mark: str = ",") -> str:
  assert_count(digits)
  arg_match(big_mark, [',', '_', ''])
  return f'{symbol}{x:{big_mark}.{digits}f}'

@np.vectorize
def num_percent(x: NumberLike, digits: int = 2) -> str:
  return f'{x:.{digits}%}'




def style_number(x: ArrayLike, digits: int = 2, big_mark: str = ",") -> pd.Series:
  x = pd.Series(x)

  assert_numeric(x)
  assert_count(digits)

  arg_match(big_mark, [',', '_', ''])

  return x.apply(lambda v: f'{v:{big_mark}.{digits}f}')

def style_currency(x: ArrayLike, symbol: str = "$", digits: int = 0, big_mark: str = ",") -> pd.Series:
  x = pd.Series(x)

  assert_numeric(x)
  assert_count(digits)

  arg_match(big_mark, [',', '_', ''])

  return x.apply(lambda v: f'{symbol}{v:{big_mark}.{digits}f}')

def style_percent(x: ArrayLike, digits: int = 2, unit: float = 100, symbol: str = "%") -> pd.Series:
  x = pd.Series(x)

  assert_numeric(x)
  assert_count(digits)

  return x.apply(lambda v: f'{v*unit:.{digits}f}{symbol}')




@np.vectorize
def pad_zero(x: Any, digits: int = 2) -> str:
    s = str(x)
    # もし s が整数値なら、何もしない。
    if s.find('.') != -1:
        s_digits = len(s[s.find('.'):])       # 小数点以下の桁数を計算
        s = s + '0' * (digits + 1 - s_digits) # 足りない分だけ0を追加
    return s




@np.vectorize
def add_big_mark(s: Any) -> str:
    return f'{s:,}'


# ## set operation of list



def list_diff(x: List[Any], y: List[Any]) -> List[Any]:
    """Set operation of list
        - `list_diff(x, y)`: The difference set between lists `x` and `y`
        - `list_intersect(x, y)`: The intersection of lists `x` and `y`
        - `list_union(x, y)`: The union of lists `x` and `y`
        - `list_xor(x, y)`: The symmetric difference of lists `x` and `y`
        - `list_unique(x)`: Creates a list consisting of the non-duplicate elements from list `x`
        - `list_dupricated(x)`: Dupricated elements of list `x`
        - `list_dropnulls(x)`: Drop missing values from list `x`
        - `list_subset(x, subset)`: A subset of list `x`
    """
    return [v for v in x if v not in y]

def list_intersect(x: List[Any], y: List[Any]) -> List[Any]:
    return [v for v in x if v in y]

def list_union(x: List[Any], y: List[Any]) -> List[Any]:
    return list_unique(x + y)

def list_xor(x: List[Any], y: List[Any]) -> List[Any]:
    return list_diff(x, y) + list_diff(y, x)

def list_unique(x: List[Any]) -> List[Any]:
    result = []
    for v in x:
        if v not in result:
            result.append(v)
    return result

def list_dupricated(x: List[Any]) -> List[Any]:
    return list_subset(x, lambda v: x.count(v) >= 2)

def list_dropnulls(x: List[Any]) -> List[Any]:
    return list_subset(x, ~is_missing(x))

def list_subset(x: List[Any], subset: Union[Callable, List[int], List[bool]]) -> List[Any]:
    if isinstance(subset, Callable):
        return list(filter(subset, x))
    if is_logical(subset):
        return [v for i, v in enumerate(x) if subset[i]]
    if is_integer(subset):
        result = []
        for i in subset:
            result.extend(x[i])
        return result

list_intersect.__doc__ = list_diff.__doc__
list_union.__doc__ = list_diff.__doc__
list_xor.__doc__ = list_diff.__doc__
list_unique.__doc__ = list_diff.__doc__
list_dupricated.__doc__ = list_diff.__doc__
list_dropnulls.__doc__ = list_diff.__doc__
list_subset.__doc__ = list_diff.__doc__




def list_flatten(x: List[Any]) -> Iterable:
    for el in x:
        if isinstance(el, (list, tuple)):
            yield from list_flatten(el)
        else:
            yield el




def list_replace(x: List[Any], mapping: Union[Dict, Callable]) -> List[Any]:
    if isinstance(mapping, dict):
        return [mapping.get(v, v) for v in x]
    if isinstance(mapping, Callable):
        return [mapping(v) for v in x]




def which(x: Iterable[bool]):
    indices = [i for i, val in enumerate(x) if val]
    return indices

