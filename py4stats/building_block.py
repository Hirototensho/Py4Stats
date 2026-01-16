#!/usr/bin/env python
# coding: utf-8



from __future__ import annotations


# # `py4stats` のプログラミングを補助する関数群
# 
# `eda_tools` や `regression_tools` で共通して使う関数をここにまとめておきます。`building_block` モジュール自体は外部から呼び出さずに使用することを想定しています。

# `building_block` モジュールに実装された主要な関数の依存関係
# 
# ```python
# arg_match()                       # 文字列やリストの引数を有効値と突き合わせてチェック
# └─ arg_match0()                  # 単一値の照合処理を担当
#    └─ oxford_comma_or()         # 候補が複数ある場合のメッセージ整形に使用
#        └─ oxford_comma()        # 英語表現の並列化（Oxford comma）
# 
# match_arg()                      # Rの match.arg に類似、部分一致で引数を照合
# └─ oxford_comma_or()            # 候補の整形出力（上記と共通）
# 
# make_assert_type()              # 型アサート用関数の function factory（文字列・論理型用など）
# └─ predicate_fun()             # 呼び出し側が指定する型チェック関数（例: is_character）
# 
# make_assert_numeric()           # 数値アサート関数のfunction factory（範囲チェック付き）
# └─ oxford_comma_or()           # エラーメッセージに候補を整形表示
#    └─ oxford_comma()
# 
# p_stars()                        # p値にアスタリスク（***など）を付与
# ├─ assert_numeric()            # 入力値チェック
# └─ pd.cut()                    # 範囲に応じたカテゴリ化
# 
# style_pvalue()                  # p値の整形出力（しきい値による置き換え）
# ├─ assert_numeric()
# ├─ assert_count()
# ├─ pd.Series()
# └─ pandas.mask()               # 条件によって文字列を変換
# 
# style_number()                  # 数値を桁区切り付きで文字列整形
# ├─ assert_numeric()
# ├─ assert_count()
# └─ arg_match()                 # 区切り記号の妥当性チェック
# 
# style_currency()                # 金額表記に整形（通貨記号 + 整数）
# ├─ assert_numeric()
# ├─ assert_count()
# └─ arg_match()
# 
# style_percent()                 # パーセント表記に整形（単位変換含む）
# ├─ assert_numeric()
# └─ assert_count()
# 
# num_comma()                     # 数値をコンマ区切り付きで整形（ベクトル化関数）
# └─ arg_match()
# 
# num_currency()                 # 金額整形（num_commaと同様）
# └─ arg_match()
# 
# pad_zero()                      # 小数点以下のゼロを桁数に応じて補完（文字列化）
# （依存なし）
# 
# add_big_mark()                 # 桁区切りを追加（f'{:,}'形式）
# （依存なし）
# 
# oxford_comma_and()              # A, B and C のような整形
# └─ oxford_comma()
# 
# oxford_comma_or()               # A, B or C のような整形
# └─ oxford_comma()
# 
# oxford_comma()                  # 英文の並列表記として候補リストを整形
# （依存なし）
# 
# ```



import pandas as pd
import numpy as np
import scipy as sp
# from varname import argname
import varname
import collections


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
StrSeq = Sequence[str]
StrOrStrSeq = Union[str, StrSeq]

# match系の戻り（single / multiple）
MatchArgReturn = Union[str, List[str]]

Inclusive = Literal["both", "neither", "left", "right"]

# 数値っぽい入力を広めに許容（pandas Series / numpy array / Python number）
NumberLike = Union[int, float, np.number]
ArrayLike = Union[NumberLike, Sequence[NumberLike], np.ndarray, pd.Series]

# 文字列配列っぽい入力
StrArrayLike = Union[str, Sequence[str], np.ndarray, pd.Series]

# p-value などは 0..1 の数値配列として扱うことが多い
ProbArrayLike = ArrayLike


# ## `oxford_comma()`
# 
# 文字列のリストを与えると、英文の並列の形に変換する関数です。表記法については[Wikipedia Serial comma](https://en.wikipedia.org/wiki/Serial_comma)を参照し、コードについては[stack overflow:Grammatical List Join in Python [duplicate]](https://stackoverflow.com/questions/19838976/grammatical-list-join-in-python)を参照しました。
# 
# ```python
# choices = ['apple', 'orange', 'grape']
# oxford_comma_or(choices)
# #> 'apple, orange or grape'
# ```



def oxford_comma(x: Union[str, Sequence[str]], sep_last: str = "and", quotation: bool = True) -> str:
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


# ## 引数の要素数に関するアサーション



def length(x):
    if x is None:
        return 0
    if isinstance(x, str):
        return 1
    if isinstance(x, collections.abc.Sized):
        return len(x)       # list, dict, pandas.Series, etc.
    return 1




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
                     f"Argument '{arg_name}' must have length {len_arg}, "
                     f"but has length {arg_length}."
                )
        if (len_max is not None) and (len_min is not None):
            if not(len_min <= arg_length <= len_max):
                raise ValueError(
                     f"Argument '{arg_name}' must have length {len_min} <= n <= {len_max}, "
                     f"but has length {arg_length}."
            )




def assert_scalar(arg: Any, arg_name:str = 'arg'):
    if not isinstance(arg, str):
        if isinstance(arg, collections.abc.Sized):
            raise ValueError(
                    f"Argument '{arg_name}' must be a scalar value, not an array-like object."
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
            f"Argument '{arg_name}' contains only missing values."
        )

    if not any_missing and any(missing): 
       raise ValueError(
            f"Argument '{arg_name}' contains missing values (element {oxford_comma_and(not_sutisfy)})."
        )


# ## 選択式のアサーション



def match_arg(arg: str, values: Sequence[str], arg_name: str = "argument") -> str:
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
              f"""'{arg}' is ambiguous arg for '{arg_name}'. Matches multiple values: {', '.join(matches)}.
              '{arg_name}' must be one of {oxford_comma_or(values)}."""
              )
      else:
          raise ValueError(f"'{arg_name}' must be one of {oxford_comma_or(values)}, not '{arg}'.")





def arg_match0(arg: str, values: Sequence[str], arg_name: Optional[str] = None) -> str:
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
            f"""'{arg_name}' must be one of {oxford_comma_or(values)}, not '{arg}'.
             Did you mean {oxford_comma_or(matches)}?"""
        )
      else:
        raise ValueError(f"'{arg_name}' must be one of {oxford_comma_or(values)}, not '{arg}'.")





def arg_match(
    arg: Union[str, Sequence[str], pd.Series, np.ndarray],
    values: Sequence[str],
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

    if (arg is None) and nullable: return

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




def make_assert_type(
    predicate_fun: Callable[[Any], bool],
    valid_type: Sequence[str],
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
      ):
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
            Exact number of elements required. If specified, the input must have
            exactly this length.
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
    if (arg is None) and nullable: return
    if scalar_only: assert_scalar(arg, arg_name = arg_name)

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

    arg = pd.Series(arg)

    if any_missing: 
      arg = arg[~is_missing(arg)]

    if not predicate_fun(arg):
      messages = f"Argument '{arg_name}' must be of type {oxford_comma_or(valid_type)}." 
      raise ValueError(messages)

  return func




assert_character = make_assert_type(is_character, valid_type = ['str'])
assert_logical = make_assert_type(is_logical, valid_type = ['bool'])


# ### 数値用の `assert_*()` 関数



def make_assert_numeric(
    predicate_fun: Callable[[Any], bool],
    valid_type: Sequence[str],
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
            Exact number of elements required. If specified, the input must have
            exactly this length.
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
    if (arg is None) and nullable: return
    if scalar_only: assert_scalar(arg, arg_name = arg_name)

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

    # 引数の値に関するアサーション ===============================================
    arg = pd.Series(arg)

    if any_missing: 
      arg = arg[~is_missing(arg)]

    inclusive_dict = {
      'both':'<= x <=',
      'neither':'< x <',
      'left':'<= x <',
      'right':'< x <='
    }

    if not predicate_fun(arg): 
       message = f"Argument '{arg_name}' must be of" +\
            f" type {oxford_comma_or(valid_type)}" + \
            f" with value(s) {lower} {inclusive_dict[inclusive]} {upper}."
       raise ValueError(message)

    cond = arg.between(lower, upper, inclusive = inclusive)
    not_sutisfy = arg[~cond].index.astype(str).to_list()

    if(len(arg) > 1):
      if not cond.all():
        message = (
            f"Argument '{arg_name}' must have value {lower} {inclusive_dict[inclusive]} {upper}\n"  +
            f"element {oxford_comma_and(not_sutisfy)} of '{arg_name}' not sutisfy the condtion."
            )
        raise ValueError(message)
    else:
      if not cond.all():
       message =  f"Argument '{arg_name}' must have value {lower} {inclusive_dict[inclusive]} {upper}."
       raise ValueError(message)

    # if(arg_name is None):
    #     arg_name = varname.argname('arg')
  return func




assert_numeric = make_assert_numeric(is_numeric, valid_type = ['int', 'float'])
assert_integer = make_assert_numeric(is_integer, valid_type = ['int'])
assert_count = make_assert_numeric(is_integer, valid_type = ['positive integer'], lower = 0)
assert_float = make_assert_numeric(is_float, valid_type = ['float'])


# ## 数値などのフォーマット



# def p_stars(p_value, stars = {'***':0.01, '**':0.05, '*':0.1}):
def p_stars(
    p_value: ProbArrayLike,
    # stars: Optional[Mapping[str, float]] = None,
    stars = {'***':0.01, '**':0.05, '*':0.1}
) -> pd.Series:
    """
    Map p-values to significance stars.

    Args:
        p_value: Scalar or array-like of p-values.
        stars: Mapping from star label to cutoff (upper bound). Defaults to
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




# def style_pvalue(p_value, digits = 3, prepend_p = False, p_min = 0.001, p_max = 0.9):
def style_pvalue(
    p_value: ProbArrayLike,
    digits: int = 3,
    prepend_p: bool = False,
    p_min: float = 0.001,
    p_max: float = 0.9,
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

