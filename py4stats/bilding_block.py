# %%
from __future__ import annotations

# %% [markdown]
# # `py4stats` のプログラミングを補助する関数群
# 
# `eda_tools` や `regression_tools` で共通して使う関数をここにまとめておきます。`bilding_block` モジュール自体は外部から呼び出さずに使用することを想定しています。

# %% [markdown]
# `bilding_block` モジュールに実装された主要な関数の依存関係
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

# %%
import pandas as pd
import numpy as np
import scipy as sp
from varname import argname

# %% [markdown]
# ## 型ヒントの準備

# %%
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

# %% [markdown]
# ## 引数のアサーション

# %%
import argparse

def match_arg(arg: str, values: Sequence[str], arg_name: str = "argument") -> str:
    """Partially match an argument against allowed values (R-like match.arg).

    This function performs case-insensitive partial matching similar to R's
    `match.arg()`. If `arg` is an exact match, it is returned as-is. Otherwise,
    `arg` is matched as a substring of candidates in `values`.

    Args:
        arg: Argument string to match. Partial matching is allowed.
        values: Sequence of allowed values.
        arg_name: Name of the argument used in error messages.

    Returns:
        The matched value from `values`.

    Raises:
        ValueError: If `arg` matches multiple candidates (ambiguous) or matches none.
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

# %%
def arg_match0(arg: str, values: Sequence[str], arg_name: Optional[str] = None) -> str:
    """Match a single argument with suggestions (R-like rlang::arg_match).

    If the argument is not exactly in `values`, the function searches for
    case-insensitive partial matches. If there are suggestions, the error
    message includes them.

    Args:
        arg: Argument string to match.
        values: Sequence of allowed values.
        arg_name: Name of the argument used in error messages. If None, inferred
            by `varname.argname("arg")`.

    Returns:
        The matched value.

    Raises:
        ValueError: If `arg` is not a valid value. The error message may include
            suggested candidates.
    """
    if(arg_name is None):
      arg_name = argname('arg')

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

# %%
from varname import argname
def arg_match(
    arg: Union[str, Sequence[str], pd.Series, np.ndarray],
    values: Sequence[str],
    arg_name: Optional[str] = None,
    multiple: bool = False,
) -> MatchArgReturn:
  """Match one or more arguments against allowed values.

  This is a user-facing helper that accepts either a single string or a
  list-like of strings. When `multiple=True`, all elements are matched and
  returned as a list.

  Args:
      arg: Argument(s) to match. Accepts a string or list-like of strings.
      values: Sequence of allowed values.
      arg_name: Name of the argument used in error messages. If None, inferred
          by `varname.argname("arg")`.
      multiple: If True, allow multiple values and return a list of matched
          strings.

  Returns:
      If `multiple=False`, returns a single matched value.
      If `multiple=True`, returns a list of matched values.

  Raises:
      ValueError: If any element is invalid or ambiguous.
  """
  if(arg_name is None):
      arg_name = argname('arg')

  arg = pd.Series(arg)
  if(multiple):
    # 複数選択可の場合
    arg = [arg_match0(val, values = values, arg_name = arg_name) for val in arg]
    return arg
  else:
    arg = arg_match0(arg[0], values = values, arg_name = arg_name)
    return arg

# %% [markdown]
# ## タイプチェックを行う関数

# %%
from varname import argname
import pandas.api.types

def is_character(x: Any) -> bool:
  """Return True if `x` is string-like according to pandas dtype rules.

  Args:
      x: Any value or array-like object.

  Returns:
      True if `pd.Series(x)` is interpreted as string dtype.
  """
  return pandas.api.types.is_string_dtype(pd.Series(x))

def is_logical(x: Any) -> bool:
  """Return True if `x` is boolean-like according to pandas dtype rules.

  Args:
      x: Any value or array-like object.

  Returns:
      True if `pd.Series(x)` is interpreted as boolean dtype.
  """
  return pandas.api.types.is_bool_dtype(pd.Series(x))

def is_numeric(x: Any) -> bool:
  """Return True if `x` is numeric-like according to pandas dtype rules.

  Args:
      x: Any value or array-like object.

  Returns:
      True if `pd.Series(x)` is interpreted as numeric dtype.
  """
  return pandas.api.types.is_numeric_dtype(pd.Series(x))

def is_integer(x: Any) -> bool:
  """Return True if `x` is integer-like according to pandas dtype rules.

  Args:
      x: Any value or array-like object.

  Returns:
      True if `pd.Series(x)` is interpreted as integer dtype.
  """
  return pandas.api.types.is_integer_dtype(pd.Series(x))

def is_float(x: Any) -> bool:
  """Return True if `x` is float-like according to pandas dtype rules.

  Args:
      x: Any value or array-like object.

  Returns:
      True if `pd.Series(x)` is interpreted as float dtype.
  """
  return pandas.api.types.is_float_dtype(pd.Series(x))

# %%
def make_assert_type(
    predicate_fun: Callable[[Any], bool],
    valid_type: Sequence[str],
) -> Callable[..., None]:
  """Create an assertion function for a given type predicate.

  The returned function asserts that the given argument satisfies
  `predicate_fun`. The error message uses `valid_type` for readability.

  Args:
      predicate_fun: Predicate that returns True if the argument has the expected type.
      valid_type: Human-readable type label(s) used in error messages.

  Returns:
      A function `(arg, arg_name=None) -> None` that raises AssertionError if
      the predicate fails.
  """

  def func(arg, arg_name = None):
    if(arg_name is None):
      arg_name = argname('arg')

    assert predicate_fun(arg), f"Argment '{arg_name}' must be of" +\
      f" type {oxford_comma_or(valid_type)}"

  return func

# %%
assert_character = make_assert_type(is_character, valid_type = ['str'])
assert_logical = make_assert_type(is_logical, valid_type = ['bool'])

# %% [markdown]
# ### 数値用の `assert_*()` 関数

# %%
def make_assert_numeric(
    predicate_fun: Callable[[Any], bool],
    valid_type: Sequence[str],
    lower: float = -float("inf"),
    upper: float = float("inf"),
) -> Callable[..., None]:
  """Create a numeric assertion function with optional range checks.

  The returned function checks:
      1) Numeric dtype via `predicate_fun`.
      2) Range constraint using pandas `.between(...)`.

  Args:
      predicate_fun: Predicate that checks numeric dtype.
      valid_type: Human-readable type label(s) used in error messages.
      lower: Default lower bound.
      upper: Default upper bound.

  Returns:
      A function `(arg, lower=..., upper=..., inclusive=..., arg_name=None) -> None`
      that raises AssertionError when checks fail.
  """
  def func(
      arg: Any,
      lower: float = lower,
      upper: float = upper,
      inclusive: Inclusive = "both",
      arg_name: Optional[str] = None,
  ) -> None:
    if(arg_name is None):
      arg_name = argname('arg')

    arg = pd.Series(arg)

    inclusive_dict = {
      'both':'<= x <=',
      'neither':'< x <',
      'left':'<= x <',
      'right':'< x <='
    }

    assert predicate_fun(arg), f"Argment '{arg_name}' must be of" +\
      f" type {oxford_comma_or(valid_type)}" + \
      f" with value(s) {lower} {inclusive_dict[inclusive]} {upper}."

    cond = arg.between(lower, upper, inclusive = inclusive)
    not_sutisfy = arg[~cond].index.astype(str).to_list()

    if(len(arg) > 1):
      assert cond.all(),\
      f"""Argment '{arg_name}' must have value {lower} {inclusive_dict[inclusive]} {upper}.
                element {oxford_comma_and(not_sutisfy)} of '{arg_name}' not sutisfy the condtion."""
    else:
      assert cond.all(),\
      f"Argment '{arg_name}' must have value {lower} {inclusive_dict[inclusive]} {upper}."

    if(arg_name is None):
        arg_name = argname('arg')
  return func

# %%
assert_numeric = make_assert_numeric(is_numeric, valid_type = ['int', 'float'])
assert_integer = make_assert_numeric(is_integer, valid_type = ['int'])
assert_count = make_assert_numeric(is_integer, valid_type = ['positive integer'], lower = 0)
assert_float = make_assert_numeric(is_float, valid_type = ['float'])

# %% [markdown]
# ## 数値などのフォーマット

# %%
# def p_stars(p_value, stars = {'***':0.01, '**':0.05, '*':0.1}):
def p_stars(
    p_value: ProbArrayLike,
    # stars: Optional[Mapping[str, float]] = None,
    stars = {'***':0.01, '**':0.05, '*':0.1}
) -> pd.Series:
    """Convert p-values to significance stars.

    This function maps p-values into categorical star labels by binning.
    By default, the mapping is:
        - '***' for p <= 0.01
        - '**'  for p <= 0.05
        - '*'   for p <= 0.10
        - ''    otherwise

    Args:
        p_value: Scalar or array-like of p-values.
        stars: Mapping from star label to cutoff value. If None, defaults to
            `{'***': 0.01, '**': 0.05, '*': 0.1}`. An empty label '' is
            automatically appended with `inf` as an upper bound.

    Returns:
        pandas.Series:
            Series of star labels (strings), aligned to the input length.

    Raises:
        AssertionError:
            If `p_value` or `stars` contains non-numeric values or invalid ranges.
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

# %%
def style_pvalue(
    p_value: ProbArrayLike,
    digits: int = 3,
    prepend_p: bool = False,
    p_min: float = 0.001,
    p_max: float = 0.9,
) -> pd.Series:
    """Format p-values as readable strings with optional clipping.

    Args:
        p_value: Scalar or array-like of p-values.
        digits: Number of decimal places for rounding.
        prepend_p: If True, prepend 'p' or 'p='.
        p_min: Values smaller than this threshold are shown as `<p_min`.
        p_max: Values larger than this threshold are shown as `>p_max`.

    Returns:
        pandas.Series: Formatted p-values as strings.

    Raises:
        AssertionError: If inputs are out of expected ranges.
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

# %%
@np.vectorize
def num_comma(x: NumberLike, digits: int = 2, big_mark: str = ",") -> str:
  """Format a number with thousands separators and fixed decimals.

  Args:
      x: Numeric value.
      digits: Number of decimal places.
      big_mark: Thousands separator style. One of ',', '_' or ''.

  Returns:
      Formatted number as a string.
  """
  assert_count(digits)
  arg_match(big_mark, [',', '_', ''])
  return f'{x:{big_mark}.{digits}f}'

@np.vectorize
def num_currency(x: NumberLike, symbol: str = "$", digits: int = 0, big_mark: str = ",") -> str:
  """Format a number as currency.

  Args:
      x: Numeric value.
      symbol: Currency symbol to prepend.
      digits: Number of decimal places.
      big_mark: Thousands separator style. One of ',', '_' or ''.

  Returns:
      Formatted currency string.
  """  
  assert_count(digits)
  arg_match(big_mark, [',', '_', ''])
  return f'{symbol}{x:{big_mark}.{digits}f}'

@np.vectorize
def num_percent(x: NumberLike, digits: int = 2) -> str:
  """Format a number as percentage using Python's percent format.

  Args:
      x: Numeric value interpreted as a fraction (e.g., 0.12 -> 12%).
      digits: Number of decimal places.

  Returns:
      Percentage string.
  """
  return f'{x:.{digits}%}'

# %%
def style_number(x: ArrayLike, digits: int = 2, big_mark: str = ",") -> pd.Series:
  """Format numeric values into fixed-decimal strings.

  Args:
      x: Numeric scalar or array-like.
      digits: Number of decimal places.
      big_mark: Thousands separator style. One of ',', '_' or ''.

  Returns:
      pandas.Series: Formatted numbers as strings.
  """
  x = pd.Series(x)

  assert_numeric(x)
  assert_count(digits)

  arg_match(big_mark, [',', '_', ''])

  return x.apply(lambda v: f'{v:{big_mark}.{digits}f}')

def style_currency(x: ArrayLike, symbol: str = "$", digits: int = 0, big_mark: str = ",") -> pd.Series:
  """Format numeric values as currency strings.

  Args:
      x: Numeric scalar or array-like.
      symbol: Currency symbol to prepend.
      digits: Number of decimal places.
      big_mark: Thousands separator style. One of ',', '_' or ''.

  Returns:
      pandas.Series: Formatted currency strings.
  """
  x = pd.Series(x)

  assert_numeric(x)
  assert_count(digits)

  arg_match(big_mark, [',', '_', ''])

  return x.apply(lambda v: f'{symbol}{v:{big_mark}.{digits}f}')

def style_percent(x: ArrayLike, digits: int = 2, unit: float = 100, symbol: str = "%") -> pd.Series:
  """Format numeric values as percent strings.

  Args:
      x: Numeric scalar or array-like.
      digits: Number of decimal places.
      unit: Scale factor applied before formatting (default 100).
      symbol: Suffix symbol (default '%').

  Returns:
      pandas.Series: Formatted percent strings.
  """
  x = pd.Series(x)

  assert_numeric(x)
  assert_count(digits)

  return x.apply(lambda v: f'{v*unit:.{digits}f}{symbol}')

# %%
@np.vectorize
def pad_zero(x: Any, digits: int = 2) -> str:
    """Pad trailing zeros for decimal representation.

    This function is useful when numeric values are stringified with varying
    decimal lengths and you want a consistent number of decimal places.

    Args:
        x: Value to format (typically numeric or numeric-like string).
        digits: Desired number of digits after the decimal point.

    Returns:
        String with padded zeros when needed.
    """
    s = str(x)
    # もし s が整数値なら、何もしない。
    if s.find('.') != -1:
        s_digits = len(s[s.find('.'):])       # 小数点以下の桁数を計算
        s = s + '0' * (digits + 1 - s_digits) # 足りない分だけ0を追加
    return s

# %%
@np.vectorize
def add_big_mark(s: Any) -> str:
    """Insert thousands separators into an integer-like string.

    Args:
        s: Value convertible to string.

    Returns:
        Formatted string with commas as thousands separators.
    """
    return f'{s:,}'

# %% [markdown]
# 　文字列のリストを与えると、英文の並列の形に変換する関数です。表記法については[Wikipedia Serial comma](https://en.wikipedia.org/wiki/Serial_comma)を参照し、コードについては[stack overflow:Grammatical List Join in Python [duplicate]](https://stackoverflow.com/questions/19838976/grammatical-list-join-in-python)を参照しました。
# 
# ```python
# choices = ['apple', 'orange', 'grape']
# oxford_comma_or(choices)
# #> 'apple, orange or grape'
# ```

# %%
def oxford_comma(x: Union[str, Sequence[str]], sep_last: str = "and", quotation: bool = True) -> str:
    """Join items into an English list with an Oxford comma.

    Examples:
        >>> oxford_comma_or(["apple", "orange", "grape"])
        " 'apple', 'orange' or 'grape' "

    Args:
        x: A single string or a sequence of strings.
        sep_last: The conjunction used before the final item ('and' or 'or').
        quotation: If True, wrap each item with single quotes.

    Returns:
        A grammatically joined string.
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
  """Join items with Oxford comma using 'and' before the last item.

  Args:
      x: A single string or sequence of strings.
      quotation: If True, wrap each item with single quotes.

  Returns:
      Joined string using 'and'.
  """
  return oxford_comma(x, quotation = quotation, sep_last = 'and')

def oxford_comma_or(x: Union[str, Sequence[str]], quotation: bool = True) -> str:
  """Join items with Oxford comma using 'or' before the last item.

  Args:
      x: A single string or sequence of strings.
      quotation: If True, wrap each item with single quotes.

  Returns:
      Joined string using 'or'.
  """
  return oxford_comma(x, quotation = quotation, sep_last = 'or')


