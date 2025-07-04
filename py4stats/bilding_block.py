# -*- coding: utf-8 -*-
"""bilding_block.ipynb

# `py4stats` のプログラミングを補助する関数群

`eda_tools` や `regression_tools` で共通して使う関数をここにまとめておきます。`bilding_block` モジュール自体は外部から呼び出さずに使用することを想定しています。

`bilding_block` モジュールに実装された主要な関数の依存関係

```python
arg_match()                       # 文字列やリストの引数を有効値と突き合わせてチェック
└─ arg_match0()                  # 単一値の照合処理を担当
   └─ oxford_comma_or()         # 候補が複数ある場合のメッセージ整形に使用
       └─ oxford_comma()        # 英語表現の並列化（Oxford comma）

match_arg()                      # Rの match.arg に類似、部分一致で引数を照合
└─ oxford_comma_or()            # 候補の整形出力（上記と共通）

make_assert_type()              # 型アサート用関数の function factory（文字列・論理型用など）
└─ predicate_fun()             # 呼び出し側が指定する型チェック関数（例: is_character）

make_assert_numeric()           # 数値アサート関数のfunction factory（範囲チェック付き）
└─ oxford_comma_or()           # エラーメッセージに候補を整形表示
   └─ oxford_comma()

p_stars()                        # p値にアスタリスク（***など）を付与
├─ assert_numeric()            # 入力値チェック
└─ pd.cut()                    # 範囲に応じたカテゴリ化

style_pvalue()                  # p値の整形出力（しきい値による置き換え）
├─ assert_numeric()
├─ assert_count()
├─ pd.Series()
└─ pandas.mask()               # 条件によって文字列を変換

style_number()                  # 数値を桁区切り付きで文字列整形
├─ assert_numeric()
├─ assert_count()
└─ arg_match()                 # 区切り記号の妥当性チェック

style_currency()                # 金額表記に整形（通貨記号 + 整数）
├─ assert_numeric()
├─ assert_count()
└─ arg_match()

style_percent()                 # パーセント表記に整形（単位変換含む）
├─ assert_numeric()
└─ assert_count()

num_comma()                     # 数値をコンマ区切り付きで整形（ベクトル化関数）
└─ arg_match()

num_currency()                 # 金額整形（num_commaと同様）
└─ arg_match()

pad_zero()                      # 小数点以下のゼロを桁数に応じて補完（文字列化）
（依存なし）

add_big_mark()                 # 桁区切りを追加（f'{:,}'形式）
（依存なし）

oxford_comma_and()              # A, B and C のような整形
└─ oxford_comma()

oxford_comma_or()               # A, B or C のような整形
└─ oxford_comma()

oxford_comma()                  # 英文の並列表記として候補リストを整形
（依存なし）

```
"""



import pandas as pd
import numpy as np
import scipy as sp
from varname import argname

"""## 引数のアサーション"""

import argparse

def match_arg(arg, values, arg_name = 'argument'):
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

def arg_match0(arg, values, arg_name = None):
    """
    Simulates the functionality of R's rlang::arg_match() function with partial matching in Python.

    Args:
    - arg: The arg to match against the values.
    - values: List of valid values.

    Returns:
    - The matched arg if found in values, otherwise raises an ArgumentError.
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

from varname import argname
def arg_match(arg, values, arg_name = None, multiple = False):
  """
  Simulates the functionality of R's rlang::arg_match() function with partial matching in Python.

  Args:
  - arg: The list or str of arg to match against the values.
  - values: List of valid values.
  - arg_name : name of argument.
  - multiple: Whether multiple values are allowed for the arg.

  Returns:
  - The matched arg if found in values, otherwise raises an ArgumentError.
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

"""## タイプチェックを行う関数"""

from varname import argname
import pandas.api.types

def is_character(x):
  return pandas.api.types.is_string_dtype(pd.Series(x))

def is_logical(x):
  return pandas.api.types.is_bool_dtype(pd.Series(x))

def is_numeric(x):
  return pandas.api.types.is_numeric_dtype(pd.Series(x))

def is_integer(x):
  return pandas.api.types.is_integer_dtype(pd.Series(x))

def is_float(x):
  return pandas.api.types.is_float_dtype(pd.Series(x))

def make_assert_type(predicate_fun, valid_type):

  def func(arg, arg_name = None):
    if(arg_name is None):
      arg_name = argname('arg')

    assert predicate_fun(arg), f"Argment '{arg_name}' must be of" +\
      f" type {oxford_comma_or(valid_type)}"

  return func

assert_character = make_assert_type(is_character, valid_type = ['str'])
assert_logical = make_assert_type(is_logical, valid_type = ['bool'])

"""### 数値用の `assert_*()` 関数"""

def make_assert_numeric(predicate_fun, valid_type, lower = -float('inf'), upper = float('inf')):

  def func(arg, lower = lower, upper = upper, inclusive = 'both', arg_name = None):
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

assert_numeric = make_assert_numeric(is_numeric, valid_type = ['int', 'float'])
assert_integer = make_assert_numeric(is_integer, valid_type = ['int'])
assert_count = make_assert_numeric(is_integer, valid_type = ['positive integer'], lower = 0)
assert_float = make_assert_numeric(is_float, valid_type = ['float'])

"""## 数値などのフォーマット"""

def p_stars(p_value, stars = {'***':0.01, '**':0.05, '*':0.1}):
  # stars のラベルに上限値を追加
  stars = stars.copy()
  stars[''] = float('inf')

  # pd.Series に変換した上で、
  # pd.cut() の bin として使用するときにエラーを生じないよう、昇順にソート
  stars = pd.Series(stars).sort_values(ascending = True)

  assert_numeric(p_value, lower = 0)
  assert_numeric(stars, lower = 0)

  # 0 に stars の値を追加したものを bins とします。
  bins = [0]
  bins.extend(stars.to_list())

  styled = pd.cut(
      p_value,
      bins = bins,
      labels = stars.index,
      ordered = True,
      duplicates = 'drop'
      ).astype(str)

  return pd.Series(styled)

def style_pvalue(p_value, digits = 3, prepend_p = False, p_min = 0.001, p_max = 0.9):
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
def num_comma(x, digits = 2, big_mark = ','):
  assert_count(digits)
  arg_match(big_mark, [',', '_', ''])
  return f'{x:{big_mark}.{digits}f}'

@np.vectorize
def num_currency(x, symbol = '$', digits = 0, big_mark = ','):
  assert_count(digits)
  arg_match(big_mark, [',', '_', ''])
  return f'{symbol}{x:{big_mark}.{digits}f}'

@np.vectorize
def num_percent(x, digits = 2): return f'{x:.{digits}%}'

def style_number(x, digits = 2, big_mark = ','):
  x = pd.Series(x)

  assert_numeric(x)
  assert_count(digits)

  arg_match(big_mark, [',', '_', ''])

  return x.apply(lambda v: f'{v:{big_mark}.{digits}f}')

def style_currency(x, symbol = '$', digits = 0, big_mark = ','):
  x = pd.Series(x)

  assert_numeric(x)
  assert_count(digits)

  arg_match(big_mark, [',', '_', ''])

  return x.apply(lambda v: f'{symbol}{v:{big_mark}.{digits}f}')

def style_percent(x, digits = 2, unit = 100, symbol = '%'):
  x = pd.Series(x)

  assert_numeric(x)
  assert_count(digits)

  return x.apply(lambda v: f'{v*unit:.{digits}f}{symbol}')

@np.vectorize
def pad_zero(x, digits = 2):
    s = str(x)
    # もし s が整数値なら、何もしない。
    if s.find('.') != -1:
        s_digits = len(s[s.find('.'):])       # 小数点以下の桁数を計算
        s = s + '0' * (digits + 1 - s_digits) # 足りない分だけ0を追加
    return s

@np.vectorize
def add_big_mark(s): return f'{s:,}'

"""　文字列のリストを与えると、英文の並列の形に変換する関数です。表記法については[Wikipedia Serial comma](https://en.wikipedia.org/wiki/Serial_comma)を参照し、コードについては[stack overflow:Grammatical List Join in Python [duplicate]](https://stackoverflow.com/questions/19838976/grammatical-list-join-in-python)を参照しました。

```python
choices = ['apple', 'orange', 'grape']
oxford_comma_or(choices)
#> 'apple, orange or grape'
```
"""

def oxford_comma(x, sep_last = 'and', quotation = True):
    if isinstance(x, str):
      if(quotation): return f"'{x}'"
      else: return x
    else:
      if(quotation): x = [f"'{s}'" for s in x]
    if(len(x) == 1):
      return f"{x[0]}"
    else:
      return ", ".join(x[:-1]) + f" {sep_last} " + x[-1]

def oxford_comma_and(x, quotation = True):
  return oxford_comma(x, quotation = quotation, sep_last = 'and')

def oxford_comma_or(x, quotation = True):
  return oxford_comma(x, quotation = quotation, sep_last = 'or')
