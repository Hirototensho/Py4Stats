# -*- coding: utf-8 -*-
"""bilding_block.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aw_42MYb_e-1UHVxos9niFVkRFmZnzIL

# `py4stats` のプログラミングを補助する関数群

`eda_tools` や `regression_tools` で共通して使う関数をここにまとめておきます。`bilding_block` モジュール自体は外部から呼び出さずに使用することを想定しています。
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

def is_numeric(x):
  return pandas.api.types.is_numeric_dtype(pd.Series(x))

def is_integer(x):
  return pandas.api.types.is_integer_dtype(pd.Series(x))

def is_float(x):
  return pandas.api.types.is_float_dtype(pd.Series(x))

def assert_character(arg, arg_name = None):
  if(arg_name is None):
      arg_name = argname('arg')
  assert is_character(arg), f"Argment '{arg_name}' must be of type 'str'."

"""### 数値用の `assert_*()` 関数"""

def make_assert_fun(predicate_fun, valid_type, lower = -float('inf'), upper = float('inf')):

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

assert_numeric = make_assert_fun(is_numeric, valid_type = ['int', 'float'])
assert_integer = make_assert_fun(is_integer, valid_type = ['int'])
assert_count = make_assert_fun(is_integer, valid_type = ['positive integer'], lower = 0)
assert_float = make_assert_fun(is_float, valid_type = ['float'])

"""## 数値などのフォーマット"""

# 有意性を表すアスタリスクを作成する関数
@np.vectorize
def p_stars(p_value):
    stars = np.where(p_value <= 0.1, ' *', '')
    stars = np.where(p_value <= 0.05, ' **', stars)
    stars = np.where(p_value <= 0.01, ' ***', stars)
    return stars

@np.vectorize
def pad_zero(x, digits = 2):
    s = str(x)
    # もし s が整数値なら、何もしない。
    if s.find('.') != -1:
        s_digits = len(s[s.find('.'):])       # 小数点以下の桁数を計算
        s = s + '0' * (digits + 1 - s_digits) # 足りない分だけ0を追加
    return s

@np.vectorize
def add_big_mark(s): return  f'{s:,}'

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
