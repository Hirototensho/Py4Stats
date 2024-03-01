# -*- coding: utf-8 -*-
"""bilding_block.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aw_42MYb_e-1UHVxos9niFVkRFmZnzIL

# `py4stats` のプログラミングを補助する関数群

`eda_tools` や `regression_tools` で共通して使う関数をここにまとめておきます。`bilding_block` モジュール自体は外部から呼び出さずに使用することを想定しています。

## 引数のアサーション
"""

import argparse

def match_arg(value, choices, arg_name = 'argument'):
    """
    Simulates the functionality of R's match.arg() function with partial matching in Python.

    Args:
    - value: The value to match against the choices (partially).
    - choices: List of valid choices.

    Returns:
    - The matched value if found in choices (partially), otherwise raises an ArgumentError.
    """
    if(value in choices):
      return value
    else:
      matches = [c for c in choices if value.lower() in c.lower()]
      if len(matches) == 1:
          return matches[0]
      elif len(matches) > 1:
          raise ValueError(
              f"""'{value}' is ambiguous value for '{arg_name}'. Matches multiple choices: {', '.join(matches)}.
              '{arg_name}' must be one of {oxford_comma_or(choices)}."""
              )
      else:
          # raise ValueError(f"No match found for value: '{value}'.")
          raise ValueError(f"'{arg_name}' must be one of {oxford_comma_or(choices)}.")

def arg_match(value, choices, arg_name = 'argument'):
    """
    Simulates the functionality of R's rlang::arg_match() function with partial matching in Python.

    Args:
    - value: The value to match against the choices.
    - choices: List of valid choices.

    Returns:
    - The matched value if found in choices (partially), otherwise raises an ArgumentError.
    """
    if(value in choices):
      return value
    else:
      matches = [c for c in choices if value.lower() in c.lower()]
      if len(matches) >= 1:
        raise ValueError(
            f"""'{arg_name}' must be one of {oxford_comma_or(choices)}.
            Did you mean {' or '.join(matches)}?"""
        )
      else:
        raise ValueError(f"'{arg_name}' must be one of {oxford_comma_or(choices)}.")

"""## 数値などのフォーマット"""

import pandas as pd
import numpy as np
import scipy as sp

# 有意性を表すアスタリスクを作成する関数
def p_stars_row(p_value):
    stars = np.where(p_value <= 0.1, ' *', '')
    stars = np.where(p_value <= 0.05, ' **', stars)
    stars = np.where(p_value <= 0.01, ' ***', stars)
    return stars

p_stars = np.vectorize(p_stars_row)

def pad_zero_row(x, digits = 2):
    s = str(x)
    # もし s が整数値なら、何もしない。
    if s.find('.') != -1:
        s_digits = len(s[s.find('.'):])       # 小数点以下の桁数を計算
        s = s + '0' * (digits + 1 - s_digits) # 足りない分だけ0を追加
    return s

pad_zero = np.vectorize(pad_zero_row, excluded = 'digits')

def add_big_mark_row(s): return  f'{s:,}'
add_big_mark = np.vectorize(add_big_mark_row)

"""　文字列のリストを与えると、英文の並列の形に変換する関数です。表記法については[Wikipedia Serial comma](https://en.wikipedia.org/wiki/Serial_comma)を参照し、コードについては[stack overflow:Grammatical List Join in Python [duplicate]](https://stackoverflow.com/questions/19838976/grammatical-list-join-in-python)を参照しました。

```python
choices = ['apple', 'orange', 'grape']
oxford_comma_or(choices)
#> 'apple, orange or grape'
```
"""

def oxford_comma_and(lst):
    if not lst:
        return ""
    elif len(lst) == 1:
        return str(lst[0])
    return "{} and {}".format(", ".join(lst[:-1]), lst[-1])

def oxford_comma_or(lst):
    if not lst:
        return ""
    elif len(lst) == 1:
        return str(lst[0])
    return "{} or {}".format(", ".join(lst[:-1]), lst[-1])
