# 引数のデータ型によるアサーション
## `bilding_block.assert_character()` `bilding_block.assert_numeric()` `bilding_block.assert_integer()` `bilding_block.assert_count()` `bilding_block.assert_float()`

## 概要

　引数に代入された値が想定されたデータ型ではないときにエラーを出力する関数です。

``` python
assert_character(arg, arg_name = None)

assert_logical(arg, arg_name = None)

assert_numeric(
    arg, 
    lower = -float('inf'), 
    upper = float('inf'), 
    inclusive = 'both', 
    arg_name = None
    )

assert_integer(
    arg, 
    lower = -float('inf'), 
    upper = float('inf'), 
    inclusive = 'both', 
    arg_name = None
    )

assert_count(
    arg, 
    lower = 0, 
    upper = float('inf'), 
    inclusive = 'both', 
    arg_name = None
    )

assert_float(
    arg, 
    lower = -float('inf'), 
    upper = float('inf'), 
    inclusive = 'both', 
    arg_name = None
    )
```

　それぞれの関数は第一引数 `arg` に代入された array-like オブジェクトの要素が、次の型ではない場合にエラーを出力します。

- `assert_character()`：**str**
- `assert_numeric()`：**int or float**
- `assert_integer()`：**int**
- `assert_count()`：**int**
- `assert_float()`：**float**

## 引数 Argument

- `arg`（必須）**array-like**</br>
　適正かどうかを判断したい引数　
- `arg_name`：**str**</br>
　エラーメッセージに表示する引数の名前。指定されなかった場合（初期設定）、引数 `arg` に代入されたオブジェクトの名称を表示します。なお、この機能は [`varname.argname()`](https://github.com/pwwang/python-varname?tab=readme-ov-file)関数を使って実装されています。
- `lower, upper` **int or float** `assert_numeric(), assert_integer(), assert_count(), assert_float()` のみ</br>
　`arg` に代入されたオブジェクトの要素が取るべき値の最大値と最小値。
- inclusive **str**
　'both', 'neither', 'left', 'right' から選択できます。引数 `arg` に代入されたオブジェクトの要素を `x` とするとき、次の条件で値の範囲を判定します。
    - `'both'`：`lower <= x <= upper`
    - `'neither'`：`lower < x < upper`
    - `'left'`：`lower <= x < upper`
    - `'right'`：`lower < x <= upper`

## 返り値 Value

　引数 `arg` に代入されたオブジェクトの全ての要素が、アサーションの条件を満たしていれば何も返さず、そうでなければエラーメッセージを出力します。

## 使用例 Examples

```python
from py4stats import bilding_block as bild
x = [1, 2, 3]
y = ['A', 'B', 'C']

bild.assert_character(x)
#> AssertionError: Argment 'x' must be of type 'str'.

bild.assert_character(y)
```

```python
bild.assert_numeric(x)

bild.assert_numeric(y)
#> AssertionError: Argment 'y' must be of type 'int' or 'float'.

z = [0.1, 0.3, 0.6]
bild.assert_numeric(z, lower = 0, upper = 1)

z.extend([2, 3])
bild.assert_numeric(z, lower = 0, upper = 1)
#> AssertionError: Argment 'z' must have value 0 <= x <= 1.
#>                element '3' and '4' of 'z' not sutisfy the condtion.

z = 1
bild.assert_numeric(z, lower = 0, upper = 1, inclusive = 'left')
#> AssertionError: Argment 'z' must have value 0 <= x < 1.
```
***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md)
