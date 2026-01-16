# 引数のデータ型によるアサーション
## `building_block.assert_character()` `building_block.assert_logical()` `building_block.assert_numeric()` `building_block.assert_integer()` `building_block.assert_count()` `building_block.assert_float()`

## 概要

　R言語の [`checkmate`](https://mllg.github.io/checkmate/index.html) パッケージの関数群をオマージュした、引数に代入された値が想定されたデータ型ではないときにエラーを出力する関数です。

``` python
assert_character(
    arg: Any, 
    arg_name: Optional[str] = None,
    len_arg: Optional[int] = None,
    len_min: int = 1,
    len_max: Optional[int] = None
    )

assert_logical(
    arg: Any, 
    arg_name: Optional[str] = None,
    len_arg: Optional[int] = None,
    len_min: int = 1,
    len_max: Optional[int] = None
    )

assert_numeric(
    arg: Any,
    arg_name: Optional[str] = None,
    lower = -float('inf'), 
    upper = float('inf'), 
    inclusive: Literal["both", "neither", "left", "right"] = "both",
    len_arg: Optional[int] = None,
    len_min: int = 1,
    len_max: Optional[int] = None
    )

assert_integer(
    arg: Any,
    arg_name: Optional[str] = None,
    lower = -float('inf'), 
    upper = float('inf'), 
    inclusive: Literal["both", "neither", "left", "right"] = "both",
    len_arg: Optional[int] = None,
    len_min: int = 1,
    len_max: Optional[int] = None
    )

assert_count(
    arg: Any,
    arg_name: Optional[str] = None,
    lower = 0, 
    upper = float('inf'), 
    inclusive: Literal["both", "neither", "left", "right"] = "both",
    len_arg: Optional[int] = None,
    len_min: int = 1,
    len_max: Optional[int] = None
    arg_name = None
    )

assert_float(
    arg: Any,
    arg_name: Optional[str] = None,
    lower = -float('inf'), 
    upper = float('inf'), 
    inclusive: Literal["both", "neither", "left", "right"] = "both",
    len_arg: Optional[int] = None,
    len_min: int = 1,
    len_max: Optional[int] = None
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
- `lower`, `upper`：**int or float** `assert_numeric(), assert_integer(), assert_count(), assert_float()` のみ</br>
　`arg` に代入されたオブジェクトの要素が取るべき値の最大値と最小値。
- inclusive：**str**</br>
　`'both', 'neither', 'left', 'right'` から選択できます。引数 `arg` に代入されたオブジェクトの要素を `x` とするとき、次の条件で値の範囲を判定します。
    - `'both'`：`lower <= x <= upper`
    - `'neither'`：`lower < x < upper`
    - `'left'`：`lower <= x < upper`
    - `'right'`：`lower < x <= upper`

## 返り値 Value

　引数 `arg` に代入されたオブジェクトの全ての要素が、アサーションの条件を満たしていれば何も返さず、そうでなければエラーメッセージを出力します。

## 使用例 Examples

```python
from py4stats import building_block as build
x = [1, 2, 3]
y = ['A', 'B', 'C']

build.assert_character(x, arg_name = 'x')
#> ValueError: Argument 'x' must be of type 'str'.

build.assert_character(y, arg_name = 'y')
```

```python
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
build.assert_numeric(
    z, arg_name = 'z', 
    lower = 0, upper = 1, 
    inclusive = 'left'
    )
#> ValueError: Argument 'z' must have value 0 <= x < 1.
```

## 参照

　データ型の判定には[こちらの関数](./is_dtype.md)を使用しています。

***
[Return to **Function reference**.](../reference.md)
