# データ型を判定する論理関数
## `bilding_block.is_character()` `bilding_block.is_logical()` `bilding_block.is_numeric()` `bilding_block.is_integer()` `bilding_block.is_float()`

## 概要

```python
is_character(x)

is_logical(x)

is_numeric(x)

is_integer(x)

is_float(x)
```

## 引数 Argument

- `x`（必須）**array, list, or pd.Series**</br>

## 返り値 Value

引数 `x` が次の型であるときに、True を返します。

- `is_character()`：**str**
- `is_logical()`：**bool**
- `is_numeric()`：**int, float or bool**
- `is_integer()`：**int or bool**
- `is_float()`：**float**

## 使用例 Examples

```python
from py4stats import bilding_block as bild
x_str = ['A', 'B']
x_bool = [True, False, True]
x_int = [1, 2, 3]
x_float = [0, 1, 2.1, 0.5]
x_list = [x_str, x_bool, x_int, x_float]

print([bild.is_character(x) for x in x_list])
#> [True, False, False, False]

print([bild.is_logical(x) for x in x_list])
#> [False, True, False, False]

print([bild.is_numeric(x) for x in x_list])
#> [False, True, True, True]

print([bild.is_integer(x) for x in x_list])
#> [False, False, True, False]

print([bild.is_float(x) for x in x_list])
#> [False, False, False, True]
```
***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md)
