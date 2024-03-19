# 数字のフォーマットを変更する関数

## 概要

``` python
style_comma(x, digits = 2, big_mark = ',')

style_currency(x, symbol = '$', digits = 0, big_mark = ',')

style_percent(x, digits = 2, unit = 100, symbol = '%')

pad_zero(x, digits = 2)
```

## 引数 Argument

- `x`：**scalar or array-like of int or float**</br>
- `p_value`：**scalar or array-like of int or float**</br>
- `digits`：**int**</br>
小数点以下の桁数
- `big_mark`：**int**</br>
3桁毎の桁区切りに使用する記号。カンマ `','`, アンダーバー `'_'`, もしくは 非表示 `''` から選ぶことができます。
- `symbol`：**str**</br>
　貨幣記号を表す文字列

## 返り値 Value

- `bilding_block.style_comma()`： 任意の数値に対して、小数点以下を桁数 `digits` に丸め、3桁区切り記号を通過した値を文字列として返します。f-string によるフォーマット `f'{x:{big_mark}.{digits}f}'` を用いて実装されています。
- `bilding_block.style_currency()`： `bild.style_comma()` と同じく任意の数値に対して、小数点以下を桁数 `digits` に丸め、3桁区切り記号を通過した値を文字列として返しますが、さらに貨幣記号を追加します。f-string によるフォーマット `f'{symbol}{x:{big_mark}.{digits}f}'` を用いて実装されています。
- `bilding_block.style_percent()`： 任意の数値をパーセンテージ表示に変換した値を文字列として返します。f-string によるフォーマット `f'{x:,.{digits}%}'` を用いて実装されています。
- `bilding_block.pad_zero()`： float 値に対して少数部分が `digits` で指定された桁数になるまで `0` を末尾に追加した文字列を返す関数です。ただし、整数値に対しては `0` の追加は行いません。


## 使用例 Examples

```python
import numpy as np
from py4stats import bilding_block as bild

x = [2000, 1000, 0.5, 0.11, 0.123]

print(bild.style_number(x).to_list())
#> ['2,000.00', '1,000.00', '0.50', '0.11', '0.12']

print(bild.style_number(x, big_mark = '').to_list())
#> ['2000.00', '1000.00', '0.50', '0.11', '0.12']

print(bild.style_currency(x).to_list())
#> ['$2,000', '$1,000', '$0', '$0', '$0']

print(bild.pad_zero(x, digits = 3))
#> ['2000' '1000' '0.500' '0.110' '0.123']
```

```python
pct = [0.11, 0.06, 0.05, 0.01, 0.00234]

print(bild.style_percent(pct).to_list())
#> ['11.00%', '6.00%', '5.00%', '1.00%', '0.23%']

print(bild.style_percent(pct, unit = 1).to_list())
#> ['0.11%', '0.06%', '0.05%', '0.01%', '0.00%']

print(bild.style_percent(pct, unit = 1000, symbol = '‰').to_list())
#> ['110.00‰', '60.00‰', '50.00‰', '10.00‰', '2.34‰']
```

***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md)
