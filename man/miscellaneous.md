# 数字のフォーマットを変更する関数

## 概要


``` python
num_comma(x, digits = 2, big_mark = ',')

num_currency(x, symbol = '$', digits = 0, big_mark = ',')

num_percent(x, digits = 2)

pad_zero(x, digits = 2)

p_stars(p_value)
```

## 引数 Argument

- `x`：**scalar or array-like of int or float**</br>
- `p_value`：**scalar or array-like of int or float**</br>
- `digits`：**int**</br>
小数点以下の桁数
- `big_mark`：**int**</br>
3桁毎の桁区切りに使用する記号。カンマ `’,’`, アンダーバー `’_’`, もしくは 非表示 `’’` から選ぶことができます。
- `symbol`：**str**</br>
　貨幣記号を表す文字列

## 返り値 Value

- `bilding_block.num_comma()`： 任意の数値に対して、小数点以下を桁数 `digits` に丸め、3桁区切り記号を通過した値を文字列として返します。f-string によるフォーマット `f'{x:{big_mark}.{digits}f}'` を用いて実装されています。
- `bilding_block.num_currency()`： `bild.num_comma()` と同じく任意の数値に対して、小数点以下を桁数 `digits` に丸め、3桁区切り記号を通過した値を文字列として返しますが、さらに貨幣記号を追加します。f-string によるフォーマット `f'{symbol}{x:{big_mark}.{digits}f}'` を用いて実装されています。
- `bilding_block.num_percent()`： 任意の数値をパーセンテージ表示に変換した値を文字列として返します。f-string によるフォーマット `f'{x:,.{digits}%}'` を用いて実装されています。
- `bilding_block.pad_zero()`： float 値に対して少数部分が `digits` で指定された桁数になるまで `0` を追加した文字列を返す関数です。
- `bilding_block.p_stars()`：`p_value` に与えられた数値を仮説検定の有意性を示すアスタリスク `*` に変換する関数です。アスタリスクはp-値の値に応じて次のように表示されます。
  - p ≤ 0.1 `*`
  - p ≤ 0.05 `**`
  - p ≤ 0.01 `***`
  - p > 0.1 表示なし

## 使用例 Examples

```python
from py4stats import bilding_block as bild

x = [2000, 1000, 0.5, 0.11, 0.123]

print(bild.num_comma(x))
#> ['2,000.00' '1,000.00' '0.50' '0.11' '0.12']

print(bild.num_comma(x, big_mark = ''))
#> ['2000.00' '1000.00' '0.50' '0.11' '0.12']

print(bild.num_currency(x))
#> ['$2,000' '$1,000' '$0' '$0' '$0']

print(bild.pad_zero(x, digits = 3))
#> ['2000' '1000' '0.500' '0.110' '0.123']
```

```python
pct = [0.11, 0.06, 0.05, 0.01, 0.00234]

print(bild.num_percent(pct))
#> ['11.00%' '6.00%' '5.00%' '1.00%' '0.23%']

print(bild.p_stars(pct))
#> ['' '*' '**' '***' '***']
```

***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md)
