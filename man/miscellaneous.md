# 表の作成を補助するその他の関数

## 概要

``` python
pad_zero(x, digits = 2)

p_stars(p_value)
```

　`regression_tools.pad_zero()` はfloat 値に対して少数部分が `digits` で指定された桁数になるまで `0` を追加した文字列を返す関数で、`regression_tools.p_stars()` は `p_value` に与えられた数値を仮説検定の有意性を示すアスタリスク `*` に変換する関数です。アスタリスクはp-値の値に応じて次のように表示されます。

- p ≤ 0.1 `*`
- p ≤ 0.05 `**`
- p ≤ 0.01 `***`
- p > 0.1 表示なし

## 引数

- `x`：**scalar or array-like**</br>
- `digits`：**scalar**</br>
- `p_value`：**scalar or array-like**</br>

## 使用例

``` python
print(reg.pad_zero([0.11, 0.123, 0.5], digits = 3))
#> ['0.110' '0.123' '0.500']

print(reg.p_stars([0.11, 0.06, 0.05, 0.01]))
#> ['' ' *' ' **' ' ***']
```
