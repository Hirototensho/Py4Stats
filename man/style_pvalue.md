# p-値のフォーマットを変更する関数

## 概要

　R言語の [`style_pvalue()`](https://www.danieldsjoberg.com/gtsummary/reference/style_pvalue.html) と [`gtools::stars.pval()`](https://search.r-project.org/CRAN/refmans/gtools/html/stars.pval.html) をオマージュした関数でp-値を見やすい形のフォーマットに変換します。


``` python
style_pvalue(
    p_value, 
    digits = 3, 
    prepend_p = False, 
    p_min = 0.001, 
    p_max = 0.9
    )

p_stars(
    p_value, 
    stars = {'***':0.01, '**':0.05, '*':0.1}
    )
```

## 引数 Argument

- `p_value`：**scalar or array-like of int or float**</br>
- `digits`：**int**（`style_pvalue()` のみ）</br>
　小数点以下の桁数
- `prepend_p`：**bool**（`style_pvalue()` のみ）</br>
　出区力に接頭辞 `’p’` を追加するかどうかを表す論理値。False であれば追加されず、True であれば追加されます。
- `p_min`：**int**（`style_pvalue()` のみ）</br>
　p-値を実数値で表示する最小値。`p_value` がこの値を下回る場合、`’<p_min’` もしくは `’p<p_min’` の形で表示されます。
- `p_max`：**int**（`style_pvalue()` のみ）</br>
　p-値を実数値で表示する最大値。`p_value` がこの値を下回る場合、`’>p_max’` もしくは `’p>p_max’` の形で表示されます。
- `stars`：**dict**（`p_stars()` のみ）</br>
　有意性を示す記号を key に、表示を切り替える閾値を値にもつ辞書オブジェクト。使用方法は下記を参照して下さい。

## 返り値 Value

　フォーマットされたp-値を表す pd.Series を出力します。`bilding_block.style_pvalue()` では引数 `p_value` に与えられた数値を指定された桁数に丸めた値を表示し、指定された範囲を外れる値については `’<p_min’` や ’>p_max’` の書式にへんかんします。  
　`bilding_block.p_stars()` では仮説検定の有意性を示すアスタリスク `*` に変換します。初期設定ではアスタリスクはp-値の値に応じて次のように表示されます。

  - p ≤ 0.1 `*`
  - p ≤ 0.05 `**`
  - p ≤ 0.01 `***`
  - p > 0.1 表示なし

## 使用例 Examples

```python

from py4stats import bilding_block as bild
p_value = [
    0.999, 0.5028, 0.2514, 0.197, 0.10, 
    0.0999, 0.06, 0.03, 0.002, 0.0002
    ]

print(bild.style_pvalue(p_value).to_list())
#> ['>0.9', '0.503', '0.251', '0.197', '0.1', '0.1', '0.06', '0.03', '0.002', '<0.001']

print(bild.style_pvalue(p_value, prepend_p = True).to_list())
#> ['p>0.9', 'p=0.503', 'p=0.251', 'p=0.197', 'p=0.1', 'p=0.1', 'p=0.06', 'p=0.03', 'p=0.002', 'p<0.001']

print(bild.p_stars(p_value).to_list())
#> ['', '', '', '', '*', '*', '*', '**', '***', '***']

# R言語の stats::summary.lm() や gtools::stars.pval() を再現する場合。
stars_dict = {'***':0.001, '**':0.01, '*': 0.05, '.':0.1}
print(bild.p_stars(p_value, stars = stars_dict).to_list())
#> ['', '', '', '', '.', '.', '.', '*', '**', '***']
```
***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md)
