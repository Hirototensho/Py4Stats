# `py4stats.diagnose()`

## 概要

　R言語の[`dlookr::diagnose()`](https://choonghyunryu.github.io/dlookr/reference/diagnose.data.frame.html)を再現した関数で、データの全般的な状態についての要約を提供します。

``` python
diagnose(self)
```

## 引数

- `self`：`IntoFrameT`（必須）

## 返り値

- `dtype`：該当する列のpandasにおけるデータの型。「〇〇の個数」や「〇〇の金額」といったデータの `dtype` が `object` や `String` になっていたら、文字列として読み込まれているので要注意です。
- `missing_count`：1列のなかで `NaN` などの欠測値になっている数
- `missing_percent`：1列のなかで欠測値が占めている割合で`missing_percent = (missing_count / 行数) * 100` として計算されます。もし `missing_percent = 100` なら、その列は完全に空白です。
- `unique_count`：その列で重複を除外したユニークな値の数。例えばある列の中身が「`a, a, b`」であればユニークな値は `a` と `b` の2つなので `unique_count = 2` です。もし `unique_count = 1` であれば、その行にはたった1種類の値しか含まれていないことが分かりますし、例えば都道府県を表す列の `unique_count` が47より多ければ、都道府県以外のものが混ざっていると考えられます。
- `unique_rate`： サンプルに占めるユニークな値の割合。 `unique_rate = unique_count / 行数` で計算されます。`unique_rate = 1` であれば、全ての行に異なる値が入っています。一般的に、実数値の列は `unique_rate` が高くなりますが、年齢の「20代」や価格の「200円代」のように階級に分けられている場合には `unique_rate` が低くなります。

## 使用例 Examples

``` python
import py4stats as py4st
import pandas as pd
from palmerpenguins import load_penguins
penguins = load_penguins() # サンプルデータの読み込み

print(py4st.diagnose(penguins).round(4))
#>              columns    dtype  missing_count  missing_percent  unique_count  unique_rate
#> 0            species   object              0           0.0000             3       0.8721
#> 1             island   object              0           0.0000             3       0.8721
#> 2     bill_length_mm  float64              2           0.5814           165      47.9651
#> 3      bill_depth_mm  float64              2           0.5814            81      23.5465
#> 4  flipper_length_mm  float64              2           0.5814            56      16.2791
#> 5        body_mass_g  float64              2           0.5814            95      27.6163
#> 6                sex   object             11           3.1977             3       0.8721
#> 7               year    int64              0           0.0000             3       0.8721
```

***
[Return to **Function reference**.](../reference.md)

