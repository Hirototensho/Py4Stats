# `eda_tools.diagnose()`

## 概要

　R言語の[`dlookr::diagnose()`](https://choonghyunryu.github.io/dlookr/reference/diagnose.data.frame.html)を再現した関数で、データの全般的な状態についての要約を提供します。

``` python
eda.diagnose(self)
```

## 引数

- `self`：`pandas DataFrame`（必須）

## 返り値

- `dtype`：該当する列のpandasにおけるデータの型。「〇〇の個数」や「〇〇の金額」といったデータの `dtype` が `object` になっていたら、文字列として読み込まれているので要注意です。
- `missing_count`：1列のなかで `NaN` などの欠測値になっている数
- `missing_percent`：1列のなかで欠測値が占めている割合で`missing_percent = (missing_count / 行数) * 100` として計算されます。もし `missing_percent = 100` なら、その列は完全に空白です。
- `unique_count`：その列で重複を除外したユニークな値の数。例えばある列の中身が「`a, a, b`」であればユニークな値は `a` と `b` の2つなので `unique_count = 2` です。もし `unique_count = 1` であれば、その行にはたった1種類の値しか含まれていないことが分かりますし、例えば都道府県を表す列の `unique_count` が47より多ければ、都道府県以外のものが混ざっていると考えられます。
- `unique_rate`： サンプルに占めるユニークな値の割合。 `unique_rate = unique_count / 行数` で計算されます。`unique_rate = 1` であれば、全ての行に異なる値が入っています。一般的に実数値の列は `unique_rate` が高くなりますが、年齢の「20代」や価格の「200円代」のように階級に分けられている場合には `unique_rate` が低くなります。

## 使用例 Examples

``` python
import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
penguins = load_penguins() # サンプルデータの読み込み

print(penguins.diagnose().round(4))
#>                      dtype  missing_count  missing_percent  unique_count  unique_rate
#> species             object              0           0.0000             3       0.8721
#> island              object              0           0.0000             3       0.8721
#> bill_length_mm     float64              2           0.5814           164      47.6744
#> bill_depth_mm      float64              2           0.5814            80      23.2558
#> flipper_length_mm  float64              2           0.5814            55      15.9884
#> body_mass_g        float64              2           0.5814            94      27.3256
#> sex                 object             11           3.1977             2       0.5814
#> year                 int64              0           0.0000             3       0.8721
```

[Return to Function reference.](https://github.com/Hirototensho/Py4Stats/blob/main/man/reference.md)
