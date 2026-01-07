# 数値変数の点推定と区間推定：`py4stats.mean_qi()` `py4stats.median_qi()` `py4stats.mean_ci()`

## 概要

　R言語の [`ggdist::mean_qi()`](https://mjskay.github.io/ggdist/reference/point_interval.html) をオマージュした数値変数の点推定と区間推定を行う関数です。

```python
mean_qi(self, width = 0.95)

median_qi(self, width = 0.95)

mean_ci(self, width = 0.95)
```

## 引数 Argument

- `self`：**pd.DataFrame or pd.Series**（必須）
- `width`：**float**<br>
　分位点区間の幅、もしくは信頼区間の計算に用いる信頼係数。

## 使用例 Examples

```python
import py4stats as py4st
import pandas as pd
from palmerpenguins import load_penguins
penguins = load_penguins() # サンプルデータの読み込み

print(penguins['bill_length_mm'].mean_qi().round(2))
#>          variable   mean  lower  upper
#> 0  bill_length_mm  43.92   34.8   53.1


print(penguins['bill_length_mm'].median_qi().round(2))
#>          variable  median  lower  upper
#> 0  bill_length_mm   44.45   34.8   53.1

print(penguins['bill_length_mm'].mean_ci().round(2))
#>          variable   mean  lower  upper
#> 0  bill_length_mm  43.92  43.26  44.58

print(penguins[['bill_length_mm', 'bill_depth_mm']].mean_ci().round(2))
#>          variable   mean  lower  upper
#> 0  bill_length_mm  43.92  43.26  44.58
#> 1   bill_depth_mm  17.15  16.91  17.39

print(penguins.groupby('species')[['bill_length_mm']].apply(py4st.median_qi).round(2))
#>                    variable  median  lower  upper
#> species                                          
#> Adelie    0  bill_length_mm   38.80  34.05  44.10
#> Chinstrap 0  bill_length_mm   49.55  42.45  55.00
#> Gentoo    0  bill_length_mm   47.30  42.65  53.85
```
***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md)
