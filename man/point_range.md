# 数値変数の点推定と区間推定：`eda_tools.mean_qi()`, `eda_tools.median_qi()`, `eda_tools.mean_ci()`

## 概要

数値変数の点推定と区間推定

```python
mean_qi(self, width = 0.95)
median_qi(self, width = 0.95)
mean_ci(self, width = 0.95)
```

## 引数 Argument

- `self`：**pd.DataFrame or pd.Series**（必須）
- `width`：**float**
　分位点区間の幅、もしくは信頼区間の計算に用いる信頼係数。

## 使用例 Examples

```python
from py4stats import eda_tools as eda
import pandas as pd
from palmerpenguins import load_penguins
penguins = load_penguins() # サンプルデータの読み込み

print(penguins['bill_length_mm'].mean_qi().round(2))
#>                  mean  lower  upper
#> variable                           
#> bill_length_mm  43.92  34.81  53.08

print(penguins['bill_length_mm'].median_qi().round(2))
#>                 median  lower  upper
#> variable                            
#> bill_length_mm   44.45  34.81  53.08

print(penguins['bill_length_mm'].mean_ci().round(2))
#>                  mean  lower  upper
#> variable                           
#> bill_length_mm  43.92  43.34   44.5

print(penguins[['bill_length_mm', 'bill_depth_mm']].mean_ci().round(2))
#>                  mean  lower  upper
#> variable                           
#> bill_length_mm  43.92  43.34  44.50
#> bill_depth_mm   17.15  16.94  17.36

print(penguins.groupby('species')[['bill_length_mm']].apply(eda.median_qi).round(2))
#>                           median  lower  upper
#> species   variable                            
#> Adelie    bill_length_mm   38.80  34.08  44.10
#> Chinstrap bill_length_mm   49.55  42.47  54.72
#> Gentoo    bill_length_mm   47.30  42.60  54.26
```
***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md)
