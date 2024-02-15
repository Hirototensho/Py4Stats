# `py4stats` 

　主にR言語でできる「アレ」を Python で再現することを目的としたモジュールです。

## Install

``` 
! pip install git+https://github.com/Hirototensho/py4stats.git
```

## 使用方法

``` python
from py4stats import eda_tools as eda        # 基本統計量やデータの要約など
from py4stats import regression_tools as reg # 回帰分析の要約
```

``` python
import pandas as pd
from palmerpenguins import load_penguins
penguins = load_penguins() # サンプルデータの読み込み

eda.diagnose(penguins)
```
|                   | dtype   |   missing_count |   missing_percent |   unique_count |   unique_rate |
|:------------------|:--------|----------------:|------------------:|---------------:|--------------:|
| species           | object  |               0 |          0        |              3 |      0.872093 |
| island            | object  |               0 |          0        |              3 |      0.872093 |
| bill_length_mm    | float64 |               2 |          0.581395 |            164 |     47.6744   |
| bill_depth_mm     | float64 |               2 |          0.581395 |             80 |     23.2558   |
| flipper_length_mm | float64 |               2 |          0.581395 |             55 |     15.9884   |
| body_mass_g       | float64 |               2 |          0.581395 |             94 |     27.3256   |
| sex               | object  |              11 |          3.19767  |              2 |      0.581395 |
| year              | int64   |               0 |          0        |              3 |      0.872093 |

``` python
# 回帰分析の実行
fit1 = smf.ols('body_mass_g ~ bill_length_mm + species', data = penguins).fit()
fit2 = smf.ols('body_mass_g ~ bill_length_mm + bill_depth_mm + species', data = penguins).fit()
fit3 = smf.ols('body_mass_g ~ bill_length_mm + bill_depth_mm + species + sex', data = penguins).fit()

# 表の作成
compare_tab1 = reg.compare_ols(list_models = [fit1, fit2, fit3])
compare_tab1
```

| term                 | model 1       | model 2        | model 3       |
|:---------------------|:--------------|:---------------|:--------------|
| Intercept            | 153.7397      | -1742.7202 *** | 843.9812 **   |
|                      | (0.5679)      | (0.0000)       | (0.0373)      |
| species[T.Chinstrap] | -885.8121 *** | -539.6864 ***  | -245.1516 *** |
|                      | (0.0000)      | (0.0000)       | (0.0040)      |
| species[T.Gentoo]    | 578.6292 ***  | 1492.8283 ***  | 1443.3525 *** |
|                      | (0.0000)      | (0.0000)       | (0.0000)      |
| bill_length_mm       | 91.4358 ***   | 55.6461 ***    | 26.5366 ***   |
|                      | (0.0000)      | (0.0000)       | (0.0003)      |
| bill_depth_mm        |               | 179.0434 ***   | 87.9328 ***   |
|                      |               | (0.0000)       | (0.0000)      |
| sex[T.male]          |               |                | 437.2007 ***  |
|                      |               |                | (0.0000)      |
| rsquared_adj         | 0.7810        | 0.8258         | 0.8613        |
| nobs                 | 342           | 342            | 333           |
| df                   | 3             | 4              | 5             |


``` python
reg.compare_ols(
    list_models = [fit1, fit2, fit3],
    model_name = ['基本モデル', '嘴の高さ追加', '性別追加'], # モデル名を変更
    stats = 'std_err',        # () 内の値を標準誤差に設定する
    add_stars = False,        # 有意性のアスタリスクなし
    table_style = 'one_line', # 表スタイルを1行表示に設定 'one' でも可能
    digits = 2,               # 小数点以下の桁数を2に設定
    stats_glance = ['rsquared_adj', 'nobs', 'df'] # 例えば nobs を n と省略しても良い
    )
```

| term                 | 基本モデル     | 嘴の高さ追加     | 性別追加        |
|:---------------------|:---------------|:-----------------|:----------------|
| Intercept            | 153.74(268.90) | -1742.72(313.77) | 843.98(403.60)  |
| species[T.Chinstrap] | -885.81(88.25) | -539.69(86.94)   | -245.15(84.60)  |
| species[T.Gentoo]    | 578.63(75.36)  | 1492.83(118.44)  | 1443.35(107.78) |
| bill_length_mm       | 91.44(6.89)    | 55.65(7.23)      | 26.54(7.24)     |
| bill_depth_mm        |                | 179.04(19.10)    | 87.93(20.22)    |
| sex[T.male]          |                |                  | 437.20(49.11)   |
| rsquared_adj         | 0.78           | 0.83             | 0.86            |
| nobs                 | 342            | 342              | 333             |
| df                   | 3              | 4                | 5               |


```python
import matplotlib.pyplot as plt
reg.plot_coef(fit3)
```
![Unknown](https://github.com/Hirototensho/Py4Stats/assets/55335752/7ac8a168-295f-4a61-80ae-fd1dfb394201)
