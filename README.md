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

### `.eda_tools`

　`eda.diagnose()`：R言語の[`dlookr::diagnose()`](https://choonghyunryu.github.io/dlookr/reference/diagnose.data.frame.html)を再現した関数です。

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

`eda.tabyl()`：R言語の [`janitor::tabyl()`](https://sfirke.github.io/janitor/reference/tabyl.html)にいくつかの `adorn_` 関数を追加した状態を再現した関数です。

``` python
eda.tabyl(penguins, 'species', 'island')
```
| species   | Biscoe       | Dream       | Torgersen   |   All |
|:----------|:-------------|:------------|:------------|------:|
| Adelie    | 44 (28.9%)   | 56 (36.8%)  | 52 (34.2%)  |   152 |
| Chinstrap | 0 (0.0%)     | 68 (100.0%) | 0 (0.0%)    |    68 |
| Gentoo    | 124 (100.0%) | 0 (0.0%)    | 0 (0.0%)    |   124 |
| All       | 168 (48.8%)  | 124 (36.0%) | 52 (15.1%)  |   344 |


### `.regression_tools`

``` python
# 回帰分析の実行
fit1 = smf.ols('body_mass_g ~ bill_length_mm + species', data = penguins).fit()
fit2 = smf.ols('body_mass_g ~ bill_length_mm + bill_depth_mm + species', data = penguins).fit()
fit3 = smf.ols('body_mass_g ~ bill_length_mm + bill_depth_mm + species + sex', data = penguins).fit()

reg.compare_ols(list_models = [fit1, fit2, fit3]) # 表の作成
```

| term                 | model 1       | model 2        | model 3       |
|:---------------------|:--------------|:---------------|:--------------|
| Intercept            | 153.7397      | -1742.7202 *** | 843.9812 **   |
|                      | (268.9012)    | (313.7697)     | (403.5956)    |
| species[T.Chinstrap] | -885.8121 *** | -539.6864 ***  | -245.1516 *** |
|                      | (88.2502)     | (86.9425)      | (84.5952)     |
| species[T.Gentoo]    | 578.6292 ***  | 1492.8283 ***  | 1443.3525 *** |
|                      | (75.3623)     | (118.4442)     | (107.7844)    |
| bill_length_mm       | 91.4358 ***   | 55.6461 ***    | 26.5366 ***   |
|                      | (6.8871)      | (7.2326)       | (7.2436)      |
| bill_depth_mm        |               | 179.0434 ***   | 87.9328 ***   |
|                      |               | (19.0997)      | (20.2192)     |
| sex[T.male]          |               |                | 437.2007 ***  |
|                      |               |                | (49.1098)     |
| rsquared_adj         | 0.7810        | 0.8258         | 0.8613        |
| nobs                 | 342           | 342            | 333           |
| df                   | 3             | 4              | 5             |


`reg.compare_ols()` の実行結果は `Pandas` の `DataFrame` として出力されるため、`.xlsx`. ファイルなどに変換することができます。また、用途に応じて表の体裁を調整できるようにしています。

``` python
reg.compare_ols(
    list_models = [fit1, fit2, fit3],
    model_name = ['基本モデル', '嘴の高さ追加', '性別追加'], # モデル名を変更
    stats = 'p_value',        # () 内の値をP-値に変更する
    add_stars = False,        # 有意性のアスタリスクなし
    table_style = 'one_line', # 表スタイルを1行表示に設定 'one' でも可能
    digits = 2                # 小数点以下の桁数を2に設定
    )
```

| term                 | 基本モデル    | 嘴の高さ追加   | 性別追加      |
|:---------------------|:--------------|:---------------|:--------------|
| Intercept            | 153.74(0.57)  | -1742.72(0.00) | 843.98(0.04)  |
| species[T.Chinstrap] | -885.81(0.00) | -539.69(0.00)  | -245.15(0.00) |
| species[T.Gentoo]    | 578.63(0.00)  | 1492.83(0.00)  | 1443.35(0.00) |
| bill_length_mm       | 91.44(0.00)   | 55.65(0.00)    | 26.54(0.00)   |
| bill_depth_mm        |               | 179.04(0.00)   | 87.93(0.00)   |
| sex[T.male]          |               |                | 437.20(0.00)  |
| rsquared_adj         | 0.78          | 0.83           | 0.86          |
| nobs                 | 342           | 342            | 333           |
| df                   | 3             | 4              | 5             |

```python
import matplotlib.pyplot as plt
reg.plot_coef(fit3)
```
![Unknown](https://github.com/Hirototensho/Py4Stats/assets/55335752/7ac8a168-295f-4a61-80ae-fd1dfb394201)
