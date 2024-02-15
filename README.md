# `py4stats` 

　回帰分析を中心とした統計分析を行う際によく使うR言語の機能を Python で再現することを目的としたモジュールです。

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

`reg.coefplot()`：回帰係数の可視化。R言語の[`coefplot::coefplot()`](https://cran.r-project.org/web/packages/coefplot/index.html)を参考にしました。

```python
import matplotlib.pyplot as plt
reg.coefplot(fit3)
```
![Unknown](https://github.com/Hirototensho/Py4Stats/assets/55335752/637437c3-f943-4817-a1ad-21bbd538e97d)


```python
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots(1, 2, figsize = (2.2 * 5, 5), dpi = 100)

reg.coefplot(fit2, ax = ax[0])
ax[0].set_xlim(-900, 1800)

reg.coefplot(fit3, ax = ax[1])
ax[1].set_xlim(-900, 1800);
```

![Unknown](https://github.com/Hirototensho/Py4Stats/assets/55335752/0f11205b-5090-4b45-9a2e-7db7be3cc0f4)

`reg.compare_mfx()` と `reg.mfxplot()`：それぞれ `reg.compare_ols()` と `reg.coefplot()` の一般化線型モデルバージョンで、`statsmodels` ライブラリの[`.get_margeff()`](https://www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.DiscreteResults.get_margeff.html)メソッドから得られた限界効果の推定結果を表示します。

```python
# ロジスティック回帰の実行
fit_logit1 = smf.logit('female ~ body_mass_g + bill_length_mm + bill_depth_mm', data = penguins).fit()
fit_logit2 = smf.logit('female ~ body_mass_g + bill_length_mm + bill_depth_mm + species', data = penguins).fit()

reg.compare_mfx([fit_logit1, fit_logit2])
```
| term                 | model 1     | model 2     |
|:---------------------|:------------|:------------|
| body_mass_g          | -0.0004 *** | -0.0003 *** |
|                      | (0.0000)    | (0.0000)    |
| bill_length_mm       | -0.0053     | -0.0357 *** |
|                      | (0.0036)    | (0.0070)    |
| bill_depth_mm        | -0.1490 *** | -0.1098 *** |
|                      | (0.0051)    | (0.0175)    |
| species[T.Chinstrap] |             | 0.4172 ***  |
|                      |             | (0.0848)    |
| species[T.Gentoo]    |             | 0.3527 ***  |
|                      |             | (0.1308)    |
| prsquared            | 0.5647      | 0.6187      |
| nobs                 | 342         | 342         |
| df                   | 3           | 5           |


```python
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots(1, 2, figsize = (2.2 * 5, 5), dpi = 100)

reg.mfxplot(fit_logit1, ax = ax[0])
ax[0].set_xlim(-0.2, 0.85)

reg.mfxplot(fit_logit2, ax = ax[1])
ax[1].set_xlim(-0.2, 0.85);
```

![Unknown](https://github.com/Hirototensho/Py4Stats/assets/55335752/17ed1a82-5e17-4933-88f5-538a0ce081e0)
