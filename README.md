# `py4stats` 

　回帰分析を中心とした統計分析の場面で、よく使うR言語の機能を Python で再現することを目的としたモジュールです。`py4stats` モジュールで実装されている関数の一覧は [Function reference](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md) を参照してください。

## Installation

``` python
pip install git+https://github.com/Hirototensho/py4stats.git
```

一部の関数は [`pandas-flavor`](https://pypi.org/project/pandas-flavor/)、[`varname`](https://github.com/pwwang/python-varname?tab=readme-ov-file) および [`pure_eval`](https://pypi.org/project/pure-eval/) ライブラリの機能を使って実装しているため、事前のインストールをお願いいたします。

``` python
pip install pandas_flavor varname pure_eval
```
## 使用方法

``` python
from py4stats import eda_tools as eda        # 基本統計量やデータの要約など
from py4stats import regression_tools as reg # 回帰分析の要約
```

### `py4stats.eda_tools`

　探索的データ解析と前処理に関する機能を提供するモジュールです。一部の関数は [`pandas-flavor`](https://pypi.org/project/pandas-flavor/)ライブラリの機能を使って実装しており、[`pandas.DataFrame`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) のメソッドと同じ構文で利用することができます。

　[`eda.diagnose()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/diagnose.md)：R言語の[`dlookr::diagnose()`](https://choonghyunryu.github.io/dlookr/reference/diagnose.data.frame.html)を再現した関数で、データの全般的な状態についての要約を提供します。

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

[`eda.tabyl()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/tabyl.md)：R言語の [`janitor::tabyl()`](https://sfirke.github.io/janitor/reference/tabyl.html)を参考にした、クロス集計表を作成する関数です。

``` python
print(penguins.tabyl('island', 'species'))
#> species         Adelie   Chinstrap       Gentoo  All
#> island                                              
#> Biscoe      44 (26.2%)    0 (0.0%)  124 (73.8%)  168
#> Dream       56 (45.2%)  68 (54.8%)     0 (0.0%)  124
#> Torgersen  52 (100.0%)    0 (0.0%)     0 (0.0%)   52
#> All        152 (44.2%)  68 (19.8%)  124 (36.0%)  344
```
　[`eda.freq_table()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/freq_table.md)：R言語の[`DescTools::Freq()`](https://cran.r-project.org/web/packages/DescTools/DescTools.pdf)をオマージュした、1変数の度数分布表を計算する関数。度数 `freq` と相対度数 `perc` に加えて、それぞれの累積値を計算します。

``` python
print(penguins.freq_table('species'))
#>            freq      perc  cumfreq   cumperc
#> species                                     
#> Adelie      152  0.441860      152  0.441860
#> Gentoo      124  0.360465      276  0.802326
#> Chinstrap    68  0.197674      344  1.000000
```

引数 `group` を指定すると、グループ別の度数分布表を計算できます。

``` python
penguins2 = penguins.assign(bill_length_mm2 = pd.cut(penguins['bill_length_mm'], 6))

print(
    penguins2.freq_table(['species', 'bill_length_mm2'], sort = False)
    )
#>                             freq      perc  cumfreq   cumperc
#> species   bill_length_mm2
#> Adelie    (32.072, 38.975]    79  0.523179       79  0.523179
#>           (38.975, 45.85]     71  0.470199      150  0.993377
#>           (45.85, 52.725]      1  0.006623      151  1.000000
#>           (52.725, 59.6]       0  0.000000      151  1.000000
#> Chinstrap (32.072, 38.975]     0  0.000000        0  0.000000
#>           (38.975, 45.85]     13  0.191176       13  0.191176
#>           (45.85, 52.725]     50  0.735294       63  0.926471
#>           (52.725, 59.6]       5  0.073529       68  1.000000
#> Gentoo    (32.072, 38.975]     0  0.000000        0  0.000000
#>           (38.975, 45.85]     40  0.325203       40  0.325203
#>           (45.85, 52.725]     78  0.634146      118  0.959350
#>           (52.725, 59.6]       5  0.040650      123  1.000000
```

　[`eda.remove_empty()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/remove_empty_constant.md)：完全に空白な列や行の削除する関数。R言語の [`janitor::remove_empty()`](https://sfirke.github.io/janitor/reference/remove_empty.html) をオマージュした関数で、全ての要素が `NaN` である列や行をデータフレームから除外します。

``` python
penguins2 = penguins.loc[:, ['species', 'body_mass_g']].copy()
penguins2.loc[:, 'empty'] = np.nan
penguins2.loc[344, :] = np.nan

print(penguins2.tail(3))
#>        species  body_mass_g  empty
#> 342  Chinstrap       4100.0    NaN
#> 343  Chinstrap       3775.0    NaN
#> 344        NaN          NaN    NaN

# 完全に空白な行と列を削除。
print(penguins2.remove_empty(quiet = False).tail(3))
#> Removing 1 empty column(s) out of 3 columns(Removed: empty).
#> Removing 1 empty row(s) out of 345 rows(Removed: 344). 
#>        species  body_mass_g
#> 341  Chinstrap       3775.0
#> 342  Chinstrap       4100.0
#> 343  Chinstrap       3775.0

# 完全に空白な列のみ削除。
print(penguins2.remove_empty(rows = False, quiet = False).tail(3))
#> Removing 1 empty column(s) out of 3 columns(Removed: empty).
#>        species  body_mass_g
#> 342  Chinstrap       4100.0
#> 343  Chinstrap       3775.0
#> 344        NaN          NaN

# 完全に空白な行のみ削除。
print(penguins2.remove_empty(cols = False, quiet = False).tail(3))
#> Removing 1 empty row(s) out of 345 rows(Removed: 344). 
#>        species  body_mass_g  empty
#> 341  Chinstrap       3775.0    NaN
#> 342  Chinstrap       4100.0    NaN
#> 343  Chinstrap       3775.0    NaN
```

　[`eda.remove_constant()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/remove_empty_constant.md)：定数列の削除。R言語の [`janitor::remove_constant()`](https://sfirke.github.io/janitor/reference/remove_constant.html) をオマージュした関数で、1種類だけの要素からなる列をデータフレームから除外します。
``` python
penguins2 = penguins.loc[:, ['species', 'body_mass_g']].copy()
penguins2.loc[:, 'constant'] = 'c'

print(penguins2.head(3))
#>   species  body_mass_g constant
#> 0  Adelie       3750.0        c
#> 1  Adelie       3800.0        c
#> 2  Adelie       3250.0        c

print(penguins2.remove_constant(quiet = False).head(3))
#> Removing 1 constant column(s) out of 3 column(s)(Removed: constant). 
#>   species  body_mass_g
#> 0  Adelie       3750.0
#> 1  Adelie       3800.0
#> 2  Adelie       3250.0
```

　[`eda.filtering_out()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/filtering_out.md)：`pandas` の [`DataFrame.filter()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.filter.html) メソッドでは引数 `like` に文字列を指定することで、列名に特定の文字列を含む列を選択できますが、反対に `eda.filtering_out()` では列名に特定の文字列を含む列を除外します。実装の一部はR言語の [`dplyr::select()`](https://dplyr.tidyverse.org/reference/select.html) を参考にしました。

```python
# 列名に 'length' を含む列を除外
print(penguins.filtering_out(contains = 'length').head(3))
#>   species     island  bill_depth_mm  body_mass_g     sex  year  female
#> 0  Adelie  Torgersen           18.7       3750.0    male  2007       0
#> 1  Adelie  Torgersen           17.4       3800.0  female  2007       1
#> 2  Adelie  Torgersen           18.0       3250.0  female  2007       1

# 列名が 'bill' から始まる列を除外
print(penguins.filtering_out(starts_with = 'bill').head(3))
#>   species     island  flipper_length_mm  body_mass_g     sex  year  female
#> 0  Adelie  Torgersen              181.0       3750.0    male  2007       0
#> 1  Adelie  Torgersen              186.0       3800.0  female  2007       1
#> 2  Adelie  Torgersen              195.0       3250.0  female  2007       1

# 列名が '_mm' で終わる列を除外
print(penguins.filtering_out(ends_with = '_mm').head(3))
#>   species     island  body_mass_g     sex  year  female
#> 0  Adelie  Torgersen       3750.0    male  2007       0
#> 1  Adelie  Torgersen       3800.0  female  2007       1
#> 2  Adelie  Torgersen       3250.0  female  2007       1
```

### `py4stats.regression_tools`

　[`statsmodels`](https://www.statsmodels.org/stable/index.html)ライブラリで作成された回帰分析の結果についての表作成と視覚化を補助する機能を提供するモジュールです。

　[`reg.compare_ols()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/compare_ols.md) 回帰分析の表を作成。：計量経済学の実証論文でよく用いられる、回帰分析の結果を縦方向に並べて比較する表を作成します。表のフォーマットについてはR言語の[`texreg::screenreg()`](https://cran.r-project.org/web/packages/texreg/index.html)や[`modelsummary::modelsummary()`](https://modelsummary.com/man/modelsummary.html)を参考にしています。同種の機能を提供する Python ライブラリーとしては、R言語の [`stargazer`](https://cran.r-project.org/web/packages/stargazer/index.html) パッケージをもとにした [`stargazer`](https://pypi.org/project/stargazer/) ライブラリがあります。

``` python
import statsmodels.formula.api as smf

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


`reg.compare_ols()` の実行結果は `Pandas` の `DataFrame` として出力されるため、`.xlsx`. ファイルなどに変換することができます。また、用途に応じて表の体裁を調整できるようにしています。詳細については [「回帰分析の比較」](https://github.com/Hirototensho/Py4Stats/blob/main/man/compare_ols.md) を参照してください。

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

[`reg.coefplot()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/coefplot.md)：回帰係数の可視化。R言語の[`coefplot::coefplot()`](https://cran.r-project.org/web/packages/coefplot/index.html)を参考にしました。

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

reg.coefplot(fit3, ax = ax[1], palette = ['#FF6F91', '#F2E5EB'])
ax[1].set_xlim(-900, 1800);
```

![Unknown](https://github.com/Hirototensho/Py4Stats/assets/55335752/4c2dbfda-c67d-45c5-ba28-0f7fc72bd7d3)

　[`reg.compare_mfx()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/compare_mfx.md)
 と [`reg.mfxplot()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/compare_mfx.md) は、それぞれ `reg.compare_ols()` と `reg.coefplot()` の一般化線型モデルバージョンです。`statsmodels` ライブラリの[`.get_margeff()`](https://www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.DiscreteResults.get_margeff.html) メソッドから得られた限界効果の推定値を表示します。

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

reg.mfxplot(fit_logit2, ax = ax[1], palette = ['#FF6F91', '#F2E5EB'])
ax[1].set_xlim(-0.2, 0.85);
```

![Unknown](https://github.com/Hirototensho/Py4Stats/assets/55335752/f62e934a-91da-4ca8-9272-3006df2383f0)

***
[Jump to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md)
