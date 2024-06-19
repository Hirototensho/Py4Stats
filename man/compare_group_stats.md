# `eda_tools.compare_group_means()`, `eda_tools.plot_mean_diff()`, `eda_tools.plot_median_diff()`, `eda_tools.compare_group_median()`

## 概要

　グループ別の記述統計量をペア比較するための関数です。

```python
compare_group_means(
    group1, group2, 
    group_names = ['group1', 'group2']
    )

compare_group_median(
    group1, group2, 
    group_names = ['group1', 'group2']
    )

plot_mean_diff(
    group1, group2, 
    stats_diff = 'norm_diff',
    ax = None
    )

plot_median_diff(
    group1, group2, 
    stats_diff = 'rel_diff',
    ax = None
    )
```

## 引数 Argument

- `group1`（必須）**a pandas.DataFrame** <br>
　数値変数を含む pandas.DataFrame で `group2` との比較対象となるもの
- `group2`（必須）**a pandas.DataFrame** <br>
　数値変数を含む pandas.DataFrame で `group1` との比較対象となるもの
- `group_names` **list of str** <br>
　表頭に表示するグループの名前。`['group1', 'group2']` のように、2つの要素をもつ文字列のリストとして指定してください。
- `stats_diff`（`plot_mean_diff()` および `plot_median_diff()` のみ） **str** <br>
　グラフの描画に使用するグループ別統計量の差の評価指標。`'norm_diff'`（`plot_mean_diff()` のみ）、`'rel_diff'`、`'abs_diff'` のいずれかから選ぶことができます。

## 返り値 Value

　`compare_group_means()`, `compare_group_median()` では次の値をもつ `pandas.DataFrame` が出力されます。

- `group1, group2`（初期設定の場合）</br>
　各グループにおける記述統計統計量の値
- `norm_diff`（`compare_group_means()` のみ）</br>
　標準化された平均値の差で、2つのグループの平均値を $\bar{X}_1$, $\bar{X}_2$、分散を $s^2_1, s^2_2$ とし、サンプルサイズを $n_1, n_2$ とするとき、次式のように定義されます。

$$
\delta = \frac{\bar{X}_1  - \bar{X}_2}{s},~~~~~ s^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}
$$

- `abs_diff`</br>
 2つのグループの記述統計量の絶対差
- `rel_diff`</br>
 2つのグループの記述統計量の相対差。2つのグループの記述統計量を $\bar{X}_1$, $\bar{X}_2$ とするとき、次式のように定義されます。

$$
\delta = \cfrac{\bar{X}_1  - \bar{X}_2}{\cfrac{\bar{X}_1  + \bar{X}_2}{2}}
= 2 \cdot \frac{\bar{X}_1  - \bar{X}_2}{\bar{X}_1  + \bar{X}_2}
$$

## 使用例 Examples

```python
import pandas as pd
from py4stats import regression_tools as reg # 回帰分析の要約
from palmerpenguins import load_penguins


penguins = load_penguins().drop('year', axis = 1) # サンプルデータの読み込み
```

```python
res1 = eda.compare_group_means(
    penguins.query('species == "Gentoo"'),
    penguins.query('species == "Adelie"')
)
print(res1.round(3))
#>                      group1    group2  norm_diff  abs_diff  rel_diff
#> bill_length_mm       47.505    38.791      3.048     8.713     0.202
#> bill_depth_mm        14.982    18.346     -3.012     3.364    -0.202
#> flipper_length_mm   217.187   189.954      4.180    27.233     0.134
#> body_mass_g        5076.016  3700.662      2.868  1375.354     0.313
```

```python
res2 = eda.compare_group_median(
    penguins.query('species == "Gentoo"'),
    penguins.query('species == "Adelie"'),
    group_names = ['Gentoo', 'Adelie']
)
print(res2.round(3))
#>                    Gentoo  Adelie  abs_diff  rel_diff
#> bill_length_mm       47.3    38.8       8.5     0.197
#> bill_depth_mm        15.0    18.4       3.4    -0.204
#> flipper_length_mm   216.0   190.0      26.0     0.128
#> body_mass_g        5000.0  3700.0    1300.0     0.299
```

```python
eda.plot_mean_diff(
    penguins.query('species == "Gentoo"'),
    penguins.query('species == "Adelie"'),
    stats_diff = 'norm_diff'
)
```

![Unknown](https://github.com/Hirototensho/Py4Stats/assets/55335752/696cbbe0-2c0c-435c-bb9c-71a59a3742f9)

```python
eda.plot_mean_diff(
    penguins.query('species == "Gentoo"'),
    penguins.query('species == "Adelie"'),
    stats_diff = 'abs_diff'
)
```

![Unknown-2](https://github.com/Hirototensho/Py4Stats/assets/55335752/735866a9-aed2-4e10-bac1-6fc7004fba8f)

```python
eda.plot_median_diff(
    penguins.query('species == "Gentoo"'),
    penguins.query('species == "Adelie"'),
    stats_diff = 'rel_diff'
)
```
![Unknown-3](https://github.com/Hirototensho/Py4Stats/assets/55335752/7a496916-e828-44e1-a0e0-d50bb22ecc12)
