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
    penguins.query('species == "Adelie"')
)
print(res2.round(3))
#>                    group1  group2  abs_diff  rel_diff
#> bill_length_mm       47.3    38.8       8.5     0.197
#> bill_depth_mm        15.0    18.4       3.4    -0.204
#> flipper_length_mm   216.0   190.0      26.0     0.128
#> body_mass_g        5000.0  3700.0    1300.0     0.299
```
