# `py4stats.tidy_test()`

## 概要

　R言語の [`broom::tidy()`](https://broom.tidymodels.org/reference/tidy.lm.html) をオマージュした [`py4stats.tidy()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/tidy.md) 関数のうち、`statsmodels` ライブラリのメソッド [`RegressionResults.t_test()`](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.RegressionResults.t_test.html#statsmodels.regression.linear_model.RegressionResults.t_test) もしくは [`RegressionResults.f_test()`](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.RegressionResults.f_test.html#statsmodels.regression.linear_model.RegressionResults.f_test) で作成された `statsmodels.stats.contrast.ContrastResults` クラスのオブジェクト専用のメソッドです。[`py4stats.tidy()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/tidy.md)はジェネリック関数として実装されているため、`py4st.tidy(x)` としてご利用いただけます。

```python
tidy_test(x, conf_level = 0.95, **kwargs)
```

## 引数 Argument

- `x`（必須）</br>
　`statsmodels` ライブラリのメソッド `RegressionResults.t_test()` もしくは`RegressionResults.f_test()` で作成された `statsmodels.stats.contrast.ContrastResults` クラスのオブジェクト。
- `conf_level`：**float**</br>
　信頼区間の計算に用いる信頼係数。ただし、`x` に代入されたオブジェクトが `f_test()` の結果である場合は、この引数は無視されます。

## 返り値 Value

　引数 `x` に代入されたオブジェクトが `t_test()` の結果である場合、次の列を含む pands.DataFrame が出力されます。

- `estimate`</br>
　帰無仮説のもとでの回帰係数（の線型結合）の推定値
- `std_err`</br>
  推定値 `estimate` の標準誤差
- `statistics`</br>
　仮説検定の標本検定統計量。
- `p_value`</br>
 　両側検定の標本p-値
- `conf_lower`</br>
　信頼区間の下側信頼限界
- `conf_higher`</br>
　信頼区間の上側信頼限界

　一方で引数 `x` に代入されたオブジェクトが `f_test()` の結果である場合、次の列を含む pands.DataFrame が出力されます。

- `statistics`</br>
　仮説検定の標本検定統計量。
- `p_value`</br>
 　F検定の標本p-値
- `df_denom`</br>
　モデルの残差自由度
- `df_denom`</br>
　帰無仮説のもとでの制約数

## 使用例 Examples

```python
import py4stats as py4st

import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
import statsmodels.formula.api as smf

penguins = load_penguins() # サンプルデータの読み込み

fit3 = smf.ols('body_mass_g ~ bill_length_mm + bill_depth_mm + species + sex', data = penguins).fit()
```

```python
hypotheses = 'bill_length_mm = 20'
print(py4st.tidy(fit3.t_test(hypotheses)).round(4))
#>       estimate  std_err  statistics  p_value  conf_lower  conf_higher
#> term                                                                 
#> c0     26.5366   7.2436      0.9024   0.3675     12.2867      40.7866
```

```python
hypotheses = 'species[T.Chinstrap] = 0, species[T.Gentoo] = 0'
print(py4st.tidy(fit3.f_test(hypotheses)).round(4))
#>           statistics  p_value  df_denom  df_num
#> term                                           
#> contrast    210.9432      0.0       327       2
```

***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md)
