# `tidy()`

## 概要

　R言語の [bloom::tidy()](https://broom.tidymodels.org/reference/tidy.lm.html) をオマージュした関数で、[`sm.ols()`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html) や [`sm.glm()`](https://www.statsmodels.org/devel/generated/statsmodels.genmod.generalized_linear_model.GLM.html) の結果をpands.DataFrame に変換します。

```python
tidy(
  x, 
  name_of_term = None,
  conf_level = 0.95
  )
```

## 引数 Argument

- `x`（必須）</br>
　[`sm.ols()`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html) で作成された `RegressionResultsWrapper` オブジェクト。
- `name_of_term`：**list of str**</br>
　`term` 列（index） として表示する説明変数の名前のリスト。指定しない場合（初期設定）、モデルの推定に使用された説明変数の名前がそのまま表示されます。
- `conf_level`：**float**</br>
　信頼区間の計算に用いる信頼係数。

## 返り値 Value

　次の列を含む pands.DataFrame が出力されます。

- `term`（index）</br>
　説明変数の名称
- `coef`</br>
　回帰係数の推定値
- `std_err`</br>
　回帰係数の標準誤差
- `statistics`</br>
　$β = 0$ を帰無仮説とする仮説検定の標本検定統計量。`x` に代入されたモデルが `sm.ols()` によって作成されたものであれば $t$ 統計量が表示され、`sm.glm()` によって作成されたものであれば $z$ 統計量が表示されます。
- `p_value`</br>
　$β = 0$ を帰無仮説とする両側検定の標本$p$-値
- conf_lower</br>
　信頼区間の下側信頼限界
- conf_higher</br>
　信頼区間の上側信頼限界

## 使用例 Examples

```python
import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
import statsmodels.formula.api as smf

from py4stats import regression_tools as reg # 回帰分析の要約
penguins = load_penguins() # サンプルデータの読み込み
```

```python
# 回帰分析の実行
fit1 = smf.ols('body_mass_g ~ bill_length_mm + species', data = penguins).fit()

print(reg.tidy(fit1).round(4))
#>                           coef   std_err  statistics  p_value  conf_lower  conf_higher
#> term                                                                                  
#> Intercept             153.7397  268.9012      0.5717   0.5679   -375.1910     682.6704
#> species[T.Chinstrap] -885.8121   88.2502    -10.0375   0.0000  -1059.4008    -712.2234
#> species[T.Gentoo]     578.6292   75.3623      7.6780   0.0000    430.3909     726.8674
#> bill_length_mm         91.4358    6.8871     13.2764   0.0000     77.8888     104.9828
```
