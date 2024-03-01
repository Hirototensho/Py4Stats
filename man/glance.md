# `regression_tools.glance()`

## 概要

　R言語の [`bloom::glance()`](https://broom.tidymodels.org/reference/glance.lm.html) をオマージュした関数で、[`sm.ols()`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html) や [`sm.glm()`](https://www.statsmodels.org/devel/generated/statsmodels.genmod.generalized_linear_model.GLM.html) の推定結果をpands.DataFrame に変換します。

```python
glance(x)
```

## 引数 Argument

- `x`（必須）</br>
　[`sm.ols()`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html)もしくは[`sm.glm()`](https://www.statsmodels.org/devel/generated/statsmodels.genmod.generalized_linear_model.GLM.html) で作成された分析結果のオブジェクト。

## 返り値 Value

　モデルの当てはまり（goodness of fit）の尺度を各列に持つ pands.DataFrame が出力されます。表示される指標はモデルの種類によって異なります。


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
# 線形回帰の場合
fit_lm1 = smf.ols('body_mass_g ~ bill_length_mm + species', data = penguins).fit()

pd.set_option('display.expand_frame_repr', False)
print(reg.glance(fit_lm1).round(4))
#>    rsquared  rsquared_adj  nobs  df     sigma  F_values  p_values        AIC        BIC
#> 0    0.7829         0.781   342   3  375.3251  406.2735       0.0  5029.1406  5044.4798
```


```python
# ロジスティック回帰の場合
penguins['female'] = np.where(penguins['sex'] == 'female', 1, 0)
fit_logit1 = smf.logit('female ~ body_mass_g + bill_length_mm + bill_depth_mm', data = penguins).fit()

print(reg.glance(fit_logit1).round(4))

#>    prsquared   LL-Null  df_null    logLik       AIC      BIC  deviance  df_resid  nobs
#> 0     0.5647 -236.8458      341 -103.1079  214.2157  229.555  206.2157       338   342
```

## 注意点

　参考にしたR言語の [`bloom::tidy()`](https://broom.tidymodels.org/reference/tidy.lm.html) は様々な種類のモデルに対応したジェネリック関数として定義されていますが、現段階では `regression_tools.glance()` は [`sm.ols()`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html) と [`sm.glm()`](https://www.statsmodels.org/devel/generated/statsmodels.genmod.generalized_linear_model.GLM.html) で推定されたモデルのみにしか対応していません。
