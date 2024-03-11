# heckit_helper.tidy_heckit()

## 概要

　R言語の [`broom::tidy()`](https://broom.tidymodels.org/reference/tidy.lm.html) をオマージュした関数で、[`sm.ols()`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html) や

```python
tidy_heckit(
    model, 
    name_selection = None, 
    name_outcome = None, 
    conf_level = 0.95
  )
```

## 引数 Argument

- `x`（必須）</br>
　[`sm.ols()`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html)もしくは[`sm.glm()`](https://www.statsmodels.org/devel/generated/statsmodels.genmod.generalized_linear_model.GLM.html) で作成された分析結果のオブジェクト。
- `name_selection`：**list of str**</br>
　`term` 列（index） のうち、第1段階の説明変数の名称として表示する文字列のリスト。指定しない場合（初期設定）、モデルの推定に使用された説明変数の名前がそのまま表示されます。
- `name_outcome`：**list of str**</br>
　`term` 列（index） のうち、第2段階の説明変数の名称として表示する文字列のリスト。指定しない場合（初期設定）、モデルの推定に使用された説明変数の名前がそのまま表示されます。
- `conf_level`：**float**</br>
　信頼区間の計算に用いる信頼係数。

## 返り値 Value

　次の列を含む pands.DataFrame が出力されます。

- `term`（index）</br>
　説明変数の名称
- `estimate`）</br>
　回帰係数の推定値
- `std_err`</br>
  推定値 `estimate` の標準誤差
- `statistics`</br>
  `estimate = 0` を帰無仮説とする仮説検定の標本検定統計量。`x` に代入されたモデルが `sm.ols()` によって作成されたものであれば $t$ 統計量が表示され、`sm.glm()` によって作成されたものであれば $z$ 統計量が表示されます。
- `p_value`</br>
  `estimate = 0` を帰無仮説とする両側検定の標本p-値
- conf_lower</br>
　信頼区間の下側信頼限界
- conf_higher</br>
　信頼区間の上側信頼限界

## 使用例 Examples

```python
import pandas as pd
import wooldridge
from py4stats import heckit_helper
from py4stats import regression_tools as reg # 回帰分析の要約

mroz = wooldridge.data('mroz') # サンプルデータの読み込み

mod_heckit, exog_outcome, exog_select =   heckit_helper.Heckit_from_formula(
    selection = 'lwage ~ educ + exper + expersq + nwifeinc + age + kidslt6 + kidsge6',
    outcome = 'lwage ~ educ + exper + expersq',
    data = mroz
)

res_heckit = mod_heckit.fit(cov_type_2 = 'HC1')

# 初期設定で使用した場合
print(reg.tidy(res_heckit).round(4))
#>               estimate  std_err  statistics  p_value  conf_lower  conf_higher
#> term                                                                         
#> O: Intercept   -0.5781   0.3050     -1.8954   0.0580     -1.1759       0.0197
#> O: educ         0.1091   0.0155      7.0261   0.0000      0.0786       0.1395
#> O: exper        0.0439   0.0163      2.6989   0.0070      0.0120       0.0758
#> O: expersq     -0.0009   0.0004     -1.9574   0.0503     -0.0017       0.0000
#> S: const        0.2701   0.5086      0.5310   0.5954     -0.7267       1.2669
#> S: x1           0.1309   0.0253      5.1835   0.0000      0.0814       0.1804
#> S: x2           0.1233   0.0187      6.5903   0.0000      0.0867       0.1600
#> S: x3          -0.0019   0.0006     -3.1452   0.0017     -0.0031      -0.0007
#> S: x4          -0.0120   0.0048     -2.4843   0.0130     -0.0215      -0.0025
#> S: x5          -0.0529   0.0085     -6.2347   0.0000     -0.0695      -0.0362
#> S: x6          -0.8683   0.1185     -7.3263   0.0000     -1.1006      -0.6360
#> S: x7           0.0360   0.0435      0.8281   0.4076     -0.0492       0.1212
```

　**注意**：内部で使用している `statsmodels.iolib.summary.summary_params_frame()` の仕様上、初期設定では第1段階の説明変数の名前が反映できないため、説明変数の名前を反映するには `name_selection` 引数で指定してください。

```python
print(reg.tidy(res_heckit, name_selection = exog_select.columns).round(4))
#>               estimate  std_err  statistics  p_value  conf_lower  conf_higher
#> term                                                                         
#> O: Intercept   -0.5781   0.3050     -1.8954   0.0580     -1.1759       0.0197
#> O: educ         0.1091   0.0155      7.0261   0.0000      0.0786       0.1395
#> O: exper        0.0439   0.0163      2.6989   0.0070      0.0120       0.0758
#> O: expersq     -0.0009   0.0004     -1.9574   0.0503     -0.0017       0.0000
#> S: Intercept    0.2701   0.5086      0.5310   0.5954     -0.7267       1.2669
#> S: educ         0.1309   0.0253      5.1835   0.0000      0.0814       0.1804
#> S: exper        0.1233   0.0187      6.5903   0.0000      0.0867       0.1600
#> S: expersq     -0.0019   0.0006     -3.1452   0.0017     -0.0031      -0.0007
#> S: nwifeinc    -0.0120   0.0048     -2.4843   0.0130     -0.0215      -0.0025
#> S: age         -0.0529   0.0085     -6.2347   0.0000     -0.0695      -0.0362
#> S: kidslt6     -0.8683   0.1185     -7.3263   0.0000     -1.1006      -0.6360
#> S: kidsge6      0.0360   0.0435      0.8281   0.4076     -0.0492       0.1212
```
***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/man/reference.md)
