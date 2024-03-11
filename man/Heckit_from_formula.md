# `heckit_helper.Heckit_from_formula()`

## 概要

``` python
Heckit_from_formula(
  selection, 
  outcome, 
  data, 
  **kwargs
  )
```

## 引数 Argument

- `selection`**str**（必須）</br>
　Type2トービットモデルのうち第1段階の selection equation(選択関数, 就業決定関数)の回帰式
- `outcome`**str**（必須）</br>
　Type2トービットモデルのうち第2段階の regression equation(賃金関数)の回帰式
- `data`：**pandas.DataFrame**（必須）</br>
- `**kwargs` </br>
  `py4etrics.heckit.Heckit()` に渡すその他の引数

## 使用例 Examples

　`heckit_helper` モジュールはヘックマンの2段階推定（Heckit）を実行するために [`Py4Etrics`](https://github.com/Py4Etrics/py4etrics) に依存しているため、事前のインストールをお願いします。

```python
pip install git+https://github.com/Py4Etrics/py4etrics.git
```

ここでは春山(2023, Chap.24)のモデルを再現するため、`wooldridge` モジュールから `mroz` データを読み込みます。

```python
import pandas as pd
import wooldridge
from py4stats import heckit_helper

mroz = wooldridge.data('mroz') # サンプルデータの読み込み
```

`Heckit_from_formula()` 関数を使い、モデルを推定します。なお、Type2トービットモデルを推定する場合、第2段階の回帰式 `outcome` で使用される説明変数は全て第1段階の回帰式 `selection` に含まれ、なおかつ `selection` に含まれるものの、`outcome` には含まれない説明変数が少なくとも1つは必要であることに注意してください(末石, 2015, p.117)。

```python
mod_heckit, exog_outcome, exog_select = \
 heckit_helper.Heckit_from_formula(
    selection = 'lwage ~ educ + exper + expersq + nwifeinc + age + kidslt6 + kidsge6',
    outcome = 'lwage ~ educ + exper + expersq',
    data = mroz
)

res_heckit = mod_heckit.fit(cov_type_2 = 'HC1')

print(res_heckit.summary())
#>                            Heckit Regression Results                            
#> ================================================================================
#> Dep. Variable:                    lwage   R-squared:                       0.156
#> Model:                           Heckit   Adj. R-squared:                  0.150
#> Method:                Heckman Two-Step   F-statistics:                   26.148
#> Date:                  Mon, 11 Mar 2024   Prob (F-statistic):              0.000
#> Time:                          08:40:39   Cov in 1st Stage:            nonrobust
#> No. Total Obs.:                     753   Cov in 2nd Stage:                  HC1
#> No. Censored Obs.:                  325                                         
#> No. Uncensored Obs.:                428                                         
#> ==============================================================================
#>                  coef    std err          z      P>|z|      [0.025      0.975]
#> ------------------------------------------------------------------------------
#> Intercept     -0.5781      0.305     -1.895      0.058      -1.176       0.020
#> educ           0.1091      0.016      7.026      0.000       0.079       0.139
#> exper          0.0439      0.016      2.699      0.007       0.012       0.076
#> expersq       -0.0009      0.000     -1.957      0.050      -0.002    1.15e-06
#> ==============================================================================
#>                  coef    std err          z      P>|z|      [0.025      0.975]
#> ------------------------------------------------------------------------------
#> Intercept      0.2701      0.509      0.531      0.595      -0.727       1.267
#> educ           0.1309      0.025      5.183      0.000       0.081       0.180
#> exper          0.1233      0.019      6.590      0.000       0.087       0.160
#> expersq       -0.0019      0.001     -3.145      0.002      -0.003      -0.001
#> nwifeinc      -0.0120      0.005     -2.484      0.013      -0.022      -0.003
#> age           -0.0529      0.008     -6.235      0.000      -0.069      -0.036
#> kidslt6       -0.8683      0.119     -7.326      0.000      -1.101      -0.636
#> kidsge6        0.0360      0.043      0.828      0.408      -0.049       0.121
#> ================================================================================
#>                    coef    std err          z      P>|z|      [0.025      0.975]
#> --------------------------------------------------------------------------------
#> IMR (Lambda)     0.0323      0.134      0.241      0.809      -0.230       0.294
#> =====================================
#> rho:                            0.049
#> sigma:                          0.664
#> =====================================
#> 
#> First table are the estimates for the regression (response) equation.
#> Second table are the estimates for the selection equation.
#> Third table is the estimate for the coef of the inverse Mills ratio (Heckman's Lambda).
```

Type2トービットモデルとヘックマンの2段階推定についての詳細は、春山(2023)の第24章や末石(2015, p.117)の第6章を参照してください。

## 参考文献
- 末石直也(2015)『計量経済学：ミクロデータ分析へのいざない』 日本評論社.
- 春山鉄源(2023) 『Pythonで学ぶ入門計量経済学』 https://py4etrics.github.io/index.html

***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/man/reference.md)
