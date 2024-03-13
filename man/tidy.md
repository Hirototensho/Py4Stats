# `regression_tools.tidy()`, `regression_tools.tidy_mfx()`

## 概要

　R言語の [`broom::tidy()`](https://broom.tidymodels.org/reference/tidy.lm.html) をオマージュした関数で、[`sm.ols()`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html) や [`sm.glm()`](https://www.statsmodels.org/devel/generated/statsmodels.genmod.generalized_linear_model.GLM.html) の推定結果を pands.DataFrame に変換します。`regression_tools.tidy()` は回帰係数と関連する検定結果を表示し、 `regression_tools.tidy_mfx()` は限界効果と関連する検定結果を表示します。

```python
tidy(
  x, 
  name_of_term = None,
  conf_level = 0.95,
  **kwargs
  )

tidy_mfx(
  x, 
  at = 'overall', 
  method = 'dydx', 
  dummy = False, 
  conf_level = 0.95, 
  **kwargs
  )
```

## 引数 Argument

- `x`（必須）</br>
　以下のクラスに該当する分析結果のオブジェクト。
    - [`sm.ols()`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html) で作成された、`RegressionResultsWrapper` クラスのオブジェクト
    - [`sm.glm()`](https://www.statsmodels.org/devel/generated/statsmodels.genmod.generalized_linear_model.GLM.html) で作成された `BinaryResultsWrapper` クラスのオブジェクト。
- `name_of_term`：**list of str**</br>
　`term` 列（index） として表示する説明変数の名前のリスト。指定しない場合（初期設定）、モデルの推定に使用された説明変数の名前がそのまま表示されます。
- `conf_level`：**float**</br>
　信頼区間の計算に用いる信頼係数。

- `at`：限界効果の集計方法（`tidy_mfx()` のみ）。内部で使用している[`statsmodels.discrete.discrete_model.DiscreteResults.get_margeff()`](https://www.statsmodels.org/devel/generated/statsmodels.discrete.discrete_model.DiscreteResults.get_margeff.html) メソッドに引数 `at` として渡されます。`method = 'coef'` を指定した場合、この引数は無視されます。
    - `'overall'`：各観測値の限界効果の平均値を表示（初期設定）
    - `'mean'`：各説明変数の平均値における限界効果を表示
    - `'median'`：各説明変数の中央値における限界効果を表示
    - `'zero'`：各説明変数の値がゼロであるときの限界効果を表示

- `method`：推定する限界効果の種類（`tidy_mfx()` のみ）。内部で使用している[`statsmodels.discrete.discrete_model.DiscreteResults.get_margeff()`](https://www.statsmodels.org/devel/generated/statsmodels.discrete.discrete_model.DiscreteResults.get_margeff.html) メソッドに引数 `method` として渡されます。ただし、`method = 'coef'` を指定した場合には限界効果を推定せずに回帰係数をそのまま表示します。
    - `'coef'`：回帰係数の推定値を表示
    - `'dydx'`：限界効果の値を変換なしでそのまま表。（初期設定）
    - `'eyex'`：弾力性 d(lny)/d(lnx) の推定値を表示
    - `'dyex'`：準弾力性 dy /d(lnx) の推定値を表示
    - `'eydx'`：準弾力性 d(lny)/dx の推定値を表示

- `dummy`：ダミー変数の限界効果の推定方法（`tidy_mfx()` のみ）。もし False （初期設定）であれば、ダミー変数を連続な数値変数として扱います。もし、True であればダミー変数が0から1へと変化したときの予測値の変化を推定します。内部で使用している[`statsmodels.discrete.discrete_model.DiscreteResults.get_margeff()`](https://www.statsmodels.org/devel/generated/statsmodels.discrete.discrete_model.DiscreteResults.get_margeff.html) メソッドに引数 `dummy` として渡されます。

## 返り値 Value

　次の列を含む pands.DataFrame が出力されます。

- `term`（index）</br>
　説明変数の名称
- `estimate`）</br>
　回帰係数(`tidy()`の場合)、もしくは限界効果(`tidy_mfx()`の場合)の推定値
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
#>                       estimate   std_err  statistics  p_value  conf_lower  conf_higher
#> term                                                                                  
#> Intercept             153.7397  268.9012      0.5717   0.5679   -375.1910     682.6704
#> species[T.Chinstrap] -885.8121   88.2502    -10.0375   0.0000  -1059.4008    -712.2234
#> species[T.Gentoo]     578.6292   75.3623      7.6780   0.0000    430.3909     726.8674
#> bill_length_mm         91.4358    6.8871     13.2764   0.0000     77.8888     104.9828
```


```python
penguins['female'] = np.where(penguins['sex'] == 'female', 1, 0)

# ロジスティック回帰の実行
fit_logit1 = smf.logit('female ~ body_mass_g + bill_length_mm + bill_depth_mm', data = penguins).fit()

print(reg.tidy_mfx(fit_logit1).round(4))
#>                 estimate  std_err  statistics  p_value  conf_lower  conf_higher
#> body_mass_g      -0.0004   0.0000    -17.6561   0.0000     -0.0004      -0.0003
#> bill_length_mm   -0.0053   0.0036     -1.4628   0.1435     -0.0123       0.0018
#> bill_depth_mm    -0.1490   0.0051    -29.1681   0.0000     -0.1591      -0.1390
```

## 注意点

　参考にしたR言語の [`broom::tidy()`](https://broom.tidymodels.org/reference/tidy.lm.html) は様々な種類のモデルに対応したジェネリック関数として定義されていますが、`regression_tools.tidy()` と `regression_tools.tidy_mfx()` では対応しているモデルは限定的であることにご注意ださい。

## 補足

 機能は限定的ですが、`functools.singledispatch` を用いたジェネリック関数として実装しています。 [`Py4Etrics`](https://github.com/Py4Etrics/py4etrics) モジュールの `py4etrics.heckit.Heckit()` で作成された `HeckitResults` クラスのオブジェクト用のメソッドについては [`heckit_helper.tidy_heckit()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/tidy_heckit.md) を参照してください。


***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md)
