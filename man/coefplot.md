# 回帰分析による推定値の視覚化：`regression_tools.coefplot(), regression_tools.mfxplot()`

## 概要

　グラフ上の縦軸が説明変数、横軸回帰係数の値です。点が回帰係数の推定値を、エラーバー（横棒）が信頼区間を表します。

```python
coefplot(
    mod, 
    subset = None, 
    conf.level = [0.95, 0.99], 
    palette = ['#1b69af', '#629CE7'], 
    show_Intercept = False,
    show_vline = True,
    ax = None,
    **kwargs
)

mfxplot(
    mod, 
    subset = None, 
    conf.level = [0.95, 0.99], 
    at = 'overall',
    method = 'dydx',
    dummy = False,
    palette = ['#1b69af', '#629CE7'], 
    show_Intercept = False,
    show_vline = True,
    ax = None,
    **kwargs
)
```

## 引数

- `mod`：`statsmodels` で作成した回帰分析の結果（必須）。
- `subset`：グラフに回帰係数を表示する説明変数のリスト。指定しなければモデルに含まれる全ての説明変数を使用します。また `subset` に指定された順番に合わせてグラフ内での回帰係数の並び順が変更されます。
- `conf.level`：信頼区間の計算に用いる信頼係数。1つ目の要素が太い方のエラーバーの幅に、2つ目の要素が細い方のエラーバーの幅に対応します。初期設定は `[0.95, 0.99]` です。
- `palette`：グラフの描画に使用する色コード。1つ目の要素が太い方のエラーバーの色に、2つ目の要素が細い方のエラーバーの色に対応します。
- `show_Intercept`：切片の係数を表示するかどうか。True だと切片の係数を表示し、False（初期設定）だと表示しません。
- `show_vline`：回帰係数 = 0 の垂直線を表示するかどうか。True （初期設定）を指定すると垂直線を表示し、False を指定すると表示されません。
- `ax`：matplotlib の ax オブジェクト。複数のグラフを並べる場合などに使用します。

- `at`：限界効果の集計方法（`mfxplot()` のみ）。内部で使用している[`statsmodels.discrete.discrete_model.DiscreteResults.get_margeff()`](https://www.statsmodels.org/devel/generated/statsmodels.discrete.discrete_model.DiscreteResults.get_margeff.html) メソッドに引数 `at` として渡されます。`method = 'coef'` を指定した場合、この引数は無視されます。
    - `'overall'`：各観測値の限界効果の平均値を表示（初期設定）
    - `'mean'`：各説明変数の平均値における限界効果を表示
    - `'median'`：各説明変数の中央値における限界効果を表示
    - `'zero'`：各説明変数の値がゼロであるときの限界効果を表示

- `method`：推定する限界効果の種類（`mfxplot()` のみ）。内部で使用している[`statsmodels.discrete.discrete_model.DiscreteResults.get_margeff()`](https://www.statsmodels.org/devel/generated/statsmodels.discrete.discrete_model.DiscreteResults.get_margeff.html) メソッドに引数 `method` として渡されます。ただし、`method = 'coef'` を指定した場合には限界効果を推定せずに回帰係数をそのまま表示します。
    - `'coef'`：回帰係数の推定値を表示
    - `'dydx'`：限界効果の値を変換なしでそのまま表。（初期設定）
    - `'eyex'`：弾力性 d(lny)/d(lnx) の推定値を表示
    - `'dyex'`：準弾力性 dy /d(lnx) の推定値を表示
    - `'eydx'`：準弾力性 d(lny)/dx の推定値を表示

- `dummy`：ダミー変数の限界効果の推定方法（`mfxplot()` のみ）。もし False （初期設定）であれば、ダミー変数を連続な数値変数として扱います。もし、True であればダミー変数が0から1へと変化したときの予測値の変化を推定します。内部で使用している[`statsmodels.discrete.discrete_model.DiscreteResults.get_margeff()`](https://www.statsmodels.org/devel/generated/statsmodels.discrete.discrete_model.DiscreteResults.get_margeff.html) メソッドに引数 `dummy` として渡されます。

## 使用例

```python
from py4stats import eda_tools as eda        # 基本統計量やデータの要約など
from py4stats import regression_tools as reg # 回帰分析の要約
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from palmerpenguins import load_penguins
penguins = load_penguins() # サンプルデータの読み込み


# 回帰分析の実行
fit2 = smf.ols('body_mass_g ~ bill_length_mm + bill_depth_mm + species', data = penguins).fit()
fit3 = smf.ols('body_mass_g ~ bill_length_mm + bill_depth_mm + species + sex', data = penguins).fit()

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

```python
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots(1, 2, figsize = (2.2 * 5, 5), dpi = 100)

reg.mfxplot(fit_logit1, ax = ax[0])
ax[0].set_xlim(-0.2, 0.85)

reg.mfxplot(fit_logit2, ax = ax[1], palette = ['#FF6F91', '#F2E5EB'])
ax[1].set_xlim(-0.2, 0.85);
```

![Unknown](https://github.com/Hirototensho/Py4Stats/assets/55335752/f62e934a-91da-4ca8-9272-3006df2383f0)
