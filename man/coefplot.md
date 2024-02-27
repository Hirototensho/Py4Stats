# `py4stats.regression_tools.coefplot()`

## 概要

　グラフ上の縦軸が説明変数、横軸回帰係数の値です。点が回帰係数の推定値を、エラーバー（横棒）が信頼区間を表します。

```python
coefplot(
    mod, 
    subset = None, 
    alpha = [0.05, 0.01], 
    palette = ['#1b69af', '#629CE7'], 
    show_Intercept = False,
    show_vline = True,
    ax = None,
    **kwargs
)

mfxplot(
    mod, 
    subset = None, 
    alpha = [0.05, 0.01], 
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

- `mod`：`statsmodels` で作成した回帰分析の結果（必須）。
- `subset = None`：グラフに回帰係数を表示する説明変数のリスト。指定しなければモデルに含まれる全ての説明変数を使用します。また `subset` に指定された順番に合わせてグラフ内での回帰係数の並び順が変更されます。
- `alpha = [0.05, 0.01]`：信頼区間の計算に用いる有意水準。1つ目の要素が太い方のエラーバーの幅に、2つ目の要素が細い方のエラーバーの幅に対応します。
- `palette = ['#1b69af', '#629CE7']`：グラフの描画に使用する色コード。1つ目の要素が太い方のエラーバーの色に、2つ目の要素が細い方のエラーバーの色に対応します。
- `show_Intercept = False`：切片の係数を表示するかどうか。True だと切片の係数を表示し、False（初期設定）だと表示しません。
- `show_vline`：回帰係数 = 0 の垂直線を表示するかどうか。True （初期設定）を指定すると垂直線を表示し、False を指定すると表示されません。
- `ax`：matplotlib の ax オブジェクト。複数のグラフを並べる場合などに使用します。


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
