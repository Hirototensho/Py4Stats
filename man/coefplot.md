# `py4stats.regression_tools.coefplot()`

## 概要

　グラフ上の縦軸が説明変数、横軸回帰係数の値です。点が回帰係数の推定値を、エラーバー（横棒）が信頼区間を表し、β = 0 を帰無仮説とする仮説検定の結果によって有意であれば青色に、有意でなければグレーに色分けされます。`coefplot()` 関数の引数は次の通りです。

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

reg.coefplot(fit3, ax = ax[1])
ax[1].set_xlim(-900, 1800);
```

![Unknown](https://github.com/Hirototensho/Py4Stats/assets/55335752/0f11205b-5090-4b45-9a2e-7db7be3cc0f4)
