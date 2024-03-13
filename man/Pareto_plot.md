# `eda_tools.Pareto_plot()`

## 概要

データフレームからパレート図を作図する関数です。

``` python
Pareto_plot(
    data, 
    group, 
    values = None, 
    top_n = None, 
    aggfunc = 'sum',
    ax = None, 
    fontsize = 12, 
    xlab_rotation = 0,
    palette = ['#478FCE', '#252525']
    )
``` 

- `data`：`pandas DataFrame`（必須）
- `group`：**str**</br>
　集計に使用するデータフレームの列名（必須）。
- `values`：**str**</br>
　集計に使用するデータフレームの列名。`values = None`（初期設定）の場合、`group` 別の度数が表示され、
`values` が指定された場合、`group` 別に `values` を `aggfunc`で集計した値がグラフに表示されます。
- `top_n`：**int**</br>
　棒グラフを表示するカテゴリーの件数。`top_n = None`（初期設定）の場合、すべてのカテゴリーを表示し、整数値が指定された場合、上位 `top_n` 件が表示されます。
- `ax`</br>
　matplotlib の ax オブジェクト。複数のグラフを並べる場合などに使用します。
- `fontsize`：**int**</br>
　軸ラベルなどのフォントサイズ。
- `xlab_rotation`：**int**</br>
　横軸ラベルの角度。matplotlib の `ax.xaxis.set_tick_params()` に引数 `rotation` として渡されます。
- `palette`：**dict of str**</br>
　グラフの描画に使用する色コード。1つ目の要素が棒グラフの色に、2つ目の累積値を表す折線グラフの色に対応します。

## 使用例

``` python
from py4stats import eda_tools as eda
import pandas as pd
from palmerpenguins import load_penguins
penguins = load_penguins() # サンプルデータの読み込

penguins['group'] = penguins['species'] + '\n' + penguins['island']

eda.Pareto_plot(penguins, group = 'group')
```

![Unknown](https://github.com/Hirototensho/Py4Stats/assets/55335752/46fca5f5-bde9-480d-b6bf-1957ac1035b5)

``` python
eda.Pareto_plot(
    penguins, group = 'group', 
    values = 'bill_length_mm',
    aggfunc = 'mean',
    palette = ['#FF6F91', '#252525']
    )
``` 
![Unknown-2](https://github.com/Hirototensho/Py4Stats/assets/55335752/5e323376-eb56-4407-a047-0fded76c6619)

***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md)
