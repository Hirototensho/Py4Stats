# `py4stats.Pareto_plot()`

## 概要

データフレームからパレート図を作図する関数です。

``` python
Pareto_plot(
    data: IntoFrameT,
    group: str,
    values: Optional[str] = None,
    top_n: Optional[int] = None,
    aggfunc: Callable[..., Any] = nw.mean,
    ax: Optional[Axes] = None,
    fontsize: int = 12,
    xlab_rotation: Union[int, float] = 0,
    palette: Sequence[str] = ("#478FCE", "#252525"),
    )
``` 

- `data`：**IntoFrameT**（必須）<br>
  入力データ。narwhals が受け入れ可能な DataFrame 互換オブジェクト<br>
  （例：`pandas.DataFrame`、`polars.DataFrame`、`pyarrow.Table`）を指定できます。
- `group`：**str**</br>
　集計に使用するデータフレームの列名（必須）。
- `values`：**str**</br>
　集計に使用するデータフレームの列名。`values = None`（初期設定）の場合、`group` 別の度数が表示され、`values` が指定された場合、`group` 別に `values` を `aggfunc`で集計した値がグラフに表示されます。
- `top_n`：**int**</br>
　棒グラフを表示するカテゴリーの件数。`top_n = None`（初期設定）の場合、すべてのカテゴリーを表示し、整数値が指定された場合、上位 `top_n` 件が表示されます。
- `ax`</br>
　matplotlib の ax オブジェクト。複数のグラフを並べる場合などに使用します。
- `fontsize`：**int**</br>
　軸ラベルなどのフォントサイズ。
- `xlab_rotation`：**int or float**</br>
　横軸ラベルの角度。matplotlib の `ax.xaxis.set_tick_params()` に引数 `rotation` として渡されます。
- `palette`：**list of str**</br>
　グラフの描画に使用する色コード。1つ目の要素が棒グラフの色に、2つ目の累積値を表す折線グラフの色に対応します。

## 使用例

``` python
import py4stats as py4st
import pandas as pd
from palmerpenguins import load_penguins
penguins = load_penguins() # サンプルデータの読み込

penguins['group'] = penguins['species'] + '\n' + penguins['island']

py4st.Pareto_plot(penguins, group = 'group')
```

![Unknown](https://github.com/Hirototensho/Py4Stats/assets/55335752/46fca5f5-bde9-480d-b6bf-1957ac1035b5)

``` python
py4st.Pareto_plot(
    penguins, group = 'group', 
    values = 'bill_length_mm',
    aggfunc = 'mean',
    palette = ['#FF6F91', '#252525']
    )
``` 
![Unknown-2](https://github.com/Hirototensho/Py4Stats/assets/55335752/5e323376-eb56-4407-a047-0fded76c6619)

***
[Return to **Function reference**.](../reference.md)
