# `eda.tabyl()`

## 概要

　データフレームのクロス集計表を作成します。R言語の [`janitor::tabyl()`](https://sfirke.github.io/janitor/reference/tabyl.html)にいくつかの `adorn_` 関数を追加した状態を再現した関数です。初期設定ではクロス集計表の各セルに度数と相対度数を 「度数(相対度数%)`」 の形式で表示します。

```python
def tabyl(
    self, 
    index, 
    columns, 
    values = None,
    aggfunc = None,
    margins = True, 
    margins_name = 'All', 
    normalize = 'index', 
    dropna = False,
    rownames = None, 
    colnames = None
    digits = 1,
):
```

## 引数

- `self`：`pandas DataFrame`
- `index`：**str**</br>
　集計に使用するデータフレームの変数名（必須）。
- `columns`：**str**</br>
　集計に使用するデータフレームの変数名（必須）。
- `values`：**str**</br>
　集計に使用するデータフレームの変数名。指定しない場合、`index` と `columns` に基づくクロス集計表が計算されます。
- `margins`：**bool**</br>
　行または列の合計を追加するかどうかを表すブール値。初期設定は True です。
- `dropna`：**bool**</br>
　全ての値が NaN である列を除外するかどうかを表すブール値
- `normalize`：**str**</br>
　丸括弧`( )`に表示する相対度数の計算方式。
    - `index` 各セルの度数を行の和で割り、横方向の相対度数の和が100%になるように計算します。
    - `columns` 各セルの度数を行の列で割り、縦方向の相対度数の和が100%になるように計算します。
    - `all` 各セルの度数を総度数で割り、全てのセルの相対度数の和が100%になるように計算します。

以上の引数は全て、内部で使用している [`pandas.crosstab`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.contains.html#pandas.Series.str.contains) の同名の引数と同じ意味を持ちます。

- `digits`：**int**</br>
　丸括弧`( )`に表示する相対度数の小数点以下の桁数。初期設定は1です。

## 使用例

```python
from py4stats import eda_tools as eda
import pandas as pd
from palmerpenguins import load_penguins
penguins = load_penguins() # サンプルデータの読み込

# 横方向の和を100%として計算（初期設定）
print(eda.tabyl(penguins, 'island', 'species', normalize = 'index'))
#> species         Adelie   Chinstrap       Gentoo  All
#> island                                              
#> Biscoe      44 (26.2%)    0 (0.0%)  124 (73.8%)  168
#> Dream       56 (45.2%)  68 (54.8%)     0 (0.0%)  124
#> Torgersen  52 (100.0%)    0 (0.0%)     0 (0.0%)   52
#> All        152 (44.2%)  68 (19.8%)  124 (36.0%)  344

# 縦方向の和を100%として計算
print(eda.tabyl(penguins, 'island', 'species', normalize = 'columns'))
#> species        Adelie    Chinstrap        Gentoo          All
#> island                                                       
#> Biscoe     44 (28.9%)     0 (0.0%)  124 (100.0%)  168 (48.8%)
#> Dream      56 (36.8%)  68 (100.0%)      0 (0.0%)  124 (36.0%)
#> Torgersen  52 (34.2%)     0 (0.0%)      0 (0.0%)   52 (15.1%)
#> All               152           68           124          344

# 全体の和を100%として計算
print(eda.tabyl(penguins, 'island', 'species', normalize = 'all'))
#> species         Adelie   Chinstrap       Gentoo           All
#> island                                                       
#> Biscoe      44 (12.8%)    0 (0.0%)  124 (36.0%)   168 (48.8%)
#> Dream       56 (16.3%)  68 (19.8%)     0 (0.0%)   124 (36.0%)
#> Torgersen   52 (15.1%)    0 (0.0%)     0 (0.0%)    52 (15.1%)
#> All        152 (44.2%)  68 (19.8%)  124 (36.0%)  344 (100.0%)
```
