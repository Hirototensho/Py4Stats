# 空白列, 定数列の削除：`eda_tools.remove_empty(), eda_tools.remove_constant()`

## 概要

　`eda_tools.remove_empty()`はR言語の [`janitor:remove_empty()`](https://sfirke.github.io/janitor/reference/remove_empty.html) をオマージュした関数で、全ての要素が `NaN` である列や行をデータフレームから除外します`eda_tools.remove_constant()`はR言語の [`janitor:remove_constant()`](https://sfirke.github.io/janitor/reference/remove_constant.html) をオマージュした関数で、1種類だけの要素からなる列をデータフレームから除外します。

``` python
remove_empty(
    self, 
    cols = True, 
    rows = True, 
    cutoff = 1, 
    quiet = True
)

remove_constant(
    self, 
    quiet = True, 
    dropna = False
)
```


## 引数

- `self`：`pandas DataFrame`
- `cols`：**bool**</br>
　空白列を削除するかどうかを表すブール値（`remove_empty()` のみ）。True（初期設定） なら空白列を削除し、Falseなら全ての要素が `NaN` の列があっても削除しません。
- `rows`：**bool**</br>
　空白行を削除するかどうかを表すブール値（`remove_empty()` のみ）。True（初期設定） なら空白行を削除し、Falseなら全ての要素が `NaN` の行があっても削除しません。
- `cutoff`：**float**</br>
　列の削除を行う閾値（`remove_empty()` のみ）。ある列（行）における `NaN` の割合が `>= cutoff` のとき、その列（行）を削除します。初期設定は1で全ての要素が `NaN` の列（行）のみ削除しますが、例えば `cutoff = 0.9` とすることで `NaN` の割合9が割以上の列（行）を削除できます。
- `quiet`：**bool**</br>
　削除した列（行）を報告するかどうかを表すブール値。`quiet = True`（初期設定） であれば何も報告せずに削除だけ行い、`quiet = False` なら、削除した列（行）の数と列名（行名）を報告します。
- `dropna`：**bool**</br>
　ユニーク値の数を計算する際に、`NaN` を除外するかどうかを表すブール値（`remove_constant()` のみ）。`dropna = True` だと `NaN` を除外し、`dropna = False`（初期設定）だと `NaN` を除外しません。データフレームに `NaN` と、 `NaN` ではない1種類の値からなる列がある場合、`dropna = False` だと削除されず、`dropna = True` だと削除されます。

## 使用例

`eda_tools.remove_empty()` の使用例。

``` python
import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins2 = penguins.loc[:, ['species', 'body_mass_g']].copy()
# 空白列を作成
penguins2.loc[:, 'empty'] = np.nan
# 空白行を作成
penguins2.loc[344, :] = np.nan

print(penguins2.tail(3))
#>        species  body_mass_g  empty
#> 342  Chinstrap       4100.0    NaN
#> 343  Chinstrap       3775.0    NaN
#> 344        NaN          NaN    NaN
```

``` python
# 完全に空白な行と列を削除。
print(penguins2.remove_empty(quiet = False).tail(3))
#> Removing 1 empty column(s) out of 3 columns(Removed: empty).
#> Removing 1 empty row(s) out of 345 rows(Removed: 344).
#>        species  body_mass_g
#> 341  Chinstrap       3775.0
#> 342  Chinstrap       4100.0
#> 343  Chinstrap       3775.0

# 完全に空白な列のみ削除。
print(penguins2.remove_empty(rows = False, quiet = False).tail(3))
#> Removing 1 empty column(s) out of 3 columns(Removed: empty).
#>        species  body_mass_g
#> 342  Chinstrap       4100.0
#> 343  Chinstrap       3775.0
#> 344        NaN          NaN

# 完全に空白な行のみ削除。
print(penguins2.remove_empty(cols = False, quiet = False).tail(3))
#> Removing 1 empty row(s) out of 345 rows(Removed: 344).
#>        species  body_mass_g  empty
#> 341  Chinstrap       3775.0    NaN
#> 342  Chinstrap       4100.0    NaN
#> 343  Chinstrap       3775.0    NaN
```

``` python
# quiet = True の場合
print(penguins2.remove_empty().tail(3))
#>        species  body_mass_g
#> 341  Chinstrap       3775.0
#> 342  Chinstrap       4100.0
#> 343  Chinstrap       3775.0
```

`eda_tools.remove_constant()` の使用例。

``` python
penguins2 = penguins.loc[:, ['species', 'body_mass_g']].copy()
penguins2.loc[:, 'constant'] = 'c'

print(penguins2.head(3))
#>   species  body_mass_g constant
#> 0  Adelie       3750.0        c
#> 1  Adelie       3800.0        c
#> 2  Adelie       3250.0        c

print(penguins2.remove_constant(quiet = False).head(3))
#> Removing 1 constant column(s) out of 3 column(s)(Removed: constant).
#>   species  body_mass_g
#> 0  Adelie       3750.0
#> 1  Adelie       3800.0
#> 2  Adelie       3250.0
```

``` python
penguins2.loc[:, 'almost_empty'] = np.nan
penguins2.loc[1, 'almost_empty'] = 'c'

# dropna = False なら、almost_empty は削除されません。
print(penguins2.remove_constant().head(3))
#>   species  body_mass_g almost_empty
#> 0  Adelie       3750.0          NaN
#> 1  Adelie       3800.0            c
#> 2  Adelie       3250.0          NaN

print(penguins2.remove_constant(dropna = True).head(3))
#>   species  body_mass_g
#> 0  Adelie       3750.0
#> 1  Adelie       3800.0
#> 2  Adelie       3250.0
```
