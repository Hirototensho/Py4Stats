# `py4stats.filtering_out()`

## 概要

　`pandas` の [`DataFrame.filter()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.filter.html) メソッドでは引数 `like` に文字列を指定することで、列名に特定の文字列を含む列を選択できますが、反対に `py4st.filtering_out()` では列名に特定の文字列を含む列を除外します。実装の一部はR言語の [`dplyr::select()`](https://dplyr.tidyverse.org/reference/select.html) を参考にしました。

```python
filtering_out(
    self, 
    contains = None, 
    starts_with = None, 
    ends_with = None, 
    axis = 1
)
```

## 引数

- `self`：`pandas DataFrame`
- `contains`：**str**</br>
　列名（行名）の検索に使用する文字列。内部で使用している [`pandas.Series.str.contains`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.contains.html#pandas.Series.str.contains) に渡され、指定された文字列を列名（行名）に含む列（行）を除外します。
- `starts_with`：**str**</br>
　列名（行名）の検索に使用する文字列。内部で使用している [`pandas.Series.str.startswith`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.startswith.html) に渡され、指定された文字列で列名（行名）が始まる列（行）を除外します。
- `ends_with`：**str**</br>
　列名（行名）の検索に使用する文字列。内部で使用している [`pandas.Series.str.endswith`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.endswith.html) に渡され、指定された文字列で列名（行名）が終わる列（行）を除外します。
- `axis`：**`{0 or 'index', 1 or 'columns'}`**</br>
 `axis = 1` または `axis = 'columns'` なら列の削除を行い、`axis = 0` または `axis = 'index'` なら行の削除を行います。


## 使用例

```python
import py4stats as py4st
import pandas as pd
from palmerpenguins import load_penguins
penguins = load_penguins() # サンプルデータの読み込み

print(penguins.head(3))
#>   species     island  bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g     sex  year  female
#> 0  Adelie  Torgersen            39.1           18.7              181.0       3750.0    male  2007       0
#> 1  Adelie  Torgersen            39.5           17.4              186.0       3800.0  female  2007       1
#> 2  Adelie  Torgersen            40.3           18.0              195.0       3250.0  female  2007       1
```

```python
# 列名に 'length' を含む列を除外
print(py4st.filtering_out(penguins, contains = 'length').head(3))
#>   species     island  bill_depth_mm  body_mass_g     sex  year  female
#> 0  Adelie  Torgersen           18.7       3750.0    male  2007       0
#> 1  Adelie  Torgersen           17.4       3800.0  female  2007       1
#> 2  Adelie  Torgersen           18.0       3250.0  female  2007       1

# 列名が 'bill' から始まる列を除外
print(py4st.filtering_out(penguins, starts_with = 'bill').head(3))
#>   species     island  flipper_length_mm  body_mass_g     sex  year  female
#> 0  Adelie  Torgersen              181.0       3750.0    male  2007       0
#> 1  Adelie  Torgersen              186.0       3800.0  female  2007       1
#> 2  Adelie  Torgersen              195.0       3250.0  female  2007       1

# 列名が '_mm' で終わる列を除外
print(py4st.filtering_out(penguins, ends_with = '_mm').head(3))
#>   species     island  body_mass_g     sex  year  female
#> 0  Adelie  Torgersen       3750.0    male  2007       0
#> 1  Adelie  Torgersen       3800.0  female  2007       1
#> 2  Adelie  Torgersen       3250.0  female  2007       1
```
***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md)
