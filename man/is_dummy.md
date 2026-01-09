# `py4stats.is_dummy()`

## 概要

　リストや pandas.Series の要素がダミー変数かどうかを判定する関数。

```python
is_dummy(self, cording = [0, 1])
```

## 引数 Argument

- `self` **list, pandas.Series or pandas.DataFrame** <br>
- `cording` **list** <br>
　ダミー変数のコーディング方式を指定するリスト。

## 返り値 Value

`py4stats.is_dummy()` は `self` が `cording`  と集合として等しければ True を、そうでなければ False を返します。

## 使用例 Examples

```python
import py4stats as py4st
import pandas as pd
from palmerpenguins import load_penguins

penguins = load_penguins() # サンプルデータの読み込み

# ダミー変数の作成
penguins2 = pd.get_dummies(
    penguins.loc[:, 'species':'bill_length_mm'], 
    columns = ['species']
    )
penguins2['Intercept'] = 1 # 定数列の作成
penguins2['female'] = penguins['sex'] == 'female' # bool 型の変数を作成

print(py4st.is_dummy(penguins2['species_Adelie']))
#> True
```

なお、初期設定では bool 型の変数についても True を返します。

```python
print(py4st.is_dummy(penguins2))
#> island               False
#> bill_length_mm       False
#> species_Adelie        True
#> species_Chinstrap     True
#> species_Gentoo        True
#> Intercept            False
#> female                True
#> Name: 0, dtype: bool
```
***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md)
