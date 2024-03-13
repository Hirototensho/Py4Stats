# `eda_tools.is_dummy()`

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

`eda_tools.is_dummy()` は `self` が `cording`  と集合として等しければ True を、そうでなければ False を返します。

## 使用例 Examples

```python
from py4stats import eda_tools as eda        # 基本統計量やデータの要約など
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

print(penguins2['species_Adelie'].is_dummy())
#> True
```

なお、初期設定では bool 型の変数についても True を返します。

```python
print(penguins2.is_dummy())
#> island               False
#> bill_length_mm       False
#> species_Adelie        True
#> species_Chinstrap     True
#> species_Gentoo        True
#> Intercept            False
#> female                True
#> dtype: bool
```
***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md)
