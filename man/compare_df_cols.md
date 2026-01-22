# `py4stats.compare_df_cols()`, `py4stats.compare_df_stats()`

## 概要

　R言語の [`janitor::compare_df_cols()`](https://sfirke.github.io/janitor/reference/compare_df_cols.html) をオマージュした関数で、
`compare_df_cols()` は複数の pandas.DataFrame に含まれる同じ名前を持つ列同士のデータ型 `dtype` を比較し、`compare_df_stats()` は同じ名前を持つ列同士の記述統計量を比較します。

```python
compare_df_cols(df_list, return_match = 'all', df_name = None, dropna = False)

compare_df_stats(
    df_list, return_match = 'all', df_name = None,
    stats = 'mean', rtol = 1e-05, atol = 1e-08,
    **kwargs
)
```

## 引数 Argument

- `df_list`（必須） **a list of pandas.DataFrame** <br>
　列を比較するデータフレームのリスト
- `df_name` **list of str** <br>
　表頭に表示するデータフレームの名前。`['df1', 'df2']` のように文字列のリストを指定してください。初期設定では、自動的に `df1, df2, df3 …` と連番が割り当てられます。
- `return_match` **str** <br>
　出力に反映する変数の範囲を表す文字列。次の値から選択できます。
    - `’all’`（初期設定）： 全ての列を表示。
    - `’match’`：全てのデータフレームで dtype が一致している列のみを表示。
    - `’mismatch’`：少なくとも1つのデータフレームで dtype が一致していない列のみを表示。
- `dropna` **bool**<br>
　データ型 `dtype` の一致判定に当たり、`NaN` を無視するかどうか。初期設定 `False` の場合、すべてのデータフレームに同名かつ同じデータ型の列を持たない限り、ミスマッチが発生したと判定されます。
- `stats`  **str or function**<br>
　比較に用いる記述統計量を表す文字列もしくは関数。初期設定は平均値 `'mean'` です。内部で使用している [`pandas.DataFrame.agg()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.agg.html) メソッドの引数 `func` に受け渡されます。

## 使用例 Examples

```python
import pandas as pd
import py4stats as py4st

df1 = pd.DataFrame({'x':[1, 2, 3], 'y':[5,   4, 2], 'z':[True, False, True]})
df2 = pd.DataFrame({'x':[1, 2, 3], 'y':[5.0, 4, 2], 'z':['True', 'False', 'True']})

print(py4st.compare_df_cols([df1, df2]))
#>   term    df1      df2  match_dtype
#> 0    x  int64    int64         True
#> 1    y  int64  float64        False
#> 2    z   bool   object        False

```
`return_match = 'mismatch'` を指定すると、データフレームの中で、`dtype` が一致していないものがある列を返します。

```python
print(py4st.compare_df_cols(
    [df1, df2], return_match = 'mismatch'
    ))
#>   term    df1      df2  match_dtype
#> 1    y  int64  float64        False
#> 2    z   bool   object        False
```

```python
df_list = [
    df2, pl.from_pandas(df2), pa.Table.from_pandas(df2)
]
df_name = ['pd', 'pl', 'pa']

print(py4st.compare_df_cols(df_list, df_name = df_name))
#>   term       pd       pl      pa  match_dtype
#> 0    x    int64    Int64   int64        False
#> 1    y  float64  Float64  double        False
#> 2    z   object   String  string        False

print(eda_nw.compare_df_cols(
    df_list, df_name = df_name,
    compar_by = 'narwhals_schema'
    ))
#>   term       pd       pl       pa  match_dtype
#> 0    x    Int64    Int64    Int64         True
#> 1    y  Float64  Float64  Float64         True
#> 2    z   String   String   String         True
```

　`py4st.compare_df_stats()` は数値変数の記述統計量を比較するため、異なる経路で行われたデータ処理の結果が一致しているかを検証する場合に便利です。

```python
from palmerpenguins import load_penguins
penguins = load_penguins()
penguins2 = penguins.copy()
vars = ['flipper_length_mm', 'body_mass_g']
penguins2.loc[:, vars] = py4st.scale(penguins2.loc[:, vars])

print(
    py4st.compare_df_stats([penguins, penguins2]).round(2)
)
#>                 term      df1      df2  match_stats
#> 0      bill_depth_mm    17.15    17.15         True
#> 1     bill_length_mm    43.92    43.92         True
#> 2        body_mass_g  4201.75     0.00        False
#> 3  flipper_length_mm   200.92    -0.00        False
#> 4               year  2008.03  2008.03         True
```
***
[Return to **Function reference**.](../reference.md)
