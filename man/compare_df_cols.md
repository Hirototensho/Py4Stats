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
- `return_match` **str** <br>
　出力に反映する変数の範囲を表す文字列。次の値から選択できます。
    - `’all’`（初期設定）： 全ての列を表示。
    - `’match’`：全てのデータフレームで dtype が一致している列のみを表示。
    - `’mismatch’`：少なくとも1つのデータフレームで dtype が一致していない列のみを表示。
- `df_name` **list of str** <br>
　表頭に表示するデータフレームの名前。`['df1', 'df2']` のように文字列のリストを指定してください。初期設定では、自動的に `df1, df2, df3 …` と連番が割り当てられます。
- `dropna` **bool**<br>
　データ型 `dtype` の一致判定に当たり、`NaN` を無視するかどうか。初期設定 `False` の場合、すべてのデータフレームに同名かつ同じデータ型の列を持たない限り、ミスマッチが発生したと判定されます。
- `stats`  **str or function**<br>
　比較に用いる記述統計量を表す文字列もしくは関数。初期設定は平均値 `'mean'` です。内部で使用している [`pandas.DataFrame.agg()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.agg.html) メソッドの引数 `func` に受け渡されます。

## 使用例 Examples

```python
import pandas as pd
import py4stats as py4st

# 厚生労働省：食中毒統計資料より
data_2019 = pd.read_excel('https://www.mhlw.go.jp/content/R1itiran.xlsx', header = 1)
data_2020 = pd.read_excel('https://www.mhlw.go.jp/content/R2itiran.xlsx', header = 1)
data_2021 = pd.read_excel('https://www.mhlw.go.jp/content/000948376.xlsx', header = 1)
```

```python
print(py4st.compare_df_cols([data_2019, data_2020, data_2021]).head())
#>                 df1      df2      df3  match_dtype
#> term
#> Unnamed: 0  float64  float64  float64         True
#> 都道府県名等       object   object   object         True
#> 発生月日          int64   object   object        False
#> 発生場所         object   object   object         True
#> 原因食品         object   object   object         True
```
`return_match = 'mismatch'` を指定すると、3つのデータフレームの中で、`dtype` が一致していないものがある列を返します。

```python
print(py4st.compare_df_cols(
    [data_2019, data_2020, data_2021], 
    return_match = 'mismatch'
    ))
#>         df1     df2     df3  match_dtype
#> term
#> 発生月日  int64  object  object        False
```

　`py4st.compare_df_stats()` は数値変数の記述統計量を比較するため、異なる経路で行われたデータ処理の結果が一致しているかを検証する場合に便利です。

```python
print(py4st.compare_df_stats([data_2019, data_2020, data_2021]))
#                df1        df2        df3  match_stats
# term                                                 
# 発生月日  43643.629699        NaN        NaN        False
# 患者数      12.238722  16.457207  15.413074        False
# 死者数       0.003759   0.003378   0.002782        False
```
***
[Return to **Function reference**.](../reference.md)
