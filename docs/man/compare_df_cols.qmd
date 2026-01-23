# `py4stats.compare_df_cols()`, `py4stats.compare_df_stats()`

## 概要

　R言語の [`janitor::compare_df_cols()`](https://sfirke.github.io/janitor/reference/compare_df_cols.html) をオマージュした関数で、`compare_df_cols()` は複数の pandas.DataFrame に含まれる同じ名前を持つ列同士のデータ型 `dtype` を比較し、`compare_df_stats()` は同じ名前を持つ列同士の記述統計量を比較します。

```python
compare_df_cols(
    df_list: Union[List[IntoFrameT], Mapping[str, IntoFrameT]],
    df_name: Optional[List[str]] = None,
    return_match: Literal["all", "match", "mismatch"] = 'all',
    dropna:bool = False,
    to_native: bool = True
)

compare_df_stats(
    df_list: List[IntoFrameT],
    df_name: Optional[List[str]] = None,
    return_match: Literal["all", "match", "mismatch"] = "all",
    stats: Callable[..., Any] = np.mean,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    to_native: bool = True,
    **kwargs: Any,
)
```

## 引数 Argument

- `df_list`（必須） **A list or dict of IntoFrameT** <br>
　列を比較するデータフレームのリストもしくは辞書オブジェクト。辞書が `df_name` が未指定の場合、辞書の keys を `df_name` として使用します。
- `df_name` **list of str** <br>
　表頭に表示するデータフレームの名前。`['df1', 'df2']` のように文字列のリストを指定してください。初期設定では、自動的に `df1, df2, df3 …` と連番が割り当てられます。
- `return_match` **str** <br>
　出力に反映する変数の範囲を表す文字列。次の値から選択できます。
    - `'all'`（初期設定）： 全ての列を表示。
    - `'match'`：全てのデータフレームで dtype が一致している列のみを表示。
    - `'mismatch'`：少なくとも1つのデータフレームで dtype が一致していない列のみを表示。
- `dropna` **bool** (`compare_df_cols()` のみ)<br>
　データ型 `dtype` の一致判定に当たり、`NaN` を無視するかどうか。初期設定 `False` の場合、すべてのデータフレームが同名かつ同じデータ型の列を持たない限り、ミスマッチが発生したと判定されます。
- `stats`  **str or function**<br>
　比較に用いる記述統計量を定義する関数。`np.mean` など `values` 列を1次元配列として受け取って単一の数値を返す任意の関数が使用できるほか、`nw.mean` など narwhals.functions モジュールで実装された関数を使用できます。初期設定は `np.mean` です。

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
