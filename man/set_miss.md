# `py4stats.set_miss()`

## 概要

この関数は、Series の非欠測要素のうち、指定された個数または割合を欠測値に置き換えます。narwhals を利用することで、複数の Series バックエンドに対応しています。主にテストデータの作成や、欠測データのシミュレーションを目的とした関数です。

``` python
set_miss(
    x: IntoSeriesT, 
    n: Optional[int] = None,
    prop: Optional[float] = None, 
    method: Literal['random', 'first', 'last'] = 'random', 
    random_state: Optional[int] = None, 
    na_value: Any = None,
    to_native: bool = True
    )
``` 

## 引数 Argument

- `data`：**IntoFrameT**（必須）<br>
  入力データ。narwhals が受け入れ可能な DataFrame 互換オブジェクト<br>
  （例：`pandas.DataFrame`、`polars.DataFrame`、`pyarrow.Table`）を指定できます。
- `n`：**int** <br>
    処理後の Series に含まれる欠測値の目標個数。すでに `n` 個以上の欠測値が含まれている場合は、新たな欠測値は追加されず、警告が発せられます。
- `prop`：**float** <br>
    処理後の Series に含まれる欠測値の目標割合。0 から 1 の間で指定してください。すでに欠測値の割合が `prop` 以上である場合は、新たな欠測値は追加されず、警告が発せられます。
- `method`: **str**:
    欠測値に置き換える要素の選択方法。
    - `'random'`: 非欠測要素の中からランダムに選択します。
    - `'first'`: Series の先頭から選択します。
    - `'last'`: Series の末尾から選択します。
    デフォルトは `'random'` です。
- `random_state` (**int**, optional):
    `method = 'random'` の場合に使用する乱数シード。再現性のある結果を得るために指定できます。
    `method` が `'random'` 以外の場合、`random_state` は無視されます。
- `na_value`: (**Any**)<br>
    欠測値として使用する値。デフォルトは `None` です。
- `to_native`（**bool**, optional）<br>
  `True` の場合、入力と同じ型の Series（e.g. pandas / polars / pyarrow）を返します。<br>
  `False` の場合、`narwhals.Series` を返します。デフォルトは `True` で、`to_native = False` は、主にライブラリ内部での利用や、`backend` に依存しない後続処理を行う場合を想定したオプションです。

## 使用例 Example

``` python
import pandas as pd
from py4stats import set_miss
s = pd.Series([1, 2, 3, 4, 5])
py4st.set_miss(s, n = 2, method='first')
#> 0    NaN
#> 1    NaN
#> 2    3.0
#> 3    4.0
#> 4    5.0
#> dtype: float64

s_miss = py4st.set_miss(s, prop=0.4, method='random', random_state=0)
#> 0    1.0
#> 1    NaN
#> 2    3.0
#> 3    NaN
#> 4    5.0
#> dtype: float64
```
`x` に代入された Series オブジェクトに、既に指定された以上の欠測値が含まれていた場合、次のように欠測値を追加せず `UserWarning` を出します。

``` python
py4st.set_miss(s_miss, n = 2)
#> UserWarning: Already contained 2(>= n) missing value(s) in `x`, 
#> no additional missing values were added.
#> 0    1.0
#> 1    NaN
#> 2    3.0
#> 3    NaN
#> 4    5.0
#> dtype: float64
```

``` python
from palmerpenguins import load_penguins
penguins = load_penguins() # サンプルデータの読み込

penguins['island'] = py4st.set_miss(
    penguins['island'], 
    n = 100, method='first'
    )
py4st.plot_miss_var(penguins, values = 'missing_count')
```
![set_miss.png](image/set_miss.png)