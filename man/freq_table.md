# `py4stats.freq_table()`

## 概要

　R言語の[`DescTools::Freq()`](https://cran.r-project.org/web/packages/DescTools/DescTools.pdf)をオマージュした、1変数の度数分布表を計算する関数。度数 `freq` と相対度数 `perc` に加えて、それぞれの累積値を計算します。

``` python
freq_table(
    self, 
    subset, 
    sort = True,
    ascending = False,
    dropna = False
)
```

## 引数 

- `self`：`pd.DataFrame`（必須）
- `subset`：**str or list of str**</br>
　集計に使用するデータフレームの列名（必須）。
- `sort`：**bool**</br>
　True（初期値）なら度数分布表を頻度に応じてソートし、False なら `subset` で指定した列の値に応じてソートします。
- `ascending`：**bool**</br>
　ソートの方式。True なら昇順でソートし、False（初期設定）なら降順でソートします。
- `dropna`：**bool**</br>
　`subset` で指定した列の値が全て NaN である観測値を除外するかどうか。初期設定は False です。

以上の引数は、基本的に [pandas.DataFrame.value_counts](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html)の同名の引数と同じですが、  `dropna` のみ初期設定を変更しています。

## 使用例

``` python
import py4stats as py4st
import pandas as pd
from palmerpenguins import load_penguins
penguins = load_penguins() # サンプルデータの読み込

print(py4st.freq_table(penguins, 'species'))
#>      species  freq      perc  cumfreq   cumperc
#> 0     Adelie   152  0.441860      152  0.441860
#> 2  Chinstrap    68  0.197674      220  0.639535
#> 1     Gentoo   124  0.360465      344  1.000000

print(py4st.freq_table(penguins, ['island', 'species']))
#>       island    species  freq      perc  cumfreq   cumperc
#> 1     Biscoe     Adelie    44  0.127907       44  0.127907
#> 3     Biscoe     Gentoo   124  0.360465      168  0.488372
#> 2      Dream     Adelie    56  0.162791      224  0.651163
#> 4      Dream  Chinstrap    68  0.197674      292  0.848837
#> 0  Torgersen     Adelie    52  0.151163      344  1.000000
``` 

``` python
penguins2 = penguins.assign(bill_length_mm2 = pd.cut(penguins['bill_length_mm'], 6))

print(
    py4st.freq_table(penguins2, ['species', 'bill_length_mm2'], sort = False)
    )
#>       species   bill_length_mm2  freq      perc  cumfreq   cumperc
#> 0      Adelie  (36.683, 41.267]    89  0.258721       89  0.258721
#> 1      Adelie               NaN     1  0.002907       90  0.261628
#> 2      Adelie  (32.072, 36.683]    36  0.104651      126  0.366279
#> 3      Adelie   (41.267, 45.85]    25  0.072674      151  0.438953
#> 4      Adelie   (45.85, 50.433]     1  0.002907      152  0.441860
#> 5      Gentoo   (45.85, 50.433]    65  0.188953      217  0.630814
#> 6      Gentoo   (41.267, 45.85]    39  0.113372      256  0.744186
#> 7      Gentoo  (36.683, 41.267]     1  0.002907      257  0.747093
#> 8      Gentoo    (55.017, 59.6]     3  0.008721      260  0.755814
#> 9      Gentoo  (50.433, 55.017]    15  0.043605      275  0.799419
#> 10     Gentoo               NaN     1  0.002907      276  0.802326
#> 11  Chinstrap   (45.85, 50.433]    29  0.084302      305  0.886628
#> 12  Chinstrap  (50.433, 55.017]    24  0.069767      329  0.956395
#> 13  Chinstrap   (41.267, 45.85]    12  0.034884      341  0.991279
#> 14  Chinstrap    (55.017, 59.6]     2  0.005814      343  0.997093
#> 15  Chinstrap  (36.683, 41.267]     1  0.002907      344  1.000000
``` 
***
[Return to **Function reference**.](../reference.md)
