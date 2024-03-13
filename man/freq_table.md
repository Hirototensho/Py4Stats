# `eda_tools.freq_table()`

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

- `self`：`pandas DataFrame`（必須）
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
from py4stats import eda_tools as eda
import pandas as pd
from palmerpenguins import load_penguins
penguins = load_penguins() # サンプルデータの読み込

print(penguins.freq_table('species'))
#>            freq      perc  cumfreq   cumperc
#> species
#> Adelie      152  0.441860      152  0.441860
#> Gentoo      124  0.360465      276  0.802326
#> Chinstrap    68  0.197674      344  1.000000

print(penguins.freq_table(['island', 'species']))
#>                      freq      perc  cumfreq   cumperc
#> island    species                                     
#> Biscoe    Gentoo      124  0.360465      124  0.360465
#> Dream     Chinstrap    68  0.197674      192  0.558140
#>           Adelie       56  0.162791      248  0.720930
#> Torgersen Adelie       52  0.151163      300  0.872093
#> Biscoe    Adelie       44  0.127907      344  1.000000
``` 

``` python
penguins2 = penguins.assign(bill_length_mm2 = pd.cut(penguins['bill_length_mm'], 6))

print(
    penguins2.freq_table(['species', 'bill_length_mm2'], sort = False)
    )
#>                             freq      perc  cumfreq   cumperc
#> species   bill_length_mm2
#> Adelie    (32.072, 38.975]    79  0.523179       79  0.523179
#>           (38.975, 45.85]     71  0.470199      150  0.993377
#>           (45.85, 52.725]      1  0.006623      151  1.000000
#>           (52.725, 59.6]       0  0.000000      151  1.000000
#> Chinstrap (32.072, 38.975]     0  0.000000        0  0.000000
#>           (38.975, 45.85]     13  0.191176       13  0.191176
#>           (45.85, 52.725]     50  0.735294       63  0.926471
#>           (52.725, 59.6]       5  0.073529       68  1.000000
#> Gentoo    (32.072, 38.975]     0  0.000000        0  0.000000
#>           (38.975, 45.85]     40  0.325203       40  0.325203
#>           (45.85, 52.725]     78  0.634146      118  0.959350
#>           (52.725, 59.6]       5  0.040650      123  1.000000
``` 
***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md)
