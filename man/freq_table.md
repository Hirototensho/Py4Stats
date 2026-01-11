# `py4stats.freq_table()`

## 概要

　R言語の[`DescTools::Freq()`](https://cran.r-project.org/web/packages/DescTools/DescTools.pdf)をオマージュした、1変数の度数分布表を計算する関数。度数 `freq` と相対度数 `perc` に加えて、それぞれの累積値を計算します。

``` python
freq_table(
    self, 
    subset: Union[str, Sequence[str]],
    sort: bool = True,
    descending: bool = False,
    dropna: bool = False,
    to_native: bool = True
)
```

## 引数 

- `self`：`IntoFrameT`（必須）
- `subset`：**str or list of str**</br>
　集計に使用するデータフレームの列名（必須）。
- `sort`：**bool**</br>
　True（初期値）なら度数分布表を頻度に応じてソートし、False なら `subset` で指定した列の値に応じてソートします。
- `descending`：**bool**</br>
　ソートの方式。True なら降順でソートし、False（初期設定）なら昇順でソートします。
- `dropna`：**bool**</br>
　欠測値（NaN）を集計から除外するかどうかを表すブール値。初期設定は False です。

以上の引数は、基本的に [pandas.DataFrame.value_counts](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html) の同名の引数と同じですが、 `dropna` のみ初期設定を変更しています。

## 使用例

``` python
import py4stats as py4st
import pandas as pd
from palmerpenguins import load_penguins
penguins = load_penguins() # サンプルデータの読み込

print(py4st.freq_table(penguins, 'species'))
#>      species  freq      perc  cumfreq   cumperc
#> 0  Chinstrap    68  0.197674       68  0.197674
#> 1     Gentoo   124  0.360465      192  0.558140
#> 2     Adelie   152  0.441860      344  1.000000


print(py4st.freq_table(penguins, ['island', 'species']))
#>       island    species  freq      perc  cumfreq   cumperc
#> 0     Biscoe     Adelie    44  0.127907       44  0.127907
#> 1  Torgersen     Adelie    52  0.151163       96  0.279070
#> 2      Dream     Adelie    56  0.162791      152  0.441860
#> 3      Dream  Chinstrap    68  0.197674      220  0.639535
#> 4     Biscoe     Gentoo   124  0.360465      344  1.000000
``` 

``` python
penguins2 = penguins.assign(bill_length_mm2 = pd.cut(penguins['bill_length_mm'], 6))

print(
    py4st.freq_table(
        penguins2, ['species', 'bill_length_mm2'], 
        sort = False, dropna = True
        )
    )
#>       species   bill_length_mm2  freq      perc  cumfreq   cumperc
#> 0      Adelie  (32.072, 36.683]    36  0.105263       36  0.105263
#> 1      Adelie  (36.683, 41.267]    89  0.260234      125  0.365497
#> 2      Adelie   (41.267, 45.85]    25  0.073099      150  0.438596
#> 3      Adelie   (45.85, 50.433]     1  0.002924      151  0.441520
#> 4   Chinstrap  (36.683, 41.267]     1  0.002924      152  0.444444
#> 5   Chinstrap   (41.267, 45.85]    12  0.035088      164  0.479532
#> 6   Chinstrap   (45.85, 50.433]    29  0.084795      193  0.564327
#> 7   Chinstrap  (50.433, 55.017]    24  0.070175      217  0.634503
#> 8   Chinstrap    (55.017, 59.6]     2  0.005848      219  0.640351
#> 9      Gentoo  (36.683, 41.267]     1  0.002924      220  0.643275
#> 10     Gentoo   (41.267, 45.85]    39  0.114035      259  0.757310
#> 11     Gentoo   (45.85, 50.433]    65  0.190058      324  0.947368
#> 12     Gentoo  (50.433, 55.017]    15  0.043860      339  0.991228
#> 13     Gentoo    (55.017, 59.6]     3  0.008772      342  1.000000
``` 
***
[Return to **Function reference**.](../reference.md)
