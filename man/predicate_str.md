# 文字列のフォーマットについての論理関数</br> `py4stats.is_number()`, `py4stats.is_ymd()`,  `py4stats.is_ymd_like()`

## 概要

　`pandas.Series` の要素が、特定のフォーマットにそった文字列かどうかを判定する関数。

```python
is_number(self, na_default = True)
is_ymd(self, na_default = True)
is_ymd_like(self, na_default = True)
```

- `py4stats.is_number()`：与えられた文字列が数字かどうかを判定します。
- `py4stats.is_ymd()`：与えられた文字列が `yyyy-mm-dd` フォーマットにそった値かどうかを判定します。
- `py4stats.is_ymd_like()`：与えられた文字列が `yyyy-mm-dd` に近いフォーマットの値かどうかを判定します。

## 引数

- `self`：`pandas.Series`（必須）
- `na_default`：**bool**</br>
 　NA値に対して関数が返す値。`na_default = True` （初期設定）であれば `None` や  `NaN` には True を返し、`na_default = False` であれば  False が返します。

## 使用例

```python
import py4stats as py4st
import pandas as pd
import numpy as np

s = pd.Series([
    '123', "0.12", "1e+07", '-31', '2個', '1A',
    "2024-03-03", "2024年3月3日", "24年3月3日", '令和6年3月3日',
    '0120-123-456', "apple", "不明", None, np.nan
    ])

print(s[s.is_number()])
#> 0       123
#> 1      0.12
#> 2     1e+07
#> 3       -31
#> 13     None
#> 14      NaN
#> dtype: object

print(s[s.is_ymd()])
#> 6     2024-03-03
#> 13          None
#> 14           NaN
#> dtype: object

print(s[s.is_ymd_like()])
#> 6     2024-03-03
#> 7      2024年3月3日
#> 8        24年3月3日
#> 9       令和6年3月3日
#> 13          None
#> 14           NaN
#> dtype: object
```

　実践的な使用例として[「厚生労働省 ４．食中毒統計資料」](https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/kenkou_iryou/shokuhin/syokuchu/04.html)のうち、2020年の食中毒事件一覧を考えます。東京都のデータを取り出て`'摂食者数'`の列を見ると、数字が並んでいるものの `dtype` は `object` となっており、数字ではない値が含まれていることが疑われます。

```python
# 厚生労働省：食中毒統計資料より
data = pd.read_excel('https://www.mhlw.go.jp/content/R2itiran.xlsx', header = 1)\
  .query('都道府県名等.str.contains("東京")')

print(data['摂食者数'])
#> 280    41
#> 281    86
#> 282     3
#> 283    10
#> 284     3
#>        ..
#> 381     2
#> 382     2
#> 383     4
#> 384     6
#> 385     4
#> Name: 摂食者数, Length: 106, dtype: object
```

`eda.is_number()` を使うと数字以外にどのような値が含まれているかを確認できるため、これをもとに「不明」となっている部分は `NaN` に置き換えるなどの対処法が考えられます。

```python
print(data.loc[~data['摂食者数'].is_number(), '摂食者数'])
#> 285    不明
#> 315    不明
#> 374    不明
#> 375    不明
#> 377    不明
#> 378    不明
#> 379    不明
#> 380    不明
#> Name: 摂食者数, dtype: object
```
***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md)
