# `eda.is_number(), eda.is_ymd(),  eda.is_ymd_like()`

## 概要

　`pandas.Series` の要素が、特定のフォーマットにそった文字列かどうかを判定する関数。

```python
is_number(self, na_default = True)
is_ymd(self, na_default = True)
is_ymd_like(self, na_default = True)
```

- `eda.is_number()`：与えられた文字列が数字かどうかを判定します。
- `eda.is_ymd()`：与えられた文字列が `yyyy-mm-dd` フォーマットにそった値かどうかを判定します。
- `eda.is_ymd_like()`：与えられた文字列が `yyyy-mm-dd` に近いフォーマットの値かどうかを判定します。

## 引数

- `self`：`pandas.Series`（必須）
- `na_default`：**bool**</br>
 　NA値に対して関数が返す値。`na_default = True` （初期設定）であれば `None` や  `NaN` には True を返し、`na_default = False` であれば  False が返します。

## 使用例

```python
import pandas as pd
import numpy as np
from py4stats import eda_tools as eda        # 基本統計量やデータの要約など

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
