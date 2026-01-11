# 引数のアサーション `building_block.arg_match()`

## 概要

　R言語の [`rlang::arg_match()`](https://rlang.r-lib.org/reference/arg_match.html) をオマージュした関数で、文字列で与えられた引数のアサーションを行います。

``` python
arg_match(
    arg, 
    values, 
    arg_name = None, 
    multiple = False
    )
``` 

## 引数 Argument

- `arg`（必須）**str or list of str**</br>
　適正かどうかを判断したい引数の値　
- `values`（必須）：**list of str**</br>
　引数 `arg` の適正な値のリスト
- `arg_name`：**str**</br>
　エラーメッセージに表示する引数の名前。指定されなかった場合（初期設定）、引数 `arg` に代入されたオブジェクトの名称を表示します。なお、この機能は [`varname.argname()`](https://github.com/pwwang/python-varname?tab=readme-ov-file)関数を使って実装されています。
- `multiple`：**bool**</br>
　引数の値として複数の値を許容するかどうかを示すブール値。`arg` にリストが代入された場合、`multiple = False`（初期設定）であれば最初の値のみを出力し、`multiple = True` であればリストの値を全て出力します。

## 返り値 Value

　引数 `arg` に代入された値が、`values` に代入されたリストに含まれていればその値を返し、そうでなければエラーメッセージを出力します。エラーメッセージでは `values` に代入されたリストの値を `arg` の適正な値の候補として提示します。


## 使用例 Examples

```python
from py4stats import building_block as build

def my_faivarit(fruits):
  fruits = build.arg_match(
      fruits, 
      values = ['apple', 'orange', 'grape'], 
      multiple = False
      )
  return fruits

my_faivarit('apple')
#> 'apple'

my_faivarit('orang')
#> ValueError: 'fruits' must be one of 'apple', 'orange' or 'grape', not 'orang'.
#>              Did you mean 'orange'?

my_faivarit('ap')
#> ValueError: 'fruits' must be one of 'apple', 'orange' or 'grape', not 'ap'.
#>              Did you mean 'apple' or 'grape'?
```

```python
# arg に list を指定した場合
# 初期設定では1つ目の要素だけ使用されます。
my_faivarit(['apple', 'orange'])
#> 'apple'

# multiple = True として再度関数を定義
def my_faivarit2(fruits):
  fruits = build.arg_match(
      fruits, 
      values = ['apple', 'orange', 'grape'], 
      multiple = True
      )
  return fruits

my_faivarit2(['apple', 'orange'])
#> ['apple', 'orange']

my_faivarit2(['apple', 'orang'])
#> ValueError: 'fruits' must be one of 'apple', 'orange' or 'grape', not 'orang'.
#>              Did you mean 'orange'?
```

　`Py4Stats` では [`eda_tools.tabyl()`](./tabyl.md)や [`regression_tools.compare_ols()`](./compare_ols.md) など、文字列で指定する引数をもつ関数で、引数のアサーションに `build.arg_match()` を使用しています。

```python
import py4stats as py4st
import pandas as pd
from palmerpenguins import load_penguins
penguins = load_penguins() # サンプルデータの読み込

py4st.tabyl(penguins, 'island', 'species', normalize = 'ind')
#> ValueError: 'normalize' must be one of 'index', 'columns' or 'all', not 'ind'.
#>              Did you mean 'index'?
```
***
[Return to **Function reference**.](../reference.md)

