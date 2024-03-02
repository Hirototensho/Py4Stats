# 引数のアサーション `bilding_block.arg_match()`

## 概要

　R言語の [`rlang::arg_match()`](https://rlang.r-lib.org/reference/arg_match.html) をオマージュした関数で文字列で与えられた引数のアサーションを行います。

``` python
arg_match(
    arg, 
    values, 
    arg_name = 'argument', 
    multiple = False
    )
``` 

## 引数 Argument

- `arg`（必須）**str or list of str**</br>
　適正かどうかを判断したい引数の値　
- `values`（必須）：**list of str**</br>
　引数 `arg` の適正な値のリスト
- `arg_name`：**str**</br>
　エラーメッセージに表示する引数 `arg` の名前
- `multiple`：**bool**</br>
　引数の値として複数の値を許容するかどうかを示すブール値。`arg` にリストが代入された場合、`multiple = False`（初期設定）であれば最初の値のみを出力し、`multiple = True` であればリストの値を全て出力します。

## 返り値 Value

　引数 `arg` に代入された値が、`values` に代入されたリストに含まれていればその値を返し、そうでなければエラーメッセージを出力します。エラーメッセージでは `values` に代入されたリストの値を `arg` の適正な値の候補として提示します。


## 使用例 Examples

```python
from py4stats import bilding_block as bild

values = ['apple', 'orange', 'grape', 'banana']

bild.arg_match('apple', values, 'fruits')
#> 'apple'

arg_match('ora', values, 'fruits')
#> ValueError: 'fruits' must be one of 'apple', 'orange', 'grape' or 'banana', not 'ora'.
#>              Did you mean 'orange'?

bild.arg_match('ap', values, 'fruits')
#> ValueError: 'fruits' must be one of 'apple', 'orange', 'grape' or 'banana', not 'ap'.
#>              Did you mean 'apple' or 'grape'?

bild.arg_match(['apple', 'orange'], values, 'fruits', multiple = False)
#> 'apple'

bild.arg_match(['apple', 'orange'], values, 'fruits', multiple = True)
#> ['apple', 'orange']

bild.arg_match(['apple', 'ora'], values, 'fruits', multiple = True)
#> ValueError: 'fruits' must be one of 'apple', 'orange', 'grape' or 'banana', not 'ora'.
#>              Did you mean 'orange'?
```
