# 引数のアサーション `bilding_block.arg_match()`


``` python
arg_match(
    arg, 
    values, 
    arg_name = 'argument', 
    multiple = False
    )
``` 

```python
from py4stats import bilding_block as bild # py4stats のプログラミングを補助する関数群

values = ['apple', 'orange', 'grape', 'banana']

bild.arg_match('apple', values, 'fruits')
#> 'apple'

arg_match('ora', values, 'fruits')
#> ValueError: 'fruits' must be one of 'apple', 'orange', 'grape' or 'banana', not 'ora'.
#>              Did you mean 'orange'?

bild.arg_match('ap', values, 'fruits')
#> ValueError: 'fruits' must be one of 'apple', 'orange', 'grape' or 'banana', not 'ap'.
#>              Did you mean 'apple' or 'grape'?

bild.arg_match(['apple', 'orange'], values, 'fruits')
#> 'apple'

bild.arg_match(['apple', 'orange'], values, 'fruits', multiple = True)
#> ['apple', 'orange']

bild.arg_match(['apple', 'ora'], values, 'fruits', multiple = True)
#> ValueError: 'fruits' must be one of 'apple', 'orange', 'grape' or 'banana', not 'ora'.
#>              Did you mean 'orange'?
```
