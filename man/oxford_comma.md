# 並列文の作成 `oxford_comma()`

## 概要

　文字列のリストを与えると、リストの要素を英文における並列文の形に変換する関数です。表記法については [Wikipedia Serial comma](https://en.wikipedia.org/wiki/Serial_comma) を参照し、コードについては [stack overflow:Grammatical List Join in Python [duplicate]](https://stackoverflow.com/questions/19838976/grammatical-list-join-in-python) を参照しました。

```python
oxford_comma(x, sep_last = 'and', quotation = True)

oxford_comma_and(x, quotation = True)

oxford_comma_or(x, quotation = True)
```

なお、`oxford_comma_and(x)` は `oxford_comma(x, 'and')` と、`oxford_comma_or(x)` は `oxford_comma(x, 'or')` と同等です。

## 引数

- `x`：**str or list of str**</br>
- `quotation`: **bool**</br>
　リストの各要素にクオーテーションマーク `’’` を追加するかどうかを表す論理値。True（初期設定）であればクオーテーションマークを追加し、False であれば追加しません。
- `sep_last`: **str** `oxford_comma()` のみ</br>
　リストの最後の要素の直前に付加する単語を表す文字列。

## 使用例

```python
from py4stats import bilding_block as bild
x = ['A', 'B', 'C']

print(bild.oxford_comma_and(x))
#> 'A', 'B' and 'C'

print(bild.oxford_comma_and(x, quotation = False))
#> A, B and C

print(bild.oxford_comma_or(x))
#> 'A', 'B' or 'C'
```

リストの要素が1つの場合、あるいは `x` に文字列が指定された場合はカンマなどを追加せずにそのまま出力します。

```python
print(bild.oxford_comma_or(['A']))
#> 'A'

print(bild.oxford_comma_or('A'))
#> 'A'
```
***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/man/reference.md)
