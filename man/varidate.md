# 簡易なルールベースのデータ検証ツール `eda_tools.check_that()` `eda_tools.check_viorate()`

## 概要

　R言語の [`varidate`](https://github.com/data-cleaning/validate)パッケージの `check_that()` 関数などをオマージュした、ごく簡易なデータ検証関数です。

```python
check_that(data, rule_dict, **kwargs)

check_viorate(data, rule_dict, **kwargs)
```

## 引数 Argument

- `data`**pd.DataFrame**（必須）<br>
　ルールに基づくデータ検証を行うデータセット。

- `rule_dict`**dict or pd.Series of str**（必須）<br>
　`pandas.eval()` メソッドで実行した結果が論理値となるような expression の文字列を値とする辞書オブジェクト。詳細は使用例も参照してください。

- `**kwargs`<br>
　[`pandas.eval()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.eval.html) に渡す追加の引数。

## 使用例 Examples1

　ここでは `eda.check_that()` 関数を使って Loo, Jonge(2022, p. 136)の結果を再現します。まずはR言語の `validate` パッケージに付属する `retailers` データを利用します。`retailers` は60件の小売業者の経営状況についてのデータで、従業員数、売上高とその他の収入、人件費、総費用、および利益がユーロ導入前の通貨単位である1000ギルダー単位で収録されています。

```python
import pandas as pd
from py4stats import eda_tools as eda        # 基本統計量やデータの要約など

URL = 'https://raw.githubusercontent.com/data-cleaning/validate/master/pkg/data/retailers.csv'
retailers = pd.read_csv(URL, sep = ';')
retailers.columns = retailers.columns.to_series().str.replace('.', '_', regex = False)
```

　`eda.check_that()` 関数は、第1引数にデータセットを、第2引数に検証ルールの辞書オブジェクトを代入して使用します。  
　まずは、検証ルールの辞書オブジェクトを定義します。辞書オブジェクトの値には `pandas.eval()` メソッドで実行可能な expression の文字列を指定し、key に検証ルールの名前を指定します。検証ルールの名前は任意の値で構いませんが、 expression は結果が論理値となるものでなければなりません。

```python
rule_dict =  {
    'to':'turnover > 0',                                     # 売上高は厳密に正である
    'sc':'staff_costs / staff < 50',                         # 従業員1人当たりの人件費は50,000ギルダー未満である
    'cd1':'staff_costs > 0 | ~(staff > 0)',                  # 従業員がいる場合、人件費は厳密に正である
    'cd2':eda.implies_exper('staff > 0', 'staff_costs > 0'), # 従業員がいる場合、人件費は厳密に正である
    'bs':'turnover + other_rev == total_rev',                # 売上高とその他の収入の合計は総収入に等しい
    'mn':'profit.mean() > 0'                                 # セクター全体の平均的な利益はゼロよりも大きい
    }
pd.Series(rule_dict)
#> to                          turnover > 0
sc              staff_costs / staff < 50
cd1       staff_costs > 0 | ~(staff > 0)
cd2       staff_costs > 0 | ~(staff > 0)
bs     turnover + other_rev == total_rev
mn                     profit.mean() > 0
dtype: object
```

`retailers` と `rule_dict` を `eda.check_that()` に代入すると、`rule_dict` に指定したルールに基づいた検証が実行されます。`item` 列はその検証ルールで生成された論理値の個数（通常はデータセットの列数と一致します）を表し、`passes` 列は検証結果が True となったレコードの数を、`fails` は False となったレコードの数を表します。また、`coutna` はルールの検証に使用した変数（データセットの列）のいずれかが欠測値であったレコードの数です。

```python
print(eda.check_that(retailers, rule_dict))
#>       item  passes  fails  coutna                         expression
#> name                                                                
#> to      60      56      0       4                       turnover > 0
#> sc      60      39      5      16           staff_costs / staff < 50
#> cd1     60      44      0      16     staff_costs > 0 | ~(staff > 0)
#> cd2     60      44      0      16     staff_costs > 0 | ~(staff > 0)
#> bs      60      19      4      37  turnover + other_rev == total_rev
#> mn       1       1      0       0                  profit.mean() > 0
```

前述の通り、`eda.check_that()` 関数ではルール検証を `pandas.eval()` メソッドで実行しているため、検証ルールに自作関数や外部のモジュールからインポート関数を使うには、関数名の前に `@` をつけて `@func(…)` と記述し、また `**kwargs` 引数に `local_dict = locals()` と指定してください。  
　次のコードで定義している `is_complet()` 関数は、代入された pd.Series が全て欠測値ではなく、指定された変数に関して完全ケースであることを判定する関数です。`turnover.notna() & total_rev.notna() & other_rev.notna()` と記述しても同じ結果が得られますが、自作関数を使うことで若干簡潔に記述できます。

```python
from pandas.api.types import is_numeric_dtype
def is_complet(*arg): return pd.concat(arg, axis = 'columns').notna().all(axis = 'columns')

pd.set_option('display.expand_frame_repr', False)

rule_dict2 =  {
    'to_num':'@is_numeric_dtype(turnover)',                      # 売上高は数値変数である
    'rev_complet':'@is_complet(turnover, total_rev, other_rev)', # 売上高と収入が全て観測されている
    }

print(eda.check_that(
    retailers, rule_dict2, local_dict = locals()
    ))
#>              item  passes  fails  coutna                                   expression
#> name                                                                                 
#> to_num          1       1      0       0                  @is_numeric_dtype(turnover)
#> rev_complet    60      23      0      37  @is_complet(turnover, total_rev, other_rev)
```

`eda.check_viorate()` の使い方も `eda.check_that()` と同様ですが、`eda.check_that()` がデータセット全体での検証結果を出力するのに対し、`eda.check_viorate()` ではレコード別の検証結果を表示します。`eda.check_viorate()` から出力されるデータフレームでは、各列が検証ルールに、各行が元データの観測値に対応し、当該ルールが満たされていない場合、True と表示されます。また、`any` 列は複数あるルールのいずれか1つでも満たされていないことを、`all` 列は全てのルールが満たされていないことを示します。

```python
rule_dict3 =  {
    'to':'turnover > 0',                                     # 売上高は厳密に正である
    'sc':'staff_costs / staff < 50',                         # 従業員1人当たりの人件費は50,000ギルダー未満である
    'rev_complet':'@is_complet(turnover, total_rev, other_rev)',# 売上高と収入が全て観測されている
    }
  
df_viorate = eda.check_viorate(retailers, rule_dict3)
print(df_viorate.head())
#>       to     sc  rev_complet   any    all
#> 0   True   True         True  True   True
#> 1  False  False         True  True  False
#> 2  False   True        False  True  False
#> 3  False   True        False  True  False
#> 4   True   True         True  True   True
```

`df_viorate` データフレームの各列は論理値であるため、次のように検証ルールを満たさない観測値を抽出することができます。

```python
print(retailers.loc[df_viorate['to'], 'size':'turnover'])
#>   size  incl_prob  staff  turnover
#> 0  sc0       0.02   75.0       NaN
#> 4  sc3       0.14    NaN       NaN
#> 6  sc3       0.14    5.0       NaN
```
***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md)
