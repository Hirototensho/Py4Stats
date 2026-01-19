# narwhals についての考察

## narwhals での再現が難しい Pandas の機能

### 異なるデータフレーム間の二項演算

Pandas の場合、2つのデータフレーム `df1` と `df2` が共通の columns と index をもつ限り、`df3 = df1 + df2` によって二項演算を行うことができ、このとき、columns と index をもつ要素同士が加算されます。しかし、narwhals には Pandas のような index が存在しないため、この計算は再現が困難です。

### データフレームへの値の代入 

Pandas の場合、`df.loc[i, j] = x` という形でデータフレーム `df` の i, j 要素に値 `x` を代入することができますが、narwhals ではこれに相当する演算 `df[i, j] = x` は禁止されています。

異なるデータフレーム間の二項演算に制約があること、そしてデータフレームへの値の代入が難しいことから、[`tabyl()`](../man/tabyl.md) 関数では、集計後の作表処理の一部を Pandas に依存しています。

### 任意の関数でグループ別集計を行う

自作関数を使ってグループ別集計を行いたい場合、Pandas であれば `df.groupby(group)[x].agg(my_func)` で行うことができます。同じく narwhals でも 

``` python
data_nw.group_by(nw.col(group)).agg(nw.col('x').mean())
````

という形でグループ別の集計がサポートされているものの、ここで使用できる集計関数は narwhals で実装されているものに限定されるようで、次のような方法で自作関数を使用することはできません。


``` python
data_nw.group_by(nw.col(group)).agg(nw.col('x').my_func())
data_nw.group_by(nw.col(group)).agg(my_func(nw.col('x')))
````

例えば Py4Stats では、[`Pareto_plot()`](../man/Pareto_plot.md) 関数の内部実装に使用している [`make_rank_table()`](../py4stats/eda_tools/_nw.py) 関数において、任意の `aggfunc()` 関数をグループ別集計に使うために、サブセッティングを使って `group_by()` メソッドの使用を回避するという変則的（かつ、おそらく非効率な）な実装を行なっています。

``` python
stat_values = [
            aggfunc(
                data_nw.filter(nw.col(group) == g)[values]
                .drop_nulls().to_native()
                ) 
            for g in group_value
            ]
```

また、上記の回避策のもう1つの問題として、`data_nw.filter(nw.col(group) == g)` では、複数の変数に基づくグループ化に対応できないことも挙げられます。`make_rank_table()` 関数については、`Pareto_plot()` 関数でパレート図を作図するときに横軸になる `group` が多変数だと対応できないので、`group` が1変数（= 引数として1つの文字列だけを受け付ける）とすることで妥協しています。

ただ、現時点で [`narwhals.GroupBy`](https://narwhals-dev.github.io/narwhals/api-reference/group_by/) クラスに実装されているメソッドは `.agg()` しかなく、開発が進めばより柔軟な関数適用が可能になるのではないかと期待しています。

## narwhals におけるバックエンドとその書き換え

### バックエンドの基本的な理解

narwhals におけるバックエンドによる型変換の基本的な理解として（不正確かもしれませんが）、`nw.from_native(data)` の実行時に `data` の型に応じて backend が記録され、`.to_native()` メソッドを呼び出すと、記録された backend に応じて元の型に変換されます。

backend の情報は `.select()` `.filter()` などのメソッドを使って `data_nw` を加工しても保持され、これによって入力された `input_pd` と同じ型のデータフレームを返すことが可能になっています。

```python
data_nw = nw.from_native(input_pd) # ここで backend が記録される
data_nw.implementation       # -> Pandas
result = data_nw.to_native() # -> pd.DataFrame が出力される
```

一方で、処理の途中で pd.DataFrame や pl.DataFrame などの native オブジェクトを経由した場合、改めて `nw.from_native()` を使って nw.DataFrame に変換し直したとしても、その時点で backend が上書きされるので、`.to_native()` メソッドを使用しても引数として入力された `input_pd` と同じ型に復元される保証はありません。

```python
data_nw = nw.from_native(input_pd)              # ここで backend が記録される
data_nw2 = nw.from_native(data_nw.to_polars())  # ここで backend が上書きされる
data_nw2.implementation        # -> polars
result = data_nw2.to_native()  # -> pl.DataFrame が出力される
```

従って、`result` が `input_pd` と同じ型をもつことを保証するには、`data_nw` を nw.DataFrame クラスのまま維持する（≒ narwhals ベースのメソッドだけで処理を書く）必要があり、これが narwhals ベースの実装としてのあるべき姿だと思われます。

一方で、一部の処理が特定のバックエンド（e.g. Pandas）に依存している場合にはどうするべきでしょうか。これには次のような2つの選択肢があると考えています。

1. 処理が依存しているバックエンドのオブジェクト（e.g. pd.DataFrame）として出力する〔推奨〕
2. narwhals の仕様を迂回してバックエンドを書き換える〔非推奨ですが次節で考察〕

これら2つの可能性の間での選択は、技術的な問題であると同時にユーザーとのコミュニケーションの問題です。入力と同型のデータフレームを返す関数の中に pd.DataFrame を返す関数が混ざっていることをユーザーにどう説明するのか。あるいは、narwhals の仕様を迂回をしたことで非効率性やカラムレベルでデータ型（dtype）の一貫性が失われる問題が生じたとして、それをユーザーにどう説明するのか、という問いです。

### バックエンドの書き換え (非推奨)

いま、`some_computation()` として実装された処理の一部が Pandas に依存しており、結果が `result_pd` という pd.DataFrame 型のオブジェクトとして得られているとします。このとき、`result_pd` をもとのデータフレーム `data_pl` と同型にする方法の1つとして、`result_pd` を `pd.Series.to_dict()` などを使って辞書のリスト（list of dict）に変換したのち、`nw.from_dicts()` を使って `data_pl` と同じバックエンドをもつ `nw.DataFrame` に変換するという方法があります。

以上の変換の実例を見てみましょう。まずは、`data_pl`

``` python
data_pl = pl.from_pandas(load_penguins())[:10, :2]

data_pl = data_pl.with_columns(
        pl.all().cast(pl.Categorical)
    )
print(type(data_pl))
#> <class 'polars.dataframe.frame.DataFrame'>
print(data_pl.schema)
#> Schema({'species': Categorical, 'island': Categorical})

data_nw_pl = nw.from_native(data_pl) # ここでバックエンドを記録、後ほど復元に使います。

# 何かしらの処理の結果 pd.DataFrame に変換されたとする
result_pd = data_nw_pl.to_pandas()
print(type(result_pd))
#> <class 'pandas.core.frame.DataFrame'>
```

次に、pl.DataFrame 型をもつ `result_pd` を pl.DataFrame に変換します。

ここでポイントとなるのが、`nw.from_dicts()` 関数の引数の (1)`schema` 引数と、(2)`backend`引数に、それぞれ `data_nw_pl` から取得した値を入力することで、`result_pl` の列が `data_pl` と同じく `Categorical` 型になるようにしています(指定しないと String 型として解釈されてしまいます)。

``` python
# Pandas -> polars の変換
dict_list = [result_pd.loc[i, :].to_dict() for i in result_pd.index]

result_nw_pl = nw.from_dicts(
    dict_list, 
    schema = data_nw_pl.schema,         # (1)
    backend = data_nw_pl.implementation # (2)
    )
result_pl = result_nw_pl.to_native()

print(type(result_pl))
#> <class 'polars.dataframe.frame.DataFrame'>

print(result_pl.schema)
#> Schema({'species': Categorical, 'island': Categorical})
```

また、Series については、`nw.Series.from_iterable()` 関数を使うことで、次のようにバックエンドを書き換えることができます。

```python
x_pl = data_pl['island']
print(type(x_pl))
#> <class 'polars.series.series.Series'>
print(x_pl.dtype)
#> Categorical

x_nw = nw.from_native(x_pl, allow_series = True)
x_pd = x_nw.to_pandas()
print(type(x_pd))
#> <class 'pandas.core.series.Series'>
```

```python
x_pl2 = nw.Series.from_iterable(
    name = x_pd.name,
    values = x_pd.to_list(),
    backend = x_nw.implementation,
    dtype = x_nw.dtype
).to_native()

print(type(x_pl2))
#> <class 'polars.series.series.Series'>
print(x_pl2.dtype)
#> Categorical
```

narwhals の仕様を迂回してバックエンドを書き換えることは可能ですが、この方法には次のような問題があります。
ただし、以上のような方法でバックエンドの書き換えは可能ですが、

1. 小さいデータフレームでない限り時間がかかる
    - 恐らく、dict_list を作成するための for ループによるもの
2. 上記の (1) に代入する正しい schema が用意できないと、カラムレベルでデータ型の一貫性保証できない。

特に2番目の問題点については、集計処理によって列名が変わった場合には正しい schema(≒ {列名:dtype} の辞書オブジェクト)を用意することが難しくなります。そして、schema を指定できないと、`pd.Categorical`、`pl.Categorical` あるいは `pl.Enum` といったカテゴリー変数は文字列型に変換されてしまい、データ型の一貫性が失われます。

カラムレベルで型の一貫性が失われると、返り値が入力値とは異なる型になるよりも把握しづらく、また挙動の予測が難しいため、上記のような処理は採用するとしても、**他に方法がないときの最終手段**として扱うべきでしょう。









