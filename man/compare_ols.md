# 回帰分析の比較：`py4stats.compare_ols()`

## 概要

　`sm.ols()` や `smf.glm()` で作成された回帰分析の結果から、推定結果を縦方向に並べて比較する表を作成します。表のフォーマットについてはR言語の [`texreg::screenreg()`](https://cran.r-project.org/web/packages/texreg/index.html)や[`modelsummary::modelsummary()`](https://modelsummary.com/man/modelsummary.html)を参考にしています。

```python
compare_ols(
    list_models: Sequence[RegressionResultsWrapper],
    model_name: Optional[Sequence[str]] = None,
    subset: Optional[Sequence[str]] = None,
    stats: Literal["std_err", "statistics", "p_value", "conf_int"] = "std_err",
    add_stars: bool = True,
    stars: Optional[Mapping[str, float]] = None,
    stats_glance: Optional[Sequence[str]] = ("rsquared_adj", "nobs", "df"),
    digits: int = 4,
    table_style: Literal["two_line", "one_line"] = "two_line",
    line_break: str = "\n",
    **kwargs: Any
)
```

## 引数

- `list_models`：**Sequence[RegressionResultsWrapper]**</br>
 推定結果を表示する分析結果のリスト（必須）。`sm.ols()` や `smf.ols()` で作成された回帰分析の結果を `list_models = [fit1, fit2]` のようにリストとして指定してください。

- `model_name`：**list of str**</br>
表頭に表示するモデルの名前。`['モデル1', 'モデル2']` のように文字列のリストを指定してください。初期設定では、自動的に `model 1, model 2, model 3 …` と連番が割り当てられます。

- `subset`：**list of str**</br>
    表示する回帰係数のリスト。指定しない場合（初期設定）、モデルに含まれる全ての回帰係数が表示されます。内部では[`pandas.DataFrame.loc`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html)メソッドを用いて処理を行っているため、`['変数1', '変数2', ...]` のような文字列のリスト、`[True, False, True, ...]` のようなブール値のリストに対応しています。文字列のリストが指定された場合、リストの並び順に合わせて回帰係数が表示されます。

- `stats`：**str**</br>
    表中の丸括弧 ( ) 内に表示する統計値の設定。次の値が指定できます。
    - `'p_value'` p-値（初期設定）
    - `'std_err'` 標準誤差
    - `'statistics'` t統計量

- `add_stars`：**bool**</br>
    回帰係数の統計的有意性を表すアスタリスク `*` を表示するかどうかを表すブール値。`add_stars = True`（初期-設定）なら表示、`add_stars = False`なら非表示となります。`table_style` に `'two_line'` を指定した場合はアスタリスクは回帰係数の直後に表示され、`'one_line'` を指定した場合は `stats` で指定した統計値の後に表示されます。アスタリスクはp-値の値に応じて次のように表示されます。

- `stars`：**dict**（`p_stars()` のみ）</br>
　有意性を示す記号を key に、表示を切り替える閾値を値(value)にもつ辞書オブジェクト。初期設定の `stars = None` の場合、下記の方式で表示されます。
    - p ≤ 0.1 `*`
    - p ≤ 0.05 `**`
    - p ≤ 0.01 `***`
    - p > 0.1 表示なし</br>
詳細は[`building_block.style_pvalue()`](man/style_pvalue.md) を参照してください。

- `stats_glance`:**list of str**</br>
- 表の下部に追加する当てはまりの尺度の種類を表す文字列のリスト。リストの値には次の値を指定できます。なお、`None` もしくは空のリスト `[ ]` が指定された場合には非表示となります。
    - `'rsquared'`：決定係数
    - `'rsquared_adj'`：自由度調整済み決定係数
    - `'nobs'`：サインプルサイズ
    - `'df'`：モデルの自由度（説明変数の数）
    - `'sigma'`：回帰式の標準誤差
    - `'F_values'`：全ての回帰係数がゼロであることを帰無仮説とするF検定の統計量
    - `'p_values'`：F検定のP-値
    - `'AIC'`：赤池情報量基準
    - `'BIC'`：ベイズ情報量基準

- `digits`：回帰係数と統計値について表示する小数点以下の桁数。初期設定は4です。

- `table_style`：表の書式設定。次の値から選択できます（部分一致可）。
    - `'two_line'`回帰係数と統計値を2行に分ける（初期設定）
    - `'one_line'`回帰係数と統計値を1行で表示する
   
- `line_break`：`table_style = 'two_line'` とした場合に使用される改行記号。`table_style = 'one_line'` とした場合、この引数は無視されます。

## 使用例 Examples

``` python
import py4stats as py4st
import statsmodels.formula.api as smf

import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
penguins = load_penguins() # サンプルデータの読み込み

# 回帰分析の実行
fit1 = smf.ols('body_mass_g ~ bill_length_mm + species', data = penguins).fit()
fit2 = smf.ols('body_mass_g ~ bill_length_mm + bill_depth_mm + species', data = penguins).fit()
fit3 = smf.ols('body_mass_g ~ bill_length_mm + bill_depth_mm + species + sex', data = penguins).fit()

compare_tab1 = py4st.compare_ols(list_models = [fit1, fit2, fit3]) # 表の作成
compare_tab1
```

| term                 | model 1        | model 2          | model 3         |
|:---------------------|:---------------|:-----------------|:----------------|
| Intercept            | 153.7397       | -1,742.7202  *** | 843.9812  **    |
|                      | (268.9012)     | (313.7697)       | (403.5956)      |
| species[T.Chinstrap] | -885.8121  *** | -539.6864  ***   | -245.1516  ***  |
|                      | (88.2502)      | (86.9425)        | (84.5952)       |
| species[T.Gentoo]    | 578.6292  ***  | 1,492.8283  ***  | 1,443.3525  *** |
|                      | (75.3623)      | (118.4442)       | (107.7844)      |
| bill_length_mm       | 91.4358  ***   | 55.6461  ***     | 26.5366  ***    |
|                      | (6.8871)       | (7.2326)         | (7.2436)        |
| bill_depth_mm        |                | 179.0434  ***    | 87.9328  ***    |
|                      |                | (19.0997)        | (20.2192)       |
| sex[T.male]          |                |                  | 437.2007  ***   |
|                      |                |                  | (49.1098)       |
| rsquared_adj         | 0.7810         | 0.8258           | 0.8613          |
| nobs                 | 342            | 342              | 333             |
| df                   | 3              | 4                | 5               |

 
  `py4st.compare_ols()` の実行結果は `Pandas` の `DataFrame` として出力されるため、`.xlsx`. ファイルなどに変換することができます。また、用途に応じて表の体裁を調整できるようにしています。

``` python
compare_tab2 = py4st.compare_ols(
    list_models = [fit1, fit2, fit3],
    model_name = ['基本モデル', '嘴の高さ追加', '性別追加'], # モデル名を変更
    stats = 'p_value',        # () 内の値をP-値に変更する
    add_stars = False,        # 有意性のアスタリスクなし
    table_style = 'one_line', # 表スタイルを1行表示に設定 'one' でも可能
    digits = 3                # 小数点以下の桁数を3に設定
    )
compare_tab2
```

| term                 | 基本モデル      | 嘴の高さ追加      | 性別追加         |
|:---------------------|:----------------|:------------------|:-----------------|
| Intercept            | 153.740(0.568)  | -1,742.720(0.000) | 843.981(0.037)   |
| species[T.Chinstrap] | -885.812(0.000) | -539.686(0.000)   | -245.152(0.004)  |
| species[T.Gentoo]    | 578.629(0.000)  | 1,492.828(0.000)  | 1,443.353(0.000) |
| bill_length_mm       | 91.436(0.000)   | 55.646(0.000)     | 26.537(0.000)    |
| bill_depth_mm        |                 | 179.043(0.000)    | 87.933(0.000)    |
| sex[T.male]          |                 |                   | 437.201(0.000)   |
| rsquared_adj         | 0.781           | 0.826             | 0.861            |
| nobs                 | 342             | 342               | 333              |
| df                   | 3               | 4                 | 5                |

`table_style = ’two_line’` のときに使用される改行記号は `line_break` で指定できます。[`great_tables`](https://posit-dev.github.io/great-tables/articles/intro.html) モジュールの `GT()` 関数と併用する場合など、html 形式で出力する場合には `line_break = '<br>' ` を指定します。

``` python
from great_tables import GT, md, html

compare_tab3 = py4st.compare_ols(
    list_models = [fit1, fit2, fit3],
    model_name = ['基本モデル', '嘴の高さ追加', '性別追加'], # モデル名を変更
    line_break = '<br>'                              # 改行文字の変更
    )

GT(compare_tab3.reset_index())\
  .tab_header(title = 'Palmer penguin データを使った回帰分析の結果')\
  .tab_source_note(
      source_note= "Signif. codes: 0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’"
      )\
  .tab_source_note(source_note = '( ) の値は標準誤差')
```
<img width="532" alt="compare_tab_gt" src="https://github.com/Hirototensho/Py4Stats/assets/55335752/51b64eaa-fb2f-45e9-ac03-16f9bd5dd3d6">

#### 回帰係数の sbusetting

 引数 `subset` を使って表示したい回帰係数を指定することで、一部の回帰係数を省略して表記することもできます。

``` python
# 説明変数に island を追加したモデルを推定
fit4 = smf.ols(
    'body_mass_g ~ bill_length_mm + bill_depth_mm + species + sex + island',
    data = penguins).fit()

var_list = [
    'species[T.Chinstrap]', 'species[T.Gentoo]',
    'bill_length_mm', 'bill_depth_mm', 'sex[T.male]'
    ]

# 全ての回帰係数を表示すると表が長すぎるので、一部を省略します
compare_tab4 = py4st.compare_ols(
    list_models = [fit2, fit3, fit4],
    subset = var_list
    )

compare_tab4.loc['島ダミー', :] = ['No', 'No', 'Yes']

compare_tab4
```
| term                 | model 1        | model 2        | model 3        |
|:---------------------|:---------------|:---------------|:---------------|
| species[T.Chinstrap] | -539.6864 ***  | -245.1516 ***  | -255.2732 ***  |
|                      | (86.9425)      | (84.5952)      | (92.4796)      |
| species[T.Gentoo]    | 1,492.8283 *** | 1,443.3525 *** | 1,446.1574 *** |
|                      | (118.4442)     | (107.7844)     | (114.1676)     |
| bill_length_mm       | 55.6461 ***    | 26.5366 ***    | 26.6643 ***    |
|                      | (7.2326)       | (7.2436)       | (7.2792)       |
| bill_depth_mm        | 179.0434 ***   | 87.9328 ***    | 88.3284 ***    |
|                      | (19.0997)      | (20.2192)      | (20.3267)      |
| sex[T.male]          |                | 437.2007 ***   | 436.0334 ***   |
|                      |                | (49.1098)      | (49.4227)      |
| rsquared_adj         | 0.8258         | 0.8613         | 0.8605         |
| nobs                 | 342            | 333            | 333            |
| df                   | 4              | 5              | 7              |
| 島ダミー             | No             | No             | Yes            |

pandas の [`pandas.DataFrame.query`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html) メソッドを使って、次のように説明変数を除外することもできます。

``` python
compare_tab4 = py4st.compare_ols(
    list_models = [fit2, fit3, fit4]
    )

compare_tab4 = compare_tab4\
  .query('~term.str.contains("Intercept|island")').copy()

compare_tab4.loc['島ダミー', :] = ['No', 'No', 'Yes']

compare_tab4 # 上記のコードと同じ結果
```

## 補足

　　`table_style = 'two_line'` としたとき、初期設定ではの回帰係数とp-値の間に改行記号 `'\n'`が挿入されます。`そのため、print()` 関数や `display()` 関数を使った出力では、改行記号 `'\n'` がそのまま表示されます。この場合でも、[`pd.DataFrame.to_excel()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_excel.html) や [`pd.DataFrame.to_markdown()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_markdown.html) を使って Excel ファイルや markdown の表に変換していただくと、改行として反映されます。

## 参照 see also

　一般化線形モデルの限界効果を比較する場合は [`py4stats.compare_mfx()`](./compare_mfx.md)をご利用ください。

***
[Return to **Function reference**.](../reference.md)
