# `py4stats.regression_tools.compare_ols()`, `py4stats.regression_tools.compare_mfx()`

## 概要

`reg.compare_ols()` 回帰分析の表を作成。：計量経済学の実証論文でよく用いられる、回帰分析の結果を縦方向に並べて比較する表を作成します。表のフォーマットについてはR言語の [`texreg::screenreg()`](https://cran.r-project.org/web/packages/texreg/index.html)や[`modelsummary::modelsummary()`](https://modelsummary.com/man/modelsummary.html)を参考にしています。

```python
compare_ols(
    list_models, 
    model_name = None,
    subset = None,
    stats = 'std_err',
    add_stars = True,
    stats_glance = ['rsquared_adj', 'nobs', 'df'],
    digits = 4, 
    table_style = 'two_line',
    line_break = '\n',
    **kwargs
)

compare_mfx(
    list_models, 
    model_name = None,
    subset = None,
    stats = 'std_err',
    add_stars = True,
    at = 'overall',
    method = 'dydx',
    dummy = False,
    digits = 4, 
    table_style = 'two_line',
    line_break = '\n',
    **kwargs
)
```

## 引数

- `list_models`：推定結果を表示する分析結果のリスト（必須）。`sm.ols()` や `smf.ols()` で作成された回帰分析の結果を `list_models = [fit1, fit2]` のようにリストとして指定してください。

- `model_name = None`：表頭に表示するモデルの名前。`['モデル1', 'モデル2']` のように文字列のリストを指定してください。初期設定では、自動的に `model 1, model 2, model 3 …` と連番が割り当てられます。

- `subset = None`：表示する回帰係数のリスト。指定しない場合（初期設定）、モデルに含まれる全ての回帰係数が表示されます。内部では[`pandas.DataFrame.loc`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html)メソッドを用いて処理を行っているため、`['変数1', '変数2', ...]` のような文字列のリスト、`[True, False, True, ...]` のようなブール値のリストに対応しています。文字列のリストが指定された場合、リストの並び順に合わせて回帰係数が表示されます。

- `stats = 'std_err'`：表中の丸括弧 ( ) 内に表示する統計値の設定。次の値が指定できます（部分一致可）。
    - `'p_value'` p-値（初期設定）
    - `'std_err'` 標準誤差
    - `'statistics'` t統計量

- `add_stars = True`：回帰係数の統計的有意性を表すアスタリスク `*` を表示するかどうかを表すブール値。`add_stars = True`（初期設定）なら表示、`add_stars = False`なら非表示となります。`table_style` に `'two_line'` を指定した場合はアスタリスクは回帰係数の直後に表示され、`'one_line'` を指定した場合は `stats` で指定した統計値の後に表示されます。アスタリスクはp-値の値に応じて次のように表示されます。
    - p ≤ 0.1 `*`
    - p ≤ 0.05 `**`
    - p ≤ 0.01 `***`
    - p > 0.1 表示なし

- `stats_glance`：表の下部に追加する回帰モデル全体に関する統計値の種類を表す文字列のリスト（`compare_ols()` のみ）。初期設定は `['rsquared_adj', 'nobs', 'df']`。リストの値には次の値を指定できます（部分一致可）。
    - `'rsquared'`：決定係数
    - `'rsquared_adj'`：自由度調整済み決定係数
    - `'nobs'`：サインプルサイズ
    - `'df'`：モデルの自由度（説明変数の数）
    - `'sigma'`：回帰式の標準誤差
    - `'F_values'`：全ての回帰係数がゼロであることを帰無仮説とするF検定の統計量
    - `'p_values'`：F検定のP-値
    - `'AIC'`：赤池情報量基準
    - `'BIC'`：ベイズ情報量基準

- `digits = 4`：回帰係数と統計値について表示する小数点以下の桁数。初期設定は4です。

- `table_style = 'two_line'`：表の書式設定。次の値から選択できます（部分一致可）。
    - `'two_line'`回帰係数と統計値を2行に分ける（初期設定）
    - `'one_line'`回帰係数と統計値を1行で表示する
   
- line_break = '\n'：`table_style = 'two_line'` とした場合に使用される改行記号。`table_style = 'one_line'` とした場合、この引数は無視されます。



- `at = 'overall'`：限界効果の集計方法（`compare_mfx()` のみ）。内部で使用している[`statsmodels.discrete.discrete_model.DiscreteResults.get_margeff()`](https://www.statsmodels.org/devel/generated/statsmodels.discrete.discrete_model.DiscreteResults.get_margeff.html) メソッドに引数 `at` として渡されます。`method = 'coef'` を指定した場合、この引数は無視されます（部分一致可）。
    - `'overall'`：各観測値の限界効果の平均値を表示（初期設定）
    - `'mean'`：各説明変数の平均値における限界効果を表示
    - `'median'`：各説明変数の中央値における限界効果を表示
    - `'zero'`：各説明変数の値がゼロであるときの限界効果を表示

- `method = 'dydx'`：推定する限界効果の種類（`compare_mfx()` のみ）。内部で使用している[`statsmodels.discrete.discrete_model.DiscreteResults.get_margeff()`](https://www.statsmodels.org/devel/generated/statsmodels.discrete.discrete_model.DiscreteResults.get_margeff.html) メソッドに引数 `method` として渡されます。ただし、`method = 'coef'` を指定した場合には限界効果を推定せずに回帰係数をそのまま表示します。
    - `'coef'`：回帰係数の推定値を表示
    - `'dydx'`：限界効果の値を変換なしでそのまま表。（初期設定）
    - `'eyex'`：弾力性 d(lny)/d(lnx) の推定値を表示
    - `'dyex'`：準弾力性 dy /d(lnx) の推定値を表示
    - `'eydx'`：準弾力性 d(lny)/dx の推定値を表示

- `dummy = False`：ダミー変数の限界効果の推定方法（`compare_mfx()` のみ）。もし False （初期設定）であれば、ダミー変数を連続な数値変数として扱います。もし、True であればダミー変数が0から1へと変化したときの予測値の変化を推定します。内部で使用している[`statsmodels.discrete.discrete_model.DiscreteResults.get_margeff()`](https://www.statsmodels.org/devel/generated/statsmodels.discrete.discrete_model.DiscreteResults.get_margeff.html) メソッドに引数 `dummy` として渡されます。

## 使用例

``` python
from py4stats import eda_tools as eda        # 基本統計量やデータの要約など
from py4stats import regression_tools as reg # 回帰分析の要約
import statsmodels.formula.api as smf

import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
penguins = load_penguins() # サンプルデータの読み込み


# 回帰分析の実行
fit1 = smf.ols('body_mass_g ~ bill_length_mm + species', data = penguins).fit()
fit2 = smf.ols('body_mass_g ~ bill_length_mm + bill_depth_mm + species', data = penguins).fit()
fit3 = smf.ols('body_mass_g ~ bill_length_mm + bill_depth_mm + species + sex', data = penguins).fit()

reg.compare_ols(list_models = [fit1, fit2, fit3]) # 表の作成
```

| term                 | model 1       | model 2        | model 3       |
|:---------------------|:--------------|:---------------|:--------------|
| Intercept            | 153.7397      | -1742.7202 *** | 843.9812 **   |
|                      | (268.9012)    | (313.7697)     | (403.5956)    |
| species[T.Chinstrap] | -885.8121 *** | -539.6864 ***  | -245.1516 *** |
|                      | (88.2502)     | (86.9425)      | (84.5952)     |
| species[T.Gentoo]    | 578.6292 ***  | 1492.8283 ***  | 1443.3525 *** |
|                      | (75.3623)     | (118.4442)     | (107.7844)    |
| bill_length_mm       | 91.4358 ***   | 55.6461 ***    | 26.5366 ***   |
|                      | (6.8871)      | (7.2326)       | (7.2436)      |
| bill_depth_mm        |               | 179.0434 ***   | 87.9328 ***   |
|                      |               | (19.0997)      | (20.2192)     |
| sex[T.male]          |               |                | 437.2007 ***  |
|                      |               |                | (49.1098)     |
| rsquared_adj         | 0.7810        | 0.8258         | 0.8613        |
| nobs                 | 342           | 342            | 333           |
| df                   | 3             | 4              | 5             |

　**注意**：pandas データフレームとしの表示では、`table_style = 'two_line'` としたときの回帰係数とp-値の間の改行が、改行記号「\n」が表示されますが、Excel ファイルとして保存すると、正しくセル内での改行として扱われます。  
　`reg.compare_ols()` の実行結果は `Pandas` の `DataFrame` として出力されるため、`.xlsx`. ファイルなどに変換することができます。また、用途に応じて表の体裁を調整できるようにしています。

``` python
reg.compare_ols(
    list_models = [fit1, fit2, fit3],
    model_name = ['基本モデル', '嘴の高さ追加', '性別追加'], # モデル名を変更
    stats = 'p_value',        # () 内の値をP-値に変更する
    add_stars = False,        # 有意性のアスタリスクなし
    table_style = 'one_line', # 表スタイルを1行表示に設定 'one' でも可能
    digits = 2                # 小数点以下の桁数を2に設定
    )
```

| term                 | 基本モデル    | 嘴の高さ追加   | 性別追加      |
|:---------------------|:--------------|:---------------|:--------------|
| Intercept            | 153.74(0.57)  | -1742.72(0.00) | 843.98(0.04)  |
| species[T.Chinstrap] | -885.81(0.00) | -539.69(0.00)  | -245.15(0.00) |
| species[T.Gentoo]    | 578.63(0.00)  | 1492.83(0.00)  | 1443.35(0.00) |
| bill_length_mm       | 91.44(0.00)   | 55.65(0.00)    | 26.54(0.00)   |
| bill_depth_mm        |               | 179.04(0.00)   | 87.93(0.00)   |
| sex[T.male]          |               |                | 437.20(0.00)  |
| rsquared_adj         | 0.78          | 0.83           | 0.86          |
| nobs                 | 342           | 342            | 333           |
| df                   | 3             | 4              | 5             |

`table_style = ’two_line’` のときに使用される改行記号は `line_break` で指定できます。[`great_tables`](https://posit-dev.github.io/great-tables/articles/intro.html) モジュールの `GT()` 関数と併用する場合など、html 形式で出力する場合には `line_break = '<br>' ` を指定します。

``` python
from great_tables import GT, md, html

compare_tab3 = reg.compare_ols(
    list_models = [fit1, fit2, fit3],
    model_name = ['基本モデル', '嘴の高さ追加', '性別追加'], # モデル名を変更
    line_break = '<br>'                                 # 改行文字の変更
    )

compare_gt = GT(compare_tab3.reset_index())\
  .tab_header(title = 'Palmer penguin データを使った回帰分析の結果')\
  .tab_source_note(
      source_note= "Signif. codes: 0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1"
      )\
  .tab_source_note(source_note = '( ) の値は標準誤差')

compare_gt
```
<img width="549" alt="compare_tab_gt" src="https://github.com/Hirototensho/Py4Stats/assets/55335752/7e189a26-c2a3-4a52-b717-61cf71317cd3">

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

# 全て表示すると表が長すぎるので、island 一部を省略したい
compare_tab4 = reg.compare_ols(
    list_models = [fit2, fit3, fit4],
    subset = var_list
    )

compare_tab4.loc['島ダミー', :] = ['No', 'No', 'Yes']

compare_tab4
```
| term                 | model 1       | model 2       | model 3       |
|:---------------------|:--------------|:--------------|:--------------|
| species[T.Chinstrap] | -539.6864 *** | -245.1516 *** | -255.2732 *** |
|                      | (86.9425)     | (84.5952)     | (92.4796)     |
| species[T.Gentoo]    | 1492.8283 *** | 1443.3525 *** | 1446.1574 *** |
|                      | (118.4442)    | (107.7844)    | (114.1676)    |
| bill_length_mm       | 55.6461 ***   | 26.5366 ***   | 26.6643 ***   |
|                      | (7.2326)      | (7.2436)      | (7.2792)      |
| bill_depth_mm        | 179.0434 ***  | 87.9328 ***   | 88.3284 ***   |
|                      | (19.0997)     | (20.2192)     | (20.3267)     |
| sex[T.male]          |               | 437.2007 ***  | 436.0334 ***  |
|                      |               | (49.1098)     | (49.4227)     |
| rsquared_adj         | 0.8258        | 0.8613        | 0.8605        |
| nobs                 | 342           | 333           | 333           |
| df                   | 4             | 5             | 7             |
| 島ダミー             | No            | No            | Yes           |

`eda.filtering_out()` 関数を使って行名に `'Intercept'` または `'island'` を含む列を除外しても同じ結果を得ることができます。

``` python
compare_tab4 = reg.compare_ols(
    list_models = [fit1, fit2, fit3]
    )

compare_tab4 = eda.filtering_out(
    compare_tab4,
    contains = 'Intercept|species', 
    axis = 'index'
    )

compare_tab4.loc['種ダミー', :] = ['Yes', 'Yes', 'Yes']

compare_tab4 # 上記のコードと同じ結果
```

`reg.compare_mfx()` は `reg.compare_ols()` の一般化線型モデルバージョンで、初期設定では `statsmodels` ライブラリの[`.get_margeff()`](https://www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.DiscreteResults.get_margeff.html) メソッドから得られた限界効果の推定値を表示します。

```python
# ロジスティック回帰の実行
fit_logit1 = smf.logit('female ~ body_mass_g + bill_length_mm + bill_depth_mm', data = penguins).fit()
fit_logit2 = smf.logit('female ~ body_mass_g + bill_length_mm + bill_depth_mm + species', data = penguins).fit()

reg.compare_mfx([fit_logit1, fit_logit2])
```
| term                 | model 1     | model 2     |
|:---------------------|:------------|:------------|
| body_mass_g          | -0.0004 *** | -0.0003 *** |
|                      | (0.0000)    | (0.0000)    |
| bill_length_mm       | -0.0053     | -0.0357 *** |
|                      | (0.0036)    | (0.0070)    |
| bill_depth_mm        | -0.1490 *** | -0.1098 *** |
|                      | (0.0051)    | (0.0175)    |
| species[T.Chinstrap] |             | 0.4172 ***  |
|                      |             | (0.0848)    |
| species[T.Gentoo]    |             | 0.3527 ***  |
|                      |             | (0.1308)    |
| prsquared            | 0.5647      | 0.6187      |
| nobs                 | 342         | 342         |
| df                   | 3           | 5           |


