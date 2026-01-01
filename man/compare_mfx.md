# 限界効果の比較：`py4stats.compare_mfx()`

## 概要

　[`sm.glm()`](https://www.statsmodels.org/devel/generated/statsmodels.genmod.generalized_linear_model.GLM.html)の推定結果を計量経済学の実証論文でよく用いられる、回帰分析の結果を縦方向に並べて比較する表を作成します。表のフォーマットについてはR言語の [`texreg::screenreg()`](https://cran.r-project.org/web/packages/texreg/index.html)や[`modelsummary::modelsummary()`](https://modelsummary.com/man/modelsummary.html)を参考にしています。

```python
compare_mfx(
    list_models, 
    model_name = None,
    subset = None,
    stats = 'std_err',
    add_stars = True,
    stats_glance = ['prsquared', 'nobs', 'df'],
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

- `list_models`：推定結果を表示する分析結果のリスト（必須）。[`sm.glm()`](https://www.statsmodels.org/devel/generated/statsmodels.genmod.generalized_linear_model.GLM.html)で作成された一般化線形モデルの結果を `list_models = [fit1, fit2]` のようにリストとして指定してください。

- `model_name`：表頭に表示するモデルの名前。`['モデル1', 'モデル2']` のように文字列のリストを指定してください。初期設定では、自動的に `model 1, model 2, model 3 …` と連番が割り当てられます。

- `subset = None`：表示する回帰係数のリスト。指定しない場合（初期設定）、モデルに含まれる全ての回帰係数が表示されます。内部では[`pandas.DataFrame.loc`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html)メソッドを用いて処理を行っているため、`['変数1', '変数2', ...]` のような文字列のリスト、`[True, False, True, ...]` のようなブール値のリストに対応しています。文字列のリストが指定された場合、リストの並び順に合わせて回帰係数が表示されます。

- `stats`：表中の丸括弧 ( ) 内に表示する統計値の設定。次の値が指定できます。
    - `'p_value'` p-値（初期設定）
    - `'std_err'` 標準誤差
    - `'statistics'` t統計量

- `add_stars`：回帰係数の統計的有意性を表すアスタリスク `*` を表示するかどうかを表すブール値。`add_stars = True`（初期設定）なら表示、`add_stars = False`なら非表示となります。`table_style` に `'two_line'` を指定した場合はアスタリスクは回帰係数の直後に表示され、`'one_line'` を指定した場合は `stats` で指定した統計値の後に表示されます。アスタリスクはp-値の値に応じて次のように表示されます。
    - p ≤ 0.1 `*`
    - p ≤ 0.05 `**`
    - p ≤ 0.01 `***`
    - p > 0.1 表示なし

- `stats_glance`**list of str**：表の下部に追加する当てはまりの尺度の種類を表す文字列のリスト。リストの値には次の値を指定できます。なお、`None` もしくは空のリスト `[]` が指定された場合には非表示となります。
    - `'prsquared'`：擬似決定係数
    - `'LL-Null'`： Null model の対数尤度
    - `'df_null'`, ：Null model の自由度 `= nobs - 1`
    - `'logLik'` ：モデルの対数尤度
    - `'AIC'`：赤池情報量基準
    - `'BIC'`：ベイズ情報量基準 
    - `'deviance'`：モデルの逸脱度  `= -2logLik`
    - `'nobs'`：サインプルサイズ
    - `'df'`：モデルの自由度（説明変数の数）
    - `'df_resid'`：残差の自由度 

- `digits`：回帰係数と統計値について表示する小数点以下の桁数。初期設定は4です。

- `table_style`：表の書式設定。次の値から選択できます（部分一致可）。
    - `'two_line'`回帰係数と統計値を2行に分ける（初期設定）
    - `'one_line'`回帰係数と統計値を1行で表示する
   
- `line_break`：`table_style = 'two_line'` とした場合に使用される改行記号。`table_style = 'one_line'` とした場合、この引数は無視されます。

- `at`：限界効果の集計方法。内部で使用している[`statsmodels.discrete.discrete_model.DiscreteResults.get_margeff()`](https://www.statsmodels.org/devel/generated/statsmodels.discrete.discrete_model.DiscreteResults.get_margeff.html) メソッドに引数 `at` として渡されます。`method = 'coef'` を指定した場合、この引数は無視されます。
    - `'overall'`：各観測値の限界効果の平均値を表示（初期設定）
    - `'mean'`：各説明変数の平均値における限界効果を表示
    - `'median'`：各説明変数の中央値における限界効果を表示
    - `'zero'`：各説明変数の値がゼロであるときの限界効果を表示

- `method`：推定する限界効果の種類。内部で使用している[`statsmodels.discrete.discrete_model.DiscreteResults.get_margeff()`](https://www.statsmodels.org/devel/generated/statsmodels.discrete.discrete_model.DiscreteResults.get_margeff.html) メソッドに引数 `method` として渡されます。ただし、`method = 'coef'` を指定した場合には限界効果を推定せずに回帰係数をそのまま表示します。
    - `'coef'`：回帰係数の推定値を表示
    - `'dydx'`：限界効果の値を変換なしでそのまま表。（初期設定）
    - `'eyex'`：弾力性 d(lny)/d(lnx) の推定値を表示
    - `'dyex'`：準弾力性 dy /d(lnx) の推定値を表示
    - `'eydx'`：準弾力性 d(lny)/dx の推定値を表示

- `dummy`：ダミー変数の限界効果の推定方法。もし False （初期設定）であれば、ダミー変数を連続な数値変数として扱います。もし、True であればダミー変数が0から1へと変化したときの予測値の変化を推定します。内部で使用している[`statsmodels.discrete.discrete_model.DiscreteResults.get_margeff()`](https://www.statsmodels.org/devel/generated/statsmodels.discrete.discrete_model.DiscreteResults.get_margeff.html) メソッドに引数 `dummy` として渡されます。

## 使用例

``` python
import py4stats as py4st
import statsmodels.formula.api as smf

import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
penguins = load_penguins() # サンプルデータの読み込み
```

　`py4st.compare_mfx()` は [`py4st.compare_ols()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/compare_ols.md) の一般化線型モデルバージョンで、初期設定では `statsmodels` ライブラリの[`.get_margeff()`](https://www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.DiscreteResults.get_margeff.html) メソッドから得られた限界効果の推定値を表示します。

```python
penguins['female'] = np.where(penguins['sex'] == 'female', 1, 0)

# ロジスティック回帰の実行
fit_logit1 = smf.logit('female ~ body_mass_g + bill_length_mm + bill_depth_mm', data = penguins).fit()
fit_logit2 = smf.logit('female ~ body_mass_g + bill_length_mm + bill_depth_mm + species', data = penguins).fit()

py4st.compare_mfx([fit_logit1, fit_logit2])
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


```python
from great_tables import GT, md, html
compare_tab = py4st.compare_mfx(
    [fit_logit1, fit_logit2],
    model_name = ['ベースモデル', 'species 追加'], # モデル名を変更
    line_break = '<br>'                         # 改行文字の変更
)

GT(compare_tab.reset_index())\
  .tab_header(title = 'ロジットモデルの限界効果')\
  .tab_source_note(
      source_note= "Signif. codes: 0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’"
      )\
  .tab_source_note(source_note = '丸括弧 ( ) の値は標準誤差')
```

<img width="387" alt="compare_tab_gt2" src="https://github.com/Hirototensho/Py4Stats/assets/55335752/98b4f543-a2f9-46d1-9495-7ebf5737330f">

## 補足

　　`table_style = 'two_line'` としたとき、初期設定ではの回帰係数とp-値の間に改行記号 `'\n'`が挿入されます。`そのため、print()` 関数や `display()` 関数を使った出力では、改行記号 `'\n'` がそのまま表示されます。この場合でも、[`pd.DataFrame.to_excel()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_excel.html) や [`pd.DataFrame.to_markdown()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_markdown.html) を使って Excel ファイルや markdown の表に変換していただくと、改行として反映されます。

***
[Return to **Function reference**.](https://github.com/Hirototensho/Py4Stats/blob/main/reference.md)
