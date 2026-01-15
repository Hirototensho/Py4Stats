# Function reference

<!--注意 main ブランチにマージする際に、`blob/experiment/narwhals/` を `blob/main/` に変更すること。-->

## Main Module

### `py4stats.eda_tools`

`py4stats.eda_tools` モジュールは、探索的データ解析と前処理に関する機能を提供します。複数の DataFrame バックエンドに対して共通の API を提供することを目的として、[`narwhals`](https://narwhals-dev.github.io/narwhals/) ライブラリを用いて実装されています。詳細は [Technical Notes: py4stats.eda_tools における narwhals ベースの実装](articles/narwhals_in_py4stats.md) を参照してください。

#### データフレームの概要
[`py4stats.diagnose()`](man/diagnose.md)

#### クロス集計

[`py4stats.tabyl()`](man/tabyl.md)

[`py4stats.freq_table()`](man/freq_table.md)

[`py4stats.Pareto_plot()`](man/Pareto_plot.md)

[`py4stats.plot_category()`](man/plot_category.md)

### 数値変数の点推定と区間推定

[`py4stats.mean_qi()`](man/point_range.md)
[`py4stats.median_qi()`](man/point_range.md)
[`py4stats.mean_ci()`](man/point_range.md)

#### データフレームの列や行の削除

[`py4stats.remove_empty()`](man/remove_empty_constant.md)  
[`py4stats.remove_constant()`](man/remove_empty_constant.md)

[`py4stats.filtering_out()`](man/filtering_out.md)

#### 複数のデータフレームの比較

[`py4stats.compare_df_cols()`](man/compare_df_cols.md)

#### 簡易なグループ別統計量の比較

[`py4stats.compare_group_means()`](man/compare_group_stats.md)
[`py4stats.compare_group_median()`](man/compare_group_stats.md)

[`py4stats.plot_mean_diff()`](man/compare_group_stats.md)
[`py4stats.plot_median_diff()`](man/compare_group_stats.md)

#### 簡易な欠測値の可視化

[`py4stats.plot_miss_var()`](man/plot_miss_var.md)

#### 数値変数の集計と標準化

[`py4stats.weighted_mean()`](man/scale_wmean.md)
[`py4stats.scale()`](man/scale_wmean.md)
[`py4stats.min_max()`](man/scale_wmean.md)

#### 論理関数

[`py4stats.is_number()`](man/predicate_str.md)
[`py4stats.is_ymd()`](man/predicate_str.md)
[`py4stats.is_ymd_like()`](man/predicate_str.md)

[`py4stats.is_dummy()`](man/is_dummy.md)

#### 簡易なルールベースのデータ検証ツール

[`py4stats.check_that()`](man/varidate.md) [`py4stats.check_viorate()`](man/varidate.md)


***
### `py4stats.regression_tools`

`py4stats.regression_tools` は [`statsmodels`](https://www.statsmodels.org/stable/index.html) ライブラリで作成された回帰分析の結果についての表作成と可視化を補助する機能を提供するモジュールです。

#### 線形モデルの比較

[`py4stats.compare_ols()`](man/compare_ols.md)

[`py4stats.compare_mfx()`](man/compare_mfx.md)

#### 線形モデルの可視化

[`py4stats.coefplot()`](man/coefplot.md) [`py4stats.mfxplot()`](man/coefplot.md)

#### 線形モデルを作表するためのバックエンド関数
[`py4stats.tidy()`](man/tidy.md)[`py4stats.tidy_mfx()`](man/tidy.md)

[`py4stats.tidy_test()`](man/tidy_test.md)

[`py4stats.glance()`](man/glance.md)

#### Blinder-Oaxaca分解

[`py4stats.Blinder_Oaxaca()`](man/Blinder_Oaxaca.md)
[`py4stats.plot_Blinder_Oaxaca()`](man/Blinder_Oaxaca.md)

## Sub Module

### `py4stats.heckit_helper`

[`heckit_helper.Heckit_from_formula()`](man/Heckit_from_formula.md)

[`heckit_helper.tidy_heckit()`](man/tidy_heckit.md)

[`heckit_helper.heckitmfx_compute()`](man/heckitmfx_compute.md)

***
### `py4stats.building_block`

`py4stats.regression_tools` の関数を [`py4etrics.heckit`](https://github.com/Py4Etrics/py4etrics) で実装された Heckit
 モデルに対応させるためのメソッドを実装したモジュールです。

### 引数のアサーション関数

[`building_block.arg_match()`](man/arg_match.md)

[`building_block.assert_character()`](man/assert_dtype.md)
[`building_block.assert_logical()`](man/assert_dtype.md)
[`building_block.assert_numeric()`](man/assert_dtype.md)
[`building_block.assert_integer()`](man/assert_dtype.md)
[`building_block.assert_count()`](man/assert_dtype.md)
[`building_block.assert_float()`](man/assert_dtype.md)

### データ型を判定する論理関数

[`building_block.is_character()`](man/is_dtype.md)
[`building_block.is_logical()`](man/is_dtype.md)
[`building_block.is_numeric()`](man/is_dtype.md)
[`building_block.is_integer()`](man/is_dtype.md)
[`building_block.is_float()`](man/is_dtype.md)

### 数字のフォーマット

[`building_block.style_number()`](man/miscellaneous.md)
[`building_block.style_currency()`](man/miscellaneous.md)
[`building_block.style_percent()`](man/miscellaneous.md)

[`building_block.style_pvalue()`](man/style_pvalue.md)
[`building_block.p_stars()`](man/style_pvalue.md)

### 並列文の作成

[`building_block.oxford_comma()`](man/oxford_comma.md)
[`building_block.oxford_comma_and()`](man/oxford_comma.md)
[`building_block.oxford_comma_or()`](man/oxford_comma.md)

***
[Jump to **Function Get started**.](..//INTRODUCTION.md)  
[Jump to **Function reference**.](../reference.md)
