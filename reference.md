# Function reference

## `py4stats.eda_tools`

### データフレームの概要
[`py4stats.diagnose()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/diagnose.md)

### クロス集計

[`py4stats.tabyl()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/tabyl.md)

[`py4stats.freq_table()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/freq_table.md)

[`py4stats.Pareto_plot()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/Pareto_plot.md)

### 数値変数の点推定と区間推定

[`py4stats.mean_qi()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/point_range.md)
[`py4stats.median_qi()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/point_range.md)
[`py4stats.mean_ci()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/point_range.md)

### データフレームの列や行の削除

[`py4stats.remove_empty()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/remove_empty_constant.md)  [`py4stats.remove_constant()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/remove_empty_constant.md)

[`py4stats.filtering_out()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/filtering_out.md)

### 複数のデータフレームの比較

[`py4stats.compare_df_cols()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/compare_df_cols.md)

### 簡易なグループ別統計量の比較

[`py4stats.compare_group_means()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/compare_group_stats.md)
[`py4stats.compare_group_median()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/compare_group_stats.md)

[`py4stats.plot_mean_diff()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/compare_group_stats.md)
[`py4stats.plot_median_diff()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/compare_group_stats.md)

### 論理関数

[`py4stats.is_number()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/predicate_str.md)
[`py4stats.is_ymd()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/predicate_str.md)
[`py4stats.is_ymd_like()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/predicate_str.md)

[`py4stats.is_dummy()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/is_dummy.md)

### 簡易なルールベースのデータ検証ツール

[`py4stats.check_that()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/varidate.md) [`py4stats.check_viorate()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/varidate.md)

## `py4stats.regression_tools`

### 分析結果の比較

[`py4stats.compare_ols()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/compare_ols.md)

[`py4stats.compare_mfx()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/compare_mfx.md)

### 分析結果の可視化

[`py4stats.coefplot()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/coefplot.md) [`py4stats.mfxplot()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/coefplot.md)

### 分析結果を作表するためのバックエンド関数
[`py4stats.tidy()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/tidy.md)[`py4stats.tidy_mfx()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/tidy.md)

[`py4stats.tidy_test()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/tidy_test.md)

[`py4stats.glance()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/glance.md)

### Blinder-Oaxaca分解

[`py4stats.Blinder_Oaxaca()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/Blinder_Oaxaca.md)
[`py4stats.plot_Blinder_Oaxaca()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/Blinder_Oaxaca.md)

## `py4stats.heckit_helper`

[`heckit_helper.Heckit_from_formula()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/Heckit_from_formula.md)

[`heckit_helper.tidy_heckit()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/tidy_heckit.md)

[`heckit_helper.heckitmfx_compute()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/heckitmfx_compute.md)

## `py4stats.bilding_block`

### 引数のアサーション関数

[`bilding_block.arg_match()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/arg_match.md)

[`bilding_block.assert_character()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/assert_dtype.md)
[`bilding_block.assert_logical()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/assert_dtype.md)
[`bilding_block.assert_numeric()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/assert_dtype.md)
[`bilding_block.assert_integer()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/assert_dtype.md)
[`bilding_block.assert_count()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/assert_dtype.md)
[`bilding_block.assert_float()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/assert_dtype.md)

### データ型を判定する論理関数

[`bilding_block.is_character()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/is_dtype.md)
[`bilding_block.is_logical()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/is_dtype.md)
[`bilding_block.is_numeric()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/is_dtype.md)
[`bilding_block.is_integer()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/is_dtype.md)
[`bilding_block.is_float()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/is_dtype.md)

### 数字のフォーマット

[`bilding_block.style_number()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/miscellaneous.md)
[`bilding_block.style_currency()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/miscellaneous.md)
[`bilding_block.style_percent()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/miscellaneous.md)

[`bilding_block.style_pvalue()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/style_pvalue.md)
[`bilding_block.p_stars()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/style_pvalue.md)

### 並列文の作成

[`bilding_block.oxford_comma()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/oxford_comma.md)
[`bilding_block.oxford_comma_and()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/oxford_comma.md)
[`bilding_block.oxford_comma_or()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/oxford_comma.md)
