# Function reference

## `py4stats.eda_tools`

### データフレームの概要
[`eda_tools.diagnose()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/diagnose.md)

### クロス集計

[`eda_tools.tabyl()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/tabyl.md)

[`eda_tools.freq_table()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/freq_table.md)

[`eda_tools.Pareto_plot()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/Pareto_plot.md)

### データフレームの列や行の削除

[`eda_tools.remove_empty()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/remove_empty_constant.md)  [`eda_tools.remove_constant()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/remove_empty_constant.md)

[`eda_tools.filtering_out()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/filtering_out.md)

### 複数のデータフレームの比較

[`eda_tools.compare_df_cols()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/compare_df_cols.md)

### 論理関数

[`eda_tools.is_number()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/predicate_str.md)
[`eda_tools.is_ymd()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/predicate_str.md)
[`eda_tools.is_ymd_like()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/predicate_str.md)

[`eda_tools.is_dummy()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/is_dummy.md)

## `py4stats.regression_tools`

### 分析結果の比較

[`regression_tools.compare_ols()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/compare_ols.md)

[`regression_tools.compare_mfx()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/compare_mfx.md)

### 分析結果の可視化

[`regression_tools.coefplot()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/coefplot.md) [`regression_tools.mfxplot()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/compare_mfx.md)

### 分析結果を作表するためのバックエンド関数
[`regression_tools.tidy()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/tidy.md)[`regression_tools.tidy_mfx()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/tidy.md)

[`regression_tools.glance()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/glance.md)

## `py4stats.heckit_helper`

[`heckit_helper.Heckit_from_formula()`](https://github.com/Hirototensho/Py4Stats/edit/main/man/Heckit_from_formula.md)

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

### 数字のフォーマット

[`bilding_block.pad_zero()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/miscellaneous.md)  [`bilding_block.p_stars()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/miscellaneous.md)

### 並列文の作成

[`bilding_block.oxford_comma()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/oxford_comma.md)
[`bilding_block.oxford_comma_and()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/oxford_comma.md)
[`bilding_block.oxford_comma_or()`](https://github.com/Hirototensho/Py4Stats/blob/main/man/oxford_comma.md)
