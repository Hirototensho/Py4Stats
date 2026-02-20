# tests/test_import.py

import importlib
import pytest

# =========================================================
# モジュールのインポートテスト
# =========================================================

modules = [
    'py4stats',
    'py4stats.building_block',
    'py4stats.heckit_helper',
    'py4stats.eda_tools._nw',
    'py4stats.eda_tools._pandas',
]

@pytest.mark.parametrize('module_name', modules)
def test_import_modules(module_name):
    importlib.import_module(module_name)

# =========================================================
# 公開 API 関数群のインポートテスト
# ただ、このテスト
# =========================================================
PUBLIC_API = [
    # regression_tools ======================================
    'Blinder_Oaxaca', 
    'add_one_sided_p_value', 
    'coefplot', 
    'compare_mfx', 
    'compare_ols', 
    'glance', 
    'glance_glm', 
    'glance_ols', 
    'log_to_pct', 
    'mfxplot', 
    'overload', 
    'plot_Blinder_Oaxaca', 
    'tidy', 
    'tidy_mfx', 
    'tidy_to_jp',
    # eda_tools ===========================================
    'Max',
    'Mean',
    'Median',
    'Min',
    'Pareto_plot',
    'Sum',
    'bind_rows',
    'check_that',
    'check_viorate',
    'compare_df_cols',
    'compare_df_record',
    'compare_df_stats',
    'compare_group_means',
    'compare_group_median',
    'crosstab',
    'diagnose',
    'diagnose_category',
    'filtering_out',
    'freq_table',
    'group_split', 
    'group_map', 
    'group_modify',
    'info_gain',
    'implies_exper',
    'is_dummy',
    'is_number',
    'is_ymd_like',
    'is_ymd',
    'mean_ci',
    'mean_qi',
    'median_qi',
    'min_max',
    'plot_category',
    'plot_mean_diff',
    'plot_median_diff',
    'plot_miss_var',
    'scale',
    'set_miss',
    'relocate',
    'remove_constant',
    'remove_empty',
    'review_wrangling',
    'review_shape',
    'review_col_addition',
    'review_casting',
    'review_missing',
    'review_category',
    'review_numeric',
    'tabyl',
    'weighted_mean',
 ]

@pytest.mark.parametrize('name', PUBLIC_API)
def test_public_api_import(name):
    module = importlib.import_module('py4stats')
    getattr(module, name)

# =========================================================
# building_block API のインポートテスト
# =========================================================

function_list = [
    'add_big_mark',
    'annotations',
    'arg_match',
    'arg_match0',
    'assert_character',
    'assert_count',
    'assert_float',
    'assert_function',
    'assert_integer',
    'assert_length',
    'assert_literal',
    'assert_literal_kyes',
    'assert_logical',
    'assert_missing',
    'assert_numeric',
    'assert_numeric_dtype',
    'assert_same_type',
    'assert_scalar',
    'assert_value_range',
    'collections',
    'is_character',
    'is_float',
    'is_function',
    'is_integer',
    'is_logical',
    'is_missing',
    'is_numeric',
    'is_pl_null',
    'length',
    'list_diff',
    'list_dropnulls',
    'list_dupricated',
    'list_flatten',
    'list_intersect',
    'list_replace',
    'list_subset',
    'list_union',
    'list_unique',
    'list_xor',
    'make_assert_numeric',
    'make_assert_type',
    'make_range_message',
    'match_arg',
    'oxford_comma',
    'oxford_comma_and',
    'oxford_comma_or',
    'oxford_comma_shorten',
    'p_stars',
    'pad_zero',
    'style_currency',
    'style_number',
    'style_percent',
    'style_pvalue',
    'which'
 ]

@pytest.mark.parametrize('name', function_list)
def test_building_block_import(name):
    module = importlib.import_module('py4stats.building_block')
    getattr(module, name)