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
    # 'py4stats.eda_tools._not',
]

@pytest.mark.parametrize('module_name', modules)
def test_import_modules(module_name):
    importlib.import_module(module_name)

# =========================================================
# 公開 API 関数群のインポートテスト
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