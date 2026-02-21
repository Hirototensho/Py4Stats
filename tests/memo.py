# tests/memo.py
import pytest
import pandas as pd
import polars as pl

from py4stats import building_block as build
from contextlib import nullcontext

# tests/test_eda_tools.py
import pytest
import pandas as pd
import numpy as np
import wooldridge
# from palmerpenguins import load_penguins
import matplotlib.pyplot as plt
from pandas.testing import assert_frame_equal
import polars as pl
import pyarrow as pa

import narwhals
import narwhals as nw
import narwhals.selectors as ncs

from typing import (Literal)

from py4stats.eda_tools import _nw as eda_nw


# サンプルデータの読み込み --------------------------------
import pathlib
tests_path = pathlib.Path(__file__).parent


# penguins = load_penguins() 
# penguins.to_csv(f'{tests_path}/fixtures/penguins.csv', index = False)
penguins = pd.read_csv(f'{tests_path}/fixtures/penguins.csv')

adelie = penguins.query("species == 'Adelie'")
gentoo = penguins.query("species == 'Gentoo'")

penguins_pa = pa.Table.from_pandas(penguins)
penguins_pl = pl.from_arrow(penguins_pa)

adelie_pl = pl.from_pandas(adelie)
adelie_pa = pa.Table.from_pandas(adelie)

gentoo_pl = pl.from_pandas(gentoo)
gentoo_pa = pa.Table.from_pandas(gentoo)

mroz = wooldridge.data('mroz')
mroz_pl = pl.from_pandas(mroz)
mroz_pa = pa.Table.from_pandas(mroz)

def _assert_df_fixture(output_df, fixture_csv: str, check_dtype: bool = False, index_col = 0, **kwarg) -> None:
    if hasattr(output_df, 'to_pandas'):
        output_df = output_df.to_pandas()
    expected_df = pd.read_csv(f'{tests_path}/fixtures/{fixture_csv}', index_col = index_col, **kwarg)
    # バックエンド差で dtype が微妙に変わりやすいので、基本は dtype を厳密に見ない運用が安定
    assert_frame_equal(output_df, expected_df, check_dtype=check_dtype)

def _assert_df_fixture_new(
        output_df,  fixture_csv: str,  
        check_dtype: bool = False, reset_index: bool = True, 
        **kwarg
        ) -> None:
    expected_df = nw.read_csv(
        f'{tests_path}/fixtures/{fixture_csv}',
        backend = output_df.implementation
        )
    
    output_df = output_df.to_pandas()
    expected_df = expected_df.to_pandas()

    if reset_index:
        output_df = output_df.reset_index(drop = True)
        expected_df = expected_df.reset_index(drop = True)

    assert_frame_equal(output_df, expected_df, check_dtype = check_dtype)

def _assert_df_eq(
        output_df,  path_fixture: str,  
        check_dtype: bool = False, 
        reset_index: bool = True, 
        update_fixture: bool = False,
        **kwarg
        ) -> None:
    
    if not isinstance(output_df, nw.DataFrame):
        output_df = nw.from_native(output_df)

    if update_fixture:
        output_df.write_csv(path_fixture)

    expected_df = nw.read_csv(path_fixture, backend = output_df.implementation)
    
    if hasattr(expected_df, 'to_pandas'): expected_df = expected_df.to_pandas()
    if hasattr(output_df, 'to_pandas'): output_df = output_df.to_pandas()

    if reset_index:
        output_df = output_df.reset_index(drop = True)
        expected_df = expected_df.reset_index(drop = True)

    assert_frame_equal(output_df, expected_df, check_dtype = check_dtype)

penguins_dict = {
    'pd':penguins,
    'pl':penguins_pl,
    'pa':penguins_pa
}

gentoo_dict = {
    'pd':gentoo,
    'pl':gentoo_pl,
    'pa':gentoo_pa
}

adelie_dict = {
    'pd':adelie,
    'pl':adelie_pl,
    'pa':adelie_pa
}

mroz = wooldridge.data('mroz')

mroz_dict = {
    'pd':mroz,
    'pl':pl.from_pandas(mroz),
    'pa':pa.Table.from_pandas(mroz)
}

list_backend = ['pd', 'pl', 'pa']

# =========================================================
# テスト用関数の定義
# =========================================================
# =========================================================
# assert_literal
# =========================================================

import importlib
import pytest

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