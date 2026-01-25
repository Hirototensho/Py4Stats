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

# ================================================================
# mean_qi / median_qi / mean_ci (DataFrame)
# ================================================================

@pytest.mark.parametrize("backend", [('pd'), ('pl'), ('pa')])

def test_mean_qi(backend) -> None:
    
    path = f'{tests_path}/fixtures/mean_qi_{backend}.csv'
    
    output_df = eda_nw.mean_qi(penguins_dict.get(backend), to_native = False)
    
    _assert_df_eq(
            output_df, path_fixture = path, 
            update_fixture = False
        )

@pytest.mark.parametrize("backend", [('pd'), ('pl'), ('pa')])
def test_median_qi(backend) -> None:
    
    path = f'{tests_path}/fixtures/median_qi_{backend}.csv'
    
    output_df = eda_nw.median_qi(penguins_dict.get(backend), to_native = False)
    
    _assert_df_eq(
            output_df, path_fixture = path, 
            update_fixture = False
        )

@pytest.mark.parametrize("backend", [('pd'), ('pl'), ('pa')])
def test_mean_ci(backend) -> None:
    
    path = f'{tests_path}/fixtures/mean_ci_{backend}.csv'
    
    output_df = eda_nw.mean_ci(penguins_dict.get(backend), to_native = False)
    
    _assert_df_eq(
            output_df, path_fixture = path, 
            update_fixture = False
        )
    
# ================================================================
# mean_qi / median_qi / mean_ci (Series)
# ================================================================

@pytest.mark.parametrize("backend", [('pd'), ('pl'), ('pa')])

def test_mean_qi_series(backend) -> None:
    
    path = f'{tests_path}/fixtures/mean_qi_series_{backend}.csv'
    
    output_df = eda_nw.mean_qi(penguins_dict.get(backend)['body_mass_g'], to_native = False)
    
    _assert_df_eq(
            output_df, path_fixture = path, 
            update_fixture = False
        )

@pytest.mark.parametrize("backend", [('pd'), ('pl'), ('pa')])
def test_median_qi_series(backend) -> None:
    
    path = f'{tests_path}/fixtures/median_qi_series_{backend}.csv'
    
    output_df = eda_nw.median_qi(penguins_dict.get(backend)['body_mass_g'], to_native = False)
    
    _assert_df_eq(
            output_df, path_fixture = path, 
            update_fixture = False
        )

@pytest.mark.parametrize("backend", [('pd'), ('pl'), ('pa')])
def test_mean_ci_series(backend) -> None:
    
    path = f'{tests_path}/fixtures/mean_ci_series_{backend}.csv'
    
    output_df = eda_nw.mean_ci(penguins_dict.get(backend)['body_mass_g'], to_native = False)
    
    _assert_df_eq(
            output_df, path_fixture = path, 
            update_fixture = False
        )