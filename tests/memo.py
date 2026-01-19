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
# ================================================================
# plot_mean_diff / plot_median_diff
# ================================================================

expect_is_number = [
        True, True, True, True, False, False, 
        False, False, False, False, False, False,
        False, False, True, True
    ]

expect_is_ymd = [
    False, False, False, False, False, False, True, False, 
    False, False, False, False, False, False, True, True]

expect_is_ymd_like = [
    False, False, False, False, False, False, True, True, 
    True, True, False, False, False, False, True, True]

expect_is_kanzi = [
    False, False, False, False, True, False, False, True, 
    True, True, False, False, False, True, True, True]

s_pd = pd.Series([
    '123', "0.12", "1e+07", '-31', '2個', '1A',
    "2024-03-03", "2024年3月3日", "24年3月3日", '令和6年3月3日',
    '0120-123-456', '15ｹ', "apple", "不明", None, np.nan
    ])
s_pl = pl.from_pandas(s_pd)
s_pa = pa.Table.from_pandas(s_pd.to_frame(name = 'string'))['string']

@pytest.mark.parametrize(
    "func, expect",
    [
        (eda_nw.is_kanzi, expect_is_kanzi),
        (eda_nw.is_ymd, expect_is_ymd),
        (eda_nw.is_ymd_like, expect_is_ymd_like),
        (eda_nw.is_number, expect_is_number),
    ],
)
def test_predicate_str_pd(func, expect) -> None:
    res = func(s_pd).to_list()
    assert (res == expect)

@pytest.mark.parametrize(
    "func, expect",
    [
        (eda_nw.is_kanzi, expect_is_kanzi),
        (eda_nw.is_ymd, expect_is_ymd),
        (eda_nw.is_ymd_like, expect_is_ymd_like),
        (eda_nw.is_number, expect_is_number),
    ],
)
def test_predicate_str_pl(func, expect) -> None:
    res = func(s_pl).to_list()
    assert (res == expect)

@pytest.mark.parametrize(
    "func, expect",
    [
        (eda_nw.is_kanzi, expect_is_kanzi),
        (eda_nw.is_ymd, expect_is_ymd),
        (eda_nw.is_ymd_like, expect_is_ymd_like),
        # (eda_nw.is_number, expect_is_number),
    ],
)
def test_predicate_str_pa(func, expect) -> None:
    res = func(s_pa).to_pylist()
    assert (res == expect)
