
# tests/test_eda_tools.py

# ↓ テストコード実装の補助関数の読み込み
import setup_test_function as tfun

from py4stats.eda_tools import _utils as eda_util
from py4stats.eda_tools import operation as eda_ops
from py4stats.eda_tools import reviewing as eda_review
from py4stats.eda_tools import visualize as eda_vis
from py4stats import building_block as build # py4stats のプログラミングを補助する関数群

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

import pathlib
from itertools import product
from contextlib import nullcontext

tests_path = pathlib.Path(__file__).parent.parent

# サンプルデータの読み込み --------------------------------

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

mroz_dict = {
    'pd':mroz,
    'pl':pl.from_pandas(mroz),
    'pa':pa.Table.from_pandas(mroz)
}

penguins_dict = {
    'pd':penguins,
    'pl':penguins_pl,
    'pa':penguins_pa
}

adelie_dict = {
    'pd':adelie,
    'pl':adelie_pl,
    'pa':adelie_pa
}

gentoo_dict = {
    'pd':gentoo,
    'pl':gentoo_pl,
    'pa':gentoo_pa
}

list_backend = ['pd', 'pl', 'pa']

# =========================================================
# diagnose
# =========================================================

@pytest.mark.parametrize("backend", list_backend)
def test_diagnose(backend) -> None:
    path = f'{tests_path}/fixtures/diagnose_{backend}.csv'
    
    output_df = eda_ops.diagnose(penguins_dict.get(backend), to_native = False)
    
    tfun._assert_df_eq(
        output_df, 
        path_fixture = path, 
        update_fixture = False
        )
# =========================================================
# freq_tabl/ crosstab
# =========================================================

@pytest.mark.parametrize("backend", list_backend)

def test_freq_table(backend) -> None:
    path = f'{tests_path}/fixtures/freq_table_{backend}.csv'
    
    output_df = eda_ops.freq_table(
        penguins_dict.get(backend), 
        'species', to_native = False
        )
    
    tfun._assert_df_eq(
        output_df, 
        path_fixture = path, 
        update_fixture = False
        )

# =========================================================
# crosstab
# =========================================================

def test_crosstab_pd():
    output_df1 = eda_ops.crosstab(penguins, 'island', 'species', margins = True, normalize = 'all')
    # output_df1.to_csv(f'{tests_path}/fixtures/crosstab_nw1.csv')
    expected_df1 = pd.read_csv(f'{tests_path}/fixtures/crosstab_nw1.csv', index_col = 0)
    test_result1 = eda_ops.compare_df_record(output_df1, expected_df1).all().all()

    output_df2 = eda_ops.crosstab(penguins, 'island', 'species', margins = True, normalize = 'columns')
    # output_df2.to_csv(f'{tests_path}/fixtures/crosstab_nw2.csv')
    expected_df2 = pd.read_csv(f'{tests_path}/fixtures/crosstab_nw2.csv', index_col = 0)
    test_result2 = eda_ops.compare_df_record(output_df2, expected_df2).all().all()

    output_df3 = eda_ops.crosstab(penguins, 'island', 'species', margins = True, normalize = 'index')
    # output_df3.to_csv(f'{tests_path}/fixtures/crosstab_nw3.csv')
    expected_df3 = pd.read_csv(f'{tests_path}/fixtures/crosstab_nw3.csv', index_col = 0)
    test_result3 = eda_ops.compare_df_record(output_df3, expected_df3).all().all()

    assert test_result1 and test_result2 and test_result3


def test_crosstab_pl():
    output_df1 = eda_ops.crosstab(penguins_pl, 'island', 'species', margins = True, normalize = 'all')
    output_df1 = output_df1.to_pandas()
    # output_df1.to_csv(f'{tests_path}/fixtures/crosstab_pl1.csv')
    expected_df1 = pd.read_csv(f'{tests_path}/fixtures/crosstab_pl1.csv', index_col = 0)
    test_result1 = eda_ops.compare_df_record(output_df1, expected_df1).all().all()

    output_df2 = eda_ops.crosstab(penguins_pl, 'island', 'species', margins = True, normalize = 'columns')
    output_df2 = output_df2.to_pandas()
    # output_df2.to_csv(f'{tests_path}/fixtures/crosstab_pl2.csv')
    expected_df2 = pd.read_csv(f'{tests_path}/fixtures/crosstab_pl2.csv', index_col = 0)
    test_result2 = eda_ops.compare_df_record(output_df2, expected_df2).all().all()

    output_df3 = eda_ops.crosstab(penguins_pl, 'island', 'species', margins = True, normalize = 'index')
    output_df3 = output_df3.to_pandas()
    # output_df3.to_csv(f'{tests_path}/fixtures/crosstab_pl3.csv')
    expected_df3 = pd.read_csv(f'{tests_path}/fixtures/crosstab_pl3.csv', index_col = 0)
    test_result3 = eda_ops.compare_df_record(output_df3, expected_df3).all().all()

    assert test_result1 and test_result2 and test_result3

def test_crosstab_pa():
    output_df1 = eda_ops.crosstab(penguins_pa, 'island', 'species', margins = True, normalize = 'all')
    output_df1 = output_df1.to_pandas()
    # output_df1.to_csv(f'{tests_path}/fixtures/crosstab_pa1.csv')
    expected_df1 = pd.read_csv(f'{tests_path}/fixtures/crosstab_pa1.csv', index_col = 0)
    test_result1 = eda_ops.compare_df_record(output_df1, expected_df1).all().all()

    output_df2 = eda_ops.crosstab(penguins_pa, 'island', 'species', margins = True, normalize = 'columns')
    output_df2 = output_df2.to_pandas()
    # output_df2.to_csv(f'{tests_path}/fixtures/crosstab_pa2.csv')
    expected_df2 = pd.read_csv(f'{tests_path}/fixtures/crosstab_pa2.csv', index_col = 0)
    test_result2 = eda_ops.compare_df_record(output_df2, expected_df2).all().all()

    output_df3 = eda_ops.crosstab(penguins_pa, 'island', 'species', margins = True, normalize = 'index')
    output_df3 = output_df3.to_pandas()
    # output_df3.to_csv(f'{tests_path}/fixtures/crosstab_pa3.csv')
    expected_df3 = pd.read_csv(f'{tests_path}/fixtures/crosstab_pa3.csv', index_col = 0)
    test_result3 = eda_ops.compare_df_record(output_df3, expected_df3).all().all()

    assert test_result1 and test_result2 and test_result3

# =========================================================
# tabyl
# =========================================================

def test_tabyl_pd():
    output_df = eda_ops.tabyl(penguins, 'island', 'species', normalize = 'index')
    # output_df.to_csv(f'{tests_path}/fixtures/tabyl_nw.csv', index = True)
    tfun._assert_df_record(output_df, 'tabyl_nw.csv', dtype = {'All':str})


def test_tabyl_pl():
    output_df = eda_ops.tabyl(
            penguins_pl, 'island', 'species', 
            normalize = 'index', to_native = False
        ).to_pandas()
    # output_df.to_csv(f'{tests_path}/fixtures/tabyl_pl.csv', index = True)
    tfun._assert_df_record(output_df, 'tabyl_pl.csv', dtype = {'All':str})

def test_tabyl_pa():
    output_df = eda_ops.tabyl(
        penguins_pa, 'island', 'species', 
        normalize = 'index', to_native = False
    ).to_pandas()
    # output_df.to_csv(f'{tests_path}/fixtures/tabyl_pa.csv', index = True)
    tfun._assert_df_record(output_df, 'tabyl_pa.csv', dtype = {'All':str})

def test_tabyl_with_boolen_col_pd():
    pm2 = penguins.copy()
    pm2['heavy'] = pm2['body_mass_g'] >= pm2['body_mass_g'].quantile(0.50)

    output_df = eda_ops.tabyl(pm2, 'heavy', 'species', normalize = 'columns')
    # output_df.to_csv(f'{tests_path}/fixtures/tabyl_nw_with_boolen.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/tabyl_nw_with_boolen.csv', index_col = 0)

    test_result = eda_ops.compare_df_record(
        output_df.astype(str), 
        expected_df.astype(str)
        )\
            .all().all()
    assert test_result

# =========================================================
# compare_df_cols
# =========================================================
@pytest.mark.parametrize("backend", list_backend)
def test_compare_df_cols(backend):

    output_df = eda_ops.compare_df_cols(
        [adelie_dict.get(backend), gentoo_dict.get(backend)],
        return_match = 'match',
        to_native = False
    )
    path = f'{tests_path}/fixtures/compare_df_cols_{backend}.csv'
    tfun._assert_df_eq(output_df, path, update_fixture = False)

# =========================================================
# compare_df_stats
# =========================================================

def test_compare_df_stats_pd():
    penguins_modify = penguins.copy()
    penguins_modify['body_mass_g'] = penguins_modify['body_mass_g'] / 1000
    penguins_modify['bill_length_mm'] = eda_ops.scale(penguins_modify['bill_length_mm'])

    output_df = eda_ops.compare_df_stats(
        [penguins, penguins_modify]
        )

    # output_df.to_csv(f'{tests_path}/fixtures/compare_df_stats_nw.csv')
    tfun._assert_df_fixture(output_df, 'compare_df_stats_nw.csv')

def test_compare_df_stats_pl():
    penguins_modify = penguins.copy()
    penguins_modify['body_mass_g'] = penguins_modify['body_mass_g'] / 1000
    penguins_modify['bill_length_mm'] = eda_ops.scale(penguins_modify['bill_length_mm'])

    output_df = eda_ops.compare_df_stats(
        [penguins_pl, pl.from_pandas(penguins_modify)],
        to_native = False
        ).to_pandas()

    # output_df.to_csv(f'{tests_path}/fixtures/compare_df_stats_pl.csv')
    tfun._assert_df_fixture(output_df, 'compare_df_stats_pl.csv')

def test_compare_df_stats_pa():
    penguins_modify = penguins.copy()
    penguins_modify['body_mass_g'] = penguins_modify['body_mass_g'] / 1000
    penguins_modify['bill_length_mm'] = eda_ops.scale(penguins_modify['bill_length_mm'])

    output_df = eda_ops.compare_df_stats(
        [penguins_pa, pa.Table.from_pandas(penguins_modify)],
        to_native = False 
        ).to_pandas()

    # output_df.to_csv(f'{tests_path}/fixtures/compare_df_stats_pa.csv')
    tfun._assert_df_fixture(output_df, 'compare_df_stats_pa.csv')

# =========================================================
# compare_df_record
# =========================================================
def test_compare_df_record_pd():
    penguins_copy = penguins.copy()
    penguins_copy.loc[200:250, 'flipper_length_mm'] = \
        2 * penguins_copy.loc[200:250, 'flipper_length_mm'] 

    output_df = eda_ops.compare_df_record(penguins, penguins_copy)

    # output_df.to_csv(f'{tests_path}/fixtures/compare_df_record_nw.csv')
    tfun._assert_df_fixture(output_df, 'compare_df_record_nw.csv')

def test_compare_df_record_pl():
    penguins_copy = penguins.copy()
    penguins_copy.loc[200:250, 'flipper_length_mm'] = \
        2 * penguins_copy.loc[200:250, 'flipper_length_mm'] 

    output_df = eda_ops.compare_df_record(
        penguins_pl, pl.from_pandas(penguins_copy),
        to_native = False
        )
    # output_df.write_csv(f'{tests_path}/fixtures/compare_df_record_pl.csv')
    tfun._assert_df_fixture_new(output_df, 'compare_df_record_pl.csv')

def test_compare_df_record_pa():
    penguins_copy = penguins.copy()
    penguins_copy.loc[200:250, 'flipper_length_mm'] = \
        2 * penguins_copy.loc[200:250, 'flipper_length_mm'] 

    output_df = eda_ops.compare_df_record(
        penguins_pa, pa.Table.from_pandas(penguins_copy),
        to_native = False
        )

    # output_df.write_csv(f'{tests_path}/fixtures/compare_df_record_pa.csv')
    tfun._assert_df_fixture_new(output_df, 'compare_df_record_pa.csv')

# =========================================================
# remove_empty
# =========================================================
penguins_empty = penguins.copy()
penguins_empty['na_col'] = pd.NA
penguins_empty['None'] = None

empty_dict = {
    'pd': penguins_empty,
    'pl': pl.from_pandas(penguins_empty),
    'pa': pa.Table.from_pandas(penguins_empty)
}

@pytest.mark.parametrize("backend", list_backend)
def test_remove_empty(backend):
    columns = eda_ops.remove_empty(
        empty_dict.get(backend), 
        to_native = False).columns
    result = all([col not in ['na_col', 'None'] for col in columns])
    
    assert result

# =========================================================
# remove_constant
# =========================================================

list_backend = ['pd', 'pl', 'pa']
penguins_constant = penguins.copy()
penguins_constant['one'] = 1
penguins_constant['two'] = 2

const_dict = {
    'pd': penguins_constant,
    'pl': pl.from_pandas(penguins_constant),
    'pa': pa.Table.from_pandas(penguins_constant)
}

@pytest.mark.parametrize("backend", list_backend)
def test_remove_constant(backend):
    columns = eda_ops.remove_constant(
        const_dict.get(backend), 
        to_native = False).columns
    result = all([col not in ['one', 'tow'] for col in columns])
    assert result

# =========================================================
# filtering_out
# =========================================================
def test_filtering_out_columns_pd() -> None:
    df = pd.DataFrame({"foo_x": [1], "foo_y": [2], "bar": [3]})
    out = eda_ops.filtering_out(df, contains="foo", axis="columns")
    assert list(out.columns) == ["bar"]

def test_filtering_out_index_pd() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]}, index=["keep", "drop_me", "drop_you"])
    out = eda_ops.filtering_out(df, starts_with="drop", axis="index")
    assert list(out.index) == ["keep"]

    df = pd.DataFrame({
        'x':range(8)
        },index = mroz.columns[:8]
    )

    res = eda_ops.filtering_out(
        df, ['inlf'], 'hours',
        starts_with = 'kid',
        ends_with = 'wage',
        axis = 'index'
        )
    assert res.index.to_list() == ['age', 'educ']


def test_filtering_out_columns_pl() -> None:
    df = pd.DataFrame({"foo_x": [1], "foo_y": [2], "bar": [3]})
    df = pl.from_pandas(df)
    out = eda_ops.filtering_out(df, contains="foo", axis="columns")
    assert list(out.columns) == ["bar"]

def test_filtering_out_columns_pa() -> None:
    df = pd.DataFrame({"foo_x": [1], "foo_y": [2], "bar": [3]})
    df = pa.Table.from_pandas(df)
    out = eda_ops.filtering_out(df, contains="foo", axis="columns")
    assert list(out.to_pandas().columns) == ["bar"]

@pytest.mark.parametrize("backend", list_backend)
def test_filtering_out_cols(backend) -> None:
    path = f'{tests_path}/fixtures/filtering_out_{backend}.csv'

    output_df = eda_ops.filtering_out(
        penguins_dict.get(backend), 'year', starts_with = 'bill', 
        contains = 'is', ends_with = '_g', to_native = False
    ).drop_nulls().head()
    
    tfun._assert_df_eq(
            output_df, path_fixture = path, 
            update_fixture = False
        )

# =========================================================
# is_dummy_nw (Series/DataFrame)
# =========================================================

@pytest.mark.parametrize("backend", list_backend)
def test_is_dummy_series(backend) -> None:
    assert eda_ops.is_dummy(mroz_dict.get(backend)['inlf']) 

    assert not eda_ops.is_dummy(mroz_dict.get(backend)['educ'])
    
    assert eda_ops.is_dummy(
        mroz_dict.get(backend)['kidslt6'],
        cording = [0, 1, 2, 3]
    )

@pytest.mark.parametrize("backend", list_backend)
def test_is_dummy_nw_dataframe(backend) -> None:
    result = eda_ops.is_dummy(mroz_dict.get(backend), to_native = False)

    expected = [
        True, False, False, False, False, False, False, False, False, 
        False, False, False, False, False, False, False, False, True, 
        False, False, False, False
    ]
    
    assert list(result) == expected 

def test_is_dummy_list():
    assert eda_ops.is_dummy([0, 1, 1, 0])
    assert not eda_ops.is_dummy([0, 1, 1, 2])
    assert eda_ops.is_dummy([1, 2, 1, 2], cording = (1, 2))

# =========================================================
# diagnose_category
# =========================================================

pm2 = penguins.copy()

pm2['species'] = pd.Categorical(pm2['species'])

pm2 = pd.get_dummies(pm2,  columns = ['sex'])

pm2['heavy'] = np.where(
    pm2['body_mass_g'] >= pm2['body_mass_g'].quantile(0.75), 
    1, 0
)

pm2_dict = {
    'pd': pm2,
    'pl': pl.from_pandas(pm2),
    'pa': pa.Table.from_pandas(pm2)
}
@pytest.mark.parametrize("backend", list_backend)
def test_diagnose_category_pd(backend):
    path = f'{tests_path}/fixtures/diagnose_category_{backend}.csv'
    output_df = eda_ops.diagnose_category(pm2_dict.get(backend), to_native = False)
    
    tfun._assert_df_eq(
        output_df, 
        path_fixture = path, 
        update_fixture = False
        )
    
    # ダミー変数がなくても動作することも確認しておきます
    eda_ops.diagnose_category(penguins_dict.get(backend)) 


# ================================================================
# compare_group_means / compare_group_median
# ================================================================
@pytest.mark.parametrize("backend", list_backend)

def test_compare_group_means(backend) -> None:
    path = f'{tests_path}/fixtures/compare_group_means_{backend}.csv'

    output_df = eda_ops.compare_group_means(
        adelie_dict.get(backend), 
        gentoo_dict.get(backend), 
        to_native = False
        ) # -> pd.DataFrame
    
    tfun._assert_df_eq(
            output_df, path_fixture = path, 
            update_fixture = False
        )

@pytest.mark.parametrize("backend", list_backend)
def test_compare_group_median(backend) -> None:
    path = f'{tests_path}/fixtures/compare_group_median_{backend}.csv'

    output_df = eda_ops.compare_group_median(
        adelie_dict.get(backend), 
        gentoo_dict.get(backend), 
        to_native = False
        ) # -> pd.DataFrame
    
    tfun._assert_df_eq(
            output_df, path_fixture = path, 
            update_fixture = False
        )
    


# ================================================================
# mean_qi / median_qi / mean_ci (DataFrame)
# ================================================================

@pytest.mark.parametrize("backend", list_backend)

def test_mean_qi(backend) -> None:
    
    path = f'{tests_path}/fixtures/mean_qi_{backend}.csv'
    
    output_df = eda_ops.mean_qi(penguins_dict.get(backend), to_native = False)
    
    tfun._assert_df_eq(
            output_df, path_fixture = path, 
            update_fixture = False
        )

@pytest.mark.parametrize("backend", list_backend)
def test_median_qi(backend) -> None:
    
    path = f'{tests_path}/fixtures/median_qi_{backend}.csv'
    
    output_df = eda_ops.median_qi(penguins_dict.get(backend), to_native = False)
    
    tfun._assert_df_eq(
            output_df, path_fixture = path, 
            update_fixture = False
        )

@pytest.mark.parametrize("backend", list_backend)
def test_mean_ci(backend) -> None:
    
    path = f'{tests_path}/fixtures/mean_ci_{backend}.csv'
    
    output_df = eda_ops.mean_ci(penguins_dict.get(backend), to_native = False)
    
    tfun._assert_df_eq(
            output_df, path_fixture = path, 
            update_fixture = False
        )
    
# ================================================================
# mean_qi / median_qi / mean_ci (Series)
# ================================================================

@pytest.mark.parametrize("backend", list_backend)

def test_mean_qi_series(backend) -> None:
    
    path = f'{tests_path}/fixtures/mean_qi_series_{backend}.csv'
    
    output_df = eda_ops.mean_qi(penguins_dict.get(backend)['body_mass_g'], to_native = False)
    
    tfun._assert_df_eq(
            output_df, path_fixture = path, 
            update_fixture = False
        )

@pytest.mark.parametrize("backend", list_backend)
def test_median_qi_series(backend) -> None:
    
    path = f'{tests_path}/fixtures/median_qi_series_{backend}.csv'
    
    output_df = eda_ops.median_qi(penguins_dict.get(backend)['body_mass_g'], to_native = False)
    
    tfun._assert_df_eq(
            output_df, path_fixture = path, 
            update_fixture = False
        )

@pytest.mark.parametrize("backend", list_backend)
def test_mean_ci_series(backend) -> None:
    
    path = f'{tests_path}/fixtures/mean_ci_series_{backend}.csv'
    
    output_df = eda_ops.mean_ci(penguins_dict.get(backend)['body_mass_g'], to_native = False)
    
    tfun._assert_df_eq(
            output_df, path_fixture = path, 
            update_fixture = False
        )


# =======================================================================
# string/regex helpers: is_number / is_ymd / is_ymd_like
# =======================================================================

def test_is_ymd_and_like_pd() -> None:
    s = pd.Series(["2025-12-30", "2025-1-2", "abc", None])
    out = eda_ops.is_ymd(s, na_default=True)
    assert list(out[:3]) == [True, True, False]
    assert out.iloc[3]

    s2 = pd.Series(["2025年12月30日", "2025-12-30", "nope"])
    out2 = eda_ops.is_ymd_like(s2)
    assert list(out2) == [True, True, False]

def test_is_number_nw_basic_pd() -> None:
    s = pd.Series(["123", "12E+3", "abc", "2025-12-30", None])
    out = eda_ops.is_number(s, na_default=False)
    assert out.iloc[0]
    assert out.iloc[1]
    assert not out.iloc[2]
    # 日付っぽいのは除外される想定
    assert not out.iloc[3]
    assert not out.iloc[4]

## テストの準備 ----------------------
expect_is_number = [
        True, True, True, True, False, False, False, False, 
        False, False, False, False, False, False, True, True]
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

## Pandas ----------------------
@pytest.mark.parametrize(
    "func, expect",
    [
        (eda_ops.is_kanzi, expect_is_kanzi),
        (eda_ops.is_ymd, expect_is_ymd),
        (eda_ops.is_ymd_like, expect_is_ymd_like),
        (eda_ops.is_number, expect_is_number),
    ],
)
def test_predicate_str_pd(func, expect) -> None:
    res = func(s_pd).to_list()
    assert (res == expect)

## pl ----------------------
@pytest.mark.parametrize(
    "func, expect",
    [
        (eda_ops.is_kanzi, expect_is_kanzi),
        (eda_ops.is_ymd, expect_is_ymd),
        (eda_ops.is_ymd_like, expect_is_ymd_like),
        (eda_ops.is_number, expect_is_number),
    ],
)
def test_predicate_str_pl(func, expect) -> None:
    res = func(s_pl).to_list()
    assert (res == expect)

## pa ----------------------
@pytest.mark.parametrize(
    "func, expect",
    [
        (eda_ops.is_kanzi, expect_is_kanzi),
        (eda_ops.is_ymd, expect_is_ymd),
        (eda_ops.is_ymd_like, expect_is_ymd_like),
        (eda_ops.is_number, expect_is_number),
    ],
)
def test_predicate_str_pa(func, expect) -> None:
    res = func(s_pa).to_pylist()
    assert (res == expect)

# ================================================================
# relocate
# ================================================================

def test_relocate_basic():
    result1 = eda_ops.relocate(penguins, 'year', 'sex').columns.to_list()
    expect1 = ['year', 'sex', 'species', 'island', 'bill_length_mm', 
            'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    assert result1 == expect1

def test_relocate_ncs():
    result2 = eda_ops.relocate(penguins_pl, ncs.numeric()).columns
    expect2 = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g',
            'year', 'species', 'island', 'sex']
    assert result2 == expect2

def test_relocate_before():
    result3 = eda_ops.relocate(
        penguins_pa, 'year', before = 'island', to_native = False
        ).columns
    expect3 = ['species', 'year', 'island', 'bill_length_mm', 'bill_depth_mm',
                'flipper_length_mm', 'body_mass_g', 'sex']
    assert result3 == expect3

def test_relocate_after():    
    result4 = eda_ops.relocate(penguins, 'year', after = 'island').columns.to_list()
    expect4 = ['species', 'island', 'year', 'bill_length_mm', 'bill_depth_mm',
                'flipper_length_mm', 'body_mass_g', 'sex']

    assert result4 == expect4

def test_relocate_error_on_invalid_selector():
    with pytest.raises(ValueError) as excinfo:
        eda_ops.relocate(penguins, 0, True)
    # 仕様：候補があると "Did you mean ..." を含む
    assert "Argument `*args` must be of type" in str(excinfo.value)
    assert "'0' and 'True' cannot be accepted" in str(excinfo.value)

@pytest.mark.parametrize(
    "arg, before, after, place, expectation",
    [
        pytest.param(
            'year', 'year', None, None,
            pytest.raises(ValueError, match = "`before` cannot be the same as"),
            id = 'arg_eq_before'
            ),
        pytest.param(
            'year', None, 'year', None,
            pytest.raises(ValueError, match = "`after` cannot be the same as"),
            id = 'arg_eq_after'
            ),
        pytest.param(
            'year', 'flipper_length_mm', 'body_mass_g', None,
            pytest.raises(ValueError, match = "`before` or `after`, not both."),
            id = 'arg_place_after'
            ),
        pytest.param(
            'year', None, 'body_mass_g', 'last',
            pytest.raises(ValueError, match = "`place` or `before`/`after`, not both"),
            id = 'arg_place_after'
            ),
    ],
)
def test_relocate_error_on_duplicated_args(arg, before, after, place, expectation):
    with expectation:
        eda_ops.relocate(
            penguins, arg, 
            before = before, 
            after = after,
            place = place
            )

# ================================================================
# weighted_mean
# ================================================================
x = penguins.groupby('species')['bill_length_mm'].mean()
w = penguins.groupby('species')['bill_length_mm'].count()
grand_mean = penguins['bill_length_mm'].mean()


def test_weighted_mean_pd():
    assert np.isclose(eda_ops.weighted_mean(x, w), grand_mean)

def test_weighted_mean_pl():
    x_pl = pl.from_pandas(x)
    w_pl = pl.from_pandas(w)
    assert np.isclose(eda_ops.weighted_mean(x_pl, w_pl), grand_mean)

def test_weighted_mean_pa():
    data_pa = pa.Table.from_pydict({
        'x':x.to_list(),
        'w':w.to_list()
    })

    assert np.isclose(eda_ops.weighted_mean(data_pa['x'], data_pa['w']), grand_mean)

# ================================================================
# scale
# ================================================================

def test_scale_pd():
    res = eda_ops.scale(penguins.select_dtypes('number'))
    assert all(np.isclose(res.mean(), 0) & np.isclose(res.std(), 1))

    res = eda_ops.scale(penguins['body_mass_g'])
    assert np.isclose(res.mean(), 0) & np.isclose(res.std(), 1)

def test_scale_pl():
    res = eda_ops.scale(penguins_pl['body_mass_g']).to_pandas()
    assert np.isclose(res.mean(), 0) & np.isclose(res.std(), 1)

def test_scale_pa():
    res = eda_ops.scale(penguins_pa['body_mass_g']).to_pandas()
    assert np.isclose(res.mean(), 0) & np.isclose(res.std(), 1)

# ================================================================
# min_max
# ================================================================

def test_min_max_pd():
    res = eda_ops.min_max(penguins.select_dtypes('number'))
    assert all(np.isclose(res.min(), 0) & np.isclose(res.max(), 1))

    res = eda_ops.min_max(penguins['body_mass_g'])
    assert np.isclose(res.min(), 0) & np.isclose(res.max(), 1)

def test_min_max_pl():
    res = eda_ops.min_max(penguins_pl['body_mass_g']).to_pandas()
    assert np.isclose(res.min(), 0) & np.isclose(res.max(), 1)

def test_min_max_pa():
    res = eda_ops.min_max(penguins_pa['body_mass_g']).to_pandas()
    assert np.isclose(res.min(), 0) & np.isclose(res.max(), 1)

# =========================================================
# set_miss
# =========================================================
@pytest.mark.parametrize("backend", list_backend)
def test_set_miss(backend):
    x = penguins_dict.get(backend)['body_mass_g']
    y = penguins_dict.get(backend)['bill_length_mm']
    miss_n = eda_ops.set_miss(x, n = 100, random_state = 123, to_native = False)
    miss_prop = eda_ops.set_miss(y, prop = 0.3, random_state = 123, to_native = False)

    assert build.is_missing(miss_n).sum() == 100
    assert np.isclose(build.is_missing(miss_prop).mean(), 0.3, atol = 0.001)

# ================================================================
# group_split, group_map, group_modify
# ================================================================
@pytest.mark.parametrize("backend", list_backend)
def test_group_split(backend):
    res = eda_ops.group_split(penguins_dict.get(backend), "species", "island")
    assert all([eda_util.is_intoframe(df) for df in res.data])
    assert sum([df.shape[0] for df in res.data]) == 344

@pytest.mark.parametrize("backend", list_backend)
def test_group_map(backend):
    res = eda_ops.group_map(
        penguins_dict.get(backend), 
        "species", "island", 
        func=lambda df: df.shape[0]
        )
    assert res.mapped == [44, 56, 52, 68, 124]

@pytest.mark.parametrize("backend", list_backend)
def test_group_modify(backend) -> None:
    path = f'{tests_path}/fixtures/group_modify_{backend}.csv'

    output_df = eda_ops.group_modify(
        penguins_dict.get(backend), 
        'species', 'sex',
        func = eda_ops.median_qi,
        to_native = False
    )
    
    tfun._assert_df_eq(
        output_df, 
        path_fixture = path, 
        update_fixture = False
        )

# ================================================================
# bind_rows
# ================================================================
from itertools import product

df1 = pl.DataFrame({'x':[1, 2], 'y':[2, 6], 'z':['a', 'c']})
df2 = pl.DataFrame({'x':[4], 'y':[3], 'z':['b']})

dict_table = {
    'pd': [df1.to_pandas(), df2.to_pandas()],
    'pl': [df1, df2],
    'pa': [df1.to_arrow(), df2.to_arrow()],
}

list_backend = ['pd', 'pl', 'pa']
test_type = ['A', 'B', 'C', 'D']

@pytest.mark.parametrize(
    "backend, test_type",
    list(product(list_backend, test_type))
)

def test_bind_rows_backend(backend, test_type) -> None:
    path = f'{tests_path}/fixtures/bind_rows_{backend}_{test_type}.csv'
    match test_type:
        case 'A': output_df = eda_ops.bind_rows(
            dict_table.get(backend)
            )
        case 'B': output_df = eda_ops.bind_rows(
            dict_table.get(backend),
            names = ['table1', 'table2'], 
            id = 'sauce'
            )
        case 'C': output_df = eda_ops.bind_rows(
            dict(zip(['table1', 'table2'], dict_table.get(backend))),
            id = 'sauce'
            )
        case 'D': output_df = eda_ops.bind_rows(
            dict_table.get(backend)[0],
            dict_table.get(backend),
            id = None
            )
    
    tfun._assert_df_eq(
            output_df, path_fixture = path, 
            update_fixture = False
        )
# ================================================================
# bind_rows (バックエンド混在に対するエラー)
# ================================================================
def test_bind_rows_mixed_backend():
    with pytest.raises(TypeError, match = "must share the same backend"):
        eda_ops.bind_rows(penguins_dict)
    
    with pytest.raises(TypeError, match = "must share the same backend"):
        eda_ops.bind_rows(list(mroz_dict.values()))

# ================================================================
# info_gain
# ================================================================

@pytest.mark.parametrize("backend", list_backend)
def test_info_gain(backend) -> None:
    path = f'{tests_path}/fixtures/info_gain_{backend}.csv'

    output_df = eda_ops.info_gain(
        penguins_dict.get(backend), 
        target = ['species', 'island'],
        features = ['island', 'sex', 'body_mass_g']
    )
    
    tfun._assert_df_eq(
        output_df, 
        path_fixture = path, 
        update_fixture = False
        )

