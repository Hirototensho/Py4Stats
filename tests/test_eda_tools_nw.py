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

import pathlib
tests_path = pathlib.Path(__file__).parent

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
mroz_pl = pl.from_pandas(mroz)
mroz_pa = pa.Table.from_pandas(mroz)

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

# =========================================================
# テスト用関数の定義
# =========================================================

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

def _assert_df_record(output_df, fixture_csv: str, index_col = 0, **kwarg) -> None:
    if hasattr(output_df, 'to_pandas'):
        output_df = output_df.to_pandas()
    expected_df = pd.read_csv(f'{tests_path}/fixtures/{fixture_csv}', index_col = index_col, **kwarg)
    
    result = eda_nw.compare_df_record(output_df, expected_df).all().all()

    assert result

# =========================================================
# 実験的実装
# =========================================================
from narwhals import testing as nw_test

# 私の手元にある環境では、`narwhals.testing.assert_frame_equal()` が読み込めないので、
# 以下の関数はまだ使えません。

# def _assert_df_eq(
#         output_df,  
#         path_fixture: str,  
#         update_fixture: bool = False, 
#         check_dtype: bool = True, 
#         **kwarg
#         ) -> None:
    
#     output_df = nw.from_native(output_df)

#     if update_fixture:
#         output_df.write_csv(f'{tests_path}/fixtures/{path_fixture}')

#     expected_df = nw.read_csv(
#         f'{tests_path}/fixtures/{path_fixture}',
#         backend = output_df.implementation
#         )
    
#     nw_test.assert_frame_equal(
#         left = output_df, 
#         right = expected_df, 
#         check_dtype = check_dtype,
#         **kwarg
#         )

# =========================================================
# diagnose
# =========================================================
def test_diagnose_pd():
    output_df = eda_nw.diagnose(penguins, to_native = False)
    # output_df.write_csv(f'{tests_path}/fixtures/diagnose_nw.csv')
    _assert_df_fixture_new(output_df, 'diagnose_nw.csv')

def test_diagnose_pl():
    output_df = eda_nw.diagnose(penguins_pl, to_native = False)
    # output_df.write_csv(f'{tests_path}/fixtures/diagnose_pl.csv')
    _assert_df_fixture_new(output_df, 'diagnose_pl.csv')

def test_diagnose_pa():
    output_df = eda_nw.diagnose(penguins_pa, to_native = False)
    _assert_df_fixture_new(output_df, 'diagnose_pl.csv')

# =========================================================
# freq_tabl/ crosstab
# =========================================================
def test_freq_table_pd():
    output_df = eda_nw.freq_table(penguins, 'species', to_native = False)
    # output_df.write_csv(f'{tests_path}/fixtures/freq_table_nw.csv')
    _assert_df_fixture_new(output_df, 'freq_table_nw.csv')

def test_freq_table_pl():
    output_df = eda_nw.freq_table(penguins_pl, 'species', to_native = False)
    # output_df.write_csv(f'{tests_path}/fixtures/freq_table_pl.csv')
    _assert_df_fixture_new(output_df, 'freq_table_pl.csv')


def test_freq_table_pa():
    output_df = eda_nw.freq_table(penguins_pa, 'species', to_native = False)
    # output_df.write_csv(f'{tests_path}/fixtures/freq_table_pa.csv')
    _assert_df_fixture_new(output_df, 'freq_table_pa.csv')

# =========================================================
# crosstab
# =========================================================

def test_crosstab_pd():
    output_df1 = eda_nw.crosstab(penguins, 'island', 'species', margins = True, normalize = 'all')
    # output_df1.to_csv(f'{tests_path}/fixtures/crosstab_nw1.csv')
    expected_df1 = pd.read_csv(f'{tests_path}/fixtures/crosstab_nw1.csv', index_col = 0)
    test_result1 = eda_nw.compare_df_record(output_df1, expected_df1).all().all()

    output_df2 = eda_nw.crosstab(penguins, 'island', 'species', margins = True, normalize = 'columns')
    # output_df2.to_csv(f'{tests_path}/fixtures/crosstab_nw2.csv')
    expected_df2 = pd.read_csv(f'{tests_path}/fixtures/crosstab_nw2.csv', index_col = 0)
    test_result2 = eda_nw.compare_df_record(output_df2, expected_df2).all().all()

    output_df3 = eda_nw.crosstab(penguins, 'island', 'species', margins = True, normalize = 'index')
    # output_df3.to_csv(f'{tests_path}/fixtures/crosstab_nw3.csv')
    expected_df3 = pd.read_csv(f'{tests_path}/fixtures/crosstab_nw3.csv', index_col = 0)
    test_result3 = eda_nw.compare_df_record(output_df3, expected_df3).all().all()

    assert test_result1 and test_result2 and test_result3


def test_crosstab_pl():
    output_df1 = eda_nw.crosstab(penguins_pl, 'island', 'species', margins = True, normalize = 'all')
    output_df1 = output_df1.to_pandas()
    # output_df1.to_csv(f'{tests_path}/fixtures/crosstab_pl1.csv')
    expected_df1 = pd.read_csv(f'{tests_path}/fixtures/crosstab_pl1.csv', index_col = 0)
    test_result1 = eda_nw.compare_df_record(output_df1, expected_df1).all().all()

    output_df2 = eda_nw.crosstab(penguins_pl, 'island', 'species', margins = True, normalize = 'columns')
    output_df2 = output_df2.to_pandas()
    # output_df2.to_csv(f'{tests_path}/fixtures/crosstab_pl2.csv')
    expected_df2 = pd.read_csv(f'{tests_path}/fixtures/crosstab_pl2.csv', index_col = 0)
    test_result2 = eda_nw.compare_df_record(output_df2, expected_df2).all().all()

    output_df3 = eda_nw.crosstab(penguins_pl, 'island', 'species', margins = True, normalize = 'index')
    output_df3 = output_df3.to_pandas()
    # output_df3.to_csv(f'{tests_path}/fixtures/crosstab_pl3.csv')
    expected_df3 = pd.read_csv(f'{tests_path}/fixtures/crosstab_pl3.csv', index_col = 0)
    test_result3 = eda_nw.compare_df_record(output_df3, expected_df3).all().all()

    assert test_result1 and test_result2 and test_result3

def test_crosstab_pa():
    output_df1 = eda_nw.crosstab(penguins_pa, 'island', 'species', margins = True, normalize = 'all')
    output_df1 = output_df1.to_pandas()
    # output_df1.to_csv(f'{tests_path}/fixtures/crosstab_pa1.csv')
    expected_df1 = pd.read_csv(f'{tests_path}/fixtures/crosstab_pa1.csv', index_col = 0)
    test_result1 = eda_nw.compare_df_record(output_df1, expected_df1).all().all()

    output_df2 = eda_nw.crosstab(penguins_pa, 'island', 'species', margins = True, normalize = 'columns')
    output_df2 = output_df2.to_pandas()
    # output_df2.to_csv(f'{tests_path}/fixtures/crosstab_pa2.csv')
    expected_df2 = pd.read_csv(f'{tests_path}/fixtures/crosstab_pa2.csv', index_col = 0)
    test_result2 = eda_nw.compare_df_record(output_df2, expected_df2).all().all()

    output_df3 = eda_nw.crosstab(penguins_pa, 'island', 'species', margins = True, normalize = 'index')
    output_df3 = output_df3.to_pandas()
    # output_df3.to_csv(f'{tests_path}/fixtures/crosstab_pa3.csv')
    expected_df3 = pd.read_csv(f'{tests_path}/fixtures/crosstab_pa3.csv', index_col = 0)
    test_result3 = eda_nw.compare_df_record(output_df3, expected_df3).all().all()

    assert test_result1 and test_result2 and test_result3

# =========================================================
# tabyl
# =========================================================

def test_tabyl_pd():
    output_df = eda_nw.tabyl(penguins, 'island', 'species', normalize = 'index')
    # output_df.to_csv(f'{tests_path}/fixtures/tabyl_nw.csv', index = True)
    _assert_df_record(output_df, 'tabyl_nw.csv', dtype = {'All':str})


def test_tabyl_pl():
    output_df = eda_nw.tabyl(
            penguins_pl, 'island', 'species', 
            normalize = 'index', to_native = False
        ).to_pandas()
    # output_df.to_csv(f'{tests_path}/fixtures/tabyl_pl.csv', index = True)
    _assert_df_record(output_df, 'tabyl_pl.csv', dtype = {'All':str})

def test_tabyl_pa():
    output_df = eda_nw.tabyl(
        penguins_pa, 'island', 'species', 
        normalize = 'index', to_native = False
    ).to_pandas()
    # output_df.to_csv(f'{tests_path}/fixtures/tabyl_pa.csv', index = True)
    _assert_df_record(output_df, 'tabyl_pa.csv', dtype = {'All':str})

def test_tabyl_with_boolen_col_pd():
    pm2 = penguins.copy()
    pm2['heavy'] = pm2['body_mass_g'] >= pm2['body_mass_g'].quantile(0.50)

    output_df = eda_nw.tabyl(pm2, 'heavy', 'species', normalize = 'columns')
    # output_df.to_csv(f'{tests_path}/fixtures/tabyl_nw_with_boolen.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/tabyl_nw_with_boolen.csv', index_col = 0)

    test_result = eda_nw.compare_df_record(
        output_df.astype(str), 
        expected_df.astype(str)
        )\
            .all().all()
    assert test_result

# =========================================================
# compare_df_cols
# =========================================================
def test_compare_df_cols_pd():
    output_df = eda_nw.compare_df_cols(
        [adelie, gentoo],
        return_match = 'match'
    )
    # output_df.to_csv(f'{tests_path}/fixtures/compare_df_cols_nw.csv')
    _assert_df_fixture(output_df, 'compare_df_cols_nw.csv')
    

def test_compare_df_cols_pl():
    output_df = eda_nw.compare_df_cols(
        [adelie_pl, gentoo_pl],
        return_match = 'match'
    )
    # output_df.to_csv(f'{tests_path}/fixtures/compare_df_cols_pl.csv')
    _assert_df_fixture(output_df, 'compare_df_cols_pl.csv')

def test_compare_df_cols_pa():
    output_df = eda_nw.compare_df_cols(
        [adelie_pa, gentoo_pa],
        return_match = 'match'
    )
    # output_df.to_csv(f'{tests_path}/fixtures/compare_df_cols_pa.csv')
    _assert_df_fixture(output_df, 'compare_df_cols_pa.csv')

# =========================================================
# compare_df_stats
# =========================================================

def test_compare_df_stats_pd():
    penguins_modify = penguins.copy()
    penguins_modify['body_mass_g'] = penguins_modify['body_mass_g'] / 1000
    penguins_modify['bill_length_mm'] = eda_nw.scale(penguins_modify['bill_length_mm'])

    output_df = eda_nw.compare_df_stats(
        [penguins, penguins_modify]
        )

    # output_df.to_csv(f'{tests_path}/fixtures/compare_df_stats_nw.csv')
    _assert_df_fixture(output_df, 'compare_df_stats_nw.csv')

def test_compare_df_stats_pl():
    penguins_modify = penguins.copy()
    penguins_modify['body_mass_g'] = penguins_modify['body_mass_g'] / 1000
    penguins_modify['bill_length_mm'] = eda_nw.scale(penguins_modify['bill_length_mm'])

    output_df = eda_nw.compare_df_stats(
        [penguins_pl, pl.from_pandas(penguins_modify)]
        )

    # output_df.to_csv(f'{tests_path}/fixtures/compare_df_stats_pl.csv')
    _assert_df_fixture(output_df, 'compare_df_stats_pl.csv')

def test_compare_df_stats_pa():
    penguins_modify = penguins.copy()
    penguins_modify['body_mass_g'] = penguins_modify['body_mass_g'] / 1000
    penguins_modify['bill_length_mm'] = eda_nw.scale(penguins_modify['bill_length_mm'])

    output_df = eda_nw.compare_df_stats(
        [penguins_pa, pa.Table.from_pandas(penguins_modify)]
        )

    # output_df.to_csv(f'{tests_path}/fixtures/compare_df_stats_pa.csv')
    _assert_df_fixture(output_df, 'compare_df_stats_pa.csv')

# =========================================================
# compare_df_record
# =========================================================
def test_compare_df_record_pd():
    penguins_copy = penguins.copy()
    penguins_copy.loc[200:250, 'flipper_length_mm'] = \
        2 * penguins_copy.loc[200:250, 'flipper_length_mm'] 

    output_df = eda_nw.compare_df_record(penguins, penguins_copy)

    # output_df.to_csv(f'{tests_path}/fixtures/compare_df_record_nw.csv')
    _assert_df_fixture(output_df, 'compare_df_record_nw.csv')

def test_compare_df_record_pl():
    penguins_copy = penguins.copy()
    penguins_copy.loc[200:250, 'flipper_length_mm'] = \
        2 * penguins_copy.loc[200:250, 'flipper_length_mm'] 

    output_df = eda_nw.compare_df_record(
        penguins_pl, pl.from_pandas(penguins_copy),
        to_native = False
        )
    # output_df.write_csv(f'{tests_path}/fixtures/compare_df_record_pl.csv')
    _assert_df_fixture_new(output_df, 'compare_df_record_pl.csv')

def test_compare_df_record_pa():
    penguins_copy = penguins.copy()
    penguins_copy.loc[200:250, 'flipper_length_mm'] = \
        2 * penguins_copy.loc[200:250, 'flipper_length_mm'] 

    output_df = eda_nw.compare_df_record(
        penguins_pa, pa.Table.from_pandas(penguins_copy),
        to_native = False
        )

    # output_df.write_csv(f'{tests_path}/fixtures/compare_df_record_pa.csv')
    _assert_df_fixture_new(output_df, 'compare_df_record_pa.csv')

# =========================================================
# remove_empty
# =========================================================
def test_remove_empty_pd():
    penguins_empty = penguins.copy()
    penguins_empty['na_col'] = pd.NA
    penguins_empty = eda_nw.remove_empty(penguins_empty)
    assert_frame_equal(penguins_empty, penguins)

def test_remove_empty_pl():
    penguins_empty = penguins.copy()
    penguins_empty['na_col'] = pd.NA
    penguins_empty = pl.from_pandas(penguins_empty)
    penguins_empty = eda_nw.remove_empty(penguins_empty)
    assert_frame_equal(penguins_empty.to_pandas(), penguins_pl.to_pandas())

def test_remove_empty_pa():
    penguins_empty = penguins.copy()
    penguins_empty['na_col'] = pd.NA
    penguins_empty = pa.Table.from_pandas(penguins_empty)
    penguins_empty = eda_nw.remove_empty(penguins_empty)
    assert_frame_equal(penguins_empty.to_pandas(), penguins_pa.to_pandas())

# =========================================================
# remove_constant
# =========================================================

def test_remove_constant_pd():
    penguins_constant = penguins.copy()
    penguins_constant['one'] = 1
    penguins_constant['two'] = 2
    output_df = eda_nw.remove_constant(penguins_constant)
    assert_frame_equal(output_df, penguins)

def test_remove_constant_pl():
    penguins_constant = penguins.copy()
    penguins_constant['one'] = 1
    penguins_constant['two'] = 2
    output_df = eda_nw.remove_constant(
        pl.from_pandas(penguins_constant)
        ).to_pandas()
    assert_frame_equal(output_df, penguins_pl.to_pandas())

def test_remove_constant_pa():
    penguins_constant = penguins.copy()
    penguins_constant['one'] = 1
    penguins_constant['two'] = 2
    output_df = eda_nw.remove_constant(
        pa.Table.from_pandas(penguins_constant)
        ).to_pandas()
    assert_frame_equal(output_df, penguins_pa.to_pandas())

# =========================================================
# filtering_out
# =========================================================
def test_filtering_out_columns_pd() -> None:
    df = pd.DataFrame({"foo_x": [1], "foo_y": [2], "bar": [3]})
    out = eda_nw.filtering_out(df, contains="foo", axis="columns")
    assert list(out.columns) == ["bar"]

def test_filtering_out_index_pd() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]}, index=["keep", "drop_me", "drop_you"])
    out = eda_nw.filtering_out(df, starts_with="drop", axis="index")
    assert list(out.index) == ["keep"]

def test_filtering_out_columns_pl() -> None:
    df = pd.DataFrame({"foo_x": [1], "foo_y": [2], "bar": [3]})
    df = pl.from_pandas(df)
    out = eda_nw.filtering_out(df, contains="foo", axis="columns")
    assert list(out.columns) == ["bar"]

def test_filtering_out_columns_pa() -> None:
    df = pd.DataFrame({"foo_x": [1], "foo_y": [2], "bar": [3]})
    df = pa.Table.from_pandas(df)
    out = eda_nw.filtering_out(df, contains="foo", axis="columns")
    assert list(out.to_pandas().columns) == ["bar"]

# =========================================================
# is_dummy_nw (Series/DataFrame)
# =========================================================

def test_is_dummy_nw_series() -> None:
    s = pd.Series([0, 1, 1, 0])
    assert eda_nw.is_dummy(s) is True

def test_is_dummy_nw_dataframe() -> None:
    result = eda_nw.is_dummy(mroz).to_list()

    expected = [
        True, False, False, False, False, False, False, False, False, 
        False, False, False, False, False, False, False, False, True, 
        False, False, False, False
    ]
    
    assert result == expected 

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
def test_diagnose_category_pd():
    output_df = eda_nw.diagnose_category(pm2, to_native = False)
    # output_df.write_csv(f'{tests_path}/fixtures/diagnose_category_nw.csv')
    _assert_df_fixture_new(output_df, 'diagnose_category_nw.csv')

def test_diagnose_category_pl():
    output_df = eda_nw.diagnose_category(pl.from_pandas(pm2)).to_pandas()
    # output_df.to_csv(f'{tests_path}/fixtures/diagnose_category_pl.csv', index = False)
    _assert_df_fixture(output_df, 'diagnose_category_pl.csv', index_col = None)

def test_diagnose_category_pa():
    output_df = eda_nw.diagnose_category(pa.Table.from_pandas(pm2)).to_pandas()
    # output_df.to_csv(f'{tests_path}/fixtures/diagnose_category_pa.csv', index = False)
    _assert_df_fixture(output_df, 'diagnose_category_pa.csv', index_col = None)

# =========================================================
# Pareto_plot_nw
# (plot は「落ちないこと」と最低限の構造だけ確認)
# =========================================================

def test_Pareto_plot() -> None:
    penguins_modify = penguins.copy()
    penguins_modify['group'] = penguins_modify['species'] + '\n' + penguins_modify['island']
    
    fig, ax = plt.subplots()
    
    eda_nw.Pareto_plot(penguins_modify, group = 'group', ax = ax)
    assert len(ax.patches) > 0

    fig, ax = plt.subplots()
    eda_nw.Pareto_plot(
        penguins_modify, group = 'group', 
        values = 'bill_length_mm',
        palette = ['#FF6F91', '#252525'],
        ax = ax
        )
    assert len(ax.patches) > 0

    fig, ax = plt.subplots()
    eda_nw.Pareto_plot(
        penguins_modify, 
        values = 'bill_length_mm',
        group = 'group',
        aggfunc = lambda x: x.std(),
        ax = ax
        )
    assert len(ax.patches) > 0

def test_make_rank_table_nw_error_on_non_exist_col():
    with pytest.raises(ValueError) as excinfo:
        eda_nw.make_rank_table(penguins, 'non_exists', 'body_mass_g')
    # 仕様：候補があると "Did you mean ..." を含む
    assert "must be one of" in str(excinfo.value)

def test_Pareto_plot_pl() -> None:
    penguins_modify = penguins.copy()
    penguins_modify['group'] = penguins_modify['species'] + '\n' + penguins_modify['island']
    
    fig, ax = plt.subplots()

    eda_nw.Pareto_plot(pl.from_pandas(penguins_modify), group = 'group', ax = ax)
    assert len(ax.patches) > 0

def test_Pareto_plot_pa() -> None:
    penguins_modify = penguins.copy()
    penguins_modify['group'] = penguins_modify['species'] + '\n' + penguins_modify['island']
    
    fig, ax = plt.subplots()

    eda_nw.Pareto_plot(pa.Table.from_pandas(penguins_modify), group = 'group', ax = ax)
    assert len(ax.patches) > 0

# ================================================================
# compare_group_means / compare_group_median
# ================================================================
@pytest.mark.parametrize(
    "backend",
    [
        ('pd'),
        ('pl'),
        ('pa'),
    ],
)
def test_compare_group_means(backend) -> None:
    output_df = eda_nw.compare_group_means(
        adelie_dict.get(backend), 
        gentoo_dict.get(backend)
        ) # -> pd.DataFrame
    
    # output_df.to_csv(f'{tests_path}/fixtures/compare_group_means_{backend}.csv')
    
    _assert_df_fixture(output_df, f'compare_group_means_{backend}.csv')

@pytest.mark.parametrize(
    "backend",
    [
        ('pd'),
        ('pl'),
        ('pa'),
    ],
)
def test_compare_group_median(backend) -> None:
    output_df = eda_nw.compare_group_median(
        adelie_dict.get(backend), 
        gentoo_dict.get(backend)
        ) # -> pd.DataFrame
    
    # output_df.to_csv(f'{tests_path}/fixtures/compare_group_median_{backend}.csv')
    
    _assert_df_fixture(output_df, f'compare_group_median_{backend}.csv')
# ================================================================
# plot_mean_diff / plot_median_diff
# ================================================================

@pytest.mark.parametrize(
    "stats_diff",
    [
        ('norm_diff'),
        ('abs_diff'),
        ('rel_diff'),
    ],
)
def test_plot_mean_diff(stats_diff) -> None:
    fig, ax = plt.subplots()
    eda_nw.plot_mean_diff(
        penguins.query('species == "Gentoo"'),
        penguins.query('species == "Adelie"'),
        stats_diff = stats_diff,
        ax = ax
    );
    assert len(ax.get_lines()) > 0 and len(ax.collections) > 0

@pytest.mark.parametrize(
    "stats_diff",
    [
        ('abs_diff'),
        ('rel_diff'),
    ],
)
def test_plot_median_diff(stats_diff) -> None:
    fig, ax = plt.subplots()
    eda_nw.plot_mean_diff(
        penguins.query('species == "Gentoo"'),
        penguins.query('species == "Adelie"'),
        stats_diff = stats_diff,
        ax = ax
    );
    assert len(ax.get_lines()) > 0 and len(ax.collections) > 0


# ================================================================
# mean_qi / median_qi / mean_ci (Pandas)
# ================================================================
def test_mean_qi_pd() -> None:
    output_df = eda_nw.mean_qi(penguins, to_native = False)
    # output_df.write_csv(f'{tests_path}/fixtures/mean_qi_nw.csv')
    _assert_df_fixture_new(output_df, 'mean_qi_nw.csv')

def test_median_qi_pd() -> None:
    output_df = eda_nw.median_qi(penguins, to_native = False)
    # output_df.write_csv(f'{tests_path}/fixtures/median_qi_nw.csv')
    _assert_df_fixture_new(output_df, 'median_qi_nw.csv')

def test_mean_ci_pd() -> None:
    output_df = eda_nw.mean_ci(penguins, to_native = False)
    # output_df.write_csv(f'{tests_path}/fixtures/mean_ci_nw.csv')
    _assert_df_fixture_new(output_df, 'mean_ci_nw.csv')

# ================================================================
# mean_qi / median_qi / mean_ci (polars)
# ================================================================
def test_mean_qi_pl() -> None:
    output_df = eda_nw.mean_qi(penguins_pl, to_native = False)
    # output_df.write_csv(f'{tests_path}/fixtures/mean_qi_pl.csv')
    _assert_df_fixture_new(output_df, 'mean_qi_pl.csv')

def test_median_qi_pl() -> None:
    output_df = eda_nw.median_qi(penguins_pl, to_native = False)
    # output_df.write_csv(f'{tests_path}/fixtures/median_qi_pl.csv')
    _assert_df_fixture_new(output_df, 'median_qi_pl.csv')

def test_mean_ci_pl() -> None:
    output_df = eda_nw.mean_ci(penguins_pl, to_native = False)
    # output_df.write_csv(f'{tests_path}/fixtures/mean_ci_pl.csv')
    _assert_df_fixture_new(output_df, 'mean_ci_pl.csv')

# ================================================================
# mean_qi / median_qi / mean_ci (pyarrow)
# ================================================================
def test_mean_qi_pa() -> None:
    output_df = eda_nw.mean_qi(penguins_pa, to_native = False)
    # output_df.write_csv(f'{tests_path}/fixtures/mean_qi_pa.csv')
    _assert_df_fixture_new(output_df, 'mean_qi_pa.csv')

def test_median_qi_pa() -> None:
    output_df = eda_nw.median_qi(penguins_pa, to_native = False)
    # output_df.write_csv(f'{tests_path}/fixtures/median_qi_pa.csv')
    _assert_df_fixture_new(output_df, 'median_qi_pa.csv')

def test_mean_ci_pa() -> None:
    output_df = eda_nw.mean_ci(penguins_pa, to_native = False)
    # output_df.write_csv(f'{tests_path}/fixtures/mean_ci_pa.csv')
    _assert_df_fixture_new(output_df, 'mean_ci_pa.csv')

# =======================================================================
# plot_miss_var
# =======================================================================
def test_plot_miss_var_pd() -> None:
    fig, ax = plt.subplots()
    eda_nw.plot_miss_var(penguins, ax = ax)
    assert len(ax.patches) > 0

def test_plot_miss_var_pl() -> None:
    fig, ax = plt.subplots()
    eda_nw.plot_miss_var(penguins_pl, ax = ax)
    assert len(ax.patches) > 0

def test_plot_miss_var_pa() -> None:
    fig, ax = plt.subplots()
    eda_nw.plot_miss_var(penguins_pa, ax = ax)
    assert len(ax.patches) > 0

# =======================================================================
# string/regex helpers: is_number / is_ymd / is_ymd_like
# =======================================================================

def test_is_ymd_and_like_pd() -> None:
    s = pd.Series(["2025-12-30", "2025-1-2", "abc", None])
    out = eda_nw.is_ymd(s, na_default=True)
    assert list(out[:3]) == [True, True, False]
    assert out.iloc[3]

    s2 = pd.Series(["2025年12月30日", "2025-12-30", "nope"])
    out2 = eda_nw.is_ymd_like(s2)
    assert list(out2) == [True, True, False]

def test_is_number_nw_basic_pd() -> None:
    s = pd.Series(["123", "12E+3", "abc", "2025-12-30", None])
    out = eda_nw.is_number(s, na_default=False)
    assert out.iloc[0]
    assert out.iloc[1]
    assert not out.iloc[2]
    # 日付っぽいのは除外される想定
    assert not out.iloc[3]
    assert not out.iloc[4]

def test_is_number_nw_extend_pd() -> None:
    s = pd.Series([
        '123', "0.12", "1e+07", '-31', '2個', '1A',
        "2024-03-03", "2024年3月3日", "24年3月3日", '令和6年3月3日',
        '0120-123-456', '15ｹ', "apple", "不明", None, np.nan
        ])

    expect = pd.Series([
        True, True, True, True, False, False, 
        False, False, False, False, False, False,
        False, False, True, True
    ])

    assert (eda_nw.is_number(s) == expect).all()

# =========================================================
# check_that_nw / check_viorate_nw
# =========================================================

def test_check_that_nw_basic() -> None:
    d = pd.DataFrame({"x": [1, 2, 3], "y": [1, 0, 1]})
    rules = {"x_pos": "x > 0", "y_is1": "y == 1"}
    out = eda_nw.check_that(d, rules)
    assert set(out.columns) == {"item", "passes", "fails", "coutna", "expression"}
    assert out.loc["x_pos", "fails"] == 0
    assert out.loc["y_is1", "fails"] == 1

def test_check_viorate_nw_flags_rows() -> None:
    d = pd.DataFrame({"x": [1, -1, 2]})
    rules = {"x_pos": "x > 0"}
    out = eda_nw.check_viorate(d, rules)
    assert out["x_pos"].tolist() == [False, True, False]
    assert out["any"].tolist() == [False, True, False]
    assert out["all"].tolist() == [False, True, False]


# URL = 'https://raw.githubusercontent.com/data-cleaning/validate/master/pkg/data/retailers.csv'
# retailers = pd.read_csv(URL, sep = ';')
# retailers.columns = retailers.columns.to_series().str.replace('.', '_', regex = False)
retailers = pd.read_csv(f'{tests_path}/fixtures/retailers.csv', index_col = 0)

rule_dict =  {
    'to':'turnover > 0',                                     # 売上高は厳密に正である
    'sc':'staff_costs / staff < 50',                         # 従業員1人当たりの人件費は50,000ギルダー未満である
    'cd1':'staff_costs > 0 | ~(staff > 0)',                  # 従業員がいる場合、人件費は厳密に正である
    'cd2':eda_nw.implies_exper('staff > 0', 'staff_costs > 0'), # 従業員がいる場合、人件費は厳密に正である
    'bs':'turnover + other_rev == total_rev',                # 売上高とその他の収入の合計は総収入に等しい
    'mn':'profit.mean() > 0'                                 # セクター全体の平均的な利益はゼロよりも大きい
    }

def test_check_that_nw_pd() -> None:
    output_df = eda_nw.check_that(retailers, rule_dict)
    # output_df.to_csv(f'{tests_path}/fixtures/check_that_nw.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/check_that_nw.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

def test_check_viorate() -> None:
    output_df = eda_nw.check_viorate(retailers, rule_dict)
    # output_df.to_csv(f'{tests_path}/fixtures/check_viorate_nw.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/check_viorate_nw.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

# ================================================================
# implies_exper / is_complete / reducers (Sum/Mean/Max/Min/Median)
# ================================================================

def test_implies_exper_string() -> None:
    assert eda_nw.implies_exper("A", "B") == "B | ~(A)"

def test_is_complete_dataframe_and_series() -> None:
    df = pd.DataFrame({"a": [1, None], "b": [2, 3]})
    out_df = eda_nw.is_complete(df)
    assert out_df.tolist() == [True, False]

    s1 = pd.Series([1, None])
    s2 = pd.Series([2, 3])
    out_s = eda_nw.is_complete(s1, s2)
    assert out_s.tolist() == [True, False]

def test_reducers() -> None:
    a = pd.Series([1, 2, np.nan])
    b = pd.Series([10, 20, 30])
    assert eda_nw.Sum(a, b).tolist() == [11, 22, 30]
    assert eda_nw.Mean(a, b).iloc[0] == pytest.approx(5.5)
    assert eda_nw.Max(a, b).tolist() == [10, 20, 30]
    assert eda_nw.Min(a, b).tolist() == [1, 2, 30]  # nan は無視される
    assert eda_nw.Median(a, b).iloc[0] == pytest.approx(5.5)

# ================================================================
# plot_category
# ================================================================
import itertools
Q1 = [70 * ['Strongly agree'], 200 * ['Agree'], 235 * ['Disagree'], 149 * ['Strongly disagree']]
Q2 = [74 * ['Strongly agree'], 209 * ['Agree'], 238 * ['Disagree'], 133 * ['Strongly disagree']]
Q3 = [59 * ['Strongly agree'], 235 * ['Agree'], 220 * ['Disagree'], 140 * ['Strongly disagree']]
Q4 = [40 * ['Strongly agree'], 72 * ['Agree'], 266 * ['Disagree'], 276 * ['Strongly disagree']]

categ_list = ['Strongly disagree', 'Disagree', 'Agree', 'Strongly agree']
data = pd.DataFrame({
    'I read only if I have to.':list(itertools.chain.from_iterable(Q1)),
    'Reading is one of my favorite hobbies.':list(itertools.chain.from_iterable(Q2)),
    'I like talking about books with other people.':list(itertools.chain.from_iterable(Q3)),
    'For me, reading is a waste of time.':list(itertools.chain.from_iterable(Q4))
})

def test_plot_category_pd() -> None:

    data_pd = data.apply(pd.Categorical, categories = categ_list)

    fig, ax = plt.subplots()
    eda_nw.plot_category(data_pd, ax = ax)

    assert len(ax.patches) > 0

def test_plot_category_pl() -> None:

    data_pl = pl.from_pandas(data)\
    .with_columns(
        pl.all().cast(pl.Enum(categ_list))
    )

    fig, ax = plt.subplots()
    eda_nw.plot_category(data_pl, ax = ax)

    assert len(ax.patches) > 0

def test_plot_category_pa() -> None:

    data_pa = pa.Table.from_pandas(data)

    fig, ax = plt.subplots()
    eda_nw.plot_category(data_pa, sort_by = 'frequency', ax = ax)

    assert len(ax.patches) > 0

# ================================================================
# relocate
# ================================================================

def test_relocate_basic():
    result1 = eda_nw.relocate(penguins, 'year', 'sex').columns.to_list()
    expect1 = ['year', 'sex', 'species', 'island', 'bill_length_mm', 
            'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    assert result1 == expect1

def test_relocate_ncs():
    result2 = eda_nw.relocate(penguins_pl, ncs.numeric()).columns
    expect2 = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g',
            'year', 'species', 'island', 'sex']
    assert result2 == expect2

def test_relocate_before():
    result3 = eda_nw.relocate(
        penguins_pa, 'year', before = 'island', to_native = False
        ).columns
    expect3 = ['species', 'year', 'island', 'bill_length_mm', 'bill_depth_mm',
                'flipper_length_mm', 'body_mass_g', 'sex']
    assert result3 == expect3

def test_relocate_after():    
    result4 = eda_nw.relocate(penguins, 'year', after = 'island').columns.to_list()
    expect4 = ['species', 'island', 'year', 'bill_length_mm', 'bill_depth_mm',
                'flipper_length_mm', 'body_mass_g', 'sex']

    assert result4 == expect4

def test_relocate_basic_error_on_invalid_selector():
    with pytest.raises(ValueError) as excinfo:
        eda_nw.relocate(penguins, 0, True)
    # 仕様：候補があると "Did you mean ..." を含む
    assert "Argument '*args' must be of type" in str(excinfo.value)
    assert "'0' and 'True' cannot be accepted" in str(excinfo.value)

# ================================================================
# weighted_mean
# ================================================================
x = penguins.groupby('species')['bill_length_mm'].mean()
w = penguins.groupby('species')['bill_length_mm'].count()
grand_mean = penguins['bill_length_mm'].mean()


def test_weighted_mean_pd():
    assert np.isclose(eda_nw.weighted_mean(x, w), grand_mean)

def test_weighted_mean_pl():
    x_pl = pl.from_pandas(x)
    w_pl = pl.from_pandas(w)
    assert np.isclose(eda_nw.weighted_mean(x_pl, w_pl), grand_mean)

def test_weighted_mean_pa():
    data_pa = pa.Table.from_pydict({
        'x':x.to_list(),
        'w':w.to_list()
    })

    assert np.isclose(eda_nw.weighted_mean(data_pa['x'], data_pa['w']), grand_mean)

# ================================================================
# scale
# ================================================================

def test_scale_pd():
    res = eda_nw.scale(penguins.select_dtypes('number'))
    assert all(np.isclose(res.mean(), 0) & np.isclose(res.std(), 1))

    res = eda_nw.scale(penguins['body_mass_g'])
    assert np.isclose(res.mean(), 0) & np.isclose(res.std(), 1)

def test_scale_pl():
    res = eda_nw.scale(penguins_pl['body_mass_g']).to_pandas()
    assert np.isclose(res.mean(), 0) & np.isclose(res.std(), 1)

def test_scale_pa():
    res = eda_nw.scale(penguins_pa['body_mass_g']).to_pandas()
    assert np.isclose(res.mean(), 0) & np.isclose(res.std(), 1)

# ================================================================
# min_max
# ================================================================

def test_min_max_pd():
    res = eda_nw.min_max(penguins.select_dtypes('number'))
    assert all(np.isclose(res.min(), 0) & np.isclose(res.max(), 1))

    res = eda_nw.min_max(penguins['body_mass_g'])
    assert np.isclose(res.min(), 0) & np.isclose(res.max(), 1)

def test_min_max_pl():
    res = eda_nw.min_max(penguins_pl['body_mass_g']).to_pandas()
    assert np.isclose(res.min(), 0) & np.isclose(res.max(), 1)

def test_min_max_pa():
    res = eda_nw.min_max(penguins_pa['body_mass_g']).to_pandas()
    assert np.isclose(res.min(), 0) & np.isclose(res.max(), 1)