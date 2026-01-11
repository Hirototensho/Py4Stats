# tests/test_eda_tools.py
import pytest
import pandas as pd
import numpy as np
import wooldridge
from palmerpenguins import load_penguins
import matplotlib.pyplot as plt
from pandas.testing import assert_frame_equal
import polars as pl
import pyarrow as pa
import narwhals as nw

from py4stats.eda_tools import _nw as eda_nw

import pathlib
tests_path = pathlib.Path(__file__).parent

# サンプルデータの読み込み --------------------------------

penguins = load_penguins() 
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


def _assert_df_record(output_df, fixture_csv: str, index_col = 0, **kwarg) -> None:
    if hasattr(output_df, 'to_pandas'):
        output_df = output_df.to_pandas()
    expected_df = pd.read_csv(f'{tests_path}/fixtures/{fixture_csv}', index_col = index_col, **kwarg)
    
    result = eda_nw.compare_df_record(output_df, expected_df).all().all()

    assert result

# =========================================================
# test_diagnose
# =========================================================
def test_diagnose_pd():
    output_df = eda_nw.diagnose(penguins)
    # output_df.to_csv(f'{tests_path}/fixtures/diagnose_nw.csv')
    _assert_df_fixture(output_df, 'diagnose_nw.csv')

def test_diagnose_pl():
    output_df = eda_nw.diagnose(penguins_pl)
    # output_df.to_pandas().to_csv(f'{tests_path}/fixtures/diagnose_pl.csv')
    _assert_df_fixture(output_df, 'diagnose_pl.csv')

def test_diagnose_pa():
    output_df = eda_nw.diagnose(penguins_pa)
    # output_df.to_pandas().to_csv(f'{tests_path}/fixtures/diagnose_pa.csv')
    _assert_df_fixture(output_df, 'diagnose_pa.csv')

# =========================================================
# freq_tabl/ crosstab
# =========================================================
def test_freq_table_pd():
    output_df = eda_nw.freq_table(penguins, 'species')
    # output_df.to_csv(f'{tests_path}/fixtures/freq_table_nw.csv')
    _assert_df_fixture(output_df, 'freq_table_nw.csv')

def test_freq_table_pl():
    output_df = eda_nw.freq_table(penguins_pl, 'species')
    # output_df.to_pandas().to_csv(f'{tests_path}/fixtures/freq_table_pl.csv')
    _assert_df_fixture(output_df, 'freq_table_pl.csv')


def test_freq_table_pa():
    output_df = eda_nw.freq_table(penguins_pa, 'species')
    # output_df.to_pandas().to_csv(f'{tests_path}/fixtures/freq_table_pa.csv')
    _assert_df_fixture(output_df, 'freq_table_pa.csv')

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


# =========================================================
# メモ：現状 narwhal ライブラリでは pa バックエンドの場合 
# `pivot`メソッドに対応していないため、使用できない。
# =========================================================
def test_crosstab_pa():
    output_df1 = eda_nw.crosstab(penguins_pa, 'island', 'species', margins = True, normalize = 'all')
    output_df1 = output_df1.to_pandas()
    output_df1.to_csv(f'{tests_path}/fixtures/crosstab_pa1.csv')
    expected_df1 = pd.read_csv(f'{tests_path}/fixtures/crosstab_pa1.csv', index_col = 0)
    test_result1 = eda_nw.compare_df_record(output_df1, expected_df1).all().all()

    output_df2 = eda_nw.crosstab(penguins_pa, 'island', 'species', margins = True, normalize = 'columns')
    output_df2 = output_df2.to_pandas()
    output_df2.to_csv(f'{tests_path}/fixtures/crosstab_pa2.csv')
    expected_df2 = pd.read_csv(f'{tests_path}/fixtures/crosstab_pa2.csv', index_col = 0)
    test_result2 = eda_nw.compare_df_record(output_df2, expected_df2).all().all()

    output_df3 = eda_nw.crosstab(penguins_pa, 'island', 'species', margins = True, normalize = 'index')
    output_df3 = output_df3.to_pandas()
    output_df3.to_csv(f'{tests_path}/fixtures/crosstab_pa3.csv')
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
    output_df = eda_nw.tabyl(penguins_pl, 'island', 'species', normalize = 'index')
    # output_df.to_csv(f'{tests_path}/fixtures/tabyl_pl.csv', index = True)
    _assert_df_record(output_df, 'tabyl_pl.csv', dtype = {'All':str})

# def test_tabyl_pa():
#     output_df = eda_nw.tabyl(penguins_pa, 'island', 'species', normalize = 'index')
#     output_df.to_csv(f'{tests_path}/fixtures/tabyl_pa.csv', index = True)
#     _assert_df_record(output_df, 'tabyl_pa.csv', dtype = {'All':str})

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
# compare_df
# =========================================================
def test_compare_df_cols_pd():
    output_df = eda_nw.compare_df_cols(
        [adelie, gentoo],
        return_match = 'match'
    )
    # display(output_df)

    # output_df.to_csv(f'{tests_path}/fixtures/compare_df_colss_nw.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/compare_df_colss_nw.csv', index_col = 0).fillna('')
    assert_frame_equal(output_df, expected_df)

def test_compare_df_stats_pd():
    penguins_modify = penguins.copy()
    penguins_modify['body_mass_g'] = penguins_modify['body_mass_g'] / 1000
    penguins_modify['bill_length_mm'] = eda_nw.scale(penguins_modify['bill_length_mm'])

    output_df = eda_nw.compare_df_stats(
        [penguins, penguins_modify]
        )

    # output_df.to_csv(f'{tests_path}/fixtures/compare_df_stats_nw.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/compare_df_stats_nw.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

def test_compare_df_record_pd():
    penguins_copy = penguins.copy()
    penguins_copy.loc[200:250, 'flipper_length_mm'] = \
        2 * penguins_copy.loc[200:250, 'flipper_length_mm'] 

    output_df = eda_nw.compare_df_record(penguins, penguins_copy)

    # output_df.to_csv(f'{tests_path}/fixtures/compare_df_record_nw.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/compare_df_record_nw.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

# =========================================================
# remove_empty_nw / test_remove_constant_nw
# =========================================================
def test_remove_empty_pd():
    penguins_empty = penguins.copy()
    penguins_empty['na_col'] = pd.NA
    penguins_empty = eda_nw.remove_empty(penguins_empty)
    assert_frame_equal(penguins_empty, penguins)

def test_remove_constant_pd():
    penguins_constant = penguins.copy()
    penguins_constant['one'] = 1
    penguins_constant['two'] = 2
    penguins_constant = eda_nw.remove_constant(penguins_constant, quiet = False)
    assert_frame_equal(penguins_constant, penguins)

# =========================================================
# filtering_out
# =========================================================

def test_filtering_out_nw_columns_pd() -> None:
    df = pd.DataFrame({"foo_x": [1], "foo_y": [2], "bar": [3]})
    out = eda_nw.filtering_out(df, contains="foo", axis="columns")
    assert list(out.columns) == ["bar"]

def test_filtering_out_nw_index_pd() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]}, index=["keep", "drop_me", "drop_you"])
    out = eda_nw.filtering_out(df, starts_with="drop", axis="index")
    assert list(out.index) == ["keep"]

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

def test_diagnose_category_pd_pd():
    pm2 = penguins.copy()
    pm2['species'] = pd.Categorical(pm2['species'])

    pm2 = pd.get_dummies(pm2,  columns = ['sex'])

    pm2['heavy'] = np.where(
        pm2['body_mass_g'] >= pm2['body_mass_g'].quantile(0.75), 
        1, 0
    )
    output_df = eda_nw.diagnose_category(pm2)
    # output_df.to_csv(f'{tests_path}/fixtures/diagnose_category_nw.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/diagnose_category_nw.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

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

def test_make_rank_table_nw_error_on_non_exist_col():
    with pytest.raises(ValueError) as excinfo:
        eda_nw.make_rank_table(penguins, 'non_exists', 'body_mass_g')
    # 仕様：候補があると "Did you mean ..." を含む
    assert "must be one of" in str(excinfo.value)


# ================================================================
# compare_group_means_nw / compare_group_median_nw
# ================================================================
def test_compare_group_means_pd() -> None:
    output_df = eda_nw.compare_group_means(adelie, gentoo)
    # output_df.to_csv(f'{tests_path}/fixtures/compare_group_means_nw.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/compare_group_means_nw.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

def test_compare_group_median_pd() -> None:
    output_df = eda_nw.compare_group_median(adelie, gentoo)
    output_df.to_csv(f'{tests_path}/fixtures/compare_group_median_nw.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/compare_group_median_nw.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

# ================================================================
# mean_qi / median_qi / mean_ci
# ================================================================
def test_mean_qi_pd() -> None:
    output_df = eda_nw.mean_qi(penguins)
    # output_df.to_csv(f'{tests_path}/fixtures/mean_qi_nw.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/mean_qi_nw.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

def test_median_qi_pd() -> None:
    output_df = eda_nw.median_qi(penguins)
    # output_df.to_csv(f'{tests_path}/fixtures/median_qi_nw.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/median_qi_nw.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

def test_mean_ci_pd() -> None:
    output_df = eda_nw.mean_ci(penguins)
    # output_df.to_csv(f'{tests_path}/fixtures/mean_ci_nw.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/mean_ci_nw.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)


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


URL = 'https://raw.githubusercontent.com/data-cleaning/validate/master/pkg/data/retailers.csv'
retailers = pd.read_csv(URL, sep = ';')
retailers.columns = retailers.columns.to_series().str.replace('.', '_', regex = False)

rule_dict =  {
    'to':'turnover > 0',                                     # 売上高は厳密に正である
    'sc':'staff_costs / staff < 50',                         # 従業員1人当たりの人件費は50,000ギルダー未満である
    'cd1':'staff_costs > 0 | ~(staff > 0)',                  # 従業員がいる場合、人件費は厳密に正である
    'cd2':eda_nw.implies_exper('staff > 0', 'staff_costs > 0'), # 従業員がいる場合、人件費は厳密に正である
    'bs':'turnover + other_rev == total_rev',                # 売上高とその他の収入の合計は総収入に等しい
    'mn':'profit.mean() > 0'                                 # セクター全体の平均的な利益はゼロよりも大きい
    }
def test_check_that_nw_basic() -> None:
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
# implies_exper / is_complet / reducers (Sum/Mean/Max/Min/Median)
# ================================================================

def test_implies_exper_string() -> None:
    assert eda_nw.implies_exper("A", "B") == "B | ~(A)"

def test_is_complet_dataframe_and_series() -> None:
    df = pd.DataFrame({"a": [1, None], "b": [2, 3]})
    out_df = eda_nw.is_complet(df)
    assert out_df.tolist() == [True, False]

    s1 = pd.Series([1, None])
    s2 = pd.Series([2, 3])
    out_s = eda_nw.is_complet(s1, s2)
    assert out_s.tolist() == [True, False]

def test_reducers() -> None:
    a = pd.Series([1, 2, np.nan])
    b = pd.Series([10, 20, 30])
    assert eda_nw.Sum(a, b).tolist() == [11, 22, 30]
    assert eda_nw.Mean(a, b).iloc[0] == pytest.approx(5.5)
    assert eda_nw.Max(a, b).tolist() == [10, 20, 30]
    assert eda_nw.Min(a, b).tolist() == [1, 2, 30]  # nan は無視される
    assert eda_nw.Median(a, b).iloc[0] == pytest.approx(5.5)