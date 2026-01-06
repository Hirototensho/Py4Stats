# tests/test_eda_tools.py
import pytest
import pandas as pd
import numpy as np
import wooldridge
from pandas.testing import assert_frame_equal
from palmerpenguins import load_penguins
import matplotlib.pyplot as plt

from py4stats import eda_tools as eda

import pathlib
tests_path = pathlib.Path(__file__).parent

penguins = load_penguins() # サンプルデータの読み込み
wage1 = wooldridge.data('wage1')
mroz = wooldridge.data('mroz')
adelie = penguins.query("species == 'Adelie'")
gentoo = penguins.query("species == 'Gentoo'")

# =========================================================
# test_diagnose
# =========================================================
def test_diagnose():
    output_df = penguins.diagnose()
    # expected_df = pd.read_csv(f'{tests_path}/fixtures/diagnose.csv', index_col = 0)
    expected_df = pd.read_csv(f'{tests_path}/fixtures/diagnose.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)
    print()

# =========================================================
# freq_tabl/ crosstab2 / tabyl
# =========================================================
def test_freq_table():
    output_df = penguins.freq_table('species')
    # output_df.to_csv(f'{tests_path}/fixtures/freq_table.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/freq_table.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

def test_crosstab2():
    output_df = penguins.crosstab2(
        index = 'sex',
        columns = 'species', 
        values = 'body_mass_g',
        aggfunc = 'mean'
    )
    output_df.round(1)
    output_df.to_csv(f'{tests_path}/fixtures/crosstab2.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/crosstab2.csv', index_col = 0)

    test_result = eda.compare_df_record(output_df, expected_df).all().all()

    assert test_result

def test_tabyl():
    output_df = penguins.tabyl('island', 'species')
    # output_df.to_csv(f'{tests_path}/fixtures/tabyl.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/tabyl.csv', index_col = 0)

    test_result = eda.compare_df_record(
        output_df.astype(str), 
        expected_df.astype(str)
        )\
            .all().all()
    assert test_result
# =========================================================
# compare_df
# =========================================================
def test_compare_df_cols():
    output_df = eda.compare_df_cols(
        [adelie, gentoo],
        return_match = 'match'
    )
    # display(output_df)

    # output_df.to_csv(f'{tests_path}/fixtures/compare_df_cols.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/compare_df_cols.csv', index_col = 0).fillna('')
    assert_frame_equal(output_df, expected_df)

def test_compare_df_stats():
    penguins_modify = penguins.copy()
    penguins_modify['body_mass_g'] = penguins_modify['body_mass_g'] / 1000
    penguins_modify['bill_length_mm'] = eda.scale(penguins_modify['bill_length_mm'])

    output_df = eda.compare_df_stats(
        [penguins, penguins_modify]
        )

    # output_df.to_csv(f'{tests_path}/fixtures/compare_df_stats.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/compare_df_stats.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

def test_compare_df_stats():
    output_df = eda.compare_group_median(
        adelie, gentoo,
        group_names = ['Adelie', 'Gentoo']
    )

    # output_df.to_csv(f'{tests_path}/fixtures/compare_group_median.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/compare_group_median.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

def test_compare_df_record():
    penguins_copy = penguins.copy()
    penguins_copy.loc[200:250, 'flipper_length_mm'] = \
        2 * penguins_copy.loc[200:250, 'flipper_length_mm'] 

    output_df = eda.compare_df_record(penguins, penguins_copy)

    # output_df.to_csv(f'{tests_path}/fixtures/compare_df_record.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/compare_df_record.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

# =========================================================
# remove_empty / test_remove_constant
# =========================================================
def test_remove_empty():
    penguins_empty = penguins.copy()
    penguins_empty['na_col'] = pd.NA
    penguins_empty = eda.remove_empty(penguins_empty)
    assert_frame_equal(penguins_empty, penguins)

def test_remove_constant():
    penguins_constant = penguins.copy()
    penguins_constant['one'] = 1
    penguins_constant['two'] = 2
    penguins_constant = eda.remove_constant(penguins_constant, quiet = False)
    assert_frame_equal(penguins_constant, penguins)

# =========================================================
# filtering_out
# =========================================================

def test_filtering_out_columns() -> None:
    df = pd.DataFrame({"foo_x": [1], "foo_y": [2], "bar": [3]})
    out = df.filtering_out(contains="foo", axis="columns")
    assert list(out.columns) == ["bar"]

def test_filtering_out_index() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]}, index=["keep", "drop_me", "drop_you"])
    out = df.filtering_out(starts_with="drop", axis="index")
    assert list(out.index) == ["keep"]

# =========================================================
# is_dummy (Series/DataFrame)
# =========================================================

def test_is_dummy_series() -> None:
    s = pd.Series([0, 1, 1, 0])
    assert eda.is_dummy(s) is True

def test_is_dummy_dataframe() -> None:
    df = pd.DataFrame({"d1": [0, 1], "d2": [0, 0], "d3": [2, 3]})
    out = eda.is_dummy(df)
    assert isinstance(out, pd.Series)
    assert out.loc["d1"]
    assert not out.loc["d2"]
    assert not out.loc["d3"]

# =========================================================
# Pareto_plot
# (plot は「落ちないこと」と最低限の構造だけ確認)
# =========================================================

def test_Pareto_plot() -> None:
    penguins_modify = penguins.copy()
    penguins_modify['group'] = penguins_modify['species'] + '\n' + penguins_modify['island']
    fig, ax = plt.subplots()
    eda.Pareto_plot(penguins_modify, group = 'group', ax = ax)
    assert len(ax.patches) > 0

# =======================================================================
# string/regex helpers: detect_Kanzi / is_number / is_ymd / is_ymd_like
# =======================================================================

def test_detect_kanzi() -> None:
    assert eda.detect_Kanzi("漢字") is True
    assert eda.detect_Kanzi("abc") is False

def test_is_ymd_and_like() -> None:
    s = pd.Series(["2025-12-30", "2025-1-2", "abc", None])
    out = s.is_ymd(na_default=True)
    assert list(out[:3]) == [True, True, False]
    assert out.iloc[3]

    s2 = pd.Series(["2025年12月30日", "2025-12-30", "nope"])
    out2 = s2.is_ymd_like()
    assert list(out2) == [True, True, False]

def test_is_number_basic() -> None:
    s = pd.Series(["123", "12E+3", "abc", "2025-12-30", None])
    out = s.is_number(na_default=False)
    assert out.iloc[0]
    assert out.iloc[1]
    assert not out.iloc[2]
    # 日付っぽいのは除外される想定
    assert not out.iloc[3]
    assert not out.iloc[4]

# =========================================================
# check_that / check_viorate
# =========================================================

def test_check_that_basic() -> None:
    d = pd.DataFrame({"x": [1, 2, 3], "y": [1, 0, 1]})
    rules = {"x_pos": "x > 0", "y_is1": "y == 1"}
    out = d.check_that(rules)
    assert set(out.columns) == {"item", "passes", "fails", "coutna", "expression"}
    assert out.loc["x_pos", "fails"] == 0
    assert out.loc["y_is1", "fails"] == 1

def test_check_viorate_flags_rows() -> None:
    d = pd.DataFrame({"x": [1, -1, 2]})
    rules = {"x_pos": "x > 0"}
    out = d.check_viorate(rules)
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
    'cd2':eda.implies_exper('staff > 0', 'staff_costs > 0'), # 従業員がいる場合、人件費は厳密に正である
    'bs':'turnover + other_rev == total_rev',                # 売上高とその他の収入の合計は総収入に等しい
    'mn':'profit.mean() > 0'                                 # セクター全体の平均的な利益はゼロよりも大きい
    }
def test_check_that_basic() -> None:
    output_df = eda.check_that(retailers, rule_dict)
    # output_df.to_csv(f'{tests_path}/fixtures/check_that.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/check_that.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

def test_check_viorate() -> None:
    output_df = eda.check_viorate(retailers, rule_dict)
    # output_df.to_csv(f'{tests_path}/fixtures/check_viorate.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/check_viorate.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

# ================================================================
# implies_exper / is_complet / reducers (Sum/Mean/Max/Min/Median)
# ================================================================

def test_implies_exper_string() -> None:
    assert eda.implies_exper("A", "B") == "B | ~(A)"

def test_is_complet_dataframe_and_series() -> None:
    df = pd.DataFrame({"a": [1, None], "b": [2, 3]})
    out_df = df.is_complet()
    assert out_df.tolist() == [True, False]

    s1 = pd.Series([1, None])
    s2 = pd.Series([2, 3])
    out_s = eda.is_complet(s1, s2)
    assert out_s.tolist() == [True, False]

def test_reducers() -> None:
    a = pd.Series([1, 2, np.nan])
    b = pd.Series([10, 20, 30])
    assert eda.Sum(a, b).tolist() == [11, 22, 30]
    assert eda.Mean(a, b).iloc[0] == pytest.approx(5.5)
    assert eda.Max(a, b).tolist() == [10, 20, 30]
    assert eda.Min(a, b).tolist() == [1, 2, 30]  # nan は無視される
    assert eda.Median(a, b).iloc[0] == pytest.approx(5.5)