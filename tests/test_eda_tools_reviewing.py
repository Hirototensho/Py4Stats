
# tests/test_eda_tools_reviewing.py

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
# check_that
# =========================================================

def test_check_that_basic() -> None:
    d = pd.DataFrame({"x": [1, 2, 3], "y": [1, 0, 1]})
    rules = {"x_pos": "x > 0", "y_is1": "y == 1"}
    out = eda_review.check_that(d, rules)
    assert set(out.columns) == {'rule', "item", "passes", "fails", "countna", "expression"}
    out = out.set_index('rule')
    assert out.loc["x_pos", "fails"] == 0
    assert out.loc["y_is1", "fails"] == 1

def test_check_viorate_flags_rows() -> None:
    d = pd.DataFrame({"x": [1, -1, 2]})
    rules = {"x_pos": "x > 0"}
    out = eda_review.check_viorate(d, rules)
    assert out["x_pos"].tolist() == [False, True, False]
    assert out["any"].tolist() == [False, True, False]
    assert out["all"].tolist() == [False, True, False]


# URL = 'https://raw.githubusercontent.com/data-cleaning/validate/master/pkg/data/retailers.csv'
# retailers = pd.read_csv(URL, sep = ';')
# retailers.columns = retailers.columns.to_series().str.replace('.', '_', regex = False)
retailers = pd.read_csv(f'{tests_path}/fixtures/retailers.csv', index_col = 0)
retailers_pl = pl.from_pandas(retailers)
retailers_pa = pa.Table.from_pandas(retailers)

rule_dict =  {
    'to':'turnover > 0',                                     # 売上高は厳密に正である
    'sc':'staff_costs / staff < 50',                         # 従業員1人当たりの人件費は50,000ギルダー未満である
    'cd1':'staff_costs > 0 | ~(staff > 0)',                  # 従業員がいる場合、人件費は厳密に正である
    'cd2':eda_review.implies_exper('staff > 0', 'staff_costs > 0'), # 従業員がいる場合、人件費は厳密に正である
    'bs':'turnover + other_rev == total_rev',                # 売上高とその他の収入の合計は総収入に等しい
    'mn':'profit.mean() > 0'                                 # セクター全体の平均的な利益はゼロよりも大きい
    }

def test_check_that_pd() -> None:
    output_df = eda_review.check_that(retailers, rule_dict)
    # output_df.to_csv(f'{tests_path}/fixtures/check_that_nw.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/check_that_nw.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

def test_check_that_pl() -> None:
    output_df = eda_review.check_that(retailers_pl, rule_dict, to_native = False)
    # output_df.write_csv(f'{tests_path}/fixtures/check_that_pl.csv')
    tfun._assert_df_fixture_new(output_df, 'check_that_pl.csv')

def test_check_that_pa() -> None:
    output_df = eda_review.check_that(retailers_pa, rule_dict, to_native = False)
    # output_df.write_csv(f'{tests_path}/fixtures/check_that_pa.csv')
    tfun._assert_df_fixture_new(output_df, 'check_that_pa.csv')

# =========================================================
# check_that
# =========================================================

def test_check_viorate_pd() -> None:
    output_df = eda_review.check_viorate(retailers, rule_dict)
    # output_df.to_csv(f'{tests_path}/fixtures/check_viorate_nw.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/check_viorate_nw.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

def test_check_viorate_pl() -> None:
    output_df = eda_review.check_viorate(retailers_pl, rule_dict, to_native = False)
    # output_df.write_csv(f'{tests_path}/fixtures/check_viorate_pl.csv')
    tfun._assert_df_fixture_new(output_df, 'check_viorate_pl.csv')

def test_check_viorate_pa() -> None:
    output_df = eda_review.check_viorate(retailers_pa, rule_dict, to_native = False)
    # output_df.write_csv(f'{tests_path}/fixtures/check_viorate_pa.csv')
    tfun._assert_df_fixture_new(output_df, 'check_viorate_pa.csv')

# ================================================================
# implies_exper / is_complete / reducers (Sum/Mean/Max/Min/Median)
# ================================================================

def test_implies_exper_string() -> None:
    assert eda_review.implies_exper("A", "B") == "B | ~(A)"

def test_is_complete_dataframe_and_series() -> None:
    df = pd.DataFrame({"a": [1, None], "b": [2, 3]})
    out_df = eda_review.is_complete(df)
    assert out_df.tolist() == [True, False]

    s1 = pd.Series([1, None])
    s2 = pd.Series([2, 3])
    out_s = eda_review.is_complete(s1, s2)
    assert out_s.tolist() == [True, False]

def test_reducers() -> None:
    a = pd.Series([1, 2, np.nan])
    b = pd.Series([10, 20, 30])
    assert eda_review.Sum(a, b).tolist() == [11, 22, 30]
    assert eda_review.Mean(a, b).iloc[0] == pytest.approx(5.5)
    assert eda_review.Max(a, b).tolist() == [10, 20, 30]
    assert eda_review.Min(a, b).tolist() == [1, 2, 30]  # nan は無視される
    assert eda_review.Median(a, b).iloc[0] == pytest.approx(5.5)

# ================================================================
# review_wrangling
# ================================================================
old = pd.read_csv(f'{tests_path}/fixtures/penguins.csv')
new = penguins.copy().dropna()
s = new['body_mass_g']
new['heavy'] = np.where(s >= s.quantile(0.75), True, False)
new['species'] = pd.Categorical(new['species'])
new['year'] = new['year'].astype(float)
new['const'] = 1
new['flipper_length_mm'] = eda_ops.set_miss(
    new['flipper_length_mm'], prop = 0.1,
    random_state = 123
    )

df_modify = {
    'pd':(old, new),
    'pl':(pl.from_pandas(old), pl.from_pandas(new)),
    'pa':(pa.Table.from_pandas(old), pa.Table.from_pandas(new))
}
update_fixture = False
# update_fixture = False

@pytest.mark.parametrize("backend", list_backend)
def test_review_wrangling(backend):
    output = eda_review.review_wrangling(*df_modify.get(backend))

    if update_fixture:
        with open(f'{tests_path}/fixtures/review_wrangling_{backend}.txt', 'w', encoding='utf-8') as f:
            f.write(output)
    
    with open(f'{tests_path}/fixtures/review_wrangling_{backend}.txt', 'r', encoding='utf-8') as f:
            expected = f.read()
    
    assert output == expected


@pytest.mark.parametrize("backend", list_backend)
def test_review_numeric(backend):
    output = eda_review.review_numeric(*df_modify.get(backend))

    if update_fixture:
        with open(f'{tests_path}/fixtures/review_numeric_{backend}.txt', 'w', encoding='utf-8') as f:
            f.write(output)
    
    with open(f'{tests_path}/fixtures/review_numeric_{backend}.txt', 'r', encoding='utf-8') as f:
            expected = f.read()
    
    assert output == expected
