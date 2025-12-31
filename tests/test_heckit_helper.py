# tests/test_heckit_helper.py
import pytest
import pandas as pd
import numpy as np
import wooldridge
from pandas.testing import assert_frame_equal
from palmerpenguins import load_penguins
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from py4stats import regression_tools as reg # 回帰分析の要約
from py4stats import heckit_helper

import os
import pathlib

tests_path = pathlib.Path(__file__).parent

# os.chdir('/Users/tenshouhiroto/Python_DS/Py4Stats開発ノート/Py4Stats')

penguins = load_penguins() # サンプルデータの読み込み
wage1 = wooldridge.data('wage1')
mroz = wooldridge.data('mroz')


mod_heckit, exog_outcome, exog_select = heckit_helper.Heckit_from_formula(
    selection = 'lwage ~ educ + exper + expersq + nwifeinc + age + kidslt6 + kidsge6',
    outcome = 'lwage ~ educ + exper + expersq',
    data = mroz
)

res_heckit = mod_heckit.fit(cov_type_2 = 'HC1')

# =========================================================
# tidy_heckit
# =========================================================

def test_tidy_heckit() -> None:
    output_df = reg.tidy(res_heckit, name_selection = exog_select.columns)

    # output_df.to_csv(f'{tests_path}/fixtures/tidy_heckit.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/tidy_heckit.csv', index_col = 0)

    assert_frame_equal(output_df, expected_df)

# =========================================================
# heckitmfx / heckitmfx_compute
# =========================================================

def test_heckitmfx_compute() -> None:
    output_df = heckit_helper.heckitmfx_compute(
        res_heckit,
        exog_select = exog_select,
        exog_outcome = exog_outcome
        )

    # output_df.to_csv(f'{tests_path}/fixtures/heckitmfx_compute.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/heckitmfx_compute.csv', index_col = 0)

    assert_frame_equal(output_df, expected_df)

def test_heckitmfx() -> None:
    output_df = heckit_helper.heckitmfx(
        res_heckit,
        exog_select = exog_select,
        exog_outcome = exog_outcome
        )
    # output_df.to_csv(f'{tests_path}/fixtures/heckitmfx.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/heckitmfx.csv', index_col = 0)

    assert_frame_equal(output_df, expected_df)