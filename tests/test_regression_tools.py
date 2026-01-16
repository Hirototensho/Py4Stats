# tests/regression_tools.py
import pytest
import pandas as pd
import numpy as np
import wooldridge
from pandas.testing import assert_frame_equal
# from palmerpenguins import load_penguins
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from py4stats import regression_tools as reg # 回帰分析の要約

import pathlib
tests_path = pathlib.Path(__file__).parent

# penguins = load_penguins() # サンプルデータの読み込み
penguins = pd.read_csv(f'{tests_path}/fixtures/penguins.csv')
wage1 = wooldridge.data('wage1')
mroz = wooldridge.data('mroz')
adelie = penguins.query("species == 'Adelie'")
gentoo = penguins.query("species == 'Gentoo'")

# 回帰分析の実行
fit1 = smf.ols('body_mass_g ~ bill_length_mm + species', data = penguins).fit()
fit2 = smf.ols('body_mass_g ~ bill_length_mm + bill_depth_mm + species', data = penguins).fit()
fit3 = smf.ols('body_mass_g ~ bill_length_mm + bill_depth_mm + species + sex', data = penguins).fit()

penguins['female'] = np.where(penguins['sex'] == 'female', 1, 0)
# ロジスティック回帰の実行
fit_logit1 = smf.logit('female ~ body_mass_g + bill_length_mm + bill_depth_mm', data = penguins).fit()
fit_logit2 = smf.logit('female ~ body_mass_g + bill_length_mm + bill_depth_mm + species', data = penguins).fit()


# =========================================================
# tidy
# =========================================================

def test_tidy_regression() -> None:
    output_df = reg.tidy(fit1)
    # output_df.to_csv(f'{tests_path}/fixtures/tidy_regression.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/tidy_regression.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

def test_tidy_mfx() -> None:
    output_df = reg.tidy_mfx(fit_logit1)
    # output_df.to_csv(f'{tests_path}/fixtures/tidy_mfx.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/tidy_mfx.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

def test_tidy_test() -> None:
    hypotheses = 'bill_length_mm = 20'
    output_df = reg.tidy(fit3.t_test(hypotheses))
    # output_df.to_csv(f'{tests_path}/fixtures/tidy_test1.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/tidy_test1.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)


    hypotheses = 'species[T.Chinstrap] = 0, species[T.Gentoo] = 0'
    output_df = reg.tidy(fit3.f_test(hypotheses))
    # output_df.to_csv(f'{tests_path}/fixtures/tidy_test2.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/tidy_test2.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

# =========================================================
# glance
# =========================================================

def test_glance() -> None:
    output_df = reg.glance(fit1)
    # output_df.to_csv(f'{tests_path}/fixtures/glance_ols.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/glance_ols.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

    output_df = reg.glance(fit_logit1)
    # output_df.to_csv(f'{tests_path}/fixtures/glance_glm.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/glance_glm.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

# =========================================================
# gazer / lineup_models
# =========================================================

def test_gazer() -> None:
    output_df = reg.gazer(reg.tidy(fit1))
    # output_df.to_csv(f'{tests_path}/fixtures/gazer.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/gazer.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

def test_gazer_change_stars() -> None:
    output_df = reg.gazer(
        reg.tidy(fit3),
        stars = {'★★★':0.001, '★★':0.01, '★': 0.05, '.':0.1}
    )
    expected_df = pd.read_csv(f'{tests_path}/fixtures/gazer_stars.csv', index_col = 0)
    assert_frame_equal(output_df, expected_df)

def test_lineup_models() -> None:
    list_models = [fit1, fit2, fit3]
    gazer_list = [reg.gazer(reg.tidy(mod)) for mod in list_models]
    output_df = reg.lineup_models(gazer_list)
    # output_df.to_csv(f'{tests_path}/fixtures/lineup_models.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/lineup_models.csv', index_col = 0).fillna('')
    assert_frame_equal(output_df, expected_df)

# =========================================================
# compare_ols
# =========================================================

def test_compare_ols() -> None:
    compare_tab1 = reg.compare_ols(list_models = [fit1, fit2, fit3]) # 表の作成
    # compare_tab1.to_csv(f'{tests_path}/fixtures/compare_ols1.csv')

    expected_df = pd.read_csv(f'{tests_path}/fixtures/compare_ols1.csv', index_col = 0).fillna('')
    assert_frame_equal(compare_tab1, expected_df)

    compare_tab2 = reg.compare_ols(
    list_models = [fit1, fit2, fit3],
    model_name = ['基本モデル', '嘴の高さ追加', '性別追加'], # モデル名を変更
    stats = 'p_value',        # () 内の値をP-値に変更する
    add_stars = False,        # 有意性のアスタリスクなし
    table_style = 'one_line', # 表スタイルを1行表示に設定 'one' でも可能
    digits = 3                # 小数点以下の桁数を3に設定
    )
    # compare_tab2.to_csv(f'{tests_path}/fixtures/compare_ols2.csv')
    expected_df = pd.read_csv(f'{tests_path}/fixtures/compare_ols2.csv', index_col = 0).fillna('')
    assert_frame_equal(compare_tab2, expected_df)

def test_compare_ols_mixed_type_models():
    output_df = reg.compare_ols(
        [fit1, fit_logit1],
        stats_glance = [
            'rsquared_adj', 
            'prsquared', 'AIC',
            'nobs', 'df'
            ]
        )
    # output_df.to_csv(f'{tests_path}/fixtures/compare_ols3.csv')

    expected_df = pd.read_csv(f'{tests_path}/fixtures/compare_ols3.csv', index_col = 0).fillna('')
    assert_frame_equal(output_df, expected_df)

def test_make_glance_tab_rejects_invalid_stats_glance():
    # 存在しない指標を指定 → エラーになるべき
    with pytest.raises(ValueError):
        reg.make_glance_tab(
            [fit1, fit_logit1],
            stats_glance=["rsquared_adj", 'prsquared', "not_exist"]
        )

# =========================================================
# compare_mfx
# =========================================================

def test_compare_mfx() -> None:

    compare_tab2 = reg.compare_mfx([fit_logit1, fit_logit2])
    # compare_tab2.to_csv(f'{tests_path}/fixtures/compare_mfx.csv')

    expected_df = pd.read_csv(f'{tests_path}/fixtures/compare_mfx.csv', index_col = 0).fillna('')
    assert_frame_equal(compare_tab2, expected_df)

# =========================================================
# Blinder_Oaxaca
# =========================================================

def test_Blinder_Oaxaca() -> None:
    fit_female = smf.ols(
        'lwage ~ educ + exper + expersq + tenure + tenursq + married', 
        data = wage1.query('female == 1')
        ).fit()

    fit_male = smf.ols(
        'lwage ~ educ + exper + expersq + tenure + tenursq + married', 
        data = wage1.query('female == 0')
        ).fit()

    wage_decomp = reg.Blinder_Oaxaca(
        model1 = fit_female,
        model2 = fit_male
    )

    # wage_decomp.to_csv(f'{tests_path}/fixtures/Blinder_Oaxaca.csv')

    expected_df = pd.read_csv(f'{tests_path}/fixtures/Blinder_Oaxaca.csv', index_col = 0)
    assert_frame_equal(wage_decomp, expected_df)