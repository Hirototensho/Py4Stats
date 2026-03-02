
# tests/test_eda_tools.py

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
from itertools import product

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
# Pareto_plot_nw
# (plot は「落ちないこと」と最低限の構造だけ確認)
# =========================================================

def test_Pareto_plot() -> None:
    penguins_modify = penguins.copy()
    penguins_modify['group'] = penguins_modify['species'] + '\n' + penguins_modify['island']
    
    fig, ax = plt.subplots()
    
    eda_vis.Pareto_plot(penguins_modify, group = 'group', ax = ax)
    assert len(ax.patches) > 0
    plt.close()

    fig, ax = plt.subplots()
    eda_vis.Pareto_plot(
        penguins_modify, group = 'group', 
        values = 'bill_length_mm',
        palette = ['#FF6F91', '#252525'],
        ax = ax
        )
    assert len(ax.patches) > 0
    plt.close()

    fig, ax = plt.subplots()
    eda_vis.Pareto_plot(
        penguins_modify, 
        values = 'bill_length_mm',
        group = 'group',
        aggfunc = lambda x: x.std(),
        ax = ax
        )
    assert len(ax.patches) > 0
    plt.close()

def test_make_rank_table_nw_error_on_non_exist_col():
    with pytest.raises(ValueError) as excinfo:
        eda_vis.make_rank_table(penguins, 'non_exists', 'body_mass_g')
    # 仕様：候補があると "Did you mean ..." を含む
    assert "must be one of" in str(excinfo.value)

def test_Pareto_plot_pl() -> None:
    penguins_modify = penguins.copy()
    penguins_modify['group'] = penguins_modify['species'] + '\n' + penguins_modify['island']
    
    fig, ax = plt.subplots()

    eda_vis.Pareto_plot(pl.from_pandas(penguins_modify), group = 'group', ax = ax)
    assert len(ax.patches) > 0
    plt.close()

def test_Pareto_plot_pa() -> None:
    penguins_modify = penguins.copy()
    penguins_modify['group'] = penguins_modify['species'] + '\n' + penguins_modify['island']
    
    fig, ax = plt.subplots()

    eda_vis.Pareto_plot(pa.Table.from_pandas(penguins_modify), group = 'group', ax = ax)
    assert len(ax.patches) > 0
    plt.close()

# ================================================================
# plot_mean_diff / plot_median_diff
# ================================================================
@pytest.mark.parametrize(
    "backend, stats_diff",
    list(product(list_backend, ['norm_diff', 'abs_diff', 'rel_diff']))
)
def test_plot_mean_diff(backend, stats_diff) -> None:
    fig, ax = plt.subplots()
    eda_vis.plot_mean_diff(
        adelie_dict.get(backend), 
        gentoo_dict.get(backend),
        stats_diff = stats_diff,
        ax = ax
    );
    assert len(ax.get_lines()) > 0 and len(ax.collections) > 0
    plt.close()

@pytest.mark.parametrize(
    "backend, stats_diff",
    list(product(list_backend, ['abs_diff', 'rel_diff']))
)
def test_plot_median_diff(backend, stats_diff) -> None:
    fig, ax = plt.subplots()
    eda_vis.plot_median_diff(
        adelie_dict.get(backend), 
        gentoo_dict.get(backend),
        stats_diff = stats_diff,
        ax = ax
    );
    assert len(ax.get_lines()) > 0 and len(ax.collections) > 0
    plt.close()



# =======================================================================
# plot_miss_var
# =======================================================================
@pytest.mark.parametrize('backend', list_backend)

def test_plot_miss_var(backend) -> None:
    fig, ax = plt.subplots()
    eda_vis.plot_miss_var(penguins_dict.get(backend), ax = ax)
    assert len(ax.patches) > 0
    plt.close()

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
    eda_vis.plot_category(data_pd, ax = ax)

    assert len(ax.patches) > 0
    plt.close()

def test_plot_category_pl() -> None:

    data_pl = pl.from_pandas(data)\
    .with_columns(
        pl.all().cast(pl.Enum(categ_list))
    )

    fig, ax = plt.subplots()
    eda_vis.plot_category(data_pl, ax = ax)

    assert len(ax.patches) > 0
    plt.close()

def test_plot_category_pa() -> None:

    data_pa = pa.Table.from_pandas(data)

    fig, ax = plt.subplots()
    eda_vis.plot_category(data_pa, sort_by = 'frequency', ax = ax)

    assert len(ax.patches) > 0
    plt.close()

# ================================================================
# test_plot_count_h
# ================================================================
@pytest.mark.parametrize('backend', list_backend)

def test_plot_count_h(backend) -> None:
    fig, ax = plt.subplots()
    eda_vis.plot_count_h(penguins_dict.get(backend), 'species', ax = ax)
    assert len(ax.patches) > 0
    plt.close()

    fig, ax = plt.subplots()
    eda_vis.plot_count_h(penguins_dict.get(backend), 'sex', ax = ax)
    assert len(ax.patches) > 0
    plt.close()
