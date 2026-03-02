# tests/memo.py

from py4stats.eda_tools import _utils as eda_util
from py4stats.eda_tools import operation as eda_ops
from py4stats.eda_tools import reviewing as eda_review
from py4stats.eda_tools import visualize as eda_vis

import pytest
import pandas as pd
import polars as pl


from py4stats import building_block as build
from contextlib import nullcontext

# tests/test_eda_tools.py
import pytest
from itertools import product
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
# ↓ テストコード実装の補助関数の読み込み
import eda_tools.setup_test_function as tfun


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

list_backend = ['pd', 'pl', 'pa']

# =========================================================
# テスト用関数の定義
# =========================================================

# ================================================================
# test_plot_count_h
# ================================================================
@pytest.mark.parametrize('backend', list_backend)

def test_plot_count_h(backend) -> None:
    fig, ax = plt.subplots()
    eda_vis.plot_count_h(penguins_dict.get(backend), 'species', ax = ax)
    assert len(ax.patches) > 0