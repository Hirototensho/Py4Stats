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
# expand
# ================================================================

implicit_miss = pd.DataFrame({
    'name': 3 * ['A'] + 3 * ['B'],
    'time': [1, 2, 3, 1, 3, 4],
    'x': [0.82, 0.21, 0.74, 0.63, 0.93, None]
})

implicit_miss_dict = {
    'pd': implicit_miss,
    'pl': pl.from_pandas(implicit_miss),
    'pa': pa.Table.from_pandas(implicit_miss)
}

@pytest.mark.parametrize("backend", list_backend)
def test_expand(backend):
    path = f'{tests_path}/fixtures/expand_{backend}.csv'

    output_df = eda_ops.expand(
        implicit_miss_dict.get(backend), 
        'name', 'time',
        to_native = False
        )
    tfun._assert_df_eq(
        output_df, path_fixture = path, 
        update_fixture = False
    )

# ================================================================
# complete
# ================================================================

@pytest.mark.parametrize(
    "backend, fill, explicit, test_id",
[
    ('pd', None, True, 1), 
    ('pd', {'x': 0}, True, 2), 
    ('pd', {'x': 0}, False,3), 
    ('pl', None, True, 1),  
    ('pl', {'x': 0}, True, 2), 
    ('pl', {'x': 0}, False, 3), 
    ('pa', None, True, 1),  
    ('pa', {'x': 0}, True, 2), 
    ('pa', {'x': 0}, False, 3)
 ]
)
def test_complete(backend, fill, explicit, test_id):
    path = f'{tests_path}/fixtures/complete_{backend}_{test_id}.csv'

    output_df = eda_ops.complete(
        implicit_miss_dict.get(backend), 'name', 'time',
        fill = fill,
        explicit = explicit,
        to_native = False
        )
    tfun._assert_df_eq(
        output_df, path_fixture = path, 
        update_fixture = False
    )