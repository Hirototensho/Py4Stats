
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
# as_nw_datarame
# =========================================================

@pytest.mark.parametrize("data, expectation", [
    pytest.param(adelie,    nullcontext()),
    pytest.param(adelie_pl, nullcontext()),
    pytest.param(adelie_pa, nullcontext()),
    pytest.param(penguins_dict, nullcontext()),
    pytest.param(list(penguins_dict.values()), nullcontext()),
    pytest.param([adelie, gentoo, '3'],  pytest.raises(TypeError, match=r"supported by narwhals")),
    pytest.param(np.array([1, 2, 3]), pytest.raises(TypeError, match=r"supported by narwhals")),
    pytest.param(123,                 pytest.raises(TypeError, match=r"supported by narwhals")),
    pytest.param(None,                pytest.raises(TypeError, match=r"supported by narwhals")),
])

def test_compare_df_cols(data, expectation):
    with expectation:
        eda_util.as_nw_datarame(data)

# =========================================================
# enframe
# =========================================================
data_list = [
    1, 'a', True, None, 
    [1, 2, 3],
    (1, 2, 3),
    pd.Series([1, 2, 3]),
    pd.Series([1, 2, 3]).to_dict()
]
@pytest.mark.parametrize("data", data_list)

def test_enframe(data):
    res = eda_util.enframe(data)
    assert type(res).__name__ == 'DataFrame'
    assert 'name' in res.columns and 'value' in res.columns 

@pytest.mark.parametrize("backend", list_backend)
def test_enframe_table(backend):
    tbl = nw.from_native(penguins_dict.get(backend))
    
    res = eda_util.enframe(
        tbl.select(ncs.numeric().mean()),
        to_native = False
        )
    assert type(res).__name__ == 'DataFrame'
    assert 'name' in res.columns and 'value' in res.columns 
