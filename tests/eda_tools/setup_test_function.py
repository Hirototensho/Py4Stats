# tests/setup_test_function.py

from py4stats.eda_tools import operation as eda_ops
import pytest
import pandas as pd
import numpy as np
# from palmerpenguins import load_penguins

from pandas.testing import assert_frame_equal
import polars as pl
import pyarrow as pa

import narwhals as nw
import narwhals.selectors as ncs

from typing import (Literal)

import pathlib

tests_path = pathlib.Path(__file__).parent.parent

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
    
    result = eda_ops.compare_df_record(output_df, expected_df).all().all()

    assert result

# =========================================================
# 実験的実装
# =========================================================
from narwhals import testing as nw_test

def _assert_df_eq(
        output_df,  path_fixture: str,  
        check_dtype: bool = False, 
        reset_index: bool = True, 
        update_fixture: bool = False,
        read_by: Literal['narwhals', 'pandas'] = 'narwhals',
        write_by: Literal['narwhals', 'pandas'] = 'narwhals',
        **kwarg
        ) -> None:
    
    if not isinstance(output_df, nw.DataFrame):
        output_df = nw.from_native(output_df)

    if update_fixture:
        if write_by == 'narwhals':
            output_df.write_csv(path_fixture)
        elif write_by == 'pandas':
            if hasattr(output_df, 'to_pandas'): output_df = output_df.to_pandas()
            output_df.to_csv(path_fixture, index = False)

    
    if read_by == 'narwhals':
        expected_df = nw.read_csv(path_fixture, backend = output_df.implementation)
        expected_df = expected_df.to_pandas()
    elif read_by == 'pandas':
        expected_df = pd.read_csv(path_fixture)
    
    if hasattr(expected_df, 'to_pandas'): expected_df = expected_df.to_pandas()
    if hasattr(output_df, 'to_pandas'): output_df = output_df.to_pandas()

    if reset_index:
        output_df = output_df.reset_index(drop = True)
        expected_df = expected_df.reset_index(drop = True)

    assert_frame_equal(output_df, expected_df, check_dtype = check_dtype)

# 私の手元にある環境では、`narwhals.testing.assert_frame_equal()` が読み込めないので、
# 以下のコードはまだ使えません。

#     nw_test.assert_frame_equal(
#         left = output_df, 
#         right = expected_df, 
#         check_dtype = check_dtype,
#         **kwarg
#         )