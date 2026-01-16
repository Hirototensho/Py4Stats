# tests/test_bilding_block.py
import pytest
import pandas as pd
import polars as pl

from py4stats import building_block as build
from contextlib import nullcontext

@pytest.mark.parametrize(
    "lower, upper, inclusive, expectation",
    [
        pytest.param(0, 1, "both", pytest.raises(ValueError, match = "0 <= x <= 1"), id = "an_0_1_b"),
        pytest.param(0, 4, "left", pytest.raises(ValueError, match = "0 <= x < 4"), id = "an_0_4_l"),
        pytest.param(-9, 1, "right", pytest.raises(ValueError, match = "-9 < x <= 1"), id = "an_9_1_r"),
        pytest.param(0, 1, "neither", pytest.raises(ValueError, match = "0 < x < 1"), id = "an_0_1_n"),

    ],
)
def test_assert_numeric_massage(lower, upper, inclusive, expectation):
    with expectation:
        build.assert_numeric(
                arg = 50, arg_name = 'x',
                lower = lower,
                upper = upper,
                inclusive = inclusive
                )