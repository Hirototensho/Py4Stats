# tests/memo.py
import pytest
import pandas as pd
import polars as pl

from py4stats import building_block as build
from contextlib import nullcontext

# =========================================================
# make_range_massage
# =========================================================

@pytest.mark.parametrize(
    "lower, upper, inclusive, expectation",
    [
        pytest.param(0, 1, "both", '0 <= x <= 1', id = "an_0_1_b"),
        pytest.param(0, 4, "left", "0 <= x < 4", id = "an_0_4_l"),
        pytest.param(-9, 1, "right", "-9 < x <= 1", id = "an_9_1_r"),
        pytest.param(0, 1, "neither", "0 < x < 1", id = "an_0_1_n"),

    ],
)
def test_make_range_massage(lower, upper, inclusive, expectation):
    res = build.make_range_massage(
            lower = lower,
            upper = upper,
            inclusive = inclusive
            )
    assert res == expectation