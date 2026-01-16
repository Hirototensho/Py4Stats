# tests/test_bilding_block.py
import pytest
import pandas as pd
import polars as pl

from py4stats import building_block as build
from contextlib import nullcontext

# =========================================================
# match_arg / arg_match0 / arg_match
# =========================================================

def test_match_arg_exact_match():
    assert build.match_arg("apple", ["apple", "orange"]) == "apple"

def test_match_arg_partial_unique_match_case_insensitive():
    assert build.match_arg("App", ["apple", "orange"]) == "apple"

def test_match_arg_ambiguous_raises_value_error():
    with pytest.raises(ValueError):
        build.match_arg("a", ["apple", "apricot", "orange"], arg_name="fruit")

def test_match_arg_no_match_raises_value_error():
    with pytest.raises(ValueError, match=r"must be one of"):
        build.match_arg("zzz", ["apple", "orange"], arg_name="fruit")


def test_arg_match0_exact_match():
    assert build.arg_match0("mean", ["mean", "median"], arg_name="stat") == "mean"

def test_arg_match0_suggests_candidates_on_partial_match():
    with pytest.raises(ValueError) as excinfo:
        build.arg_match0("mea", ["mean", "median"], arg_name="stat")
    # 仕様：候補があると "Did you mean ..." を含む
    assert "Did you mean" in str(excinfo.value)

def test_arg_match0_no_suggestion_when_none():
    with pytest.raises(ValueError) as excinfo:
        build.arg_match0("zzz", ["mean", "median"], arg_name="stat")
    assert "Did you mean" not in str(excinfo.value)


def test_arg_match_single_returns_str():
    out = build.arg_match("mean", ["mean", "median"], arg_name="stat", multiple=False)
    assert isinstance(out, str)
    assert out == "mean"

def test_arg_match_multiple_returns_list():
    out = build.arg_match(["mean", "median"], ["mean", "median"], arg_name="stat", multiple=True)
    assert out == ["mean", "median"]


# =========================================================
# type predicate functions
# =========================================================

@pytest.mark.parametrize(
    "x, expected",
    [
        (["a", "b"], True),
        (pd.Series(["a", "b"]), True),
        (["a", 1], False),
    ],
)
def test_is_character(x, expected):
    assert build.is_character(x) is expected

@pytest.mark.parametrize(
    "x, expected",
    [
        ([True, False], True),
        (pd.Series([True, False]), True),
        ([1, 0], False),
    ],
)
def test_is_logical(x, expected):
    assert build.is_logical(x) is expected

@pytest.mark.parametrize(
    "x, expected",
    [
        ([1, 2, 3], True),
        ([1.0, 2.0], True),
        (["1", "2"], False),
    ],
)
def test_is_numeric(x, expected):
    assert build.is_numeric(x) is expected

# =========================================================
# length
# =========================================================

@pytest.mark.parametrize(
    "x, expected",
    [
        (None, 0),
        ('str', 1),
        ([1], 1),
        ([1, 2], 2),
        ([1, 2, 3], 3),
        ([1, pd.NA, None], 3)
    ],
)
def test_length(x, expected):
    assert build.length(x) is expected

# =========================================================
# assert_length
# =========================================================

def test_assert_length_len_arg():
    l = ['a', 'b', 'c']
    with pytest.raises(ValueError):
        build.assert_length(l, arg_name = 'l', len_arg = 1)

def test_assert_length_len_max():
    l = ['a', 'b', 'c']
    with pytest.raises(ValueError):
        build.assert_length(l, arg_name = 'l', len_arg = 2)

def test_assert_length_len_min():
    l = ['a', 'b', 'c']
    with pytest.raises(ValueError):
        build.assert_length(l, arg_name = 'l', len_arg = 4)

# =========================================================
# assert_scalar
# =========================================================
def test_assert_scalar_not_raise():
    assert build.assert_scalar('x') is None
    assert build.assert_scalar(1) is None
    assert build.assert_scalar(True) is None

def test_assert_scalar_raise():
    with pytest.raises(ValueError):
        build.assert_scalar(['x'])

# =========================================================
# assert_missing
# =========================================================
def test_assert_missing_any_missing_False():
    arg = [1, 2 ,3, None, pd.NA]
    with pytest.raises(ValueError) as excinfo:
        build.assert_missing(arg, 'arg')
    
    assert "contains missing values (element '3' and '4')" in str(excinfo.value)

def test_assert_missing_any_missing_True():
    arg = [1, 2 ,3, None, pd.NA]
    assert build.assert_missing(arg, 'arg', any_missing = True) is None

def test_assert_missing_all_missing_False():
    arg = [None, pd.NA, pl.Null]
    with pytest.raises(ValueError) as excinfo:
        build.assert_missing(arg, 'arg')
    
    assert "contains only missing values" in str(excinfo.value)

# =========================================================
# assert_numeric_dtype (raises on invalid)
# =========================================================
@pytest.mark.parametrize(
    "arg, predicate_fun, expectation",
    [
        pytest.param([1, 0.1], build.is_numeric, nullcontext(), id = "numeric_ok"),
        pytest.param([1, '0.1'], build.is_numeric, pytest.raises(ValueError), id = "numeric_ng"),
        pytest.param([1, 0.1], build.is_float, nullcontext(), id = "float_ok"),
        pytest.param([1, 2], build.is_float, pytest.raises(ValueError), id = "float_ng"),
        pytest.param([1, 2], build.is_integer, nullcontext(), id = "int_ok"),
        pytest.param([1, 0.1], build.is_integer, pytest.raises(ValueError), id = "int_ng"),
    ],
)
def test_assert_numeric_dtype(arg, predicate_fun, expectation):
    with expectation:
        build.assert_numeric_dtype(
            arg, arg_name = 'x', 
            predicate_fun = predicate_fun, 
            valid_type = ['test']
        )

# =========================================================
# assert_value_range
# =========================================================
@pytest.mark.parametrize(
    "lower, upper, inclusive, expectation",
    [
        pytest.param(0, 3, "both", nullcontext(), id = "range_0_3_b"),
        pytest.param(1, 3, "both", pytest.raises(ValueError, match = "element '0'"), id = "range_1_3_b"),
        pytest.param(0, 3, "right", pytest.raises(ValueError, match = "element '0'"), id = "range_1_3_r"),
        pytest.param(0, 2, "right", pytest.raises(ValueError, match = "element '0' and '3'"), id = "range_0_2_b"),
        pytest.param(0, 3, "left", pytest.raises(ValueError, match = "element '3'"), id = "range_0_3_l"),
        pytest.param(-1, 4, "neither", nullcontext(), id = "range_-1_4_n"),
        pytest.param(1, 4, "neither", pytest.raises(ValueError, match = "element '0' and '1'"), id = "range_-1_4_n"),
    ],
)
def test_assert_value_range(lower, upper, inclusive, expectation):
    x = [0, 1, 2, 3]
    with expectation:
        build.assert_value_range(
                arg = x, arg_name = 'x',
                lower = lower,
                upper = upper,
                inclusive = inclusive
                )

# =========================================================
# assert_dtypes (raises on invalid)
# =========================================================

def test_assert_character_raises():
    with pytest.raises(ValueError):
        build.assert_character([1, 2, 3], arg_name="x")

def test_assert_logical_raises():
    with pytest.raises(ValueError):
        build.assert_logical([1, 0], arg_name="x")

def test_assert_numeric_passes_on_numeric():
    build.assert_numeric([1, 2, 3], arg_name="x")

def test_assert_numeric_raises_on_non_numeric():
    with pytest.raises(ValueError):
        build.assert_numeric(["a", "b"], arg_name="x")

@pytest.mark.parametrize("inclusive", ["both", "neither", "left", "right"])
def test_assert_numeric_inclusive_argument_accepted(inclusive):
    # inclusive 引数が受理されることを確認（範囲が十分広いので通る）
    build.assert_numeric([0.5, 0.9], lower=0.0, upper=1.0, inclusive=inclusive, arg_name="x")

def test_assert_numeric_range_raises():
    with pytest.raises(ValueError):
        build.assert_numeric([0, 2], lower=0, upper=1, inclusive="both", arg_name="x")

def test_assert_count_requires_nonnegative_integer():
    build.assert_count([0, 1, 10], arg_name="n")
    with pytest.raises(ValueError):
        build.assert_count([-1, 1], arg_name="n")

# =========================================================
# assert_numeric (value range message)
# =========================================================

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

# =========================================================
# p_stars / style_pvalue
# =========================================================

def test_p_stars_default_mapping():
    s = build.p_stars([0.001, 0.03, 0.08, 0.2])
    assert list(s) == ["***", "**", "*", ""]

def test_p_stars_custom_mapping():
    s = build.p_stars([0.04, 0.2], stars={"!": 0.05})
    assert list(s) == ["!", ""]

def test_style_pvalue_rounding_and_prefix():
    s = build.style_pvalue([0.01234, 0.5], digits=2, prepend_p=True)
    assert list(s) == ['p=0.01', 'p=0.5']

def test_style_pvalue_clipping():
    s = build.style_pvalue([1e-6, 0.95], p_min=0.001, p_max=0.9)
    assert list(s) == ["<0.001", ">0.9"]

# =========================================================
# number formatting (vectorized + Series returning)
# =========================================================

def test_num_comma_scalar():
    assert build.num_comma(1234.5, digits=1) == "1,234.5"

def test_num_currency_scalar():
    assert build.num_currency(1234, symbol="¥", digits=0) == "¥1,234"

def test_num_percent_scalar():
    assert build.num_percent(0.125, digits=1) == "12.5%"

def test_style_number_series():
    s = build.style_number([1, 2.345], digits=2)
    assert isinstance(s, pd.Series)
    assert list(s) == ["1.00", "2.35"]

def test_style_currency_series():
    s = build.style_currency([1000, 2500], symbol="$", digits=0)
    assert list(s) == ["$1,000", "$2,500"]

def test_style_percent_series():
    s = build.style_percent([0.1, 0.25], digits=1, unit=100, symbol="%")
    assert list(s) == ["10.0%", "25.0%"]


# =========================================================
# pad_zero / add_big_mark
# =========================================================

def test_pad_zero_adds_trailing_zeros_for_decimals():
    assert build.pad_zero("1.2", digits=3) == "1.200"
    assert build.pad_zero("1.23", digits=3) == "1.230"

def test_pad_zero_keeps_integers():
    assert build.pad_zero("10", digits=3) == "10"

def test_add_big_mark():
    assert build.add_big_mark(1234567) == "1,234,567"


# =========================================================
# Oxford comma helpers
# =========================================================

def test_oxford_comma_single_string():
    assert build.oxford_comma("apple", sep_last="and", quotation=True) == "'apple'"

def test_oxford_comma_two_items():
    assert build.oxford_comma(["apple", "orange"], sep_last="and", quotation=False) == "apple and orange"

def test_oxford_comma_three_items():
    assert build.oxford_comma(["apple", "orange", "grape"], sep_last="or", quotation=True) == "'apple', 'orange' or 'grape'"

def test_oxford_comma_and_or_wrappers():
    assert " and " in build.oxford_comma_and(["a", "b"], quotation=False)
    assert " or " in build.oxford_comma_or(["a", "b"], quotation=False)
