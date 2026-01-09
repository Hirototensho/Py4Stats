# tests/test_bilding_block.py
import pytest
import pandas as pd
from py4stats import bilding_block as bild

# =========================================================
# match_arg / arg_match0 / arg_match
# =========================================================

def test_match_arg_exact_match():
    assert bild.match_arg("apple", ["apple", "orange"]) == "apple"

def test_match_arg_partial_unique_match_case_insensitive():
    assert bild.match_arg("App", ["apple", "orange"]) == "apple"

def test_match_arg_ambiguous_raises_value_error():
    with pytest.raises(ValueError):
        bild.match_arg("a", ["apple", "apricot", "orange"], arg_name="fruit")

def test_match_arg_no_match_raises_value_error():
    with pytest.raises(ValueError, match=r"must be one of"):
        bild.match_arg("zzz", ["apple", "orange"], arg_name="fruit")


def test_arg_match0_exact_match():
    assert bild.arg_match0("mean", ["mean", "median"], arg_name="stat") == "mean"

def test_arg_match0_suggests_candidates_on_partial_match():
    with pytest.raises(ValueError) as excinfo:
        bild.arg_match0("mea", ["mean", "median"], arg_name="stat")
    # 仕様：候補があると "Did you mean ..." を含む
    assert "Did you mean" in str(excinfo.value)

def test_arg_match0_no_suggestion_when_none():
    with pytest.raises(ValueError) as excinfo:
        bild.arg_match0("zzz", ["mean", "median"], arg_name="stat")
    assert "Did you mean" not in str(excinfo.value)


def test_arg_match_single_returns_str():
    out = bild.arg_match("mean", ["mean", "median"], arg_name="stat", multiple=False)
    assert isinstance(out, str)
    assert out == "mean"

def test_arg_match_multiple_returns_list():
    out = bild.arg_match(["mean", "median"], ["mean", "median"], arg_name="stat", multiple=True)
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
    assert bild.is_character(x) is expected

@pytest.mark.parametrize(
    "x, expected",
    [
        ([True, False], True),
        (pd.Series([True, False]), True),
        ([1, 0], False),
    ],
)
def test_is_logical(x, expected):
    assert bild.is_logical(x) is expected

@pytest.mark.parametrize(
    "x, expected",
    [
        ([1, 2, 3], True),
        ([1.0, 2.0], True),
        (["1", "2"], False),
    ],
)
def test_is_numeric(x, expected):
    assert bild.is_numeric(x) is expected


# =========================================================
# assert_* (raises on invalid)
# 注意：Python は -O で assert を無効化できるので、
# テストは通常モードで実行する前提です。
# =========================================================

def test_assert_character_raises():
    with pytest.raises(ValueError):
        bild.assert_character([1, 2, 3], arg_name="x")

def test_assert_logical_raises():
    with pytest.raises(ValueError):
        bild.assert_logical([1, 0], arg_name="x")

def test_assert_numeric_passes_on_numeric():
    bild.assert_numeric([1, 2, 3], arg_name="x")

def test_assert_numeric_raises_on_non_numeric():
    with pytest.raises(ValueError):
        bild.assert_numeric(["a", "b"], arg_name="x")

@pytest.mark.parametrize("inclusive", ["both", "neither", "left", "right"])
def test_assert_numeric_inclusive_argument_accepted(inclusive):
    # inclusive 引数が受理されることを確認（範囲が十分広いので通る）
    bild.assert_numeric([0.5, 0.9], lower=0.0, upper=1.0, inclusive=inclusive, arg_name="x")

def test_assert_numeric_range_raises():
    with pytest.raises(ValueError):
        bild.assert_numeric([0, 2], lower=0, upper=1, inclusive="both", arg_name="x")

def test_assert_count_requires_nonnegative_integer():
    bild.assert_count([0, 1, 10], arg_name="n")
    with pytest.raises(ValueError):
        bild.assert_count([-1, 1], arg_name="n")


# =========================================================
# p_stars / style_pvalue
# =========================================================

def test_p_stars_default_mapping():
    s = bild.p_stars([0.001, 0.03, 0.08, 0.2])
    assert list(s) == ["***", "**", "*", ""]

def test_p_stars_custom_mapping():
    s = bild.p_stars([0.04, 0.2], stars={"!": 0.05})
    assert list(s) == ["!", ""]

def test_style_pvalue_rounding_and_prefix():
    s = bild.style_pvalue([0.01234, 0.5], digits=2, prepend_p=True)
    assert list(s) == ['p=0.01', 'p=0.5']

def test_style_pvalue_clipping():
    s = bild.style_pvalue([1e-6, 0.95], p_min=0.001, p_max=0.9)
    assert list(s) == ["<0.001", ">0.9"]

# =========================================================
# number formatting (vectorized + Series returning)
# =========================================================

def test_num_comma_scalar():
    assert bild.num_comma(1234.5, digits=1) == "1,234.5"

def test_num_currency_scalar():
    assert bild.num_currency(1234, symbol="¥", digits=0) == "¥1,234"

def test_num_percent_scalar():
    assert bild.num_percent(0.125, digits=1) == "12.5%"

def test_style_number_series():
    s = bild.style_number([1, 2.345], digits=2)
    assert isinstance(s, pd.Series)
    assert list(s) == ["1.00", "2.35"]

def test_style_currency_series():
    s = bild.style_currency([1000, 2500], symbol="$", digits=0)
    assert list(s) == ["$1,000", "$2,500"]

def test_style_percent_series():
    s = bild.style_percent([0.1, 0.25], digits=1, unit=100, symbol="%")
    assert list(s) == ["10.0%", "25.0%"]


# =========================================================
# pad_zero / add_big_mark
# =========================================================

def test_pad_zero_adds_trailing_zeros_for_decimals():
    assert bild.pad_zero("1.2", digits=3) == "1.200"
    assert bild.pad_zero("1.23", digits=3) == "1.230"

def test_pad_zero_keeps_integers():
    assert bild.pad_zero("10", digits=3) == "10"

def test_add_big_mark():
    assert bild.add_big_mark(1234567) == "1,234,567"


# =========================================================
# Oxford comma helpers
# =========================================================

def test_oxford_comma_single_string():
    assert bild.oxford_comma("apple", sep_last="and", quotation=True) == "'apple'"

def test_oxford_comma_two_items():
    assert bild.oxford_comma(["apple", "orange"], sep_last="and", quotation=False) == "apple and orange"

def test_oxford_comma_three_items():
    assert bild.oxford_comma(["apple", "orange", "grape"], sep_last="or", quotation=True) == "'apple', 'orange' or 'grape'"

def test_oxford_comma_and_or_wrappers():
    assert " and " in bild.oxford_comma_and(["a", "b"], quotation=False)
    assert " or " in bild.oxford_comma_or(["a", "b"], quotation=False)
