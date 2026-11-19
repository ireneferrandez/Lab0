"""Unit tests for src/preprocessing.py."""

import math

import pytest

from src import preprocessing as prep


@pytest.fixture
def numbers():
    """A reusable list of numbers."""
    return [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.fixture
def dirty_values():
    """A list with every kind of missing value."""
    return [1, None, "", math.nan, "a", 0]


def test_remove_missing(dirty_values):
    assert prep.remove_missing(dirty_values) == [1, "a", 0]


def test_remove_missing_empty():
    assert prep.remove_missing([]) == []


def test_fill_missing_default(dirty_values):
    assert prep.fill_missing(dirty_values) == [1, 0, 0, 0, "a", 0]


def test_fill_missing_custom(dirty_values):
    assert prep.fill_missing(dirty_values, "X") == [1, "X", "X", "X", "a", 0]


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1, 2, 2, 3, 1], [1, 2, 3]),
        (["a", "b", "a"], ["a", "b"]),
        ([], []),
        ([5], [5]),
    ],
)
def test_remove_duplicates(values, expected):
    assert prep.remove_duplicates(values) == expected


def test_normalize_default(numbers):
    assert prep.normalize_min_max(numbers) == [0.0, 0.25, 0.5, 0.75, 1.0]


@pytest.mark.parametrize(
    "new_min, new_max, expected",
    [
        (0.0, 10.0, [0.0, 2.5, 5.0, 7.5, 10.0]),
        (-1.0, 1.0, [-1.0, -0.5, 0.0, 0.5, 1.0]),
        (5.0, 5.0, [5.0] * 5),
    ],
)
def test_normalize_range(numbers, new_min, new_max, expected):
    assert prep.normalize_min_max(numbers, new_min, new_max) == pytest.approx(expected)


def test_normalize_constant_and_empty():
    assert prep.normalize_min_max([3, 3, 3]) == [0.0, 0.0, 0.0]
    assert prep.normalize_min_max([]) == []


def test_standardize(numbers):
    result = prep.standardize_z_score(numbers)
    assert sum(result) == pytest.approx(0.0)
    assert math.sqrt(sum(r**2 for r in result) / len(result)) == pytest.approx(1.0)


def test_standardize_constant_and_empty():
    assert prep.standardize_z_score([2, 2]) == [0.0, 0.0]
    assert prep.standardize_z_score([]) == []


@pytest.mark.parametrize(
    "min_value, max_value, expected",
    [
        (2, 4, [2, 2, 3, 4, 4]),
        (0, 10, [1, 2, 3, 4, 5]),
        (3, 3, [3, 3, 3, 3, 3]),
    ],
)
def test_clip_values(numbers, min_value, max_value, expected):
    assert prep.clip_values(numbers, min_value, max_value) == expected


def test_clip_invalid_range(numbers):
    with pytest.raises(ValueError):
        prep.clip_values(numbers, 5, 1)


@pytest.mark.parametrize(
    "values, expected",
    [
        (["1", "2", "3"], [1, 2, 3]),
        (["1", "abc", "4"], [1, 4]),
        (["2.7", "-3", "nan", "inf"], [2, -3]),
        (["a", "b"], []),
        ([], []),
    ],
)
def test_convert_to_integers(values, expected):
    assert prep.convert_to_integers(values) == expected


def test_log_transform():
    result = prep.log_transform([1, math.e, 0, -3])
    assert result == pytest.approx([0.0, 1.0])


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello, World!", ["hello", "world"]),
        ("snake_case 2026-11", ["snake", "case", "2026", "11"]),
        ("   ", []),
    ],
)
def test_tokenize(text, expected):
    assert prep.tokenize(text) == expected


def test_select_alphanumeric():
    assert prep.select_alphanumeric("Hi, there! 42?") == "Hi there 42"


def test_remove_stopwords():
    result = prep.remove_stopwords("The Cat and THE dog", ["the", "AND"])
    assert result == "cat dog"


def test_remove_stopwords_no_stopwords():
    assert prep.remove_stopwords("Hello World", []) == "hello world"


def test_flatten():
    assert prep.flatten([[1, 2], [], [3]]) == [1, 2, 3]


def test_shuffle_reproducible(numbers):
    first = prep.shuffle(numbers, seed=42)
    second = prep.shuffle(numbers, seed=42)
    assert first == second
    assert sorted(first) == numbers


def test_shuffle_does_not_modify_input(numbers):
    original = list(numbers)
    prep.shuffle(numbers, seed=1)
    assert numbers == original