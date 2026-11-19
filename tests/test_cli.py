"""Integration tests between the CLI (src/cli.py) and the logic."""

import pytest
from click.testing import CliRunner

from src.cli import cli


@pytest.fixture
def runner():
    """CliRunner shared by all the tests."""
    return CliRunner()


def run(runner, args):
    """Invoke the CLI and check it finished correctly."""
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    return result.output.strip()


def test_help_lists_groups(runner):
    output = run(runner, ["--help"])
    for group in ("clean", "numeric", "text", "struct"):
        assert group in output


def test_remove_missing(runner):
    assert run(runner, ["clean", "remove-missing", "a", "None", "", "nan", "b"]) == (
        "['a', 'b']"
    )


@pytest.mark.parametrize(
    "extra, expected",
    [
        ([], "['a', '0', '0', 'b']"),
        (["--fill-value", "X"], "['a', 'X', 'X', 'b']"),
    ],
)
def test_fill_missing(runner, extra, expected):
    args = ["clean", "fill-missing", "a", "None", "nan", "b"] + extra
    assert run(runner, args) == expected


def test_normalize_default(runner):
    out = run(runner, ["numeric", "normalize", "1", "2", "3"])
    assert out == "[0.0, 0.5, 1.0]"


def test_normalize_options(runner):
    args = ["numeric", "normalize", "1", "2", "3", "--new-min", "0", "--new-max", "10"]
    assert run(runner, args) == "[0.0, 5.0, 10.0]"


def test_standardize(runner):
    out = run(runner, ["numeric", "standardize", "1", "1", "1"])
    assert out == "[0.0, 0.0, 0.0]"


@pytest.mark.parametrize(
    "args, expected",
    [
        (["-2", "0.5", "7"], "[0.0, 0.5, 1.0]"),
        (
            ["-2", "0.5", "7", "--min-value", "-1", "--max-value", "5"],
            "[-1.0, 0.5, 5.0]",
        ),
    ],
)
def test_clip(runner, args, expected):
    assert run(runner, ["numeric", "clip"] + args) == expected


def test_clip_invalid_range(runner):
    result = runner.invoke(
        cli, ["numeric", "clip", "1", "--min-value", "5", "--max-value", "1"]
    )
    assert result.exit_code != 0
    assert "min_value" in result.output


def test_to_int(runner):
    assert run(runner, ["numeric", "to-int", "1", "abc", "4"]) == "[1, 4]"


def test_log(runner):
    assert run(runner, ["numeric", "log", "1", "-5", "1"]) == "[0.0, 0.0]"


def test_tokenize(runner):
    out = run(runner, ["text", "tokenize", "Hello, World!"])
    assert out == "['hello', 'world']"


def test_remove_punctuation(runner):
    assert run(runner, ["text", "remove-punctuation", "Hi, you!"]) == "Hi you"


def test_remove_stopwords(runner):
    args = ["text", "remove-stopwords", "The cat and the dog"]
    args += ["--stop-word", "the", "--stop-word", "and"]
    assert run(runner, args) == "cat dog"

def test_shuffle_with_seed_is_reproducible(runner):
    args = ["struct", "shuffle", "a", "b", "c", "d", "--seed", "42"]
    assert run(runner, args) == run(runner, args)


def test_flatten(runner):
    out = run(runner, ["struct", "flatten", "1,2", "3", "4,5"])
    assert out == "['1', '2', '3', '4', '5']"


def test_unique(runner):
    assert run(runner, ["struct", "unique", "a", "b", "a"]) == "['a', 'b']"