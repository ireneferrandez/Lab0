"""Command line interface (click) for the pre-processing functions."""

import math

import click

from src import preprocessing as prep

NUMBERS = {"ignore_unknown_options": True}  # allows negative numbers as arguments
MISSING_STRINGS = {"none": None, "nan": math.nan}


def _to_missing(value: str):
    """Translate the strings 'None' / 'nan' into real missing values."""
    return MISSING_STRINGS.get(value.strip().lower(), value)


def _echo(result) -> None:
    """Print the result of a command."""
    click.echo(result)


@click.group()
def cli():
    """Command line interface for data pre-processing."""


@cli.group()
def clean():
    """Functions related to data cleaning."""


@cli.group()
def numeric():
    """Functions to deal with numerical attributes."""


@cli.group()
def text():
    """Functions to deal with textual information."""


@cli.group()
def struct():
    """Functions related to the structure of data."""


@clean.command("remove-missing")
@click.argument("values", nargs=-1)
def remove_missing(values):
    """Remove missing values ('None', 'nan' and empty strings).

    Example: cli clean remove-missing a None "" nan b
    """
    _echo(prep.remove_missing([_to_missing(v) for v in values]))


@clean.command("fill-missing")
@click.argument("values", nargs=-1)
@click.option(
    "--fill-value", default="0", show_default=True, help="Value used to fill gaps."
)
def fill_missing(values, fill_value):
    """Fill missing values ('None', 'nan' and empty strings).

    Example: cli clean fill-missing a None nan b --fill-value X
    """
    _echo(prep.fill_missing([_to_missing(v) for v in values], fill_value))


@numeric.command(context_settings=NUMBERS)
@click.argument("values", nargs=-1, type=float)
@click.option("--new-min", default=0.0, show_default=True, help="New minimum.")
@click.option("--new-max", default=1.0, show_default=True, help="New maximum.")
def normalize(values, new_min, new_max):
    """Normalize numbers with the min-max method.

    Example: cli numeric normalize 1 2 3 4 5 --new-min 0 --new-max 10
    """
    _echo(prep.normalize_min_max(list(values), new_min, new_max))


@numeric.command(context_settings=NUMBERS)
@click.argument("values", nargs=-1, type=float)
def standardize(values):
    """Standardize numbers with the z-score method.

    Example: cli numeric standardize 1 2 3 4 5
    """
    _echo(prep.standardize_z_score(list(values)))


@numeric.command(context_settings=NUMBERS)
@click.argument("values", nargs=-1, type=float)
@click.option("--min-value", default=0.0, show_default=True, help="Lower bound.")
@click.option("--max-value", default=1.0, show_default=True, help="Upper bound.")
def clip(values, min_value, max_value):
    """Clip numbers to the range [min-value, max-value].

    Example: cli numeric clip -2 0.5 7 --min-value 0 --max-value 1
    """
    try:
        _echo(prep.clip_values(list(values), min_value, max_value))
    except ValueError as error:
        raise click.UsageError(str(error)) from error


@numeric.command("to-int", context_settings=NUMBERS)
@click.argument("values", nargs=-1)
def to_int(values):
    """Convert values to integers (non-numerical values are discarded).

    Example: cli numeric to-int 1 2.7 abc 4
    """
    _echo(prep.convert_to_integers(values))


@numeric.command("log", context_settings=NUMBERS)
@click.argument("values", nargs=-1, type=float)
def log_scale(values):
    """Transform positive numbers to logarithmic scale.

    Example: cli numeric log 1 10 100 -5
    """
    _echo(prep.log_transform(list(values)))


@text.command()
@click.argument("content")
def tokenize(content):
    """Tokenize text into lower-cased alphanumeric words.

    Example: cli text tokenize "Hello, World! 2026"
    """
    _echo(prep.tokenize(content))


@text.command("remove-punctuation")
@click.argument("content")
def remove_punctuation(content):
    """Keep only alphanumeric characters and spaces.

    Example: cli text remove-punctuation "Hello, World!"
    """
    _echo(prep.select_alphanumeric(content))


@text.command("remove-stopwords")
@click.argument("content")
@click.option(
    "--stop-word",
    "stop_words",
    multiple=True,
    help="Stop-word to remove (repeat the option for several words).",
)
def remove_stopwords(content, stop_words):
    """Remove stop-words from a (lower-cased) text.

    Example: cli text remove-stopwords "The cat and the dog" --stop-word the --stop-word and
    """
    _echo(prep.remove_stopwords(content, stop_words))


@struct.command()
@click.argument("values", nargs=-1)
@click.option("--seed", type=int, default=None, help="Seed for reproducibility.")
def shuffle(values, seed):
    """Randomly shuffle a list of values.

    Example: cli struct shuffle a b c d --seed 42
    """
    _echo(prep.shuffle(list(values), seed))


@struct.command()
@click.argument("values", nargs=-1)
def flatten(values):
    """Flatten a list of lists. Each list is given as a comma-separated string.

    Example: cli struct flatten 1,2 3 4,5
    """
    _echo(prep.flatten([v.split(",") for v in values]))


@struct.command()
@click.argument("values", nargs=-1)
def unique(values):
    """Remove duplicated values.

    Example: cli struct unique a b a c b
    """
    _echo(prep.remove_duplicates(list(values)))


if __name__ == "__main__":
    cli()
