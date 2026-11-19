"""Data pre-processing functions for numerical, textual and structured data."""

import math
import random
import re
from typing import Any, Iterable, Optional


def is_missing(value: Any) -> bool:
    """Return True if ``value`` is a missing value (None, '' or NaN)."""
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    return isinstance(value, float) and math.isnan(value)


def remove_missing(values: list) -> list:
    """Remove missing values (None, '' and NaN) from a list."""
    return [v for v in values if not is_missing(v)]


def fill_missing(values: list, fill_value: Any = 0) -> list:
    """Replace missing values (None, '' and NaN) with ``fill_value``."""
    return [fill_value if is_missing(v) else v for v in values]


def remove_duplicates(values: list) -> list:
    """Return the unique values of a list, keeping the order of appearance."""
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def normalize_min_max(
    values: list[float], new_min: float = 0.0, new_max: float = 1.0
) -> list[float]:
    """Rescale values to the range [new_min, new_max] using min-max."""
    if not values:
        return []
    old_min, old_max = min(values), max(values)
    if old_max == old_min:
        return [float(new_min)] * len(values)
    scale = (new_max - new_min) / (old_max - old_min)
    return [new_min + (v - old_min) * scale for v in values]


def standardize_z_score(values: list[float]) -> list[float]:
    """Standardize values with the z-score: (x - mean) / std (population std)."""
    if not values:
        return []
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    if std == 0:
        return [0.0] * len(values)
    return [(v - mean) / std for v in values]


def clip_values(values: list[float], min_value: float, max_value: float) -> list[float]:
    """Clip values so that they lie within [min_value, max_value]."""
    if min_value > max_value:
        raise ValueError("min_value cannot be greater than max_value")
    return [min(max(v, min_value), max_value) for v in values]


def convert_to_integers(values: Iterable[str]) -> list[int]:
    """Convert strings to integers, excluding the non-numerical ones."""
    result = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            try:
                result.append(int(float(value)))
            except (TypeError, ValueError, OverflowError):
                continue
    return result


def log_transform(values: list[float]) -> list[float]:
    """Natural logarithm of the strictly positive values."""
    return [math.log(v) for v in values if v > 0]


def tokenize(text: str) -> list[str]:
    """Split text into lower-cased words made only of alphanumeric characters."""
    return re.findall(r"[^\W_]+", text.lower())


def select_alphanumeric(text: str) -> str:
    """Keep only alphanumeric characters and spaces."""
    return "".join(c for c in text if c.isalnum() or c == " ")


def remove_stopwords(text: str, stop_words: Iterable[str]) -> str:
    """Lower-case the text and remove the given stop-words."""
    stop = {word.lower() for word in stop_words}
    return " ".join(w for w in text.lower().split() if w not in stop)


def flatten(list_of_lists: list[list]) -> list:
    """Flatten a list of lists into a single list."""
    return [item for sublist in list_of_lists for item in sublist]


def shuffle(values: list, seed: Optional[int] = None) -> list:
    """Return a shuffled copy of the list, ``seed`` makes it reproducible."""
    shuffled = list(values)
    random.Random(seed).shuffle(shuffled)
    return shuffled
