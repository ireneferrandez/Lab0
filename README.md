Assignment: Fundamentals of Continuous Integration

Small data pre-processing project with a command line interface (click),
linting (pylint), formatting (black) and testing (pytest).

## Installation

```bash
uv sync
```

## Project structure

```
Lab0/
├── pyproject.toml
├── pytest.ini
├── README.md
├── src/
│   ├── __init__.py
│   ├── cli.py
│   └── preprocessing.py
└── tests/
    ├── test_cli.py
    └── test_logic.py
```

## Usage

The CLI has four groups: `clean`, `numeric`, `text` and `struct`.

```bash
uv run python -m src.cli --help
uv run python -m src.cli clean remove-missing a None "" nan b
uv run python -m src.cli numeric normalize 1 2 3 --new-min 0 --new-max 10
```

## Quality checks

```bash
uv run python -m pylint src/*.py
uv run black src/*.py
uv run python -m pytest -v --cov=src
```