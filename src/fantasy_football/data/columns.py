"""Select explicitly approved source columns before saving a dataset."""

from pathlib import Path
import tomllib

import polars as pl

from fantasy_football.config import PROJECT_ROOT


def load_column_selection(path: Path | None = None) -> dict[str, list[str]]:
    """Keep column choices in a readable file instead of inside download code."""
    path = path or PROJECT_ROOT / "config" / "columns.toml"
    with path.open("rb") as stream:
        selections = tomllib.load(stream)
    for name, columns in selections.items():
        if not isinstance(columns, list) or not columns or not all(isinstance(c, str) for c in columns):
            raise ValueError(f"{name}: expected a nonempty list of column names")
        if len(columns) != len(set(columns)):
            raise ValueError(f"{name}: column selection contains duplicates")
    return selections


def select_columns(frame: pl.DataFrame, name: str, selections: dict[str, list[str]]) -> pl.DataFrame:
    """Reject missing expected fields rather than silently changing the dataset."""
    columns = selections[name]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: expected columns missing from source: {missing}")
    # Selecting by name also gives every season the same predictable column order.
    return frame.select(columns)
