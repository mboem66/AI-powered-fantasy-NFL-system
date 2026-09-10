"""Protect the column contract as source files and configuration change."""

import polars as pl
import pytest

from fantasy_football.data.columns import load_column_selection, select_columns


def test_selection_removes_extras_without_changing_rows_or_types():
    frame = pl.DataFrame({"weekday": ["Sunday"], "season": [2025], "gameday": ["2025-09-07"]})
    selected = select_columns(frame, "schedules", {"schedules": ["gameday", "season"]})
    assert selected.columns == ["gameday", "season"]
    assert selected.to_dicts() == [{"gameday": "2025-09-07", "season": 2025}]
    assert selected.schema["season"] == frame.schema["season"]


def test_missing_column_fails_instead_of_silently_disappearing():
    with pytest.raises(ValueError, match="game_id"):
        select_columns(pl.DataFrame({"season": [2025]}), "schedules", {"schedules": ["game_id"]})


def test_duplicate_configuration_names_fail(tmp_path):
    path = tmp_path / "columns.toml"
    path.write_text('schedules = ["season", "season"]', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates"):
        load_column_selection(path)
