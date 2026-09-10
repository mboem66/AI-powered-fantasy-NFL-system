"""Exercise failure paths without requiring network access."""

import polars as pl
import pytest

from fantasy_football.data.download import download_one
from fantasy_football.data.storage import save_parquet, validate_download


@pytest.fixture(autouse=True)
def small_column_contract(monkeypatch):
    # Small source fixtures test orchestration independently of the real schema.
    monkeypatch.setattr("fantasy_football.data.download.load_column_selection",
                        lambda: {"player_stats": ["season", "targets"]})


def test_failed_refresh_preserves_existing_data(tmp_path, monkeypatch):
    path = tmp_path / "data/raw/player_stats/2025.parquet"
    original = pl.DataFrame({"season": [2025], "targets": [8]})
    save_parquet(original, path)
    monkeypatch.setattr("fantasy_football.data.download.fetch_dataset", lambda *_: pl.DataFrame())
    result = download_one("player_stats", 2025, tmp_path, refresh=True)
    assert result["status"] == "failed"
    assert result["previous_file_preserved"]
    assert pl.read_parquet(path).equals(original)


def test_repeated_refresh_replaces_instead_of_appending(tmp_path, monkeypatch):
    frame = pl.DataFrame({"season": [2025], "targets": [8]})
    monkeypatch.setattr("fantasy_football.data.download.fetch_dataset", lambda *_: frame)
    for _ in range(2):
        assert download_one("player_stats", 2025, tmp_path, True)["status"] == "downloaded"
    assert pl.read_parquet(tmp_path / "data/raw/player_stats/2025.parquet").height == 1


def test_cached_tables_are_trimmed_without_network_access(tmp_path, monkeypatch):
    path = tmp_path / "data/raw/player_stats/2025.parquet"
    save_parquet(pl.DataFrame({"season": [2025], "targets": [8], "unneeded": [1]}), path)
    def no_download(*args):
        pytest.fail("Cached selection must not make a network request")
    monkeypatch.setattr("fantasy_football.data.download.fetch_dataset", no_download)
    result = download_one("player_stats", 2025, tmp_path, refresh=False)
    assert result["status"] == "cached"
    assert result["removed_columns"] == ["unneeded"]
    assert "retrieved_at_utc" not in result
    assert pl.read_parquet(path).columns == ["season", "targets"]


def test_missing_selected_field_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "data/raw/player_stats/2025.parquet"
    original = pl.DataFrame({"season": [2025], "targets": [8]})
    save_parquet(original, path)
    monkeypatch.setattr("fantasy_football.data.download.fetch_dataset",
                        lambda *_: pl.DataFrame({"season": [2025]}))
    result = download_one("player_stats", 2025, tmp_path, refresh=True)
    assert result["status"] == "failed"
    assert pl.read_parquet(path).equals(original)


@pytest.mark.parametrize("seasons", [[2024], [2025, 2024], [None]])
def test_wrong_or_unknown_season_is_rejected(seasons):
    with pytest.raises(ValueError):
        validate_download(pl.DataFrame({"season": seasons}), 2025)
