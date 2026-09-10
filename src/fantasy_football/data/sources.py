"""Keep nflverse-specific calls separate from local file handling."""

import nflreadpy as nfl
from nflreadpy.config import update_config

# Player IDs allow later joins between GSIS statistics and PFR snap counts.
SEASON_DATASETS = ("player_stats", "team_stats", "schedules", "rosters", "snap_counts")
REFERENCE_DATASETS = ("players", "teams")


def configure_source() -> None:
    """Always request fresh data when the user explicitly runs an update."""
    update_config(cache_mode="off", timeout=60, verbose=False)


def fetch_dataset(name: str, season: int | None = None):
    """Return a Polars table using the documented nflreadpy loaders."""
    loaders = {
        "player_stats": nfl.load_player_stats,
        "team_stats": nfl.load_team_stats,
        "schedules": nfl.load_schedules,
        "rosters": nfl.load_rosters,
        "snap_counts": nfl.load_snap_counts,
        "players": nfl.load_players,
        "teams": nfl.load_teams,
    }
    loader = loaders[name]
    return loader() if season is None else loader(season)
