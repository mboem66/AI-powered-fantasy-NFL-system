"""Read project settings and keep paths independent of the terminal directory."""

from pathlib import Path
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_settings() -> dict:
    """Load the small, human-readable configuration file."""
    with (PROJECT_ROOT / "config" / "project.toml").open("rb") as stream:
        settings = tomllib.load(stream)
    data = settings["data"]
    if not 2012 <= data["historical_start"] <= data["historical_end"] < data["current_season"]:
        raise ValueError("Expected 2012 <= historical_start <= historical_end < current_season")
    return settings
