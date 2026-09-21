"""Build historical player-game data and verify scoring from local source files."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import polars as pl

from fantasy_football.config import PROJECT_ROOT, load_settings
from fantasy_football.data.cleaning import clean_player_games, player_id_map
from fantasy_football.data.scoring import verify_ppr
from fantasy_football.data.storage import save_parquet, validate_download


def build(root: Path, settings: dict) -> dict:
    """Use available seasons, require historical stats, and audit input snapshots."""
    data = settings["data"]
    history = list(range(data["historical_start"], data["historical_end"] + 1))
    inputs = []

    def read(name, label):
        path = root / "data" / "raw" / name / f"{label}.parquet"
        frame = pl.read_parquet(path)
        validate_download(frame, None if label == "latest" else label)
        inputs.append({
            "path": str(path.relative_to(root)), "rows": frame.height,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        return frame

    current = data["current_season"]
    current_available = (root / "data/raw/player_stats" / f"{current}.parquet").exists()
    seasons = history + ([current] if current_available else [])
    stats = pl.concat([read("player_stats", s) for s in seasons], how="diagonal_relaxed")
    schedules = pl.concat([read("schedules", s) for s in seasons], how="diagonal_relaxed")
    rosters = pl.concat([read("rosters", s) for s in seasons], how="diagonal_relaxed")
    mapping = player_id_map(read("players", "latest"), rosters)
    snap_seasons = [s for s in seasons if (root / "data/raw/snap_counts" / f"{s}.parquet").exists()]
    if snap_seasons:
        snaps = pl.concat([read("snap_counts", s) for s in snap_seasons], how="diagonal_relaxed")
    else:
        snaps = pl.DataFrame(schema={
            "game_id": pl.String, "season": pl.Int32, "week": pl.Int32,
            "game_type": pl.String, "team": pl.String, "pfr_player_id": pl.String,
            "offense_snaps": pl.Float64, "offense_pct": pl.Float64,
        })
    result = clean_player_games(stats, schedules, mapping, snaps, data["positions"], snap_seasons)
    if result.is_empty():
        raise ValueError("No regular-season records for the configured positions")
    result = verify_ppr(result, settings["scoring"])
    issues = result.filter(pl.col("ppr_status") != "matched")
    now = datetime.now(timezone.utc)
    report = {
        "processed_at_utc": now.isoformat(), "input_files": inputs,
        "scoring_rules": settings["scoring"], "rows": result.height,
        "id_mapping_policy": "Player reference first; unique roster fallback only for unused GSIS/PFR IDs",
        "current_season_file_present": current_available,
        "coverage": result.group_by("season").agg(
            pl.len().alias("player_games"), pl.col("game_id").n_unique().alias("games"),
            pl.col("week").max().alias("last_week_in_file"),
            pl.col("game_date").max().cast(pl.String).alias("last_game_date_in_file"),
        ).sort("season").to_dicts(),
        "snap_status": result.group_by("snap_status").len().to_dicts(),
        "ppr_status": result.group_by("ppr_status").len().to_dicts(),
        "null_counts": result.null_count().row(0, named=True),
        "unmatched_snap_records": result.filter(pl.col("snap_status") != "matched").select(
            "player_id", "player_display_name", "game_id", "snap_status"
        ).to_dicts(),
        "scoring_issue_examples": issues.select(
            "player_id", "game_id", "calculated_ppr_points", "fantasy_points_ppr", "ppr_status"
        ).head(20).to_dicts(),
    }
    report_folder = root / "data/manifests"
    report_folder.mkdir(parents=True, exist_ok=True)
    report_path = report_folder / f"build_{now.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    output = root / "data/processed/player_games.parquet"
    report["output_written"] = False
    if issues.is_empty():
        # Full rebuilds replace the table instead of appending duplicate rows.
        save_parquet(result, output)
        report["output_written"] = True
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not issues.is_empty():
        raise ValueError(f"{issues.height} unverified PPR scores; previous output preserved. See {report_path}")
    print(f"Saved {result.height:,} player-game rows to {output}")
    print(f"PPR: all rows matched. Snap coverage: {report['snap_status']}")
    print(f"Build report: {report_path}")
    return report


if __name__ == "__main__":
    build(PROJECT_ROOT, load_settings())
