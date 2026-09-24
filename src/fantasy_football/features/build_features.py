"""Create a modest historical feature table and the next-game player pool."""

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
from zoneinfo import ZoneInfo

import polars as pl

from fantasy_football.config import PROJECT_ROOT, load_settings
from fantasy_football.data.cleaning import normalize_teams, require_unique, schedule_context
from fantasy_football.data.storage import save_parquet
from fantasy_football.features.recent import make_index, features_for

EASTERN = ZoneInfo("America/New_York")
SCHEMA = {
    "player_id": pl.String, "player_display_name": pl.String, "position": pl.String,
    "season": pl.Int32, "week": pl.Int32, "game_id": pl.String, "game_date": pl.Date,
    "team": pl.String, "opponent_team": pl.String, "is_home": pl.Boolean,
    "is_neutral_site": pl.Boolean, "rest_days": pl.Int32, "opponent_rest_days": pl.Int32,
    "ppr_last_game": pl.Float64, "ppr_recent_avg": pl.Float64,
    "targets_recent_avg": pl.Float64, "carries_recent_avg": pl.Float64,
    "snap_share_recent_avg": pl.Float64, "history_games": pl.Int32,
    "targets_trend": pl.Float64,
    "carries_trend": pl.Float64,
    "snap_share_trend": pl.Float64,
    "snap_history_games": pl.Int32, "history_crosses_season": pl.Boolean,
    "opponent_ppr_allowed_recent_avg": pl.Float64, "opponent_history_games": pl.Int32,
    "eligible_recent_participation": pl.Boolean,
    "target_ppr_points": pl.Float64,
}


def history_indexes(games, schedules, positions, today):
    """Use scored games before today"""
    completed = schedules.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null() &
        (pl.col("gameday").str.to_date() < today)
    )
    completed_ids = set(completed["game_id"])
    rows = [r for r in games.to_dicts() if r["game_id"] in completed_ids]
    totals = defaultdict(float)
    represented = defaultdict(set)
    for r in rows:
        totals[(r["game_id"], r["opponent_team"], r["position"])] += r["calculated_ppr_points"]
        represented[r["game_id"]].add(r["team"])
    team_rows = schedule_context(completed).to_dicts()
    defense_rows = []
    for r in team_rows:
        # Only use a zero position total when BOTH teams have source records.
        # An entire missing game must not make a defense look better.
        covered = {r["team"], r["opponent_team"]} <= represented[r["game_id"]]
        for position in positions:
            defense_rows.append(r | {"position": position, "allowed_ppr":
                totals[(r["game_id"], r["team"], position)] if covered else None})
    return (make_index(rows, lambda r: r["player_id"]),
            make_index(defense_rows, lambda r: (r["team"], r["position"])),
            make_index(team_rows, lambda r: r["team"]))


def next_games(schedules, now):
    future = []
    for row in schedules.to_dicts():
        if row["home_score"] is not None or row["away_score"] is not None:
            continue
        if not row["gameday"] or not row["gametime"]:
            continue
        kickoff = datetime.fromisoformat(f"{row['gameday']}T{row['gametime']}").replace(tzinfo=EASTERN)
        if kickoff > now:
            future.append(row)
    if not future:
        return {}
    context = schedule_context(pl.DataFrame(future)).sort("game_date", "gametime")
    result = {}
    for row in context.to_dicts():
        result.setdefault(row["team"], row)
    return result


def candidates(rosters, positions, active_status):
    """Current roster"""
    rosters = normalize_teams(rosters, ["team"]).filter(
        pl.col("gsis_id").is_not_null() & pl.col("position").is_in(positions)
    )
    # Season rosters may contain more than one observation of a player.
    latest = rosters.with_columns(pl.col("week").max().over("gsis_id").alias("latest_week"))
    latest = latest.filter(pl.col("week").eq_missing(pl.col("latest_week")))
    active = latest.filter(pl.col("status") == active_status).select(
        pl.col("gsis_id").alias("player_id"), pl.col("full_name").alias("player_display_name"),
        "team", "position",
    ).unique()
    require_unique(active, ["player_id"], "Current active roster")
    return active.to_dicts()


def build(root: Path = PROJECT_ROOT, settings=None, now=None):
    settings = settings or load_settings()
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(EASTERN).date()
    options = settings["features"]
    if options["recent_games"] < 1 or options["history_days"] < 1:
        raise ValueError("Feature window and history_days must be positive")
    data = settings["data"]
    games = pl.read_parquet(root / "data/processed/player_games.parquet")
    seasons = sorted(set(games["season"].to_list()) | {data["current_season"]})
    schedules = pl.concat([pl.read_parquet(root / f"data/raw/schedules/{s}.parquet")
                           for s in seasons], how="diagonal_relaxed").filter(pl.col("game_type") == "REG")
    indexes = history_indexes(games, schedules, data["positions"], today)
    history = []
    # Keep all observed historical rows; eligibility is a pregame flag, not a
    # filter based on how well the player ultimately performed.
    for row in games.filter(pl.col("game_date") < today).to_dicts():
        values = features_for(row, *indexes, options, today)
        history.append(row | values | {"target_ppr_points": row["calculated_ppr_points"]})
    roster = pl.read_parquet(root / f"data/raw/rosters/{data['current_season']}.parquet")
    pool = candidates(roster, data["positions"], options["active_roster_status"])
    upcoming_games = next_games(schedules.filter(pl.col("season") == data["current_season"]), now)
    upcoming, excluded = [], []
    for player in pool:
        game = upcoming_games.get(player["team"])
        if not game:
            excluded.append({"player_id": player["player_id"], "reason": "no_future_game"})
            continue
        row = player | game
        values = features_for(row, *indexes, options, today)
        if values["eligible_recent_participation"]:
            upcoming.append(row | values | {"target_ppr_points": None})
        else:
            excluded.append({"player_id": player["player_id"], "reason": "no_recent_participation"})
    outputs = {}
    for name, rows in [("historical_features", history), ("upcoming_features", upcoming)]:
        frame = pl.DataFrame(rows, schema=SCHEMA, strict=False).sort("season", "week", "player_id")
        require_unique(frame, ["player_id", "game_id"], name)
        save_parquet(frame, root / f"data/processed/{name}.parquet")
        outputs[name] = frame.height
    report = {"built_at_utc": now.isoformat(), "history_cutoff_exclusive": str(today),
              "settings": options, "outputs": outputs, "excluded_candidates": excluded,
              "latest_source_game_date": str(games["game_date"].max())}
    folder = root / "data/manifests"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"features_{now.strftime('%Y%m%dT%H%M%S%fZ')}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"Features saved: {outputs}. Inputs include games through {report['latest_source_game_date']}.")
    return report


if __name__ == "__main__":
    build()
