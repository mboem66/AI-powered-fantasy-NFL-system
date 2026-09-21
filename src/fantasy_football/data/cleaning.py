"""Combine historical game records without dropping unmatched players."""

import polars as pl

TEAM_ALIASES = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS"}


def normalize_teams(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Use nflverse team abbreviations consistently across sources."""
    return frame.with_columns([
        pl.col(c).str.strip_chars().str.to_uppercase().replace(TEAM_ALIASES)
        for c in columns
    ])


def require_unique(frame: pl.DataFrame, keys: list[str], label: str) -> None:
    """Fail on ambiguous joins rather than multiplying player records."""
    if frame.select(pl.any_horizontal(pl.col(keys).is_null()).any()).item():
        raise ValueError(f"{label}: null key in {keys}")
    if frame.select(keys).is_duplicated().any():
        raise ValueError(f"{label}: duplicate key in {keys}")


def player_id_map(players: pl.DataFrame, rosters: pl.DataFrame) -> pl.DataFrame:
    """Use only identity mappings from references, never current team/status."""
    def pairs(frame):
        return frame.select("gsis_id", "pfr_id").drop_nulls().with_columns(
            pl.all().str.strip_chars()
        ).filter((pl.col("gsis_id") != "") & (pl.col("pfr_id") != "")).unique()

    primary = pairs(players)
    require_unique(primary, ["gsis_id"], "Player reference")
    # Older rosters contain some stale/wrong PFR IDs. Prefer the dedicated
    # reference; use roster mappings only when the reference has no mapping.
    fallback = pairs(rosters).join(primary, on="gsis_id", how="anti").join(
        primary, on="pfr_id", how="anti"
    )
    mapping = pl.concat([
        primary.with_columns(pl.lit("player_reference").alias("id_mapping_source")),
        fallback.with_columns(pl.lit("roster_fallback").alias("id_mapping_source")),
    ])
    require_unique(mapping, ["gsis_id"], "GSIS-to-PFR mapping")
    require_unique(mapping, ["pfr_id"], "PFR-to-GSIS mapping")
    return mapping.rename({"gsis_id": "player_id", "pfr_id": "pfr_player_id"})


def schedule_context(schedules: pl.DataFrame) -> pl.DataFrame:
    """Turn each game into home and away team rows for an exact context join."""
    schedules = normalize_teams(schedules, ["home_team", "away_team"])
    schedules = schedules.filter(pl.col("game_type") == "REG")
    require_unique(schedules, ["game_id"], "Schedules")
    rows = []
    for side, other in [("home", "away"), ("away", "home")]:
        rows.append(schedules.select(
            "game_id", "season", "week",
            pl.col("gameday").str.to_date().alias("game_date"), "gametime",
            pl.col(f"{side}_team").alias("team"),
            pl.col(f"{other}_team").alias("opponent_team"),
            pl.lit(side == "home").alias("is_home"),
            (pl.col("location") == "Neutral").alias("is_neutral_site"),
            pl.col(f"{side}_rest").alias("rest_days"),
            pl.col(f"{other}_rest").alias("opponent_rest_days"),
            pl.lit(True).alias("schedule_matched"),
        ))
    return pl.concat(rows)


def clean_player_games(stats, schedules, mapping, snaps, positions, snap_seasons):
    """Keep observed regular-season position records, with nullable snap data."""
    frame = stats.with_columns(
        pl.col("position").str.strip_chars().str.to_uppercase(),
        pl.col("player_id").str.strip_chars(),
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
    ).filter(pl.col("position").is_in(positions) & (pl.col("season_type") == "REG"))
    frame = normalize_teams(frame, ["team", "opponent_team"])
    require_unique(frame, ["player_id", "game_id"], "Player games")
    require_unique(frame, ["player_id", "season", "week"], "Player weeks")
    if frame.filter(pl.col("player_id") == "").height:
        raise ValueError("Player games: empty player ID")
    context_keys = ["game_id", "season", "week", "team", "opponent_team"]
    frame = frame.join(schedule_context(schedules), on=context_keys, how="left", validate="m:1")
    if frame["schedule_matched"].null_count():
        raise ValueError("Player games have missing or inconsistent schedule matches")
    frame = frame.drop("schedule_matched").join(mapping, on="player_id", how="left", validate="m:1")
    snaps = normalize_teams(snaps, ["team"]).filter(pl.col("game_type") == "REG")
    snap_keys = ["game_id", "season", "week", "team", "pfr_player_id"]
    require_unique(snaps, snap_keys, "Snap counts")
    invalid_snaps = snaps.filter(
        (pl.col("offense_snaps") < 0) | (~pl.col("offense_snaps").is_finite()) |
        (~pl.col("offense_pct").is_between(0, 1)) | (~pl.col("offense_pct").is_finite())
    )
    if invalid_snaps.height:
        raise ValueError("Invalid offensive snap count or share")
    frame = frame.join(
        snaps.select(*snap_keys, "offense_snaps", "offense_pct", pl.lit(True).alias("snap_matched")),
        on=snap_keys, how="left", validate="m:1",
    )
    # Missing snaps are unknown, not proof that a player played zero snaps.
    frame = frame.with_columns(
        pl.when(~pl.col("season").is_in(snap_seasons)).then(pl.lit("season_unavailable"))
        .when(pl.col("pfr_player_id").is_null()).then(pl.lit("player_id_unmapped"))
        .when(pl.col("snap_matched").is_null()).then(pl.lit("record_missing"))
        .when(pl.col("offense_snaps").is_null() | pl.col("offense_pct").is_null())
        .then(pl.lit("values_missing"))
        .otherwise(pl.lit("matched")).alias("snap_status")
    ).drop("snap_matched")
    return frame.sort("season", "week", "game_id", "player_id")
