"""Independently reproduce nflverse PPR points and compare source totals."""

import polars as pl

SCORING_FIELDS = {
    "receptions": "reception", "passing_yards": "passing_yard",
    "passing_tds": "passing_td", "passing_interceptions": "interception",
    "rushing_yards": "rushing_yard", "rushing_tds": "rushing_td",
    "receiving_yards": "receiving_yard", "receiving_tds": "receiving_td",
    "special_teams_tds": "special_teams_td",
    "sack_fumbles_lost": "fumble_lost", "rushing_fumbles_lost": "fumble_lost",
    "receiving_fumbles_lost": "fumble_lost",
    "passing_2pt_conversions": "two_point_conversion",
    "rushing_2pt_conversions": "two_point_conversion",
    "receiving_2pt_conversions": "two_point_conversion",
}


def verify_ppr(frame: pl.DataFrame, rules: dict) -> pl.DataFrame:
    """Leave incomplete scores null and label missing or disagreeing totals."""
    inputs = list(SCORING_FIELDS)
    missing = set(inputs + ["fantasy_points_ppr"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing scoring fields: {sorted(missing)}")
    complete = pl.all_horizontal([
        pl.col(c).is_not_null() & pl.col(c).cast(pl.Float64).is_finite() for c in inputs
    ])
    score = sum(pl.col(c).cast(pl.Float64) * rules[rule] for c, rule in SCORING_FIELDS.items())
    frame = frame.with_columns(
        pl.when(complete).then(score.round(4)).otherwise(None).alias("calculated_ppr_points")
    ).with_columns(
        (pl.col("calculated_ppr_points") - pl.col("fantasy_points_ppr")).alias("ppr_difference")
    ).with_columns(
        pl.when(pl.col("calculated_ppr_points").is_null()).then(pl.lit("missing_inputs"))
        .when(pl.col("fantasy_points_ppr").is_null() | ~pl.col("fantasy_points_ppr").is_finite())
        .then(pl.lit("missing_source_score"))
        .when(pl.col("ppr_difference").abs() <= 0.0001).then(pl.lit("matched"))
        .otherwise(pl.lit("mismatch")).alias("ppr_status")
    )
    return frame
