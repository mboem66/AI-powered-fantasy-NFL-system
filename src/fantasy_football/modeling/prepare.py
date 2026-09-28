import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fantasy_football.data.cleaning import require_unique

TRAIN_SEASON, VALIDATION_SEASON = 2023, 2024
POSITIONS = ["QB", "RB", "WR", "TE"]
TARGET = "target_ppr_points"
IDENTIFIERS = ["player_id", "player_display_name", "position", "season", "week",
               "game_id", "game_date", "team", "opponent_team"]
NUMERIC_FEATURES = [
    "ppr_last_game", "ppr_recent_avg", "targets_recent_avg", "carries_recent_avg",
    "snap_share_recent_avg", "targets_trend", "carries_trend", "snap_share_trend",
    "passing_attempts_recent_avg", "passing_yards_recent_avg", "passing_tds_recent_avg",
    "opponent_ppr_allowed_recent_avg", "is_home", "is_neutral_site",
    "history_games", "snap_history_games", "opponent_history_games", "history_crosses_season",
]


def split_records(frame):
    required = set(IDENTIFIERS + NUMERIC_FEATURES + [TARGET, "eligible_recent_participation"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing feature columns: {missing}. Run fantasy_football.update --local first.")
    splits, counts = {}, {}
    for season in (TRAIN_SEASON, VALIDATION_SEASON):
        rows = frame.filter(pl.col("season") == season)
        report = {"available": rows.height, "excluded": {}}
        checks = [
            ("invalid_position", pl.col("position").is_in(POSITIONS)),
            ("missing_or_nonfinite_target", pl.col(TARGET).is_finite()),
            ("no_usable_prior_history", (pl.col("history_games") > 0) & pl.col("ppr_recent_avg").is_finite()),
            ("no_recent_participation", pl.col("eligible_recent_participation")),
        ]
        for reason, valid in checks:
            kept = rows.filter(valid.fill_null(False))
            report["excluded"][reason] = rows.height - kept.height
            rows = kept
        if rows.is_empty():
            raise ValueError(f"No eligible records for {season}; check the historical feature file.")
        require_unique(rows, ["player_id", "game_id"], f"Model records {season}")
        splits[season] = rows.sort("game_date", "game_id", "player_id")
        counts[str(season)] = report | {"included": rows.height}
    return splits[TRAIN_SEASON], splits[VALIDATION_SEASON], counts


def model_inputs(frame):
    """convert unknown numeric values to NaN."""
    numeric = frame.select([
        pl.when(pl.col(c).cast(pl.Float64).is_finite())
        .then(pl.col(c).cast(pl.Float64)).otherwise(None).alias(c)
        for c in NUMERIC_FEATURES
    ]).to_numpy()
    return np.column_stack([numeric, frame["position"].to_numpy()])


def make_model():
    """One shared model, with position indicators and training-only numeric medians."""
    numeric_columns = list(range(len(NUMERIC_FEATURES)))
    preprocessing = ColumnTransformer([
        # An entirely empty training feature is retained and filled with zero.
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric_columns),
        # QB is the reference category; numeric feature weights are shared.
        ("position", OneHotEncoder(categories=[POSITIONS], drop="first", sparse_output=False),
         [len(NUMERIC_FEATURES)]),
    ])
    return Pipeline([("preprocess", preprocessing), ("regression", LinearRegression())])
