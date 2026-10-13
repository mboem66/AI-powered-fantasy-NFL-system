import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from fantasy_football.data.cleaning import require_unique

TRAIN_SEASONS, VALIDATION_SEASON = (2021, 2022, 2023), 2024
# Initial settings only. All models use the same records without a tuning search.
MODEL_SETTINGS = {
    "linear_regression": {},
    "random_forest": {"n_estimators": 200, "min_samples_leaf": 5,
                      "random_state": 42, "n_jobs": 1},
    "xgboost": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05,
                "objective": "reg:squarederror", "tree_method": "hist",
                "random_state": 42, "n_jobs": 1},
}
METHODS = ("baseline", *MODEL_SETTINGS)
POSITIONS = ["QB", "RB", "WR", "TE"]
TARGET = "target_ppr_points"
IDENTIFIERS = ["player_id", "player_display_name", "position", "season", "week",
               "game_id", "game_date", "team", "opponent_team"]
RECENT_FEATURES = [
    "ppr_last_game", "ppr_recent_avg", "targets_recent_avg", "carries_recent_avg",
    "snap_share_recent_avg", "targets_trend", "carries_trend", "snap_share_trend",
    "passing_attempts_recent_avg", "passing_yards_recent_avg", "passing_tds_recent_avg",
    "opponent_ppr_allowed_recent_avg", "is_home", "is_neutral_site",
    "history_games", "snap_history_games", "opponent_history_games", "history_crosses_season",
]
SEASON_FEATURES = [
    "ppr_season_avg", "targets_season_avg", "carries_season_avg", "snap_share_season_avg",
    "passing_attempts_season_avg", "passing_yards_season_avg", "passing_tds_season_avg",
    "season_history_games",
]
NUMERIC_FEATURES = RECENT_FEATURES + SEASON_FEATURES


def split_records(frame, training_seasons=TRAIN_SEASONS, numeric_features=NUMERIC_FEATURES):
    if not training_seasons or any(s >= VALIDATION_SEASON for s in training_seasons):
        raise ValueError("Training seasons must precede the 2024 validation season")
    required = set(IDENTIFIERS + list(numeric_features) + [TARGET, "eligible_recent_participation"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing feature columns: {missing}. Run fantasy_football.update --local first.")
    splits, counts = {}, {}
    for season in (*training_seasons, VALIDATION_SEASON):
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
    training = pl.concat([splits[s] for s in training_seasons]).sort("game_date", "game_id", "player_id")
    require_unique(training, ["player_id", "game_id"], "Training records")
    return training, splits[VALIDATION_SEASON], counts


def model_inputs(frame, numeric_features=NUMERIC_FEATURES):
    """convert unknown numeric values to NaN."""
    numeric = frame.select([
        pl.when(pl.col(c).cast(pl.Float64).is_finite())
        .then(pl.col(c).cast(pl.Float64)).otherwise(None).alias(c)
        for c in numeric_features
    ]).to_numpy()
    return np.column_stack([numeric, frame["position"].to_numpy()])


def make_model(name="linear_regression", numeric_features=NUMERIC_FEATURES):
    """Give each estimator its own preprocessing, fitted on training records only."""
    estimators = {"linear_regression": LinearRegression,
                  "random_forest": RandomForestRegressor, "xgboost": XGBRegressor}
    if name not in estimators:
        raise ValueError(f"Unknown model: {name}")
    numeric_columns = list(range(len(numeric_features)))
    preprocessing = ColumnTransformer([
        # An entirely empty training feature is retained and filled with zero.
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric_columns),
        # Use the same position encoding for every estimator, with QB as reference.
        ("position", OneHotEncoder(categories=[POSITIONS], drop="first", sparse_output=False),
         [len(numeric_features)]),
    ])
    estimator = estimators[name](**MODEL_SETTINGS[name])
    return Pipeline([("preprocess", preprocessing), ("regression", estimator)])
