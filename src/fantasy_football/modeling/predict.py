"""Fit the selected model on completed history and save dated next-game predictions."""

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import polars as pl

from fantasy_football.config import PROJECT_ROOT, load_settings
from fantasy_football.data.cleaning import require_unique
from fantasy_football.features.build_features import build, EASTERN, next_games
from fantasy_football.modeling.final_test import SELECTED_MODEL
from fantasy_football.modeling.prepare import (
    IDENTIFIERS, NUMERIC_FEATURES, POSITIONS, TARGET, make_model, model_inputs, select_records,
)


def run(root=PROJECT_ROOT, settings=None, now=None):
    settings = settings or load_settings()
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(EASTERN).date()
    final_path = root / "artifacts/task2/final_test/summary.json"
    if not final_path.exists():
        raise ValueError("Record the final test first: python -m fantasy_football.modeling.final_test")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if final["model"] != SELECTED_MODEL or final["numeric_features"] != NUMERIC_FEATURES:
        raise ValueError("Selected model/features differ from the recorded final test")

    # Rebuild pregame features at today's cutoff, using the saved source files.
    feature_report = build(root, settings, now)
    source = root / "data/processed/historical_features.parquet"
    history = pl.read_parquet(source)
    upcoming = pl.read_parquet(root / "data/processed/upcoming_features.parquet")
    current_season = settings["data"]["current_season"]
    schedules = pl.concat([pl.read_parquet(root / f"data/raw/schedules/{season}.parquet")
                           for season in range(2021, current_season + 1)], how="diagonal_relaxed")
    schedules = schedules.filter(pl.col("game_type") == "REG")
    completed = schedules.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null() &
        (pl.col("gameday").str.to_date() < today)
    )
    history = history.filter(
        pl.col("game_id").is_in(completed["game_id"].to_list()) &
        pl.col("season").is_between(2021, current_season) & (pl.col("game_date") < today)
    )
    seasons = sorted(history["season"].unique().to_list())
    if not set(range(2021, 2026)).issubset(seasons):
        raise ValueError("Production training requires completed history for 2021-2025")
    training, counts = select_records(history, seasons)
    future = next_games(schedules.filter(pl.col("season") == current_season), now)
    future_ids = {game["game_id"] for game in future.values()}
    upcoming = upcoming.filter(
        pl.col("game_id").is_in(list(future_ids)) & pl.col("position").is_in(POSITIONS) &
        pl.col("eligible_recent_participation") & (pl.col("history_games") > 0) &
        pl.col("ppr_recent_avg").is_finite()
    ).sort("game_date", "game_id", "position", "player_id")
    require_unique(upcoming, ["player_id", "game_id"], "Upcoming predictions")
    if upcoming.is_empty():
        print("No eligible players in unstarted upcoming games; previous predictions preserved.")
        return None
    if upcoming[TARGET].null_count() != upcoming.height:
        raise ValueError("Upcoming outcomes must be blank")
    model = make_model(SELECTED_MODEL)
    model.fit(model_inputs(training), training[TARGET].to_numpy())
    metadata = {
        "created_at_utc": now.isoformat(), "model": SELECTED_MODEL, "model_settings": {},
        "numeric_features": NUMERIC_FEATURES, "positions": POSITIONS,
        "training_seasons": seasons, "records": counts,
        "training_cutoff_exclusive": str(today), "training_latest_game": str(training["game_date"].max()),
        "source_latest_game": feature_report["latest_source_game_date"],
        "source_sha256": sha256(source.read_bytes()).hexdigest(),
        "scikit_learn_version": version("scikit-learn"), "prediction_rows": upcoming.height,
        "final_test_summary_sha256": sha256(final_path.read_bytes()).hexdigest(),
        "note": "Uses saved sources; refresh with fantasy_football.update first. Negative predictions retained.",
    }
    output = root / "artifacts/task2/weekly" / now.strftime("%Y%m%dT%H%M%S%fZ")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix="pending_") as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir()
        joblib.dump({"pipeline": model, "metadata": metadata}, staging / "model.joblib")
        # Use the saved model for prediction, checking its serialization round trip.
        saved = joblib.load(staging / "model.joblib")
        inputs = model_inputs(upcoming, saved["metadata"]["numeric_features"])
        predicted = saved["pipeline"].predict(inputs)
        np.testing.assert_allclose(predicted, model.predict(inputs), rtol=1e-12, atol=1e-12)
        if not np.isfinite(predicted).all():
            raise ValueError("Upcoming predictions contain nonfinite values")
        predictions = upcoming.select(IDENTIFIERS).with_columns(
            pl.Series("predicted_ppr_points", predicted),
            pl.lit(now.isoformat()).alias("predicted_at_utc"),
        )
        predictions.write_parquet(staging / "predictions.parquet", compression="zstd")
        predictions.write_csv(staging / "predictions.csv")
        (staging / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        staging.rename(output)
    print(f"Training: {training.height:,} records, through {metadata['training_latest_game']}.")
    print(f"Saved {upcoming.height} next-game predictions and model to {output}")
    print("These are each team's next game, so weeks may differ around byes. Saved sources were not downloaded.")
    return output


if __name__ == "__main__":
    run()
