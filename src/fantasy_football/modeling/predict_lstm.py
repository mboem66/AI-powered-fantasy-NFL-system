"""Optional weekly LSTM forecasts, with an explicit Linear fallback."""

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import polars as pl

from fantasy_football.config import PROJECT_ROOT, load_settings
from fantasy_football.data.cleaning import require_unique
from fantasy_football.features.build_features import build, EASTERN, next_games
from fantasy_football.modeling import lstm
from fantasy_football.modeling.prepare import IDENTIFIERS, TARGET, POSITIONS, make_model, model_inputs, select_records
from fantasy_football.modeling.sequences import make_sequences, save_sequences
from fantasy_football.modeling.train_lstm import load_history


def run(root=PROJECT_ROOT, model_run=None, settings=None, now=None):
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(EASTERN).date()
    settings = settings or load_settings()
    if model_run is None:
        runs = sorted(p.parent for p in (root / "artifacts/task3/runs").glob("*/summary.json"))
        if not runs:
            raise ValueError("Run fantasy_football.modeling.train_lstm first")
        model_run = runs[-1]
    else:
        model_run = root / model_run
    selection = json.loads((model_run / "summary.json").read_text(encoding="utf-8"))
    options, epochs = selection["options"], selection["selected_epochs"]
    # Freeze the training-target cutoff once the prospective period begins.
    training_cutoff = min(today, date.fromisoformat(options["future_test_start"]))
    feature_report = build(root, settings, now)
    games, features = load_history(root, cutoff=today)
    historical = features.filter((pl.col("game_date") < training_cutoff) & (pl.col("season") >= 2021))
    seasons = sorted(historical["season"].unique().to_list())
    if not set(range(2021, 2026)).issubset(seasons):
        raise ValueError("Weekly training requires completed 2021-2025 history")
    training, counts = select_records(historical, seasons)
    upcoming = pl.read_parquet(root / "data/processed/upcoming_features.parquet")
    schedules = pl.read_parquet(root / f"data/raw/schedules/{settings['data']['current_season']}.parquet")
    future = next_games(schedules.filter(pl.col("game_type") == "REG"), now)
    upcoming = upcoming.filter(
        pl.col("game_id").is_in(list({r["game_id"] for r in future.values()})) &
        pl.col("position").is_in(POSITIONS) & pl.col("eligible_recent_participation") &
        (pl.col("history_games") > 0) & pl.col("ppr_recent_avg").is_finite()
    ).sort("game_date", "game_id", "position", "player_id")
    require_unique(upcoming, ["player_id", "game_id"], "LSTM upcoming pool")
    if upcoming.is_empty():
        print("No eligible unstarted games; previous forecasts preserved.")
        return None
    if upcoming[TARGET].null_count() != upcoming.height:
        raise ValueError("Upcoming targets must be blank")
    train_data = make_sequences(training, games, cutoff=training_cutoff)
    future_data = make_sequences(upcoming, games, cutoff=today)
    print(f"Refitting LSTM for the fixed {epochs} epochs on completed history.", flush=True)
    bundle, curve = lstm.fit(train_data, options, epochs=epochs)
    linear = make_model().fit(model_inputs(training), training[TARGET].to_numpy())
    output = root / "artifacts/task3/weekly" / now.strftime("%Y%m%dT%H%M%S%fZ")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix="pending_") as temporary:
        folder = Path(temporary) / "result"
        folder.mkdir()
        lstm.save(bundle, folder / "model")
        joblib.dump(linear, folder / "linear.joblib")
        raw = lstm.predict(lstm.load(folder / "model"), future_data)
        linear_predictions = joblib.load(folder / "linear.joblib").predict(model_inputs(upcoming))
        has_history = future_data["lengths"] > 0
        combined = np.where(has_history, raw, linear_predictions)
        if not np.isfinite(combined).all():
            raise ValueError("Nonfinite weekly predictions")
        predictions = upcoming.select(IDENTIFIERS).with_columns(
            pl.Series("sequence_length", future_data["lengths"]),
            pl.Series("lstm_prediction", raw).fill_nan(None),
            pl.Series("linear_regression_prediction", linear_predictions),
            upcoming["ppr_recent_avg"].alias("baseline_prediction"),
            pl.Series("predicted_ppr_points", combined),
            pl.Series("prediction_method", np.where(has_history, "lstm", "linear_fallback")),
            pl.lit(now.isoformat()).alias("predicted_at_utc"),
        )
        predictions.write_csv(folder / "predictions.csv")
        predictions.write_parquet(folder / "predictions.parquet", compression="zstd")
        save_sequences(future_data, folder, "upcoming")
        pl.DataFrame(curve).write_csv(folder / "training_curve.csv")
        report = {"created_at_utc": now.isoformat(), "selection_run": str(model_run),
                  "options": options, "epochs": epochs, "training_cutoff_exclusive": str(training_cutoff),
                  "training_latest_game": str(training["game_date"].max()), "training_records": counts,
                  "feature_cutoff_exclusive": str(today), "source_latest_game": feature_report["latest_source_game_date"],
                  "prediction_rows": upcoming.height, "lstm_rows": int(has_history.sum()),
                  "fallback_rows": int((~has_history).sum()),
                  "note": "Experimental LSTM; Task 2 Linear remains available. Saved sources only; refresh with update first."}
        (folder / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        folder.rename(output)
    print(f"Saved {upcoming.height} predictions ({report['fallback_rows']} Linear fallbacks) to {output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Saved Task 3 run folder; defaults to the latest completed run")
    args = parser.parse_args()
    run(model_run=args.run)
