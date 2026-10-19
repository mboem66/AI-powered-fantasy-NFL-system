"""Develop on 2024, then report a separate retrospective 2025 comparison."""

from datetime import date, datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import tomllib

import joblib
import numpy as np
import polars as pl

from fantasy_football.config import PROJECT_ROOT
from fantasy_football.features.build_features import EASTERN
from fantasy_football.modeling import lstm
from fantasy_football.modeling.prepare import IDENTIFIERS, TARGET, make_model, model_inputs, select_records
from fantasy_football.modeling.sequences import make_sequences, save_sequences, SEQUENCE_FIELDS, CONTEXT_FIELDS
from fantasy_football.modeling.train import evaluate


def load_history(root, cutoff=None):
    """Only scored regular-season games can supply outcomes or sequence steps."""
    games = pl.read_parquet(root / "data/processed/player_games.parquet")
    schedules = pl.concat([pl.read_parquet(root / f"data/raw/schedules/{s}.parquet")
                           for s in sorted(games["season"].unique())], how="diagonal_relaxed")
    completed = schedules.filter((pl.col("game_type") == "REG") &
                                 pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
    games = games.filter(pl.col("game_id").is_in(completed["game_id"].to_list()))
    if cutoff:
        games = games.filter(pl.col("game_date") < cutoff)
    features = pl.read_parquet(root / "data/processed/historical_features.parquet")
    return games, features.filter(pl.col("game_id").is_in(games["game_id"].to_list()))


def compare(data, bundle, linear, folder, label):
    raw = lstm.predict(bundle, data)
    linear_predictions = linear.predict(model_inputs(data["rows"]))
    predictions = data["rows"].select(*IDENTIFIERS, TARGET).with_columns(
        pl.Series("sequence_length", data["lengths"]),
        data["rows"]["ppr_recent_avg"].alias("baseline_prediction"),
        pl.Series("linear_regression_prediction", linear_predictions),
        pl.Series("lstm_prediction", raw).fill_nan(None),
        pl.Series("lstm_with_fallback_prediction", np.where(data["lengths"] > 0, raw, linear_predictions)),
    )
    metrics = []
    # Every model within a scope is evaluated on the identical records.
    scopes = {"all_eligible": pl.lit(True), "lstm_eligible": pl.col("sequence_length") > 0,
              "history_0": pl.col("sequence_length") == 0, "history_1": pl.col("sequence_length") == 1,
              "history_2_4": pl.col("sequence_length").is_between(2, 4),
              "history_5_9": pl.col("sequence_length").is_between(5, 9),
              "history_10_plus": pl.col("sequence_length") >= 10}
    for scope, condition in scopes.items():
        subset = predictions.filter(condition)
        if subset.is_empty():
            continue
        methods = ["baseline", "linear_regression", "lstm_with_fallback"]
        if subset["sequence_length"].min() > 0:
            methods.append("lstm")
        metrics.append(evaluate(subset, methods).with_columns(pl.lit(scope).alias("scope")))
    metrics = pl.concat(metrics).with_columns(pl.lit(label).alias("evaluation"))
    predictions.write_parquet(folder / f"{label}_predictions.parquet", compression="zstd")
    predictions.write_csv(folder / f"{label}_predictions.csv")
    metrics.write_csv(folder / f"{label}_metrics.csv")
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=-1, tbl_cols=-1, tbl_width_chars=150, float_precision=3):
        print(metrics.filter((pl.col("position") == "overall") & pl.col("scope").is_in(["all_eligible", "lstm_eligible"])))
    return {"rows": data["rows"].height, "lstm_rows": int((data["lengths"] > 0).sum()),
            "fallback_rows": int((data["lengths"] == 0).sum())}


def plot_curve(curve, best_epoch, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [r["epoch"] for r in curve]
    axes[0].plot(epochs, [r["training_mse_scaled"] for r in curve])
    axes[0].set(title="Training loss", ylabel="MSE (scaled target)", xlabel="Epoch")
    axes[1].plot(epochs, [r["validation_mae"] for r in curve], label="Validation MAE")
    axes[1].plot(epochs, [r["validation_rmse"] for r in curve], label="Validation RMSE")
    axes[1].axvline(best_epoch, color="gray", linestyle="--", label="Selected epoch")
    axes[1].set(title="2024 validation", ylabel="PPR points", xlabel="Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(root=PROJECT_ROOT):
    started = perf_counter()
    now = datetime.now(timezone.utc)
    with (root / "config/lstm.toml").open("rb") as stream:
        options = tomllib.load(stream)
    if now.astimezone(EASTERN).date() >= date.fromisoformat(options["future_test_start"]):
        raise ValueError("Future evaluation has started; use the frozen saved run for predictions, not new model selection")
    games, features = load_history(root)
    training, train_counts = select_records(features, [2021, 2022, 2023])
    validation, valid_counts = select_records(features, [2024])
    train_data = make_sequences(training, games)
    valid_data = make_sequences(validation, games)
    output = root / "artifacts/task3/runs" / now.strftime("%Y%m%dT%H%M%S%fZ")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix="pending_") as temporary:
        folder = Path(temporary) / "result"
        folder.mkdir()
        save_sequences(train_data, folder, "training")
        save_sequences(valid_data, folder, "validation")
        print("Training LSTM on 2021-2023; early stopping uses 2024 MAE.", flush=True)
        bundle, curve = lstm.fit(train_data, options, valid_data)
        lstm.save(bundle, folder / "validation_model")
        # Compare saved weights with the in-memory model before reporting results.
        loaded = lstm.load(folder / "validation_model")
        np.testing.assert_allclose(lstm.predict(bundle, valid_data), lstm.predict(loaded, valid_data), equal_nan=True)
        pl.DataFrame(curve).write_csv(folder / "learning_curve.csv")
        plot_curve(curve, bundle["epochs"], folder / "learning_curve.png")
        linear = make_model().fit(model_inputs(training), training[TARGET].to_numpy())
        joblib.dump(linear, folder / "validation_linear.joblib")
        validation_coverage = compare(valid_data, loaded, linear, folder, "validation_2024")

        # Epoch count is fixed before the retrospective 2025 outcomes are used.
        retraining, refit_counts = select_records(features, [2021, 2022, 2023, 2024])
        refit_data = make_sequences(retraining, games)
        retrospective, refit_curve = lstm.fit(refit_data, options, epochs=bundle["epochs"])
        lstm.save(retrospective, folder / "retrospective_model")
        pl.DataFrame(refit_curve).write_csv(folder / "retrospective_training_curve.csv")
        linear = make_model().fit(model_inputs(retraining), retraining[TARGET].to_numpy())
        joblib.dump(linear, folder / "retrospective_linear.joblib")
        testing, test_counts = select_records(features, [2025])
        test_data = make_sequences(testing, games)
        save_sequences(test_data, folder, "retrospective")
        test_coverage = compare(test_data, retrospective, linear, folder, "retrospective_2025")
        report = {"created_at_utc": now.isoformat(), "options": options, "selected_epochs": bundle["epochs"],
                  "epochs_run": len(curve), "sequence_fields": SEQUENCE_FIELDS, "context_fields": CONTEXT_FIELDS,
                  "training_counts": train_counts, "validation_counts": valid_counts,
                  "lstm_training_rows": int((train_data["lengths"] > 0).sum()),
                  "refit_lstm_training_rows": int((refit_data["lengths"] > 0).sum()),
                  "refit_counts": refit_counts, "retrospective_counts": test_counts,
                  "validation_coverage": validation_coverage, "retrospective_coverage": test_coverage,
                  "versions": {name: version(name) for name in ["torch", "numpy", "scikit-learn"]},
                  "saved_input_sha256": {p.name: sha256(p.read_bytes()).hexdigest()
                                         for p in folder.iterdir() if p.suffix == ".npz" or p.name.endswith("_rows.parquet")},
                  "elapsed_seconds": perf_counter() - started,
                  "evaluation_note": "2024 development; 2025 retrospective, not a new untouched test. Future forecasts from the configured start date remain unevaluated.",
                  "future_policy": "Fixed architecture/settings/epochs; production training targets strictly before future_test_start. Prior completed games may still enter prediction sequences."}
        (folder / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        folder.rename(output)
    print(f"Selected epoch {bundle['epochs']}; saved Task 3 run: {output}", flush=True)
    return output


if __name__ == "__main__":
    run()
