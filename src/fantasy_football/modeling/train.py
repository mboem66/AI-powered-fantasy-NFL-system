"""Compare baseline, Linear Regression, Random Forest, and XGBoost on 2024."""

from datetime import datetime, timezone
import argparse
from importlib.metadata import version
import json
from shutil import copy2
from time import perf_counter

import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fantasy_football.config import PROJECT_ROOT
from fantasy_football.data.storage import save_parquet
from fantasy_football.modeling.prepare import (
    IDENTIFIERS, NUMERIC_FEATURES, RECENT_FEATURES, POSITIONS, TARGET, TRAIN_SEASONS, VALIDATION_SEASON,
    make_model, model_inputs, split_records,
    METHODS, MODEL_SETTINGS,
)


def evaluate(predictions):
    """Score all four approaches on identical rows, overall and by position."""
    results = []
    for group in ["overall", *POSITIONS]:
        rows = predictions if group == "overall" else predictions.filter(pl.col("position") == group)
        if rows.is_empty():
            continue
        actual = rows[TARGET].to_numpy()
        for method in METHODS:
            predicted = rows[f"{method}_prediction"].to_numpy()
            results.append({
                "model": method, "position": group, "records": rows.height,
                "mae": float(mean_absolute_error(actual, predicted)),
                "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
                # R-squared is undefined for one outcome or a constant target.
                "r2": float(r2_score(actual, predicted)) if rows[TARGET].n_unique() > 1 else None,
            })
    return pl.DataFrame(results)


def backup_results(output):
    """Preserve existing results before replacing them after a successful fit."""
    existing = [output / name for name in
                ("metrics.csv", "validation_predictions.parquet", "run_summary.json")
                if (output / name).exists()]
    if not existing:
        return None
    folder = output / "backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True)
    for source in existing:
        copy2(source, folder / source.name)
    return folder


def run(root=PROJECT_ROOT, training_seasons=TRAIN_SEASONS, use_season_averages=True, output=None):
    started = perf_counter()
    features = NUMERIC_FEATURES if use_season_averages else RECENT_FEATURES
    source = root / "data/processed/historical_features.parquet"
    if not source.exists():
        raise FileNotFoundError("Build features first with: python -m fantasy_football.update")
    # 2025 is reserved for the final test; 2026 is not part of this experiment.
    frame = pl.scan_parquet(source).filter(
        pl.col("season").is_in([*training_seasons, VALIDATION_SEASON])
    ).collect()
    training, validation, counts = split_records(frame, training_seasons, features)
    x_train, x_validation = model_inputs(training, features), model_inputs(validation, features)
    predictions = validation.select(*IDENTIFIERS, TARGET).with_columns(
        validation["ppr_recent_avg"].alias("baseline_prediction"),
    )
    timings = {"baseline": 0.0}  # The baseline has no fitted estimator.
    for name in MODEL_SETTINGS:
        print(f"Fitting {name} on {list(training_seasons)}...")
        model = make_model(name, features)
        fit_started = perf_counter()
        # Do not pass validation outcomes to fitting or early stopping.
        model.fit(x_train, training[TARGET].to_numpy())
        timings[name] = perf_counter() - fit_started
        predictions = predictions.with_columns(
            pl.Series(f"{name}_prediction", model.predict(x_validation))
        )
    predictions = predictions.with_columns([
        (pl.col(f"{method}_prediction") - pl.col(TARGET)).abs().alias(f"{method}_absolute_error")
        for method in METHODS
    ])
    # Keep the raw predictions, including negatives; do not clip away errors.
    metrics = evaluate(predictions).with_columns(
        pl.col("model").replace_strict(timings, return_dtype=pl.Float64).alias("training_seconds")
    )
    output = root / (output or "artifacts/task2")
    output.mkdir(parents=True, exist_ok=True)
    backup = backup_results(output)
    save_parquet(predictions, output / "validation_predictions.parquet")
    metrics.write_csv(output / "metrics.csv")
    report = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source.relative_to(root)), "scikit_learn_version": version("scikit-learn"),
        "xgboost_version": version("xgboost"), "model_settings": MODEL_SETTINGS,
        "training_seasons": list(training_seasons), "validation_season": VALIDATION_SEASON,
        "reserved_test_season": 2025, "numeric_features": features,
        "season_averages": use_season_averages,
        "categorical_features": ["position"], "target": TARGET, "records": counts,
        "population": "Observed player-games with finite targets, prior PPR history and recent participation",
        "preprocessing": "Training-only medians; empty training columns use zero; position indicators with QB reference",
        "training_seconds": timings, "elapsed_seconds": perf_counter() - started,
        "previous_results_backup": str(backup.relative_to(root)) if backup else None,
    }
    (output / "run_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Training: {list(training_seasons)} ({training.height:,} records)")
    print(f"Numeric inputs: {len(features)}; season-to-date averages: {use_season_averages}")
    print(f"Validation: {VALIDATION_SEASON} ({validation.height:,} records); 2025 remains reserved.")
    for season, count in counts.items():
        print(f"{season}: excluded {count['available'] - count['included']:,} records {count['excluded']}")
    print("Training seconds include preprocessing and fitting; baseline requires no fit.")
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=-1, tbl_cols=-1, tbl_width_chars=120, float_precision=3):
        print(metrics)
    print(f"Results saved to {output}")
    if backup:
        print(f"Previous results preserved in {backup}")
    print("Comparison only: no winner selected, no 2025 test, and no upcoming-game predictions.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-start", type=int, choices=[2021, 2022, 2023], default=2021,
                        help="First training season; training always ends in 2023")
    parser.add_argument("--recent-only", action="store_true", help="Compare without season-to-date features")
    parser.add_argument("--output", default="artifacts/task2", help="Results folder relative to the project")
    args = parser.parse_args()
    run(training_seasons=tuple(range(args.train_start, 2024)),
        use_season_averages=not args.recent_only, output=args.output)
