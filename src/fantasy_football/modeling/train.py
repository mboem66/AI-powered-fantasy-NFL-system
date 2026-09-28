"""Compare recent-average PPR with Linear Regression on a later season."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from time import perf_counter

import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fantasy_football.config import PROJECT_ROOT
from fantasy_football.data.storage import save_parquet
from fantasy_football.modeling.prepare import (
    IDENTIFIERS, NUMERIC_FEATURES, POSITIONS, TARGET, TRAIN_SEASON, VALIDATION_SEASON,
    make_model, model_inputs, split_records,
)


def evaluate(predictions):
    """Score both predictions on identical rows, overall and within each position."""
    results = []
    for group in ["overall", *POSITIONS]:
        rows = predictions if group == "overall" else predictions.filter(pl.col("position") == group)
        if rows.is_empty():
            continue
        actual = rows[TARGET].to_numpy()
        for method in ("baseline", "linear_regression"):
            predicted = rows[f"{method}_prediction"].to_numpy()
            results.append({
                "model": method, "position": group, "records": rows.height,
                "mae": float(mean_absolute_error(actual, predicted)),
                "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
                # R-squared is undefined for one outcome or a constant target.
                "r2": float(r2_score(actual, predicted)) if rows[TARGET].n_unique() > 1 else None,
            })
    return pl.DataFrame(results)


def run(root=PROJECT_ROOT):
    started = perf_counter()
    source = root / "data/processed/historical_features.parquet"
    if not source.exists():
        raise FileNotFoundError("Build features first with: python -m fantasy_football.update")
    # 2025 is reserved for the final test; 2026 is not part of this experiment.
    frame = pl.scan_parquet(source).filter(
        pl.col("season").is_in([TRAIN_SEASON, VALIDATION_SEASON])
    ).collect()
    training, validation, counts = split_records(frame)
    x_train, x_validation = model_inputs(training), model_inputs(validation)
    model = make_model()
    fit_started = perf_counter()
    model.fit(x_train, training[TARGET].to_numpy())
    fit_seconds = perf_counter() - fit_started
    predictions = validation.select(*IDENTIFIERS, TARGET).with_columns(
        validation["ppr_recent_avg"].alias("baseline_prediction"),
        pl.Series("linear_regression_prediction", model.predict(x_validation)),
    ).with_columns([
        (pl.col(f"{method}_prediction") - pl.col(TARGET)).abs().alias(f"{method}_absolute_error")
        for method in ("baseline", "linear_regression")
    ])
    # Keep the raw predictions, including negatives; do not clip away errors.
    metrics = evaluate(predictions)
    output = root / "artifacts/task2"
    output.mkdir(parents=True, exist_ok=True)
    save_parquet(predictions, output / "validation_predictions.parquet")
    metrics.write_csv(output / "metrics.csv")
    report = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source.relative_to(root)), "scikit_learn_version": version("scikit-learn"),
        "training_season": TRAIN_SEASON, "validation_season": VALIDATION_SEASON,
        "reserved_test_season": 2025, "numeric_features": NUMERIC_FEATURES,
        "categorical_features": ["position"], "target": TARGET, "records": counts,
        "population": "Observed player-games with finite targets, prior PPR history and recent participation",
        "preprocessing": "2023 medians; empty training columns use zero; position indicators with QB reference",
        "training_seconds": fit_seconds, "elapsed_seconds": perf_counter() - started,
    }
    (output / "run_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Training: {TRAIN_SEASON} ({training.height:,} records)")
    print(f"Validation: {VALIDATION_SEASON} ({validation.height:,} records); 2025 remains reserved.")
    for season, count in counts.items():
        print(f"{season}: excluded {count['available'] - count['included']:,} records {count['excluded']}")
    print(f"Training time (including preprocessing): {fit_seconds:.3f} seconds")
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=-1, tbl_cols=-1, tbl_width_chars=120, float_precision=3):
        print(metrics)
    print(f"Results saved to {output}")
    print("This is historical validation, not upcoming-game predictions. Re-running replaces these results.")
    return report


if __name__ == "__main__":
    run()
