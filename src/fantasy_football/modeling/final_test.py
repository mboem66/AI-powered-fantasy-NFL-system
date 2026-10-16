"""Record the fixed Linear Regression setup's final 2025 test once."""

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from tempfile import TemporaryDirectory
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from fantasy_football.config import PROJECT_ROOT
from fantasy_football.modeling.prepare import (
    IDENTIFIERS, NUMERIC_FEATURES, POSITIONS, TARGET, make_model, model_inputs, split_records,
)
from fantasy_football.modeling.train import evaluate

SELECTED_MODEL = "linear_regression"


def run(root=PROJECT_ROOT):
    output = root / "artifacts/task2/final_test"
    if output.exists():
        # Re-running must not silently replace the first held-out evaluation.
        report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        print("Final test already recorded; preserved results:", output)
        return report
    source = root / "data/processed/historical_features.parquet"
    frame = pl.scan_parquet(source).filter(pl.col("season").is_between(2021, 2025)).collect()
    training, testing, counts = split_records(frame, (2021, 2022, 2023, 2024), evaluation_season=2025)
    model = make_model(SELECTED_MODEL)
    model.fit(model_inputs(training), training[TARGET].to_numpy())
    predicted = model.predict(model_inputs(testing))
    if not np.isfinite(predicted).all():
        raise ValueError("Final-test predictions contain nonfinite values")
    predictions = testing.select(*IDENTIFIERS, TARGET).with_columns(
        testing["ppr_recent_avg"].alias("baseline_prediction"),
        pl.Series("linear_regression_prediction", predicted),
    ).with_columns([
        (pl.col(f"{name}_prediction") - pl.col(TARGET)).abs().alias(f"{name}_absolute_error")
        for name in ("baseline", SELECTED_MODEL)
    ])
    metrics = evaluate(predictions, ("baseline", SELECTED_MODEL))
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": SELECTED_MODEL, "model_settings": {}, "numeric_features": NUMERIC_FEATURES,
        "positions": POSITIONS, "training_seasons": [2021, 2022, 2023, 2024], "test_season": 2025,
        "records": counts, "training_latest_game": str(training["game_date"].max()),
        "source_sha256": sha256(source.read_bytes()).hexdigest(),
        "scikit_learn_version": version("scikit-learn"),
        "selection": "Selected before viewing 2025 results, based on 2024 validation and simplicity",
        "overall_metrics": metrics.filter(pl.col("position") == "overall").to_dicts(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish the directory only when every artifact has been written successfully.
    with TemporaryDirectory(dir=output.parent, prefix="final_test_pending_") as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir()
        predictions.write_parquet(staging / "predictions.parquet", compression="zstd")
        metrics.write_csv(staging / "metrics.csv")
        joblib.dump({"pipeline": model, "metadata": report}, staging / "model.joblib")
        (staging / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.rename(output)
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_rows=-1, float_precision=3):
        print(metrics)
    print(f"Training: {training.height:,}; final test: {testing.height:,}. Results: {output}")
    return report


if __name__ == "__main__":
    run()
