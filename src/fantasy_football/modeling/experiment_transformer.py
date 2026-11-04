"""Compare original and deeper/wider Transformers with LSTM on 2024 data."""

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import tomllib

import numpy as np
import polars as pl

from fantasy_football.config import PROJECT_ROOT
from fantasy_football.modeling import lstm, transformer
from fantasy_football.modeling.prepare import IDENTIFIERS, TARGET, select_records
from fantasy_football.modeling.sequences import make_sequences, save_sequences
from fantasy_football.modeling.train import evaluate
from fantasy_football.modeling.train_lstm import load_history, plot_curve


def run(root=PROJECT_ROOT):
    with (root / "config/lstm.toml").open("rb") as stream:
        options = tomllib.load(stream)
    games, features = load_history(root)
    games = games.filter(pl.col("season").is_between(2021, 2024))
    training, train_counts = select_records(features, [2021, 2022, 2023])
    validation, valid_counts = select_records(features, [2024])
    train_data, valid_data = make_sequences(training, games), make_sequences(validation, games)
    now = datetime.now(timezone.utc)
    output = root / "artifacts/task3/experiments" / now.strftime("transformer_%Y%m%dT%H%M%S%fZ")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix="pending_") as temporary:
        folder = Path(temporary) / "result"
        folder.mkdir()
        save_sequences(train_data, folder, "training")
        save_sequences(valid_data, folder, "validation")
        predictions = validation.select(*IDENTIFIERS, TARGET).with_columns(pl.Series("sequence_length", valid_data["lengths"]))
        metrics, runs = [], {}
        # All candidates use the same data, seed, optimizer and stopping rule.
        for name, module, settings in [
            ("lstm", lstm, options),
            ("transformer", transformer, options | {"attention_heads": 4, "feedforward_size": 64, "dropout": 0.1}),
            ("transformer_deep_wide", transformer, options | {
                "num_layers": 2, "hidden_size": 64, "attention_heads": 4,
                "feedforward_size": 128, "dropout": 0.2,
            }),
        ]:
            print(f"Training {name} on 2021-2023; validating on 2024", flush=True)
            started = perf_counter()
            bundle, curve = module.fit(train_data, settings, validation=valid_data)
            seconds = perf_counter() - started
            model_folder = folder / name
            module.save(bundle, model_folder)
            raw = module.predict(bundle, valid_data)
            np.testing.assert_allclose(raw, module.predict(module.load(model_folder), valid_data), equal_nan=True)
            predictions = predictions.with_columns(pl.Series(f"{name}_prediction", raw).fill_nan(None))
            usable = predictions.filter(pl.col("sequence_length") > 0)
            metrics.append(evaluate(usable, [name]).with_columns(
                pl.lit(seconds).alias("training_seconds"), pl.lit(bundle["epochs"]).alias("selected_epoch")))
            pl.DataFrame(curve).write_csv(model_folder / "learning_curve.csv")
            plot_curve(curve, bundle["epochs"], model_folder / "learning_curve.png")
            runs[name] = {"options": settings, "selected_epochs": bundle["epochs"],
                          "epochs_run": len(curve), "training_seconds": seconds}
        comparison = pl.concat(metrics)
        comparison.write_csv(folder / "comparison.csv")
        predictions.write_parquet(folder / "validation_predictions.parquet", compression="zstd")
        report = {"created_at_utc": now.isoformat(), "runs": runs,
                  "training_counts": train_counts, "validation_counts": valid_counts,
                  "sequence_training_rows": int((train_data["lengths"] > 0).sum()),
                  "sequence_validation_rows": int((valid_data["lengths"] > 0).sum()),
                  "versions": {name: version(name) for name in ["torch", "numpy", "scikit-learn"]},
                  "input_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()
                                   if p.suffix == ".npz" or p.name.endswith("_rows.parquet")},
                  "note": "2024 development comparison, one seed. No 2025/2026 evaluation. Experimental models are not selected for weekly predictions."}
        (folder / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        folder.rename(output)
    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_cols=-1, tbl_width_chars=160, float_precision=3):
        print(comparison.filter(pl.col("position") == "overall"))
    print(f"Saved Transformer experiment: {output}")
    return output


if __name__ == "__main__":
    run()
